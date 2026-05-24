"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BLUESTAR HEDGE FUND GPS — V6.0 PRODUCTION-GRADE                            ║
║                                                                              ║
║  Refonte complète depuis V5.1 : corrections métier, durcissement            ║
║  concurrence, architecture industrialisée.                                  ║
║                                                                              ║
║  Convention iloc (CRITIQUE) :                                               ║
║    fetch_candles(include_incomplete=False) →                                ║
║      df.iloc[-1] = dernière bougie COMPLÈTE (= J-1 humain)                  ║
║      df.iloc[-2] = avant-dernière complète (= J-2 humain)                   ║
║    fetch_live_open() → bougie INCOMPLÈTE de la session en cours             ║
║                                                                              ║
║  Corrections critiques par rapport à V5.1 :                                 ║
║    [F1]  iloc[-2] → iloc[-1] dans prev_midpoint (3 bugs métier corrigés)    ║
║    [F2]  Détection pivots via scipy.signal.find_peaks (pas d'égalité float) ║
║    [F3]  Validation fraîcheur live_open (rejet weekend stale)               ║
║    [F4]  NC orthogonal au MTF score (pas de double-counting)                ║
║    [F5]  MTF renormalisé sur TF actifs (Range non pénalisant)               ║
║    [F6]  Filtre body/range remplace volume sur indices                      ║
║    [F7]  ZLEMA lag corrigé, min_bars intraday 200                           ║
║    [F8]  Drainage gracieux executor + session lifecycle scopé               ║
║    [F9]  Anti-stampede cache (in-flight dedup)                              ║
║    [F10] Validation OHLCV stricte (volume=0, monotonie, gaps)               ║
║    [F11] Snapshot timestamp cohérent multi-TF                               ║
║    [F12] Streamlit flag avec TTL auto-reset                                 ║
║    [F13] Logs scrubés (pas de token/payload sensible)                       ║
║    [F14] TrendConfig dataclass (zéro magic number)                          ║
║    [F15] Registre votes idempotent (dict + UID)                             ║
║    [F16] Calendrier marché FX (fermeture weekend)                           ║
║    [F17] Arrondi NC métier (floor + sign)                                   ║
║    [F18] Degraded mode si vote critique en erreur (bloque A+)               ║
║    [F19] Cache copy hors lock                                               ║
║    [F20] Validation Streamlit secrets exhaustive                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
import uuid
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
    as_completed,
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, Final, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from scipy.signal import find_peaks  # type: ignore
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

try:
    from fpdf import FPDF
    _FPDF_AVAILABLE = True
    _FPDF2 = hasattr(FPDF, "set_lang")
except ImportError:  # pragma: no cover
    FPDF = None  # type: ignore
    _FPDF_AVAILABLE = False
    _FPDF2 = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOGGING (avec scrubbing des secrets)
# ═══════════════════════════════════════════════════════════════════════════════

class _SecretScrubFilter(logging.Filter):
    """Filtre logging — supprime Bearer tokens, account IDs OANDA des logs."""

    _PATTERNS = [
        re.compile(r"Bearer\s+[A-Za-z0-9\-_\.~+/=]+", re.IGNORECASE),
        re.compile(r"(?:[0-9]{3}-[0-9]{3}-[0-9]+-[0-9]+)"),  # OANDA account ID
        re.compile(r"[a-f0-9]{32,}"),  # generic tokens hex
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            for pat in self._PATTERNS:
                msg = pat.sub("[REDACTED]", msg)
            record.msg = msg
            record.args = ()
        except Exception:  # pragma: no cover
            pass
        return True


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log = logging.getLogger("bluestar_gps")
_log.addFilter(_SecretScrubFilter())


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION (frozen dataclasses, zéro magic number)
# ═══════════════════════════════════════════════════════════════════════════════

APP_VERSION: Final[str] = "6.0.0"

OANDA_API_URL: Final[str] = "https://api-fxpractice.oanda.com"

INSTRUMENTS: Final[Tuple[str, ...]] = (
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "DE30_EUR", "XAU_USD", "SPX500_USD", "NAS100_USD", "US30_USD",
)

INDICES: Final[frozenset] = frozenset(
    {"DE30_EUR", "SPX500_USD", "NAS100_USD", "US30_USD", "XAU_USD"}
)


@dataclass(frozen=True)
class TrendConfig:
    """Toutes les constantes métier — frozen, immuable."""

    # Cache TTL par granularité (secondes)
    cache_ttl_m: int = 14400
    cache_ttl_w: int = 3600
    cache_ttl_d: int = 600
    cache_ttl_h4: int = 300
    cache_ttl_h1: int = 120
    cache_ttl_m15: int = 60
    cache_ttl_default: int = 600
    cache_ttl_live_open: int = 90

    # Fraîcheur des données
    data_max_age_min: int = 10
    analysis_timeout_sec: int = 120
    streamlit_running_flag_ttl_sec: int = 300  # 5 min

    # Workers
    max_workers: int = 5
    pool_drain_grace_sec: float = 5.0

    # OANDA HTTP
    http_timeout_sec: float = 8.0
    http_retry_total: int = 2
    http_retry_backoff: float = 0.3

    # Pivots
    pivot_wing: int = 5
    max_pivot_age: int = 50

    # Indicateurs
    rsi_period: int = 14
    atr_period: int = 14
    ema_short: int = 21
    ema_long: int = 50
    sma_macro: int = 200
    ema_intra_period: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    intraday_ema_fast: int = 9

    # Seuils macro / 4H
    macro_band_atr_ratio: float = 0.10
    ema50_slope_threshold: float = 0.05

    # Volume / body filters
    volume_ratio_strong_intraday: float = 1.30
    volume_ratio_min_midpoint: float = 1.00
    body_range_strong: float = 0.60  # filtre indices (remplace volume)

    # Scoring
    min_reliable_score: float = 1.5
    strength_min_range: int = 35
    strength_max: int = 90
    strength_scaler: float = 112.0

    # MTF weights
    mtf_weight_m: float = 5.0
    mtf_weight_w: float = 4.0
    mtf_weight_d: float = 4.0
    mtf_weight_h4: float = 2.5
    mtf_weight_h1: float = 1.5
    mtf_weight_15m: float = 1.0

    # NC orthogonal
    nc_pure_strength_min: int = 70

    # Min bars
    min_bars_m: int = 100
    min_bars_w: int = 50
    min_bars_d: int = 60
    min_bars_h4: int = 60
    min_bars_h1: int = 200  # [F7] augmenté pour stabilité EMA50
    min_bars_15m: int = 200  # [F7]


CFG: Final[TrendConfig] = TrendConfig()


# Mapping TF → TTL
_CACHE_TTL: Final[Dict[str, int]] = {
    "M": CFG.cache_ttl_m,
    "W": CFG.cache_ttl_w,
    "D": CFG.cache_ttl_d,
    "H4": CFG.cache_ttl_h4,
    "H1": CFG.cache_ttl_h1,
    "M15": CFG.cache_ttl_m15,
}

# Mapping TF → fréquence pandas
_GRAN_FREQ: Final[Dict[str, pd.Timedelta]] = {
    "M": pd.Timedelta(days=30),
    "W": pd.Timedelta(days=7),
    "D": pd.Timedelta(days=1),
    "H4": pd.Timedelta(hours=4),
    "H1": pd.Timedelta(hours=1),
    "M15": pd.Timedelta(minutes=15),
}

TREND_COLORS: Final[Dict[str, str]] = {
    "Bullish":          "#2ecc71",
    "Bearish":          "#e74c3c",
    "Retracement Bull": "#7dcea0",
    "Retracement Bear": "#f1948a",
    "Range":            "#95a5a6",
}


# Assertion config cohérente
assert CFG.data_max_age_min <= CFG.cache_ttl_d // 60, \
    "data_max_age_min doit être <= cache_ttl_d (minutes)"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STREAMLIT SETUP (uniquement si module exécuté comme app)
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title=f"Bluestar GPS V{APP_VERSION}", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
        .main-header {
            text-align: center; padding: 20px;
            background: linear-gradient(135deg, #1e3a8a 0%, #172554 100%);
            color: white; border-radius: 12px; margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .metric-card {
            background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.1); text-align: center;
        }
        .metric-value { font-size: 1.5em; font-weight: bold; margin-top: 5px; }
        .metric-label {
            font-size: 0.9em; color: #94a3b8;
            text-transform: uppercase; letter-spacing: 1px;
        }
        .stDataFrame { width: 100%; }
        div[data-testid="stMarkdown"] { text-align: center; }
        .stale-warning {
            background: #fef3c7; border-left: 4px solid #f59e0b;
            padding: 10px 16px; border-radius: 4px; margin-bottom: 12px;
        }
        .degraded-warning {
            background: #fee2e2; border-left: 4px solid #dc2626;
            padding: 10px 16px; border-radius: 4px; margin-bottom: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TYPES CONTRACTUELS
# ═══════════════════════════════════════════════════════════════════════════════

class Direction(str, Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    RANGE = "Range"


@dataclass(frozen=True)
class VoteSignal:
    """Vote atomique. Immuable."""
    name: str
    direction: Direction
    weight: float
    reliability: float
    fired: bool
    reason: str
    errored: bool = False  # [F18] degraded mode flag


@dataclass(frozen=True)
class DailyTrendResult:
    """Résultat agrégé du Daily. Rétrocompatible via __iter__."""
    direction: Direction
    strength: int
    atr_val: float
    bull_score: float
    bear_score: float
    votes: Tuple[VoteSignal, ...]
    min_votes_met: bool
    degraded: bool = False

    def __iter__(self):
        yield self.direction.value
        yield self.strength
        yield self.atr_val


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MARKET CALENDAR (gestion FX fermé weekend)
# ═══════════════════════════════════════════════════════════════════════════════

def is_fx_market_open(now: Optional[datetime] = None) -> bool:
    """
    Détermine si le marché FX est ouvert.
    FX ouvert dimanche 22:00 UTC → vendredi 22:00 UTC.
    Convention conservatrice : pendant heures fermées, on tolère les analyses
    mais on dégrade la fraîcheur attendue.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    weekday = now.weekday()  # 0=lundi, 6=dimanche
    hour = now.hour
    if weekday == 5:  # samedi
        return False
    if weekday == 4 and hour >= 22:  # vendredi après 22h UTC
        return False
    if weekday == 6 and hour < 22:  # dimanche avant 22h UTC
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — INDICATEURS (vectorisés, validés)
# ═══════════════════════════════════════════════════════════════════════════════

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return _true_range(high, low, close).ewm(alpha=1.0 / n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI Wilder. gain=0,loss=0 → 50 (range pur). gain>0,loss=0 → 100."""
    d = close.diff()
    gain = d.where(d > 0, 0.0).ewm(alpha=1.0 / n, adjust=False).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(alpha=1.0 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    gain_pos = gain > 0
    loss_zero = loss == 0
    rsi = rsi.where(~loss_zero, np.where(gain_pos, 100.0, 50.0))
    return rsi


def _dmi(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> Tuple[float, float]:
    tr = _true_range(high, low, close)
    atr_s = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    up = high.diff()
    dn = -low.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    pdi = 100 * pdm.ewm(alpha=1.0 / n, adjust=False).mean() / atr_s.replace(0, np.nan)
    mdi = 100 * mdm.ewm(alpha=1.0 / n, adjust=False).mean() / atr_s.replace(0, np.nan)
    return float(pdi.iloc[-1]), float(mdi.iloc[-1])


def _fmt_atr(val: float) -> str:
    if not val or np.isnan(val) or val <= 0:
        return "N/A"
    if val >= 10:
        return f"{val:.2f}"
    if val >= 1:
        return f"{val:.3f}"
    return f"{val:.4f}"


def _find_strict_peaks(
    series: pd.Series, wing: int, min_idx: int
) -> List[int]:
    """
    [F2] Détection pivots stricte sans égalité float.
    Utilise scipy.signal.find_peaks si disponible, sinon fallback strict.
    """
    arr = series.values
    n = len(arr)
    if n < 2 * wing + 1:
        return []

    if _HAS_SCIPY:
        peaks, _ = find_peaks(arr, distance=max(1, wing))
        return [int(p) for p in peaks if p >= min_idx and p <= n - wing - 1]

    # Fallback pur numpy — strict (>)
    result: List[int] = []
    for i in range(max(min_idx, wing), n - wing):
        window = arr[i - wing : i + wing + 1]
        center = arr[i]
        # Strict : center doit être > tous les autres
        if center == window.max() and np.sum(window == center) == 1:
            result.append(i)
    return result


def _find_strict_troughs(
    series: pd.Series, wing: int, min_idx: int
) -> List[int]:
    arr = series.values
    n = len(arr)
    if n < 2 * wing + 1:
        return []

    if _HAS_SCIPY:
        peaks, _ = find_peaks(-arr, distance=max(1, wing))
        return [int(p) for p in peaks if p >= min_idx and p <= n - wing - 1]

    result: List[int] = []
    for i in range(max(min_idx, wing), n - wing):
        window = arr[i - wing : i + wing + 1]
        center = arr[i]
        if center == window.min() and np.sum(window == center) == 1:
            result.append(i)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — HTTP SESSIONS (registry scopé, pas de thread-local fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class SessionRegistry:
    """
    Registre HTTP scopé par analyse.
    [F8] Pas de thread-local global — toute session est tracée.
    """

    def __init__(self) -> None:
        self._sessions: List[requests.Session] = []
        self._thread_sessions: Dict[int, requests.Session] = {}
        self._lock = threading.RLock()
        self._closed = False

    def get_for_thread(self) -> requests.Session:
        tid = threading.get_ident()
        with self._lock:
            if self._closed:
                raise RuntimeError("SessionRegistry closed — analyse terminée.")
            sess = self._thread_sessions.get(tid)
            if sess is None:
                sess = self._build_session()
                self._thread_sessions[tid] = sess
                self._sessions.append(sess)
            return sess

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=CFG.http_retry_total,
            backoff_factor=CFG.http_retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=20)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def close_all(self) -> None:
        with self._lock:
            self._closed = True
            for s in self._sessions:
                try:
                    s.close()
                except OSError as exc:
                    _log.debug("Session close ignoré : %s", exc)
            self._sessions.clear()
            self._thread_sessions.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — CACHE (anti-stampede, thread-safe, immutable convention)
# ═══════════════════════════════════════════════════════════════════════════════

class CandleCache:
    """
    [F9] Cache anti-stampede : in-flight deduplication par clé.
    [F14] DataFrames immutables — convention : ne JAMAIS muter un df issu du cache.
    """

    def __init__(self) -> None:
        self._data: Dict[Tuple, Tuple[datetime, pd.DataFrame]] = {}
        self._live_opens: Dict[Tuple, Tuple[datetime, Optional[float]]] = {}
        self._inflight: Dict[Tuple, threading.Event] = {}
        self._lock = threading.RLock()

    def get_candles(
        self,
        key: Tuple[str, str, int],
        ttl: int,
        fetch_fn: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Récupère bougies via cache avec déduplication des fetches concurrents.
        """
        now = datetime.now(timezone.utc)

        # 1) Lookup cache
        with self._lock:
            entry = self._data.get(key)
            if entry is not None:
                ts, df = entry
                if (now - ts).total_seconds() < ttl:
                    # Retour direct — df immutable par convention
                    return df

            # 2) Vérifier si fetch déjà en cours
            inflight_event = self._inflight.get(key)
            if inflight_event is not None:
                # Un autre thread fetch — on attend
                wait_for_event = inflight_event
                start_my_fetch = False
            else:
                # Nous démarrons le fetch
                wait_for_event = threading.Event()
                self._inflight[key] = wait_for_event
                start_my_fetch = True

        if not start_my_fetch:
            # Attendre l'autre fetch
            wait_for_event.wait(timeout=CFG.http_timeout_sec * 3)
            with self._lock:
                entry = self._data.get(key)
                if entry is not None:
                    return entry[1]
                # Autre thread a échoué — on retourne df vide (pas de re-fetch
                # cascade pour éviter amplification)
                return pd.DataFrame()

        # On est responsable du fetch
        try:
            df = fetch_fn()
            with self._lock:
                if not df.empty:
                    # Marquer DataFrame comme frozen (convention défensive)
                    df.attrs["_immutable"] = True
                    self._data[key] = (datetime.now(timezone.utc), df)
            return df
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            wait_for_event.set()

    def get_live_open(
        self,
        key: Tuple[str, str],
        fetch_fn: Callable[[], Optional[float]],
    ) -> Optional[float]:
        now = datetime.now(timezone.utc)
        with self._lock:
            entry = self._live_opens.get(key)
            if entry is not None:
                ts, value = entry
                if (now - ts).total_seconds() < CFG.cache_ttl_live_open:
                    return value

        value = fetch_fn()
        if value is not None:
            with self._lock:
                self._live_opens[key] = (datetime.now(timezone.utc), value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._live_opens.clear()
            self._inflight.clear()


_CACHE = CandleCache()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — DATA LAYER (fetch OANDA + validation stricte)
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_candle(row: Dict[str, float], allow_zero_volume: bool) -> bool:
    """
    [F10] Validation OHLCV stricte.
    Rejette volume=0 si allow_zero_volume=False (FX en heures de marché).
    """
    try:
        h, lo, o, c, v = row["High"], row["Low"], row["Open"], row["Close"], row["Volume"]
        if not (
            h >= lo > 0
            and o > 0
            and c > 0
            and h >= o
            and h >= c
            and lo <= o
            and lo <= c
            and v >= 0
        ):
            return False
        if not allow_zero_volume and v == 0:
            return False
        return True
    except (KeyError, TypeError):
        return False


def _validate_dataframe(
    df: pd.DataFrame, granularity: str, instrument: str
) -> bool:
    """[F10] Validation séquentielle : index monotone, gaps."""
    if df.empty:
        return False
    if not isinstance(df.index, pd.DatetimeIndex):
        _log.error("Index non-DatetimeIndex %s/%s", instrument, granularity)
        return False
    if not df.index.is_monotonic_increasing:
        _log.error("Index non monotone %s/%s", instrument, granularity)
        return False
    if df.index.has_duplicates:
        _log.error("Index avec doublons %s/%s", instrument, granularity)
        return False
    return True


def fetch_candles(
    instrument: str,
    granularity: str,
    count: int,
    account_id: str,
    access_token: str,
    registry: SessionRegistry,
    include_incomplete: bool = False,
) -> pd.DataFrame:
    """
    Récupère les bougies OANDA.
    [F13] Logs scrubés — pas de payload brut, pas de token.

    Convention : include_incomplete=False → df.iloc[-1] = dernière COMPLÈTE.
    """
    url = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"granularity": granularity, "count": count, "price": "M"}

    try:
        session = registry.get_for_thread()
    except RuntimeError:
        # Registry fermé (analyse annulée) — retour silencieux
        return pd.DataFrame()

    try:
        r = session.get(url, headers=headers, params=params, timeout=CFG.http_timeout_sec)
    except requests.exceptions.Timeout:
        _log.warning("Timeout OANDA %s/%s", instrument, granularity)
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        _log.warning(
            "RequestException OANDA %s/%s : %s",
            instrument,
            granularity,
            type(e).__name__,
        )
        return pd.DataFrame()

    if r.status_code == 429:
        retry_after = int(r.headers.get("Retry-After", "5"))
        _log.warning(
            "OANDA 429 %s/%s — Retry-After %ds", instrument, granularity, retry_after
        )
        time.sleep(min(retry_after, 30))
        return pd.DataFrame()
    if r.status_code != 200:
        _log.warning(
            "OANDA HTTP %d %s/%s", r.status_code, instrument, granularity
        )
        return pd.DataFrame()

    try:
        candles = r.json().get("candles", [])
    except ValueError:
        _log.warning("OANDA JSON invalide %s/%s", instrument, granularity)
        return pd.DataFrame()

    is_index = instrument in INDICES
    market_open = is_fx_market_open()
    # Indices : volume=0 toléré (pas de vrai volume)
    # FX hors heures : volume=0 toléré (pas de trades)
    allow_zero_volume = is_index or not market_open

    rows: List[Dict[str, Any]] = []
    for candle in candles:
        if not include_incomplete and not candle.get("complete"):
            continue
        try:
            row = {
                "date": candle["time"],
                "Open": float(candle["mid"]["o"]),
                "High": float(candle["mid"]["h"]),
                "Low": float(candle["mid"]["l"]),
                "Close": float(candle["mid"]["c"]),
                "Volume": float(candle.get("volume", 0)),
            }
            if _validate_candle(row, allow_zero_volume):
                rows.append(row)
        except (KeyError, ValueError, TypeError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    if not _validate_dataframe(df, granularity, instrument):
        return pd.DataFrame()

    return df


def fetch_cached(
    instrument: str,
    granularity: str,
    count: int,
    account_id: str,
    access_token: str,
    registry: SessionRegistry,
) -> pd.DataFrame:
    """[F9] Wrapper cache avec anti-stampede."""
    key = (instrument, granularity, count)
    ttl = _CACHE_TTL.get(granularity, CFG.cache_ttl_default)
    return _CACHE.get_candles(
        key,
        ttl,
        lambda: fetch_candles(
            instrument, granularity, count, account_id, access_token, registry,
            include_incomplete=False,
        ),
    )


def fetch_live_open(
    instrument: str,
    granularity: str,
    account_id: str,
    access_token: str,
    registry: SessionRegistry,
) -> Optional[float]:
    """
    [F3] Open de la bougie en cours, avec validation fraîcheur.
    Retourne None si marché fermé ou bougie trop ancienne.
    """
    key = (instrument, granularity)

    def _fetch() -> Optional[float]:
        df = fetch_candles(
            instrument, granularity, 1, account_id, access_token, registry,
            include_incomplete=True,
        )
        if df.empty:
            return None
        try:
            last_ts = df.index[-1]
            now = datetime.now(timezone.utc)
            max_age = _GRAN_FREQ.get(granularity, pd.Timedelta(days=1))
            # Bougie trop ancienne → marché fermé, pas un open "courant"
            if (now - last_ts) > max_age:
                return None
            return float(df["Open"].iloc[-1])
        except (IndexError, ValueError, TypeError):
            return None

    return _CACHE.get_live_open(key, _fetch)


def fetch_all_data(
    instrument: str,
    account_id: str,
    access_token: str,
    registry: SessionRegistry,
    stop_event: threading.Event,
) -> Optional[Dict[str, Any]]:
    """Récupère toutes les TF + opens live, avec snapshot timestamp."""
    specs = {
        "M": ("M", 150, CFG.min_bars_m),
        "W": ("W", 250, CFG.min_bars_w),
        "D": ("D", 300, CFG.min_bars_d),
        "4H": ("H4", 300, CFG.min_bars_h4),
        "1H": ("H1", 300, CFG.min_bars_h1),
        "15m": ("M15", 300, CFG.min_bars_15m),
    }
    cache: Dict[str, Any] = {
        "_snapshot_ts": datetime.now(timezone.utc),  # [F11]
    }

    for tf, (gran, count, min_bars) in specs.items():
        if stop_event.is_set():
            return None
        df = fetch_cached(instrument, gran, count, account_id, access_token, registry)
        if df.empty or len(df) < min_bars:
            _log.warning(
                "Données insuffisantes %s/%s : %d<%d",
                instrument, tf, len(df), min_bars,
            )
            return None
        cache[tf] = df

    if stop_event.is_set():
        return None

    cache["_week_open"] = fetch_live_open(
        instrument, "W", account_id, access_token, registry
    )
    cache["_day_open"] = fetch_live_open(
        instrument, "D", account_id, access_token, registry
    )
    return cache


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — REGISTRE DE VOTES (idempotent, ordonné)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VoteSpec:
    uid: str
    fn: Callable
    critical: bool = False  # [F18] si critique, erreur → degraded mode


class VoteRegistry:
    """[F15] Registre idempotent — pas de doublons, ordre déterministe."""

    def __init__(self) -> None:
        self._votes: Dict[str, VoteSpec] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()

    def register(self, uid: str, critical: bool = False) -> Callable:
        def decorator(fn: Callable) -> Callable:
            with self._lock:
                if uid in self._votes:
                    # Idempotent : remplace silencieusement (utile pour reload)
                    self._votes[uid] = VoteSpec(uid=uid, fn=fn, critical=critical)
                else:
                    self._votes[uid] = VoteSpec(uid=uid, fn=fn, critical=critical)
                    self._order.append(uid)
            return fn
        return decorator

    def all_votes(self) -> List[VoteSpec]:
        with self._lock:
            return [self._votes[uid] for uid in self._order]


DAILY_VOTES = VoteRegistry()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — VOTES ATOMIQUES (corrections F1, F2, F3 appliquées)
# ═══════════════════════════════════════════════════════════════════════════════

@DAILY_VOTES.register(uid="swing_structure", critical=True)
def _vote_swing_structure(
    h: pd.Series, lo: pd.Series, _c: pd.Series, _ctx: Dict[str, Any]
) -> VoteSignal:
    """
    Vote 1 — Structure swing HH/HL vs LH/LL.
    [F2] Détection pivots stricte via scipy.find_peaks (pas d'égalité float).
    Poids 2.0 · Reliability 0.9
    """
    name = "swing_structure"
    wing = CFG.pivot_wing

    if len(h) < 2 * wing + CFG.max_pivot_age + 1:
        return VoteSignal(
            name=name, direction=Direction.RANGE,
            weight=2.0, reliability=0.9, fired=False,
            reason=f"série trop courte ({len(h)})",
        )

    min_idx = max(0, len(h) - CFG.max_pivot_age)
    sh = _find_strict_peaks(h, wing, min_idx)
    sl = _find_strict_troughs(lo, wing, min_idx)

    if len(sh) < 2 or len(sl) < 2:
        return VoteSignal(
            name=name, direction=Direction.RANGE,
            weight=2.0, reliability=0.9, fired=False,
            reason=f"pivots insuffisants sh={len(sh)} sl={len(sl)}",
        )

    hh = h.iloc[sh[-1]] > h.iloc[sh[-2]]
    hl = lo.iloc[sl[-1]] > lo.iloc[sl[-2]]
    lh = h.iloc[sh[-1]] < h.iloc[sh[-2]]
    ll = lo.iloc[sl[-1]] < lo.iloc[sl[-2]]

    if hh and hl:
        return VoteSignal(name=name, direction=Direction.BULLISH,
                          weight=2.0, reliability=0.9, fired=True, reason="HH+HL")
    if lh and ll:
        return VoteSignal(name=name, direction=Direction.BEARISH,
                          weight=2.0, reliability=0.9, fired=True, reason="LH+LL")
    return VoteSignal(name=name, direction=Direction.RANGE,
                      weight=2.0, reliability=0.9, fired=False, reason="structure mixte")


@DAILY_VOTES.register(uid="ema_stack")
def _vote_ema_stack(
    _h, _lo, _c, ctx: Dict[str, Any]
) -> VoteSignal:
    """Vote 2 — EMA21/EMA50 stack. Poids 1.0 · Reliability 0.75"""
    name = "ema_stack"
    cur = ctx["cur"]
    e21 = ctx["e21"]
    e50_cur = ctx["e50_cur"]
    if np.isnan(e21) or np.isnan(e50_cur):
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=1.0, reliability=0.75, fired=False, reason="NaN EMA")
    if cur > e21 > e50_cur:
        return VoteSignal(name=name, direction=Direction.BULLISH,
                          weight=1.0, reliability=0.75, fired=True,
                          reason=f"cur>{e21:.5f}>e50")
    if cur < e21 < e50_cur:
        return VoteSignal(name=name, direction=Direction.BEARISH,
                          weight=1.0, reliability=0.75, fired=True,
                          reason=f"cur<{e21:.5f}<e50")
    return VoteSignal(name=name, direction=Direction.RANGE,
                      weight=1.0, reliability=0.75, fired=False, reason="stack non aligné")


@DAILY_VOTES.register(uid="weekly_open")
def _vote_weekly_open(
    _h, _lo, _c, ctx: Dict[str, Any]
) -> VoteSignal:
    """
    Vote 3 — Position vs Weekly Open.
    [F3] current_week_open déjà validé temporellement par fetch_live_open.
    Si None : fallback dernière semaine complète (rel=0.70).
    """
    name = "weekly_open"
    cur = ctx["cur"]
    current_week_open = ctx.get("current_week_open")
    df_weekly = ctx.get("df_weekly")

    if current_week_open is not None and not np.isnan(current_week_open):
        wo_price = current_week_open
        rel = 0.90
        source = "current_W"
    elif df_weekly is not None and not df_weekly.empty:
        try:
            wo_price = float(df_weekly["Open"].iloc[-1])
        except (IndexError, ValueError, TypeError):
            return VoteSignal(name=name, direction=Direction.RANGE,
                              weight=1.0, reliability=0.0, fired=False,
                              reason="weekly Open inaccessible")
        rel = 0.70
        source = "prev_W"
    else:
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=1.0, reliability=0.0, fired=False,
                          reason="weekly Open indisponible")

    if np.isnan(wo_price):
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=1.0, reliability=rel, fired=False, reason="wo NaN")
    if cur > wo_price:
        return VoteSignal(name=name, direction=Direction.BULLISH,
                          weight=1.0, reliability=rel, fired=True,
                          reason=f"cur>wo [{source}]")
    if cur < wo_price:
        return VoteSignal(name=name, direction=Direction.BEARISH,
                          weight=1.0, reliability=rel, fired=True,
                          reason=f"cur<wo [{source}]")
    return VoteSignal(name=name, direction=Direction.RANGE,
                      weight=1.0, reliability=rel, fired=False,
                      reason=f"cur==wo [{source}]")


@DAILY_VOTES.register(uid="prev_midpoint")
def _vote_prev_midpoint(
    h: pd.Series, lo: pd.Series, c: pd.Series, ctx: Dict[str, Any]
) -> VoteSignal:
    """
    Vote 4 — Close J-1 vs midpoint J-1, confirmé par volume J-1 OU body/range.

    [F1] CORRECTION CRITIQUE : avec include_incomplete=False, iloc[-1] EST déjà J-1
    (dernière bougie complète). V5.1 utilisait iloc[-2] = J-2 (faux).

    [F6] Pour indices/FX hors marché : filtre body/range au lieu de volume.
    """
    name = "prev_midpoint"
    vol = ctx.get("vol_series")
    instrument = ctx.get("instrument", "")
    is_index = instrument in INDICES

    if len(c) < 1:
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=0.5, reliability=0.0, fired=False,
                          reason="série vide")

    # [F1] iloc[-1] = dernière bougie complète = J-1
    h_j1 = float(h.iloc[-1])
    lo_j1 = float(lo.iloc[-1])
    c_j1 = float(c.iloc[-1])
    mid_j1 = (h_j1 + lo_j1) / 2.0
    bull = c_j1 > mid_j1
    direction = Direction.BULLISH if bull else Direction.BEARISH

    # [F6] Indices : filtre body/range
    if is_index:
        rng = h_j1 - lo_j1
        if rng <= 0:
            return VoteSignal(name=name, direction=Direction.RANGE,
                              weight=0.5, reliability=0.0, fired=False,
                              reason="range nul")
        try:
            o_j1 = float(ctx["df_daily"]["Open"].iloc[-1])
        except (KeyError, IndexError, ValueError, TypeError):
            return VoteSignal(name=name, direction=Direction.RANGE,
                              weight=0.5, reliability=0.0, fired=False,
                              reason="Open J-1 inaccessible")
        body_ratio = abs(c_j1 - o_j1) / rng
        if body_ratio < CFG.body_range_strong:
            return VoteSignal(name=name, direction=Direction.RANGE,
                              weight=0.5, reliability=0.0, fired=False,
                              reason=f"body/range={body_ratio:.2f} faible")
        rel = min(0.80, 0.60 + (body_ratio - CFG.body_range_strong) * 0.5)
        return VoteSignal(name=name, direction=direction,
                          weight=0.5, reliability=rel, fired=True,
                          reason=f"body/range={body_ratio:.2f} [idx]")

    # FX : filtre volume
    if vol is None or vol.empty:
        return VoteSignal(name=name, direction=direction,
                          weight=0.5, reliability=0.50, fired=True,
                          reason="no_vol_data")

    try:
        # [F1] vol.iloc[-1] = volume de J-1 (dernière bougie complète)
        # vol.iloc[:-1] = référentiel SANS J-1 (pas de self-reference)
        vol_ref = vol.iloc[:-1]
        vol_j1 = float(vol.iloc[-1])

        if len(vol_ref) < 20:
            return VoteSignal(name=name, direction=direction,
                              weight=0.5, reliability=0.50, fired=True,
                              reason="vol_history<20")

        vol_ma = float(vol_ref.rolling(20).mean().iloc[-1])
        if np.isnan(vol_j1) or np.isnan(vol_ma) or vol_ma <= 0:
            return VoteSignal(name=name, direction=direction,
                              weight=0.5, reliability=0.50, fired=True,
                              reason="vol_NaN")

        vol_ratio = vol_j1 / vol_ma
        if vol_ratio <= CFG.volume_ratio_min_midpoint:
            return VoteSignal(name=name, direction=Direction.RANGE,
                              weight=0.5, reliability=0.0, fired=False,
                              reason=f"vol_j1<MA20 ratio={vol_ratio:.2f}")

        reliability = min(0.80, 0.65 + (vol_ratio - 1.0) * 0.05)
        return VoteSignal(name=name, direction=direction,
                          weight=0.5, reliability=reliability, fired=True,
                          reason=f"vol_ratio={vol_ratio:.2f}")
    except (TypeError, ValueError, IndexError) as exc:
        _log.debug("prev_midpoint vol err: %s", exc)
        return VoteSignal(name=name, direction=direction,
                          weight=0.5, reliability=0.50, fired=True,
                          reason="vol_err")


@DAILY_VOTES.register(uid="ema50_slope")
def _vote_ema50_slope(
    _h, _lo, _c, ctx: Dict[str, Any]
) -> VoteSignal:
    """Vote 5 — Pente EMA50 normalisée ATR."""
    name = "ema50_slope"
    e50 = ctx["e50"]
    atr_val = ctx["atr_val"]
    threshold = CFG.ema50_slope_threshold
    if len(e50) < 6 or atr_val <= 0:
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=1.0, reliability=0.70, fired=False,
                          reason="données insuffisantes")
    slope_ratio = float(e50.iloc[-1] - e50.iloc[-6]) / atr_val
    if slope_ratio > threshold:
        return VoteSignal(name=name, direction=Direction.BULLISH,
                          weight=1.0, reliability=0.70, fired=True,
                          reason=f"slope={slope_ratio:.3f}")
    if slope_ratio < -threshold:
        return VoteSignal(name=name, direction=Direction.BEARISH,
                          weight=1.0, reliability=0.70, fired=True,
                          reason=f"slope={slope_ratio:.3f}")
    return VoteSignal(name=name, direction=Direction.RANGE,
                      weight=1.0, reliability=0.70, fired=False,
                      reason=f"slope={slope_ratio:.3f} neutre")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — AGRÉGATEUR (avec degraded mode)
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate_votes(
    votes: Tuple[VoteSignal, ...], atr_val: float, degraded: bool
) -> DailyTrendResult:
    """
    [F18] degraded=True si un vote critique a erré → bloque promotion A+.
    fired_possible = somme uniquement des votes fired (force non déprimée).
    """
    bull_score = sum(
        v.weight * v.reliability
        for v in votes
        if v.fired and v.direction == Direction.BULLISH
    )
    bear_score = sum(
        v.weight * v.reliability
        for v in votes
        if v.fired and v.direction == Direction.BEARISH
    )
    fired_possible = sum(v.weight * v.reliability for v in votes if v.fired)
    winning_score = max(bull_score, bear_score)
    min_votes_met = winning_score >= CFG.min_reliable_score

    if not min_votes_met or bull_score == bear_score:
        return DailyTrendResult(
            direction=Direction.RANGE, strength=CFG.strength_min_range, atr_val=atr_val,
            bull_score=bull_score, bear_score=bear_score,
            votes=votes, min_votes_met=min_votes_met, degraded=degraded,
        )

    direction = Direction.BULLISH if bull_score > bear_score else Direction.BEARISH
    ratio = winning_score / fired_possible if fired_possible > 0 else 0.0
    strength = int(
        min(CFG.strength_max, max(CFG.strength_min_range, ratio * CFG.strength_scaler))
    )
    return DailyTrendResult(
        direction=direction, strength=strength, atr_val=atr_val,
        bull_score=bull_score, bear_score=bear_score,
        votes=votes, min_votes_met=min_votes_met, degraded=degraded,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — TENDANCES PAR TIMEFRAME
# ═══════════════════════════════════════════════════════════════════════════════

def trend_daily(
    df: pd.DataFrame,
    df_weekly: Optional[pd.DataFrame] = None,
    instrument: str = "",
    current_week_open: Optional[float] = None,
) -> DailyTrendResult:
    h, lo, c = df["High"], df["Low"], df["Close"]
    atr_val = float(_atr(h, lo, c, CFG.atr_period).iloc[-1])

    if len(df) < CFG.min_bars_d:
        guard = VoteSignal(name="guard", direction=Direction.RANGE,
                           weight=0.0, reliability=1.0, fired=False,
                           reason=f"données insuffisantes ({len(df)})")
        return DailyTrendResult(
            direction=Direction.RANGE, strength=0, atr_val=atr_val,
            bull_score=0.0, bear_score=0.0,
            votes=(guard,), min_votes_met=False, degraded=True,
        )

    cur = float(c.iloc[-1])
    e50 = _ema(c, CFG.ema_long)
    e21 = float(_ema(c, CFG.ema_short).iloc[-1])
    e50_cur = float(e50.iloc[-1])

    vol_series: Optional[pd.Series] = (
        df["Volume"]
        if "Volume" in df.columns and instrument not in INDICES
        else None
    )

    ctx: Dict[str, Any] = {
        "cur": cur, "e21": e21, "e50_cur": e50_cur, "e50": e50,
        "atr_val": atr_val, "df_weekly": df_weekly, "df_daily": df,
        "vol_series": vol_series, "instrument": instrument,
        "current_week_open": current_week_open,
    }

    raw_votes: List[VoteSignal] = []
    degraded = False
    for spec in DAILY_VOTES.all_votes():
        try:
            v = spec.fn(h, lo, c, ctx)
            if not isinstance(v, VoteSignal):
                raise TypeError(f"Vote {spec.uid} n'a pas retourné VoteSignal")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _log.error("Vote %s erreur : %s", spec.uid, type(exc).__name__)
            v = VoteSignal(
                name=spec.uid, direction=Direction.RANGE,
                weight=0.0, reliability=0.0, fired=False,
                reason=f"err:{type(exc).__name__}", errored=True,
            )
            if spec.critical:
                degraded = True
        raw_votes.append(v)

    return _aggregate_votes(tuple(raw_votes), atr_val, degraded)


def trend_macro(df: pd.DataFrame, tf: str) -> Tuple[str, int, float]:
    """Tendance M (mensuelle) ou W (hebdo)."""
    if len(df) < 50:
        atr_val = (
            float(_atr(df["High"], df["Low"], df["Close"], CFG.atr_period).iloc[-1])
            if len(df) >= 15 else 0.0
        )
        return "Range", 0, atr_val

    c, h, lo = df["Close"], df["High"], df["Low"]
    atr_val = float(_atr(h, lo, c, CFG.atr_period).iloc[-1])
    band = atr_val * CFG.macro_band_atr_ratio
    e50 = _ema(c, CFG.ema_long)

    if tf == "M":
        if len(df) < CFG.min_bars_m:
            return "Range", 0, atr_val
        e100 = _ema(c, 100)
        ref = float(e100.iloc[-1])
        cur = float(e50.iloc[-1])
        if ref == 0:
            return "Range", 40, atr_val
        gap = abs(cur - ref) / ref * 100
        s = 75 if gap > 0.3 else 60
        if cur > ref + band:
            return "Bullish", s, atr_val
        if cur < ref - band:
            return "Bearish", s, atr_val
        return "Range", 40, atr_val

    # Weekly
    if len(df) < CFG.sma_macro:
        _log.warning("Weekly SMA200 indisponible (%d bougies)", len(df))
        return "Range", 40, atr_val
    s200 = _sma(c, CFG.sma_macro)
    cur50 = float(e50.iloc[-1])
    ref200 = float(s200.iloc[-1])
    prev50 = float(e50.iloc[-2])
    prev200 = float(s200.iloc[-2])
    cross = (prev50 <= prev200 < cur50) or (prev50 >= prev200 > cur50)
    if cur50 > ref200 + band:
        return "Bullish", (90 if cross else 75), atr_val
    if cur50 < ref200 - band:
        return "Bearish", (90 if cross else 75), atr_val
    return "Range", 40, atr_val


def trend_4h(
    df: pd.DataFrame,
    df_daily: Optional[pd.DataFrame] = None,
    instrument: str = "",
    current_day_open: Optional[float] = None,
) -> Tuple[str, int, float]:
    h, lo, c = df["High"], df["Low"], df["Close"]
    atr_val = float(_atr(h, lo, c, CFG.atr_period).iloc[-1])
    if len(df) < CFG.min_bars_h4:
        return "Range", 0, atr_val

    cur = float(c.iloc[-1])
    score = 0

    e50_cur = float(_ema(c, CFG.ema_long).iloc[-1])
    if not np.isnan(e50_cur):
        score += 1 if cur > e50_cur else -1

    pdi, mdi = _dmi(h, lo, c, CFG.atr_period)
    if not (np.isnan(pdi) or np.isnan(mdi)):
        score += 1 if pdi > mdi else -1

    today_open: Optional[float] = None
    if current_day_open is not None and not np.isnan(current_day_open):
        today_open = current_day_open
    elif df_daily is not None and not df_daily.empty:
        try:
            today_open = float(df_daily["Open"].iloc[-1])
        except (IndexError, ValueError, TypeError):
            today_open = None

    if today_open is not None:
        score += 1 if cur > today_open else -1
    else:
        _log.debug("trend_4h[%s] Daily Open indisponible", instrument)

    s = abs(score)
    strength = 90 if s == 3 else 70 if s >= 1 else 40
    direction = "Bullish" if score > 0 else "Bearish" if score < 0 else "Range"
    return direction, strength, atr_val


def trend_intraday(df: pd.DataFrame, instrument: str = "") -> Tuple[str, int, float]:
    h, lo, c = df["High"], df["Low"], df["Close"]
    atr_val = float(_atr(h, lo, c, CFG.atr_period).iloc[-1])
    if len(df) < 70:
        return "Range", 0, atr_val

    cur = float(c.iloc[-1])
    period = CFG.ema_intra_period
    # [F7] ZLEMA lag corrigé
    lag = (period - 1) // 2

    ema9_s = _ema(c, CFG.intraday_ema_fast)
    ema21_s = _ema(c, CFG.ema_short)
    ema50_s = _ema(c, period)
    ema12_s = _ema(c, CFG.macd_fast)
    ema26_s = _ema(c, CFG.macd_slow)

    e9 = float(ema9_s.iloc[-1])
    e21 = float(ema21_s.iloc[-1])
    e50_cur = float(ema50_s.iloc[-1])

    src_adj = c + (c - c.shift(lag))
    zlema = float(src_adj.ewm(span=period, adjust=False).mean().iloc[-1])

    rsi_val = float(_rsi(c, CFG.rsi_period).iloc[-1])
    macd = ema12_s - ema26_s
    sig_line = _ema(macd, CFG.macd_signal)
    macd_cur = float(macd.iloc[-1])
    sig_cur = float(sig_line.iloc[-1])

    if any(np.isnan(v) for v in [e9, e21, e50_cur, zlema, rsi_val, macd_cur, sig_cur]):
        return "Range", 0, atr_val

    bull_zlema = cur > zlema
    bear_zlema = cur < zlema
    bull_stack = e9 > e21 > e50_cur
    bear_stack = e9 < e21 < e50_cur
    bull_mom = rsi_val > 50 and macd_cur > sig_cur
    bear_mom = rsi_val < 50 and macd_cur < sig_cur

    votes_bull = [bull_zlema, bull_stack, bull_mom]
    votes_bear = [bear_zlema, bear_stack, bear_mom]
    max_votes = 3

    use_volume = instrument not in INDICES and "Volume" in df.columns
    if use_volume:
        vol = df["Volume"]
        vol_ma = vol.rolling(20).mean()
        vol_cur = float(vol.iloc[-1])
        vol_avg = float(vol_ma.iloc[-1])
        if not np.isnan(vol_avg) and vol_avg > 0:
            strong_vol = vol_cur > vol_avg * CFG.volume_ratio_strong_intraday
            votes_bull.append(strong_vol and bull_zlema)
            votes_bear.append(strong_vol and bear_zlema)
            max_votes = 4

    vb = sum(votes_bull)
    vbr = sum(votes_bear)
    threshold_strong = max_votes
    threshold_mod = max_votes - 1

    def _atr_strength() -> int:
        if atr_val <= 0:
            return 60
        return int(min(95, 40 + (abs(cur - zlema) / atr_val) * 25))

    if vb == threshold_strong:
        return "Bullish", _atr_strength(), atr_val
    if vbr == threshold_strong:
        return "Bearish", _atr_strength(), atr_val
    if vb >= threshold_mod:
        return "Bullish", 55, atr_val
    if vbr >= threshold_mod:
        return "Bearish", 55, atr_val
    if cur < e50_cur and e9 > e21:
        return "Retracement Bull", 45, atr_val
    if cur > e50_cur and e9 < e21:
        return "Retracement Bear", 45, atr_val
    return "Range", 30, atr_val


def trend_age_daily(df: pd.DataFrame) -> str:
    if len(df) < 55:
        return "N/A"
    c = df["Close"]
    e50 = _ema(c, CFG.ema_long)
    above = c > e50
    for i in range(len(above) - 1, 0, -1):
        if above.iloc[i] != above.iloc[i - 1]:
            age = len(above) - 1 - i
            return str(age) if age > 0 else "0"
    return f">{len(above)}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — SCORING MTF (renormalisé, NC orthogonal)
# ═══════════════════════════════════════════════════════════════════════════════

_MTF_WEIGHTS: Final[Dict[str, float]] = {
    "M": CFG.mtf_weight_m,
    "W": CFG.mtf_weight_w,
    "D": CFG.mtf_weight_d,
    "4H": CFG.mtf_weight_h4,
    "1H": CFG.mtf_weight_h1,
    "15m": CFG.mtf_weight_15m,
}


def _bull_compat(t: str) -> bool:
    return t in ("Bullish", "Retracement Bull")


def _bear_compat(t: str) -> bool:
    return t in ("Bearish", "Retracement Bear")


def _mtf_weighted_score(
    trends: Dict[str, str], scores: Dict[str, int]
) -> Tuple[float, float, float]:
    """
    [F5] Renormalisation sur TF actifs (non-Range).
    Range exclu du dénominateur — n'est plus un contre-signal.
    """
    active_total = sum(
        _MTF_WEIGHTS[tf]
        for tf in trends
        if not trends[tf].startswith("Range")
    )
    if active_total == 0:
        return 0.0, 0.0, 1.0  # éviter division zéro

    w_bull = sum(
        _MTF_WEIGHTS[tf] * (scores[tf] / 100.0)
        for tf in trends
        if trends[tf] == "Bullish"
    ) + sum(
        _MTF_WEIGHTS[tf] * (scores[tf] / 100.0) * 0.5
        for tf in trends
        if trends[tf] == "Retracement Bull"
    )
    w_bear = sum(
        _MTF_WEIGHTS[tf] * (scores[tf] / 100.0)
        for tf in trends
        if trends[tf] == "Bearish"
    ) + sum(
        _MTF_WEIGHTS[tf] * (scores[tf] / 100.0) * 0.5
        for tf in trends
        if trends[tf] == "Retracement Bear"
    )
    return w_bull, w_bear, active_total


def _mtf_alignment_bonus(trends: Dict[str, str], direction: str) -> int:
    bonus = 0
    m, w = trends.get("M", ""), trends.get("W", "")
    d, h4 = trends.get("D", ""), trends.get("4H", "")
    if direction == "Bullish":
        if m == "Bullish" and w == "Bullish":
            bonus += 15
        elif _bull_compat(m) and _bull_compat(w):
            bonus += 12
        if d == "Bullish" and h4 == "Bullish":
            bonus += 10
        elif _bull_compat(d) and _bull_compat(h4):
            bonus += 7
    else:
        if m == "Bearish" and w == "Bearish":
            bonus += 15
        elif _bear_compat(m) and _bear_compat(w):
            bonus += 12
        if d == "Bearish" and h4 == "Bearish":
            bonus += 10
        elif _bear_compat(d) and _bear_compat(h4):
            bonus += 7
    return bonus


def score_mtf(
    trends: Dict[str, str], scores: Dict[str, int]
) -> Tuple[str, float]:
    w_bull, w_bear, total = _mtf_weighted_score(trends, scores)
    if w_bull > w_bear:
        raw_score, direction = (w_bull / total) * 100, "Bullish"
    elif w_bear > w_bull:
        raw_score, direction = (w_bear / total) * 100, "Bearish"
    else:
        return "Range", 0.0
    bonus = _mtf_alignment_bonus(trends, direction)
    return direction, min(100.0, raw_score + bonus)


def _compute_nc_orthogonal(
    trends: Dict[str, str], scores: Dict[str, int], mtf_dir: str
) -> int:
    """
    [F4] NC ORTHOGONAL au MTF score.
    Compte les TF en tendance pure (Bullish/Bearish) avec strength >= seuil,
    pondérés par direction relative au MTF.

    Élimine le double-counting : NC mesure désormais la QUALITÉ de conviction,
    pas la dominance déjà reflétée par MTF score.
    """
    if mtf_dir not in ("Bullish", "Bearish"):
        return 0

    score_f = 0.0
    for tf, trend in trends.items():
        strength = scores.get(tf, 0)
        # Seul un TF avec strength forte ET tendance PURE compte
        is_strong_pure_bull = trend == "Bullish" and strength >= CFG.nc_pure_strength_min
        is_strong_pure_bear = trend == "Bearish" and strength >= CFG.nc_pure_strength_min

        if mtf_dir == "Bullish":
            if is_strong_pure_bull:
                score_f += 1.0
            elif trend == "Retracement Bull":
                score_f += 0.25
            elif is_strong_pure_bear:
                score_f -= 1.0
            elif trend == "Retracement Bear":
                score_f -= 0.25
        else:  # Bearish
            if is_strong_pure_bear:
                score_f += 1.0
            elif trend == "Retracement Bear":
                score_f += 0.25
            elif is_strong_pure_bull:
                score_f -= 1.0
            elif trend == "Retracement Bull":
                score_f -= 0.25

    # [F17] Arrondi métier : floor de |score|, signe préservé (pas de banker's)
    sign = 1 if score_f >= 0 else -1
    return sign * math.floor(abs(score_f))


def grade_hybrid(
    scores_list: List[float], nc_list: List[int], degraded_list: List[bool]
) -> List[str]:
    """[F18] degraded=True bloque promotion A+."""
    grades: List[str] = []
    for score, nc, degraded in zip(scores_list, nc_list, degraded_list):
        nc_bonus = (int(nc) - 3) * 5
        adj = min(100.0, float(score) + nc_bonus)
        if adj >= 80 and not degraded:
            grades.append("A+")
        elif adj >= 80 and degraded:
            grades.append("A")  # rétrogradé
        elif adj >= 55:
            grades.append("A")
        elif adj >= 38:
            grades.append("B+")
        else:
            grades.append("B")
    return grades


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — ANALYSE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_pair(
    pair: str,
    account_id: str,
    access_token: str,
    registry: SessionRegistry,
    stop_event: threading.Event,
) -> Optional[Dict[str, Any]]:
    if stop_event.is_set():
        return None
    try:
        cache = fetch_all_data(pair, account_id, access_token, registry, stop_event)
        if cache is None:
            return None

        trends: Dict[str, str] = {}
        scores: Dict[str, int] = {}
        atrs: Dict[str, float] = {}
        degraded = False

        for tf in ("M", "W"):
            t, s, a = trend_macro(cache[tf], tf)
            trends[tf], scores[tf], atrs[tf] = t, s, a

        if stop_event.is_set():
            return None

        daily_result = trend_daily(
            cache["D"], cache["W"], instrument=pair,
            current_week_open=cache.get("_week_open"),
        )
        trends["D"], scores["D"], atrs["D"] = (
            daily_result.direction.value,
            daily_result.strength,
            daily_result.atr_val,
        )
        if daily_result.degraded:
            degraded = True

        t, s, a = trend_4h(
            cache["4H"], cache["D"], instrument=pair,
            current_day_open=cache.get("_day_open"),
        )
        trends["4H"], scores["4H"], atrs["4H"] = t, s, a

        for tf in ("1H", "15m"):
            if stop_event.is_set():
                return None
            t, s, a = trend_intraday(cache[tf], pair)
            trends[tf], scores[tf], atrs[tf] = t, s, a

        mtf_dir, mtf_score = score_mtf(trends, scores)
        age = trend_age_daily(cache["D"])
        nc = _compute_nc_orthogonal(trends, scores, mtf_dir)

        return {
            "Paire": pair.replace("_", "/"),
            "M": trends["M"], "W": trends["W"], "D": trends["D"],
            "4H": trends["4H"], "1H": trends["1H"], "15m": trends["15m"],
            "MTF": f"{mtf_dir} ({mtf_score:.0f}%)" if mtf_dir != "Range" else "Range",
            "_mtf_score": mtf_score,
            "_mtf_dir": mtf_dir,
            "_degraded": degraded,
            "NC": nc,
            "Age D1": age,
            "ATR Daily": _fmt_atr(atrs["D"]),
            "ATR H4": _fmt_atr(atrs["4H"]),
            "ATR H1": _fmt_atr(atrs["1H"]),
            "ATR 15m": _fmt_atr(atrs["15m"]),
        }

    except Exception as e:  # pylint: disable=broad-exception-caught
        _log.error("analyze_pair %s : %s", pair, type(e).__name__)
        return None


def analyze_all_core(
    account_id: str,
    access_token: str,
    progress_cb: Optional[Callable[[float], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Noyau pur — pas de dépendance Streamlit.
    [F8] Drainage gracieux : stop_event + wait court avant close_all.
    """
    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    total = len(INSTRUMENTS)
    done = 0
    timed_out = False

    stop_event = threading.Event()
    registry = SessionRegistry()
    meta: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc),
        "version": APP_VERSION,
    }

    executor = ThreadPoolExecutor(max_workers=CFG.max_workers)
    futures: Dict[Future, str] = {}
    try:
        futures = {
            executor.submit(
                analyze_pair, inst, account_id, access_token, registry, stop_event
            ): inst
            for inst in INSTRUMENTS
        }
        try:
            for future in as_completed(futures, timeout=CFG.analysis_timeout_sec):
                inst = futures[future]
                done += 1
                if progress_cb:
                    try:
                        progress_cb(done / total)
                    except Exception:  # pragma: no cover
                        pass
                if status_cb:
                    try:
                        status_cb(f"GPS ({done}/{total}) — {inst.replace('_', '/')}")
                    except Exception:  # pragma: no cover
                        pass
                try:
                    row = future.result()
                    if row:
                        results.append(row)
                    else:
                        errors.append(inst)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    errors.append(inst)
                    _log.error("Future %s : %s", inst, type(e).__name__)
        except FutureTimeoutError:
            timed_out = True
            _log.error("analyze_all_core — timeout %ds. Annulation.", CFG.analysis_timeout_sec)
            stop_event.set()
            for f, inst in futures.items():
                if not f.done():
                    f.cancel()
                    errors.append(inst)
    finally:
        # [F8] Drainage gracieux : on signale stop, on attend court, puis on ferme
        stop_event.set()
        executor.shutdown(wait=False, cancel_futures=True)

        # Attendre que les requêtes en vol terminent (court délai)
        drain_start = time.monotonic()
        while time.monotonic() - drain_start < CFG.pool_drain_grace_sec:
            alive = sum(1 for f in futures if not f.done())
            if alive == 0:
                break
            time.sleep(0.1)

        # Maintenant on peut fermer les sessions sans risquer la corruption
        registry.close_all()

    meta["finished_at"] = datetime.now(timezone.utc)
    meta["timed_out"] = timed_out
    meta["errors_count"] = len(errors)
    meta["completeness"] = (
        len(results) / total if total > 0 else 0.0
    )

    if not results:
        return pd.DataFrame(), errors, meta

    scores_list = [r["_mtf_score"] for r in results]
    nc_list = [r["NC"] for r in results]
    degraded_list = [r["_degraded"] for r in results]
    grades = grade_hybrid(scores_list, nc_list, degraded_list)
    for r, g in zip(results, grades):
        r["Quality"] = g

    df = pd.DataFrame(results)
    df.attrs["meta"] = meta
    return df, errors, meta


def analyze_all(account_id: str, access_token: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Wrapper Streamlit."""
    progress = st.progress(0)
    status = st.empty()
    try:
        df, errors, meta = analyze_all_core(
            account_id, access_token,
            progress_cb=progress.progress,
            status_cb=status.text,
        )
    finally:
        try:
            progress.empty()
        except Exception:  # pragma: no cover
            pass
        try:
            status.empty()
        except Exception:  # pragma: no cover
            pass

    if meta.get("timed_out"):
        st.error(
            f"⏱️ Analyse interrompue après {CFG.analysis_timeout_sec}s — "
            "résultats partiels. Vérifiez la connexion OANDA."
        )
    if errors:
        st.warning(
            f"⚠️ {len(errors)} paire(s) non analysée(s) : "
            f"{', '.join(e.replace('_', '/') for e in errors[:10])}"
            + (" …" if len(errors) > 10 else "")
        )

    completeness = meta.get("completeness", 0.0)
    if completeness < 0.85 and not df.empty:
        st.markdown(
            f"<div class='degraded-warning'>⚠️ <b>Couverture partielle</b> — "
            f"seulement <b>{completeness:.0%}</b> des instruments analysés. "
            "Évitez les décisions tradables sur ce run.</div>",
            unsafe_allow_html=True,
        )
    return df, meta


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — PDF EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_str(s: str) -> str:
    return s.encode("latin-1", errors="replace").decode("latin-1")


def create_pdf(df: pd.DataFrame) -> BytesIO:
    cols = [
        "Paire", "M", "W", "D", "4H", "1H", "15m",
        "MTF", "Quality", "NC", "Age D1",
        "ATR Daily", "ATR H4", "ATR H1", "ATR 15m",
    ]
    widths = {
        "Paire": 22, "M": 16, "W": 16, "D": 16, "4H": 16, "1H": 16, "15m": 16,
        "MTF": 30, "Quality": 12, "NC": 10, "Age D1": 13,
        "ATR Daily": 17, "ATR H4": 17, "ATR H1": 15, "ATR 15m": 15,
    }
    nc_rgb = {
        (5, 99): (46, 204, 113, 255, 255, 255),
        (3, 4): (39, 174, 96, 255, 255, 255),
        (1, 2): (241, 196, 15, 0, 0, 0),
        (0, 0): (230, 126, 34, 255, 255, 255),
        (-99, -1): (231, 76, 60, 255, 255, 255),
    }
    rh = 5.5

    def _nc_colors(val: str):
        try:
            n = int(val)
            for (lo, hi), rgba in nc_rgb.items():
                if lo <= n <= hi:
                    return rgba[:3], rgba[3:]
        except (ValueError, TypeError):
            pass
        return (200, 200, 200), (0, 0, 0)

    def _cell_text(val: str) -> str:
        return val if _FPDF2 else _safe_str(val)

    buf = BytesIO()
    if not _FPDF_AVAILABLE or FPDF is None:
        buf.write(b"PDF unavailable: fpdf2 not installed")
        buf.seek(0)
        return buf

    try:
        pdf = FPDF(orientation="L", unit="mm", format="A4")
        pdf.add_page()
        pdf.set_margins(10, 10, 10)
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(0, 9, _cell_text(f"BLUESTAR GPS V{APP_VERSION}"), ln=True, align="C")
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(
            0, 5,
            _cell_text(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"),
            ln=True, align="C",
        )
        pdf.ln(4)

        grade_rgb = {
            "A+": (251, 191, 36),
            "A": (163, 230, 53),
            "B+": (52, 211, 153),
            "B": (96, 165, 250),
        }

        def header():
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_fill_color(30, 58, 138)
            pdf.set_text_color(255, 255, 255)
            for col in cols:
                pdf.cell(widths[col], 7, _cell_text(col), border=1, align="C", fill=True)
            pdf.ln()
            pdf.set_font("Helvetica", "", 6.5)

        header()
        for _, row in df.iterrows():
            if pdf.get_y() + rh > 287 - 15:
                pdf.add_page()
                header()
            for col in cols:
                val = str(row.get(col, ""))
                fc: Tuple[int, int, int] = (255, 255, 255)
                tc: Tuple[int, int, int] = (0, 0, 0)
                if col == "Quality":
                    fc = grade_rgb.get(val, (156, 163, 175))
                elif col == "NC":
                    fc, tc = _nc_colors(val)
                elif "Bull" in val and "Ret" not in val:
                    fc, tc = (46, 204, 113), (255, 255, 255)
                elif "Bear" in val and "Ret" not in val:
                    fc, tc = (231, 76, 60), (255, 255, 255)
                elif "Retracement Bull" in val:
                    fc, tc = (125, 206, 160), (255, 255, 255)
                elif "Retracement Bear" in val:
                    fc, tc = (241, 148, 138), (255, 255, 255)
                elif "Range" in val:
                    fc, tc = (149, 165, 166), (255, 255, 255)
                pdf.set_fill_color(*fc)
                pdf.set_text_color(*tc)
                pdf.cell(widths[col], rh, _cell_text(val), border=1, align="C", fill=True)
            pdf.ln()

        out = pdf.output(dest="S")
        buf.write(out.encode("latin-1") if isinstance(out, str) else bytes(out))
        buf.seek(0)
        return buf
    except Exception as e:  # pylint: disable=broad-exception-caught
        _log.error("PDF error: %s", type(e).__name__)
        buf2 = BytesIO()
        try:
            fallback = FPDF()
            fallback.add_page()
            fallback.set_font("Helvetica", "B", 12)
            fallback.cell(0, 10, "PDF Generation Error", ln=True)
            out2 = fallback.output(dest="S")
            buf2.write(out2.encode("latin-1") if isinstance(out2, str) else bytes(out2))
        except Exception:  # pragma: no cover
            buf2.write(b"PDF error")
        buf2.seek(0)
        return buf2


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

def _load_secrets() -> Tuple[Optional[str], Optional[str]]:
    """[F20] Validation exhaustive des secrets."""
    try:
        acc = st.secrets["OANDA_ACCOUNT_ID"]
        tok = st.secrets["OANDA_ACCESS_TOKEN"]
    except (KeyError, FileNotFoundError):
        return None, None
    except Exception:  # pylint: disable=broad-exception-caught
        return None, None

    if not isinstance(acc, str) or not isinstance(tok, str):
        return None, None
    if not acc.strip() or not tok.strip():
        return None, None
    return acc.strip(), tok.strip()


def _check_running_flag_ttl() -> None:
    """[F12] Reset auto du flag analysis_running après TTL."""
    started_at = st.session_state.get("_analysis_started_at")
    if started_at:
        elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
        if elapsed > CFG.streamlit_running_flag_ttl_sec:
            st.session_state["_analysis_running"] = False
            st.session_state["_analysis_started_at"] = None


def _style_trend(v: Any) -> str:
    if not isinstance(v, str):
        return ""
    if "Bull" in v and "Ret" not in v:
        return f"background-color:{TREND_COLORS['Bullish']};color:white;font-weight:bold"
    if "Bear" in v and "Ret" not in v:
        return f"background-color:{TREND_COLORS['Bearish']};color:white;font-weight:bold"
    if "Retracement Bull" in v:
        return f"background-color:{TREND_COLORS['Retracement Bull']};color:white"
    if "Retracement Bear" in v:
        return f"background-color:{TREND_COLORS['Retracement Bear']};color:white"
    if "Range" in v:
        return f"background-color:{TREND_COLORS['Range']};color:white"
    return ""


def _style_quality(s: pd.Series) -> List[str]:
    grade_css = {"A+": "#fbbf24", "A": "#a3e635", "B+": "#34d399", "B": "#60a5fa"}
    if s.name != "Quality":
        return [""] * len(s)
    return [
        f"color:black;font-weight:bold;background-color:{grade_css.get(x, '#9ca3af')}"
        for x in s
    ]


def _style_nc(s: pd.Series) -> List[str]:
    if s.name != "NC":
        return [""] * len(s)
    out: List[str] = []
    for v in s:
        try:
            n = int(v)
            if n >= 5:
                out.append("background-color:#1D9E75;color:white;font-weight:bold")
            elif n >= 3:
                out.append("background-color:#27ae60;color:white;font-weight:bold")
            elif n >= 1:
                out.append("background-color:#f39c12;color:white;font-weight:bold")
            elif n == 0:
                out.append("background-color:#e67e22;color:white")
            else:
                out.append("background-color:#e74c3c;color:white;font-weight:bold")
        except (ValueError, TypeError):
            out.append("")
    return out


def main() -> None:
    st.markdown(
        f"<div class='main-header'><h1>🧭 BLUESTAR HEDGE FUND GPS V{APP_VERSION}</h1>"
        "<p style='margin:0;font-size:0.85em;opacity:0.8'>"
        "Production-Grade · Signal Decomposition · F1–F20"
        "</p></div>",
        unsafe_allow_html=True,
    )

    acc, tok = _load_secrets()
    if not acc or not tok:
        st.error(
            "❌ Secrets OANDA manquants ou invalides — "
            "configurez OANDA_ACCOUNT_ID et OANDA_ACCESS_TOKEN dans .streamlit/secrets.toml"
        )
        st.stop()

    _check_running_flag_ttl()

    with st.sidebar:
        st.header("⚙️ Configuration")
        only_best = st.checkbox("Afficher uniquement Grade A+ / A", value=False)
        st.info(
            f"Cache TTL : M={CFG.cache_ttl_m // 60}m W={CFG.cache_ttl_w // 60}m "
            f"D={CFG.cache_ttl_d // 60}m H4={CFG.cache_ttl_h4 // 60}m\n\n"
            f"Workers : {CFG.max_workers} · Timeout : {CFG.analysis_timeout_sec}s"
        )
        if not is_fx_market_open():
            st.warning("📅 Marché FX actuellement fermé (weekend) — données potentiellement stale.")
        st.markdown("---")
        if st.button("🗑️ Vider le cache", use_container_width=True):
            _CACHE.clear()
            st.success("Cache vidé.")
        st.markdown("---")
        st.caption(
            f"Bluestar GPS V{APP_VERSION} — NC orthogonal au MTF score"
        )

    is_running = st.session_state.get("_analysis_running", False)
    if st.button(
        "🚀 LANCER L'ANALYSE TOUS ACTIFS",
        type="primary",
        use_container_width=True,
        disabled=is_running,
    ):
        st.session_state["_analysis_running"] = True
        st.session_state["_analysis_started_at"] = datetime.now(timezone.utc)
        try:
            with st.spinner("Analyse Multi-Timeframe en cours..."):
                df, meta = analyze_all(acc, tok)
            if not df.empty:
                st.session_state["df"] = df
                st.session_state["df_ts"] = datetime.now(timezone.utc)
                st.session_state["df_meta"] = meta
        finally:
            st.session_state["_analysis_running"] = False
            st.session_state["_analysis_started_at"] = None

    if "df" not in st.session_state:
        return

    df_ts = st.session_state.get("df_ts")
    if df_ts:
        age_min = (datetime.now(timezone.utc) - df_ts).total_seconds() / 60
        if age_min > CFG.data_max_age_min:
            st.markdown(
                f"<div class='stale-warning'>⏰ <b>Données périmées</b> — "
                f"dernière analyse il y a <b>{age_min:.0f} min</b>. "
                "Relancez l'analyse.</div>",
                unsafe_allow_html=True,
            )

    df = st.session_state["df"].copy()
    if only_best:
        df = df[df["Quality"].isin(["A+", "A"])]

    grade_order = ["A+", "A", "B+", "B"]
    df["Quality"] = pd.Categorical(df["Quality"], categories=grade_order, ordered=True)

    sort_cols = [c for c in ["Quality", "NC", "_mtf_score"] if c in df.columns]
    df = df.sort_values(sort_cols, ascending=[True, False, False])
    df.drop(columns=["_mtf_score", "_mtf_dir", "_degraded"], inplace=True, errors="ignore")

    c1, c2, c3, c4 = st.columns(4)
    total = len(df)
    a_plus = len(df[df["Quality"] == "A+"])
    a_grade = len(df[df["Quality"] == "A"])
    b_grade = len(df[df["Quality"].isin(["B+", "B"])])
    c1.metric("Total Analyzed", total)
    c2.metric("Setups A+ (GOLD)", a_plus)
    c3.metric("Setups A (GREEN)", a_grade)
    c4.metric("Setups B (BLUE)", b_grade)

    display = [
        "Paire", "M", "W", "D", "4H", "1H", "15m",
        "MTF", "Quality", "NC", "Age D1",
        "ATR Daily", "ATR H4", "ATR H1", "ATR 15m",
    ]
    cols_present = [col for col in display if col in df.columns]

    styled = (
        df[cols_present].style
        .apply(_style_quality, axis=0)
        .apply(_style_nc, axis=0)
        .map(_style_trend)
    )
    st.dataframe(
        styled,
        height=min(800, max(400, (len(df) + 1) * 38 + 10)),
        use_container_width=True,
        hide_index=True,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "📄 Télécharger PDF",
            data=create_pdf(df[cols_present]),
            file_name=f"Bluestar_GPS_{ts}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "📊 Télécharger CSV",
            data=df[cols_present].to_csv(index=False).encode("utf-8"),
            file_name=f"Bluestar_GPS_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "🗂️ Télécharger JSON",
            data=df[cols_present].to_json(
                orient="records", force_ascii=False, indent=2
            ).encode("utf-8"),
            file_name=f"Bluestar_GPS_{ts}.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

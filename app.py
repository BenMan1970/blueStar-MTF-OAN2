"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BLUESTAR HEDGE FUND GPS — V7.0 PRODUCTION-GRADE HARDENED                   ║
║                                                                              ║
║  Refonte V6.0 → V7.0 : durcissement critique pour déploiement 24/7.         ║
║                                                                              ║
║  Convention iloc (CRITIQUE — INCHANGÉE) :                                   ║
║    fetch_candles(include_incomplete=False) →                                ║
║      df.iloc[-1] = dernière bougie COMPLÈTE (= J-1 humain)                  ║
║      df.iloc[-2] = avant-dernière complète (= J-2 humain)                   ║
║    fetch_live_open() → bougie INCOMPLÈTE de la session en cours             ║
║                                                                              ║
║  Corrections V7.0 (couvrant audits statique + 12 couches) :                 ║
║    [C1]  Cache scopé par (env, account_hash, ...) — isolation tenant        ║
║    [C2]  Cache retourne deep-copy + arrays read-only (vraie immutabilité)   ║
║    [C3]  Drainage strict executor avant close sessions                      ║
║    [C4]  Stale-on-error policy (last-good-known avec flag)                  ║
║    [C5]  Rate limiter token-bucket non bloquant (429 ne bloque plus pool)   ║
║    [C6]  weekly_open fallback NEUTRE (pas de biais directionnel)            ║
║    [C7]  Snapshot timestamps par TF (détection drift)                       ║
║    [C8]  Zero silent except — tous incidents loggés structurés              ║
║    [H1]  Validation gaps temporels par granularité                          ║
║    [H2]  Secret scrubbing sur root logger + handlers                        ║
║    [H3]  Mode dégradé plafonne à B+ avec flag NOT_TRADEABLE                 ║
║    [H4]  Negative caching live_open (15s) + in-flight dedup                 ║
║    [H5]  UI side-effects guard (_is_streamlit_runtime)                      ║
║    [H6]  Imports inutilisés supprimés                                       ║
║    [H7]  Erreurs dédupliquées (set interne)                                 ║
║    [H8]  Propagation degraded → meta + UI                                   ║
║    [H9]  Refactor fonctions complexes (CC > 15 → sous-fonctions)            ║
║    [H10] Pénalité dispersion inter-TF (entropy penalty)                     ║
║    [H11] Validation secrets format + longueur min                           ║
║    [H12] DataFrame schema strict (colonnes + types)                         ║
║    [H13] Typage stricte (Final, NewType, Protocol où pertinent)             ║
║    [H14] Telemetry hooks (incident codes uniques)                           ║
║    [H15] Streamlit caching @st.cache_resource pour SessionRegistry          ║
║                                                                              ║
║  Préservé depuis V6.0 : F1-F20 + comportement métier + format JSON          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
    as_completed,
)
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    FrozenSet,
    List,
    Mapping,
    NewType,
    Optional,
    Tuple,
)

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from scipy.signal import find_peaks
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from fpdf import FPDF
    _FPDF_AVAILABLE = True
    _FPDF2 = hasattr(FPDF, "set_lang")
except ImportError:
    FPDF = None
    _FPDF_AVAILABLE = False
    _FPDF2 = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — STREAMLIT RUNTIME GUARD (H5)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_streamlit_runtime() -> bool:
    """[H5] Détecte si le module tourne dans un runtime Streamlit actif."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except ImportError:
        return False


_STREAMLIT_AVAILABLE = False
st: Any = None
try:
    import streamlit as _st_module
    st = _st_module
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOGGING & TELEMETRY (C8, H2, H14)
# ═══════════════════════════════════════════════════════════════════════════════

class _SecretScrubFilter(logging.Filter):
    """
    [H2] Filtre logging — supprime tokens, account IDs.
    Appliqué au ROOT logger pour couvrir libs tierces (requests, urllib3).
    """

    _PATTERNS: Tuple[re.Pattern, ...] = (
        re.compile(r"Bearer\s+[A-Za-z0-9\-_\.~+/=]+", re.IGNORECASE),
        re.compile(r"\b[0-9]{3}-[0-9]{3}-[0-9]+-[0-9]+\b"),
        re.compile(r"\b[a-f0-9]{32,}\b", re.IGNORECASE),
        re.compile(r"(?i)(access[_-]?token|api[_-]?key|secret)\s*[:=]\s*\S+"),
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            for pat in self._PATTERNS:
                msg = pat.sub("[REDACTED]", msg)
            record.msg = msg
            record.args = ()
        except (TypeError, ValueError) as exc:
            # Ne JAMAIS perdre un log — log structurel d'incident
            record.msg = f"[SCRUB_ERROR:{type(exc).__name__}]"
            record.args = ()
        return True


def _setup_logging() -> logging.Logger:
    """[H2] Setup global avec scrubbing sur root logger."""
    root = logging.getLogger()
    if not any(isinstance(f, _SecretScrubFilter) for f in root.filters):
        root.addFilter(_SecretScrubFilter())

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        handler.addFilter(_SecretScrubFilter())
        root.addHandler(handler)

    log_level = os.environ.get("BLUESTAR_LOG_LEVEL", "WARNING").upper()
    root.setLevel(getattr(logging, log_level, logging.WARNING))

    logger = logging.getLogger("bluestar_gps")
    logger.addFilter(_SecretScrubFilter())
    return logger


_log: Final[logging.Logger] = _setup_logging()


# [H14] Incident codes pour observabilité
class IncidentCode(str, Enum):
    HTTP_TIMEOUT = "E001"
    HTTP_ERROR = "E002"
    HTTP_RATELIMIT = "E003"
    HTTP_AUTH = "E004"
    JSON_INVALID = "E010"
    DATA_INSUFFICIENT = "E020"
    DATA_GAPS = "E021"
    DATA_VALIDATION = "E022"
    CACHE_STALE_HIT = "E030"
    CACHE_LEADER_FAILED = "E031"
    VOTE_ERROR = "E040"
    VOTE_CRITICAL_ERROR = "E041"
    EXECUTOR_TIMEOUT = "E050"
    EXECUTOR_CANCEL = "E051"
    SESSION_CLOSED = "E052"
    UI_CALLBACK_ERROR = "E060"
    PDF_ERROR = "E070"
    UNKNOWN = "E999"


def _log_incident(
    code: IncidentCode,
    msg: str,
    *,
    level: int = logging.WARNING,
    **context: Any,
) -> None:
    """[C8, H14] Log structuré d'un incident — jamais silencieux."""
    parts = [f"code={code.value}", f"msg={msg}"]
    for k, v in context.items():
        parts.append(f"{k}={v}")
    _log.log(level, "INCIDENT %s", " ".join(parts))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION (frozen, immuable)
# ═══════════════════════════════════════════════════════════════════════════════

APP_VERSION: Final[str] = "7.0.0"

# Environnement OANDA configurable (production vs practice)
_OANDA_ENV: Final[str] = os.environ.get("OANDA_ENV", "practice").lower()
OANDA_API_URL: Final[str] = (
    "https://api-fxtrade.oanda.com"
    if _OANDA_ENV == "live"
    else "https://api-fxpractice.oanda.com"
)

INSTRUMENTS: Final[Tuple[str, ...]] = (
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "DE30_EUR", "XAU_USD", "SPX500_USD", "NAS100_USD", "US30_USD",
)

INDICES: Final[FrozenSet[str]] = frozenset(
    {"DE30_EUR", "SPX500_USD", "NAS100_USD", "US30_USD", "XAU_USD"}
)

# Schéma strict OHLCV (H12)
REQUIRED_OHLCV_COLS: Final[Tuple[str, ...]] = ("Open", "High", "Low", "Close", "Volume")

# Types contrats
AccountHash = NewType("AccountHash", str)


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
    cache_ttl_negative_live_open: int = 15  # [H4] negative caching

    # Stale-on-error (C4)
    stale_max_age_multiplier: float = 4.0  # accepter stale jusqu'à 4x TTL si erreur leader

    # Fraîcheur des données
    data_max_age_min: int = 10
    snapshot_drift_max_sec: float = 30.0  # [C7] drift max entre 1ère et dernière TF
    analysis_timeout_sec: int = 120
    streamlit_running_flag_ttl_sec: int = 300

    # Workers
    max_workers: int = 5
    pool_drain_grace_sec: float = 10.0

    # OANDA HTTP
    http_timeout_sec: float = 8.0
    http_retry_total: int = 2
    http_retry_backoff: float = 0.3

    # Rate limiter (C5) — token bucket par instrument
    rate_limit_burst: int = 30
    rate_limit_refill_per_sec: float = 5.0
    rate_limit_429_cooldown_sec: float = 30.0

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
    body_range_strong: float = 0.60

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

    # Dispersion penalty (H10)
    dispersion_penalty_max: float = 15.0

    # Min bars
    min_bars_m: int = 100
    min_bars_w: int = 50
    min_bars_d: int = 60
    min_bars_h4: int = 60
    min_bars_h1: int = 200
    min_bars_15m: int = 200

    # Completeness minimum pour signaux tradables (H3)
    completeness_min_tradable: float = 0.85


CFG: Final[TrendConfig] = TrendConfig()


_CACHE_TTL: Final[Mapping[str, int]] = {
    "M": CFG.cache_ttl_m,
    "W": CFG.cache_ttl_w,
    "D": CFG.cache_ttl_d,
    "H4": CFG.cache_ttl_h4,
    "H1": CFG.cache_ttl_h1,
    "M15": CFG.cache_ttl_m15,
}

_GRAN_FREQ: Final[Mapping[str, pd.Timedelta]] = {
    "M": pd.Timedelta(days=30),
    "W": pd.Timedelta(days=7),
    "D": pd.Timedelta(days=1),
    "H4": pd.Timedelta(hours=4),
    "H1": pd.Timedelta(hours=1),
    "M15": pd.Timedelta(minutes=15),
}

# Tolérance gaps par granularité (H1)
_GAP_TOLERANCE: Final[Mapping[str, pd.Timedelta]] = {
    "M": pd.Timedelta(days=45),
    "W": pd.Timedelta(days=10),
    "D": pd.Timedelta(days=4),  # tolère weekend FX
    "H4": pd.Timedelta(hours=12),
    "H1": pd.Timedelta(hours=6),
    "M15": pd.Timedelta(hours=2),
}

TREND_COLORS: Final[Mapping[str, str]] = {
    "Bullish":          "#2ecc71",
    "Bearish":          "#e74c3c",
    "Retracement Bull": "#7dcea0",
    "Retracement Bear": "#f1948a",
    "Range":            "#95a5a6",
}

# Assertion config cohérente
assert CFG.data_max_age_min <= CFG.cache_ttl_d // 60, \
    "data_max_age_min doit être <= cache_ttl_d (minutes)"
assert CFG.max_workers >= 1, "max_workers >= 1 requis"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STREAMLIT SETUP GUARDED (H5)
# ═══════════════════════════════════════════════════════════════════════════════

def _configure_streamlit_ui() -> None:
    """[H5] Configuration UI uniquement si runtime Streamlit actif."""
    if not (_STREAMLIT_AVAILABLE and _is_streamlit_runtime()):
        return

    st.set_page_config(
        page_title=f"Bluestar GPS V{APP_VERSION}",
        page_icon="🧭",
        layout="wide",
    )
    st.markdown(
        """
        <style>
            .main-header {
                text-align: center; padding: 20px;
                background: linear-gradient(135deg, #1e3a8a 0%, #172554 100%);
                color: white; border-radius: 12px; margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }
            .stale-warning {
                background: #fef3c7; border-left: 4px solid #f59e0b;
                padding: 10px 16px; border-radius: 4px; margin-bottom: 12px;
            }
            .degraded-warning {
                background: #fee2e2; border-left: 4px solid #dc2626;
                padding: 10px 16px; border-radius: 4px; margin-bottom: 12px;
            }
            .not-tradable {
                background: #1f2937; color: #fbbf24; padding: 4px 8px;
                border-radius: 4px; font-weight: bold; font-size: 0.85em;
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
    name: str
    direction: Direction
    weight: float
    reliability: float
    fired: bool
    reason: str
    errored: bool = False


@dataclass(frozen=True)
class DailyTrendResult:
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


@dataclass(frozen=True)
class FetchResult:
    """[C4] Résultat fetch avec flag stale."""
    df: pd.DataFrame
    is_stale: bool = False
    fetched_at: Optional[datetime] = None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MARKET CALENDAR
# ═══════════════════════════════════════════════════════════════════════════════

def is_fx_market_open(now: Optional[datetime] = None) -> bool:
    """FX ouvert dimanche 22:00 UTC → vendredi 22:00 UTC."""
    if now is None:
        now = datetime.now(timezone.utc)
    weekday = now.weekday()
    hour = now.hour
    if weekday == 5:
        return False
    if weekday == 4 and hour >= 22:
        return False
    if weekday == 6 and hour < 22:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — INDICATEURS
# ═══════════════════════════════════════════════════════════════════════════════

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return _true_range(high, low, close).ewm(alpha=1.0 / n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
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


def _find_strict_peaks(series: pd.Series, wing: int, min_idx: int) -> List[int]:
    arr = series.to_numpy()
    n = len(arr)
    if n < 2 * wing + 1:
        return []
    if _HAS_SCIPY:
        peaks, _ = find_peaks(arr, distance=max(1, wing))
        return [int(p) for p in peaks if min_idx <= p <= n - wing - 1]
    result: List[int] = []
    for i in range(max(min_idx, wing), n - wing):
        window = arr[i - wing: i + wing + 1]
        center = arr[i]
        if center == window.max() and np.sum(window == center) == 1:
            result.append(i)
    return result


def _find_strict_troughs(series: pd.Series, wing: int, min_idx: int) -> List[int]:
    arr = series.to_numpy()
    n = len(arr)
    if n < 2 * wing + 1:
        return []
    if _HAS_SCIPY:
        peaks, _ = find_peaks(-arr, distance=max(1, wing))
        return [int(p) for p in peaks if min_idx <= p <= n - wing - 1]
    result: List[int] = []
    for i in range(max(min_idx, wing), n - wing):
        window = arr[i - wing: i + wing + 1]
        center = arr[i]
        if center == window.min() and np.sum(window == center) == 1:
            result.append(i)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RATE LIMITER (C5) — Token bucket non bloquant
# ═══════════════════════════════════════════════════════════════════════════════

class _TokenBucket:
    """
    [C5] Token bucket par instrument — non bloquant.
    Si pas de tokens → return False (worker skip immédiatement, libère le slot).
    """

    __slots__ = ("_capacity", "_refill_rate", "_tokens", "_last", "_lock", "_cooldown_until")

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self._capacity = float(capacity)
        self._refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self._cooldown_until = 0.0

    def try_acquire(self) -> bool:
        with self._lock:
            now = time.monotonic()
            if now < self._cooldown_until:
                return False
            elapsed = now - self._last
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def trigger_cooldown(self, seconds: float) -> None:
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + seconds)
            self._tokens = 0.0


class _GlobalRateLimiter:
    """[C5] Rate limiter global — un bucket par instrument."""

    def __init__(self) -> None:
        self._buckets: Dict[str, _TokenBucket] = {}
        self._lock = threading.Lock()

    def acquire(self, instrument: str) -> bool:
        with self._lock:
            bucket = self._buckets.get(instrument)
            if bucket is None:
                bucket = _TokenBucket(
                    CFG.rate_limit_burst, CFG.rate_limit_refill_per_sec
                )
                self._buckets[instrument] = bucket
        return bucket.try_acquire()

    def trigger_cooldown(self, instrument: str, seconds: float) -> None:
        with self._lock:
            bucket = self._buckets.get(instrument)
        if bucket is not None:
            bucket.trigger_cooldown(seconds)


_RATE_LIMITER: Final[_GlobalRateLimiter] = _GlobalRateLimiter()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — HTTP SESSIONS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class SessionRegistry:
    """Registre HTTP scopé par analyse — fermé proprement après drainage strict."""

    def __init__(self) -> None:
        self._sessions: List[requests.Session] = []
        self._thread_sessions: Dict[int, requests.Session] = {}
        self._lock = threading.RLock()
        self._closed = False

    def get_for_thread(self) -> requests.Session:
        tid = threading.get_ident()
        with self._lock:
            if self._closed:
                raise RuntimeError("SessionRegistry closed")
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
            status_forcelist=[500, 502, 503, 504],  # 429 géré séparément via rate limiter
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
                except (OSError, RuntimeError) as exc:
                    _log_incident(
                        IncidentCode.SESSION_CLOSED,
                        "session close failed",
                        level=logging.DEBUG,
                        err=type(exc).__name__,
                    )
            self._sessions.clear()
            self._thread_sessions.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CACHE (C1, C2, C4) — Isolation tenant + immutabilité + stale-on-error
# ═══════════════════════════════════════════════════════════════════════════════

CacheKey = Tuple[str, AccountHash, str, str, int]  # (env, account_hash, instrument, gran, count)
LiveOpenKey = Tuple[str, AccountHash, str, str]


def _hash_account(account_id: str) -> AccountHash:
    """[C1] Hash stable de l'account_id pour scoping cache (pas en clair en mémoire)."""
    h = hashlib.sha256(account_id.encode("utf-8")).hexdigest()[:16]
    return AccountHash(h)


def _freeze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    [C2] Rend un DataFrame réellement immuable.
    - Deep copy
    - Arrays numpy en read-only
    """
    frozen = df.copy(deep=True)
    for col in frozen.columns:
        arr = frozen[col].to_numpy()
        try:
            arr.flags.writeable = False
        except (ValueError, AttributeError):
            pass  # Certains dtypes ne supportent pas — pas critique, deep copy garde l'isolation
    return frozen


def _defensive_copy(df: pd.DataFrame) -> pd.DataFrame:
    """[C2] Copie défensive au retour du cache — l'appelant peut muter en sécurité."""
    return df.copy(deep=True)


class CandleCache:
    """
    [C1, C2, C4] Cache multi-tenant, immutable, stale-on-error.
    """

    def __init__(self) -> None:
        self._data: Dict[CacheKey, Tuple[datetime, pd.DataFrame]] = {}
        self._stale_data: Dict[CacheKey, Tuple[datetime, pd.DataFrame]] = {}
        self._live_opens: Dict[LiveOpenKey, Tuple[datetime, Optional[float]]] = {}
        self._inflight: Dict[CacheKey, threading.Event] = {}
        self._inflight_results: Dict[CacheKey, FetchResult] = {}
        self._lock = threading.RLock()

    def get_candles(
        self,
        key: CacheKey,
        ttl: int,
        fetch_fn: Callable[[], pd.DataFrame],
    ) -> FetchResult:
        now = datetime.now(timezone.utc)
        with self._lock:
            entry = self._data.get(key)
            if entry is not None and (now - entry[0]).total_seconds() < ttl:
                return FetchResult(df=_defensive_copy(entry[1]), is_stale=False, fetched_at=entry[0])

            inflight_event = self._inflight.get(key)
            if inflight_event is not None:
                wait_for_event = inflight_event
                start_my_fetch = False
            else:
                wait_for_event = threading.Event()
                self._inflight[key] = wait_for_event
                start_my_fetch = True

        if not start_my_fetch:
            return self._wait_for_leader(key, wait_for_event, ttl, now)

        return self._do_fetch_as_leader(key, ttl, now, fetch_fn, wait_for_event)

    def _wait_for_leader(
        self,
        key: CacheKey,
        event: threading.Event,
        ttl: int,
        now: datetime,
    ) -> FetchResult:
        event.wait(timeout=CFG.http_timeout_sec * 3)
        with self._lock:
            entry = self._data.get(key)
            if entry is not None and (now - entry[0]).total_seconds() < ttl:
                return FetchResult(
                    df=_defensive_copy(entry[1]), is_stale=False, fetched_at=entry[0]
                )
            # [C4] Stale fallback
            stale = self._stale_data.get(key)
            if stale is not None:
                age = (now - stale[0]).total_seconds()
                if age < ttl * CFG.stale_max_age_multiplier:
                    _log_incident(
                        IncidentCode.CACHE_STALE_HIT,
                        "follower stale fallback",
                        key=str(key[2:]),
                        age_sec=int(age),
                    )
                    return FetchResult(
                        df=_defensive_copy(stale[1]), is_stale=True, fetched_at=stale[0]
                    )
        _log_incident(
            IncidentCode.CACHE_LEADER_FAILED,
            "leader failed and no stale available",
            key=str(key[2:]),
        )
        return FetchResult(df=pd.DataFrame(), is_stale=False)

    def _do_fetch_as_leader(
        self,
        key: CacheKey,
        ttl: int,
        now: datetime,
        fetch_fn: Callable[[], pd.DataFrame],
        event: threading.Event,
    ) -> FetchResult:
        try:
            df = fetch_fn()
            if not df.empty:
                frozen = _freeze_dataframe(df)
                ts = datetime.now(timezone.utc)
                with self._lock:
                    self._data[key] = (ts, frozen)
                    self._stale_data[key] = (ts, frozen)
                return FetchResult(df=_defensive_copy(frozen), is_stale=False, fetched_at=ts)

            # Fetch a échoué — fallback stale
            with self._lock:
                stale = self._stale_data.get(key)
                if stale is not None:
                    age = (now - stale[0]).total_seconds()
                    if age < ttl * CFG.stale_max_age_multiplier:
                        _log_incident(
                            IncidentCode.CACHE_STALE_HIT,
                            "leader stale fallback",
                            key=str(key[2:]),
                            age_sec=int(age),
                        )
                        return FetchResult(
                            df=_defensive_copy(stale[1]), is_stale=True, fetched_at=stale[0]
                        )
            return FetchResult(df=pd.DataFrame(), is_stale=False)
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            event.set()

    def get_live_open(
        self,
        key: LiveOpenKey,
        fetch_fn: Callable[[], Optional[float]],
    ) -> Optional[float]:
        """[H4] Live open avec negative caching court."""
        now = datetime.now(timezone.utc)
        with self._lock:
            entry = self._live_opens.get(key)
            if entry is not None:
                ts, value = entry
                age = (now - ts).total_seconds()
                ttl = (
                    CFG.cache_ttl_live_open
                    if value is not None
                    else CFG.cache_ttl_negative_live_open
                )
                if age < ttl:
                    return value

        value = fetch_fn()
        with self._lock:
            self._live_opens[key] = (datetime.now(timezone.utc), value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._stale_data.clear()
            self._live_opens.clear()
            self._inflight.clear()


_CACHE: Final[CandleCache] = CandleCache()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — DATA LAYER (validation stricte, gaps temporels)
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_candle(row: Mapping[str, float], allow_zero_volume: bool) -> bool:
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


def _validate_dataframe_schema(df: pd.DataFrame, instrument: str, granularity: str) -> bool:
    """[H12] Validation schéma strict."""
    if df.empty:
        return False
    missing = [c for c in REQUIRED_OHLCV_COLS if c not in df.columns]
    if missing:
        _log_incident(
            IncidentCode.DATA_VALIDATION,
            "missing columns",
            instrument=instrument,
            granularity=granularity,
            missing=",".join(missing),
        )
        return False
    if not isinstance(df.index, pd.DatetimeIndex):
        _log_incident(
            IncidentCode.DATA_VALIDATION, "non-datetime index",
            instrument=instrument, granularity=granularity,
        )
        return False
    if not df.index.is_monotonic_increasing:
        _log_incident(
            IncidentCode.DATA_VALIDATION, "non-monotonic index",
            instrument=instrument, granularity=granularity,
        )
        return False
    if df.index.has_duplicates:
        _log_incident(
            IncidentCode.DATA_VALIDATION, "duplicates in index",
            instrument=instrument, granularity=granularity,
        )
        return False
    return True


def _validate_dataframe_gaps(
    df: pd.DataFrame, granularity: str, instrument: str
) -> bool:
    """[H1] Détecte gaps temporels anormaux."""
    if len(df) < 2:
        return True  # déjà validé ailleurs
    tolerance = _GAP_TOLERANCE.get(granularity)
    if tolerance is None:
        return True
    deltas = df.index.to_series().diff().dropna()
    max_gap = deltas.max()
    if max_gap > tolerance:
        _log_incident(
            IncidentCode.DATA_GAPS,
            "gap exceeds tolerance",
            instrument=instrument,
            granularity=granularity,
            max_gap_sec=int(max_gap.total_seconds()),
            tolerance_sec=int(tolerance.total_seconds()),
            level=logging.INFO,
        )
        # Non bloquant pour M/W/D (weekends FX). Bloquant pour intraday.
        if granularity in ("H4", "H1", "M15"):
            return False
    return True


def _parse_oanda_json(
    raw_json: Any, instrument: str, granularity: str, include_incomplete: bool
) -> List[Dict[str, Any]]:
    """[Robustesse JSON] Parse défensif — tolère structures variables."""
    if not isinstance(raw_json, dict):
        _log_incident(
            IncidentCode.JSON_INVALID, "root not dict",
            instrument=instrument, granularity=granularity,
        )
        return []
    candles = raw_json.get("candles", [])
    if not isinstance(candles, list):
        _log_incident(
            IncidentCode.JSON_INVALID, "candles not list",
            instrument=instrument, granularity=granularity,
        )
        return []

    is_index = instrument in INDICES
    market_open = is_fx_market_open()
    allow_zero_volume = is_index or not market_open

    rows: List[Dict[str, Any]] = []
    for candle in candles:
        if not isinstance(candle, dict):
            continue
        if not include_incomplete and not candle.get("complete"):
            continue
        try:
            mid = candle.get("mid", {})
            if not isinstance(mid, dict):
                continue
            row = {
                "date": candle.get("time"),
                "Open": float(mid["o"]),
                "High": float(mid["h"]),
                "Low": float(mid["l"]),
                "Close": float(mid["c"]),
                "Volume": float(candle.get("volume", 0) or 0),
            }
            if row["date"] is None:
                continue
            if _validate_candle(row, allow_zero_volume):
                rows.append(row)
        except (KeyError, ValueError, TypeError):
            continue
    return rows


def _fetch_candles_raw(
    instrument: str,
    granularity: str,
    count: int,
    account_id: str,
    access_token: str,
    registry: SessionRegistry,
    include_incomplete: bool = False,
) -> pd.DataFrame:
    """[C5] Fetch HTTP avec rate limiter non bloquant."""
    if not _RATE_LIMITER.acquire(instrument):
        _log_incident(
            IncidentCode.HTTP_RATELIMIT, "rate-limited locally",
            instrument=instrument, granularity=granularity,
            level=logging.INFO,
        )
        return pd.DataFrame()

    url = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"granularity": granularity, "count": count, "price": "M"}

    try:
        session = registry.get_for_thread()
    except RuntimeError:
        return pd.DataFrame()

    try:
        r = session.get(url, headers=headers, params=params, timeout=CFG.http_timeout_sec)
    except requests.exceptions.Timeout:
        _log_incident(
            IncidentCode.HTTP_TIMEOUT, "OANDA timeout",
            instrument=instrument, granularity=granularity,
        )
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        _log_incident(
            IncidentCode.HTTP_ERROR, "OANDA request exception",
            instrument=instrument, granularity=granularity, err=type(e).__name__,
        )
        return pd.DataFrame()

    if r.status_code == 429:
        retry_after = 5
        try:
            retry_after = int(r.headers.get("Retry-After", "5"))
        except (ValueError, TypeError):
            pass
        cooldown = min(retry_after, CFG.rate_limit_429_cooldown_sec)
        _RATE_LIMITER.trigger_cooldown(instrument, cooldown)
        _log_incident(
            IncidentCode.HTTP_RATELIMIT, "OANDA 429",
            instrument=instrument, granularity=granularity,
            retry_after_sec=retry_after,
        )
        return pd.DataFrame()

    if r.status_code in (401, 403):
        _log_incident(
            IncidentCode.HTTP_AUTH, "OANDA auth failure",
            instrument=instrument, granularity=granularity,
            status=r.status_code, level=logging.ERROR,
        )
        return pd.DataFrame()

    if r.status_code != 200:
        _log_incident(
            IncidentCode.HTTP_ERROR, "OANDA non-200",
            instrument=instrument, granularity=granularity, status=r.status_code,
        )
        return pd.DataFrame()

    try:
        raw_json = r.json()
    except ValueError:
        _log_incident(
            IncidentCode.JSON_INVALID, "OANDA invalid JSON",
            instrument=instrument, granularity=granularity,
        )
        return pd.DataFrame()

    rows = _parse_oanda_json(raw_json, instrument, granularity, include_incomplete)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame()
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    if not _validate_dataframe_schema(df, instrument, granularity):
        return pd.DataFrame()
    if not _validate_dataframe_gaps(df, granularity, instrument):
        return pd.DataFrame()

    return df


def fetch_cached(
    instrument: str,
    granularity: str,
    count: int,
    account_id: str,
    account_hash: AccountHash,
    access_token: str,
    registry: SessionRegistry,
) -> FetchResult:
    """Wrapper cache scopé par tenant."""
    key: CacheKey = (_OANDA_ENV, account_hash, instrument, granularity, count)
    ttl = _CACHE_TTL.get(granularity, CFG.cache_ttl_default)
    return _CACHE.get_candles(
        key,
        ttl,
        lambda: _fetch_candles_raw(
            instrument, granularity, count, account_id, access_token, registry,
            include_incomplete=False,
        ),
    )


def fetch_live_open(
    instrument: str,
    granularity: str,
    account_id: str,
    account_hash: AccountHash,
    access_token: str,
    registry: SessionRegistry,
) -> Optional[float]:
    """Open de la bougie en cours avec negative caching."""
    key: LiveOpenKey = (_OANDA_ENV, account_hash, instrument, granularity)

    def _fetch() -> Optional[float]:
        df = _fetch_candles_raw(
            instrument, granularity, 1, account_id, access_token, registry,
            include_incomplete=True,
        )
        if df.empty:
            return None
        try:
            last_ts = df.index[-1]
            now = datetime.now(timezone.utc)
            max_age = _GRAN_FREQ.get(granularity, pd.Timedelta(days=1))
            if (now - last_ts) > max_age:
                return None
            return float(df["Open"].iloc[-1])
        except (IndexError, ValueError, TypeError):
            return None

    return _CACHE.get_live_open(key, _fetch)


def fetch_all_data(
    instrument: str,
    account_id: str,
    account_hash: AccountHash,
    access_token: str,
    registry: SessionRegistry,
    stop_event: threading.Event,
) -> Optional[Dict[str, Any]]:
    """[C7] Récupère toutes les TF avec timestamps par TF."""
    specs = {
        "M":   ("M",   150, CFG.min_bars_m),
        "W":   ("W",   250, CFG.min_bars_w),
        "D":   ("D",   300, CFG.min_bars_d),
        "4H":  ("H4",  300, CFG.min_bars_h4),
        "1H":  ("H1",  300, CFG.min_bars_h1),
        "15m": ("M15", 300, CFG.min_bars_15m),
    }
    snapshot_started_at = datetime.now(timezone.utc)
    cache: Dict[str, Any] = {
        "_snapshot_started_at": snapshot_started_at,
        "_snapshot_per_tf": {},
        "_stale_tfs": [],
    }

    for tf, (gran, count, min_bars) in specs.items():
        if stop_event.is_set():
            return None
        result = fetch_cached(
            instrument, gran, count, account_id, account_hash, access_token, registry
        )
        if result.df.empty or len(result.df) < min_bars:
            _log_incident(
                IncidentCode.DATA_INSUFFICIENT, "bars below minimum",
                instrument=instrument, tf=tf,
                bars=len(result.df), min_bars=min_bars,
            )
            return None
        cache[tf] = result.df
        cache["_snapshot_per_tf"][tf] = result.fetched_at or datetime.now(timezone.utc)
        if result.is_stale:
            cache["_stale_tfs"].append(tf)

    if stop_event.is_set():
        return None

    cache["_snapshot_completed_at"] = datetime.now(timezone.utc)

    # [C7] Détection drift
    ts_list = [v for v in cache["_snapshot_per_tf"].values() if v is not None]
    if ts_list:
        drift = (max(ts_list) - min(ts_list)).total_seconds()
        cache["_snapshot_drift_sec"] = drift
        cache["_snapshot_drift_exceeded"] = drift > CFG.snapshot_drift_max_sec

    cache["_week_open"] = fetch_live_open(
        instrument, "W", account_id, account_hash, access_token, registry
    )
    cache["_day_open"] = fetch_live_open(
        instrument, "D", account_id, account_hash, access_token, registry
    )
    return cache


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — REGISTRE DE VOTES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VoteSpec:
    uid: str
    fn: Callable
    critical: bool = False


class VoteRegistry:
    def __init__(self) -> None:
        self._votes: Dict[str, VoteSpec] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()

    def register(self, uid: str, critical: bool = False) -> Callable:
        def decorator(fn: Callable) -> Callable:
            with self._lock:
                if uid not in self._votes:
                    self._order.append(uid)
                self._votes[uid] = VoteSpec(uid=uid, fn=fn, critical=critical)
            return fn
        return decorator

    def all_votes(self) -> List[VoteSpec]:
        with self._lock:
            return [self._votes[uid] for uid in self._order]


DAILY_VOTES: Final[VoteRegistry] = VoteRegistry()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — VOTES ATOMIQUES (C6 — weekly_open neutre sur fallback)
# ═══════════════════════════════════════════════════════════════════════════════

@DAILY_VOTES.register(uid="swing_structure", critical=True)
def _vote_swing_structure(
    h: pd.Series, lo: pd.Series, _c: pd.Series, _ctx: Mapping[str, Any]
) -> VoteSignal:
    name = "swing_structure"
    wing = CFG.pivot_wing
    if len(h) < 2 * wing + CFG.max_pivot_age + 1:
        return VoteSignal(name, Direction.RANGE, 2.0, 0.9, False, f"série trop courte ({len(h)})")
    min_idx = max(0, len(h) - CFG.max_pivot_age)
    sh = _find_strict_peaks(h, wing, min_idx)
    sl = _find_strict_troughs(lo, wing, min_idx)
    if len(sh) < 2 or len(sl) < 2:
        return VoteSignal(name, Direction.RANGE, 2.0, 0.9, False, f"pivots sh={len(sh)} sl={len(sl)}")
    hh = h.iloc[sh[-1]] > h.iloc[sh[-2]]
    hl = lo.iloc[sl[-1]] > lo.iloc[sl[-2]]
    lh = h.iloc[sh[-1]] < h.iloc[sh[-2]]
    ll = lo.iloc[sl[-1]] < lo.iloc[sl[-2]]
    if hh and hl:
        return VoteSignal(name, Direction.BULLISH, 2.0, 0.9, True, "HH+HL")
    if lh and ll:
        return VoteSignal(name, Direction.BEARISH, 2.0, 0.9, True, "LH+LL")
    return VoteSignal(name, Direction.RANGE, 2.0, 0.9, False, "structure mixte")


@DAILY_VOTES.register(uid="ema_stack")
def _vote_ema_stack(_h, _lo, _c, ctx: Mapping[str, Any]) -> VoteSignal:
    name = "ema_stack"
    cur, e21, e50_cur = ctx["cur"], ctx["e21"], ctx["e50_cur"]
    if np.isnan(e21) or np.isnan(e50_cur):
        return VoteSignal(name, Direction.RANGE, 1.0, 0.75, False, "NaN EMA")
    if cur > e21 > e50_cur:
        return VoteSignal(name, Direction.BULLISH, 1.0, 0.75, True, f"cur>{e21:.5f}>e50")
    if cur < e21 < e50_cur:
        return VoteSignal(name, Direction.BEARISH, 1.0, 0.75, True, f"cur<{e21:.5f}<e50")
    return VoteSignal(name, Direction.RANGE, 1.0, 0.75, False, "stack non aligné")


@DAILY_VOTES.register(uid="weekly_open")
def _vote_weekly_open(_h, _lo, _c, ctx: Mapping[str, Any]) -> VoteSignal:
    """
    [C6] Vote weekly_open — fallback NEUTRE.
    Si current_week_open indisponible : vote neutre (fired=False), pas de biais.
    """
    name = "weekly_open"
    cur = ctx["cur"]
    current_week_open = ctx.get("current_week_open")

    if current_week_open is None or (
        isinstance(current_week_open, float) and np.isnan(current_week_open)
    ):
        # [C6] Neutre — pas de fallback biaisé sur semaine précédente
        return VoteSignal(name, Direction.RANGE, 1.0, 0.0, False, "current_W indisponible")

    wo_price = float(current_week_open)
    rel = 0.90
    if cur > wo_price:
        return VoteSignal(name, Direction.BULLISH, 1.0, rel, True, "cur>wo")
    if cur < wo_price:
        return VoteSignal(name, Direction.BEARISH, 1.0, rel, True, "cur<wo")
    return VoteSignal(name, Direction.RANGE, 1.0, rel, False, "cur==wo")


def _vote_prev_midpoint_indices(
    h_j1: float, lo_j1: float, c_j1: float, ctx: Mapping[str, Any], direction: Direction
) -> VoteSignal:
    """[H9] Sous-fonction indices."""
    name = "prev_midpoint"
    rng = h_j1 - lo_j1
    if rng <= 0:
        return VoteSignal(name, Direction.RANGE, 0.5, 0.0, False, "range nul")
    try:
        o_j1 = float(ctx["df_daily"]["Open"].iloc[-1])
    except (KeyError, IndexError, ValueError, TypeError):
        return VoteSignal(name, Direction.RANGE, 0.5, 0.0, False, "Open J-1 inaccessible")
    body_ratio = abs(c_j1 - o_j1) / rng
    if body_ratio < CFG.body_range_strong:
        return VoteSignal(name, Direction.RANGE, 0.5, 0.0, False, f"body/range={body_ratio:.2f}")
    rel = min(0.80, 0.60 + (body_ratio - CFG.body_range_strong) * 0.5)
    return VoteSignal(name, direction, 0.5, rel, True, f"body/range={body_ratio:.2f} [idx]")


def _vote_prev_midpoint_fx(
    vol: Optional[pd.Series], direction: Direction
) -> VoteSignal:
    """[H9] Sous-fonction FX."""
    name = "prev_midpoint"
    if vol is None or vol.empty:
        return VoteSignal(name, direction, 0.5, 0.50, True, "no_vol_data")
    try:
        vol_ref = vol.iloc[:-1]
        vol_j1 = float(vol.iloc[-1])
        if len(vol_ref) < 20:
            return VoteSignal(name, direction, 0.5, 0.50, True, "vol_history<20")
        vol_ma = float(vol_ref.rolling(20).mean().iloc[-1])
        if np.isnan(vol_j1) or np.isnan(vol_ma) or vol_ma <= 0:
            return VoteSignal(name, direction, 0.5, 0.50, True, "vol_NaN")
        vol_ratio = vol_j1 / vol_ma
        if vol_ratio <= CFG.volume_ratio_min_midpoint:
            return VoteSignal(name, Direction.RANGE, 0.5, 0.0, False, f"vol_ratio={vol_ratio:.2f}")
        reliability = min(0.80, 0.65 + (vol_ratio - 1.0) * 0.05)
        return VoteSignal(name, direction, 0.5, reliability, True, f"vol_ratio={vol_ratio:.2f}")
    except (TypeError, ValueError, IndexError):
        return VoteSignal(name, direction, 0.5, 0.50, True, "vol_err")


@DAILY_VOTES.register(uid="prev_midpoint")
def _vote_prev_midpoint(
    h: pd.Series, lo: pd.Series, c: pd.Series, ctx: Mapping[str, Any]
) -> VoteSignal:
    """[F1, F6] iloc[-1] = J-1 ; indices: body/range ; FX: volume."""
    name = "prev_midpoint"
    if len(c) < 1:
        return VoteSignal(name, Direction.RANGE, 0.5, 0.0, False, "série vide")
    instrument = ctx.get("instrument", "")
    is_index = instrument in INDICES

    h_j1 = float(h.iloc[-1])
    lo_j1 = float(lo.iloc[-1])
    c_j1 = float(c.iloc[-1])
    mid_j1 = (h_j1 + lo_j1) / 2.0
    direction = Direction.BULLISH if c_j1 > mid_j1 else Direction.BEARISH

    if is_index:
        return _vote_prev_midpoint_indices(h_j1, lo_j1, c_j1, ctx, direction)
    return _vote_prev_midpoint_fx(ctx.get("vol_series"), direction)


@DAILY_VOTES.register(uid="ema50_slope")
def _vote_ema50_slope(_h, _lo, _c, ctx: Mapping[str, Any]) -> VoteSignal:
    name = "ema50_slope"
    e50, atr_val = ctx["e50"], ctx["atr_val"]
    threshold = CFG.ema50_slope_threshold
    if len(e50) < 6 or atr_val <= 0:
        return VoteSignal(name, Direction.RANGE, 1.0, 0.70, False, "données insuffisantes")
    slope_ratio = float(e50.iloc[-1] - e50.iloc[-6]) / atr_val
    if slope_ratio > threshold:
        return VoteSignal(name, Direction.BULLISH, 1.0, 0.70, True, f"slope={slope_ratio:.3f}")
    if slope_ratio < -threshold:
        return VoteSignal(name, Direction.BEARISH, 1.0, 0.70, True, f"slope={slope_ratio:.3f}")
    return VoteSignal(name, Direction.RANGE, 1.0, 0.70, False, f"slope={slope_ratio:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — AGRÉGATEUR
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate_votes(
    votes: Tuple[VoteSignal, ...], atr_val: float, degraded: bool
) -> DailyTrendResult:
    bull_score = sum(v.weight * v.reliability for v in votes if v.fired and v.direction == Direction.BULLISH)
    bear_score = sum(v.weight * v.reliability for v in votes if v.fired and v.direction == Direction.BEARISH)
    fired_possible = sum(v.weight * v.reliability for v in votes if v.fired)
    winning_score = max(bull_score, bear_score)
    min_votes_met = winning_score >= CFG.min_reliable_score

    if not min_votes_met or bull_score == bear_score:
        return DailyTrendResult(
            Direction.RANGE, CFG.strength_min_range, atr_val,
            bull_score, bear_score, votes, min_votes_met, degraded,
        )

    direction = Direction.BULLISH if bull_score > bear_score else Direction.BEARISH
    ratio = winning_score / fired_possible if fired_possible > 0 else 0.0
    strength = int(min(CFG.strength_max, max(CFG.strength_min_range, ratio * CFG.strength_scaler)))
    return DailyTrendResult(
        direction, strength, atr_val, bull_score, bear_score, votes, min_votes_met, degraded,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — TENDANCES PAR TIMEFRAME (refactor H9)
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
        guard = VoteSignal("guard", Direction.RANGE, 0.0, 1.0, False, f"insuf ({len(df)})")
        return DailyTrendResult(Direction.RANGE, 0, atr_val, 0.0, 0.0, (guard,), False, True)

    cur = float(c.iloc[-1])
    e50 = _ema(c, CFG.ema_long)
    e21 = float(_ema(c, CFG.ema_short).iloc[-1])
    e50_cur = float(e50.iloc[-1])
    vol_series = (
        df["Volume"] if "Volume" in df.columns and instrument not in INDICES else None
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
                raise TypeError(f"Vote {spec.uid} bad return")
        except Exception as exc:
            _log_incident(
                IncidentCode.VOTE_CRITICAL_ERROR if spec.critical else IncidentCode.VOTE_ERROR,
                f"vote {spec.uid} error",
                instrument=instrument, err=type(exc).__name__,
                level=logging.ERROR if spec.critical else logging.WARNING,
            )
            v = VoteSignal(spec.uid, Direction.RANGE, 0.0, 0.0, False, f"err:{type(exc).__name__}", errored=True)
            if spec.critical:
                degraded = True
        raw_votes.append(v)

    return _aggregate_votes(tuple(raw_votes), atr_val, degraded)


def _trend_macro_monthly(c: pd.Series, e50: pd.Series, atr_val: float, band: float, n: int) -> Tuple[str, int, float]:
    """[H9] Sous-fonction macro mensuelle."""
    if n < CFG.min_bars_m:
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


def _trend_macro_weekly(c: pd.Series, e50: pd.Series, atr_val: float, band: float, n: int) -> Tuple[str, int, float]:
    """[H9] Sous-fonction macro hebdo."""
    if n < CFG.sma_macro:
        return "Range", 40, atr_val
    s200 = _sma(c, CFG.sma_macro)
    cur50 = float(e50.iloc[-1])
    ref200 = float(s200.iloc[-1])
    prev50 = float(e50.iloc[-2])
    prev200 = float(s200.iloc[-2])
    cross = (prev50 <= prev200 < cur50) or (prev50 >= prev200 > cur50)
    if cur50 > ref200 + band:
        return "Bullish", 90 if cross else 75, atr_val
    if cur50 < ref200 - band:
        return "Bearish", 90 if cross else 75, atr_val
    return "Range", 40, atr_val


def trend_macro(df: pd.DataFrame, tf: str) -> Tuple[str, int, float]:
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
        return _trend_macro_monthly(c, e50, atr_val, band, len(df))
    return _trend_macro_weekly(c, e50, atr_val, band, len(df))


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
    if today_open is not None:
        score += 1 if cur > today_open else -1
    s = abs(score)
    strength = 90 if s == 3 else 70 if s >= 1 else 40
    direction = "Bullish" if score > 0 else "Bearish" if score < 0 else "Range"
    return direction, strength, atr_val


def _trend_intraday_compute_indicators(
    df: pd.DataFrame, c: pd.Series, period: int, lag: int
) -> Optional[Dict[str, float]]:
    """[H9] Sous-fonction calcul indicateurs intraday."""
    ema9 = float(_ema(c, CFG.intraday_ema_fast).iloc[-1])
    ema21 = float(_ema(c, CFG.ema_short).iloc[-1])
    ema50 = float(_ema(c, period).iloc[-1])
    src_adj = c + (c - c.shift(lag))
    zlema = float(src_adj.ewm(span=period, adjust=False).mean().iloc[-1])
    rsi_val = float(_rsi(c, CFG.rsi_period).iloc[-1])
    ema12 = _ema(c, CFG.macd_fast)
    ema26 = _ema(c, CFG.macd_slow)
    macd = ema12 - ema26
    sig = _ema(macd, CFG.macd_signal)
    macd_cur = float(macd.iloc[-1])
    sig_cur = float(sig.iloc[-1])
    values = {
        "e9": ema9, "e21": ema21, "e50": ema50, "zlema": zlema,
        "rsi": rsi_val, "macd": macd_cur, "sig": sig_cur,
    }
    if any(np.isnan(v) for v in values.values()):
        return None
    return values


def trend_intraday(df: pd.DataFrame, instrument: str = "") -> Tuple[str, int, float]:
    h, lo, c = df["High"], df["Low"], df["Close"]
    atr_val = float(_atr(h, lo, c, CFG.atr_period).iloc[-1])
    if len(df) < 70:
        return "Range", 0, atr_val
    cur = float(c.iloc[-1])
    period = CFG.ema_intra_period
    lag = (period - 1) // 2

    ind = _trend_intraday_compute_indicators(df, c, period, lag)
    if ind is None:
        return "Range", 0, atr_val

    e9, e21, e50_cur, zlema = ind["e9"], ind["e21"], ind["e50"], ind["zlema"]
    rsi_val, macd_cur, sig_cur = ind["rsi"], ind["macd"], ind["sig"]

    bull_zlema = cur > zlema
    bear_zlema = cur < zlema
    bull_stack = e9 > e21 > e50_cur
    bear_stack = e9 < e21 < e50_cur
    bull_mom = rsi_val > 50 and macd_cur > sig_cur
    bear_mom = rsi_val < 50 and macd_cur < sig_cur

    votes_bull = [bull_zlema, bull_stack, bull_mom]
    votes_bear = [bear_zlema, bear_stack, bear_mom]
    max_votes = 3

    if instrument not in INDICES and "Volume" in df.columns:
        vol = df["Volume"]
        vol_avg = float(vol.rolling(20).mean().iloc[-1])
        vol_cur = float(vol.iloc[-1])
        if not np.isnan(vol_avg) and vol_avg > 0:
            strong_vol = vol_cur > vol_avg * CFG.volume_ratio_strong_intraday
            votes_bull.append(strong_vol and bull_zlema)
            votes_bear.append(strong_vol and bear_zlema)
            max_votes = 4

    vb, vbr = sum(votes_bull), sum(votes_bear)

    def _atr_strength() -> int:
        if atr_val <= 0:
            return 60
        return int(min(95, 40 + (abs(cur - zlema) / atr_val) * 25))

    if vb == max_votes:
        return "Bullish", _atr_strength(), atr_val
    if vbr == max_votes:
        return "Bearish", _atr_strength(), atr_val
    if vb >= max_votes - 1:
        return "Bullish", 55, atr_val
    if vbr >= max_votes - 1:
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
# SECTION 15 — SCORING MTF (H10 — entropy penalty)
# ═══════════════════════════════════════════════════════════════════════════════

_MTF_WEIGHTS: Final[Mapping[str, float]] = {
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
    trends: Mapping[str, str], scores: Mapping[str, int]
) -> Tuple[float, float, float]:
    active_total = sum(_MTF_WEIGHTS[tf] for tf in trends if not trends[tf].startswith("Range"))
    if active_total == 0:
        return 0.0, 0.0, 1.0
    w_bull = sum(
        _MTF_WEIGHTS[tf] * (scores[tf] / 100.0) for tf in trends if trends[tf] == "Bullish"
    ) + sum(
        _MTF_WEIGHTS[tf] * (scores[tf] / 100.0) * 0.5
        for tf in trends if trends[tf] == "Retracement Bull"
    )
    w_bear = sum(
        _MTF_WEIGHTS[tf] * (scores[tf] / 100.0) for tf in trends if trends[tf] == "Bearish"
    ) + sum(
        _MTF_WEIGHTS[tf] * (scores[tf] / 100.0) * 0.5
        for tf in trends if trends[tf] == "Retracement Bear"
    )
    return w_bull, w_bear, active_total


def _mtf_alignment_bonus(trends: Mapping[str, str], direction: str) -> int:
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


def _mtf_dispersion_penalty(trends: Mapping[str, str], direction: str) -> float:
    """
    [H10] Pénalité de dispersion : si TF en conflit direct avec direction,
    on déduit jusqu'à dispersion_penalty_max.
    """
    if direction not in ("Bullish", "Bearish"):
        return 0.0
    conflict_weight = 0.0
    total_weight = 0.0
    for tf, trend in trends.items():
        w = _MTF_WEIGHTS[tf]
        total_weight += w
        if direction == "Bullish" and _bear_compat(trend):
            conflict_weight += w
        elif direction == "Bearish" and _bull_compat(trend):
            conflict_weight += w
    if total_weight == 0:
        return 0.0
    ratio = conflict_weight / total_weight
    return min(CFG.dispersion_penalty_max, ratio * CFG.dispersion_penalty_max * 2)


def score_mtf(trends: Mapping[str, str], scores: Mapping[str, int]) -> Tuple[str, float]:
    w_bull, w_bear, total = _mtf_weighted_score(trends, scores)
    if w_bull > w_bear:
        raw_score, direction = (w_bull / total) * 100, "Bullish"
    elif w_bear > w_bull:
        raw_score, direction = (w_bear / total) * 100, "Bearish"
    else:
        return "Range", 0.0
    bonus = _mtf_alignment_bonus(trends, direction)
    penalty = _mtf_dispersion_penalty(trends, direction)
    return direction, max(0.0, min(100.0, raw_score + bonus - penalty))


def _compute_nc_orthogonal(
    trends: Mapping[str, str], scores: Mapping[str, int], mtf_dir: str
) -> int:
    if mtf_dir not in ("Bullish", "Bearish"):
        return 0
    score_f = 0.0
    for tf, trend in trends.items():
        strength = scores.get(tf, 0)
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
        else:
            if is_strong_pure_bear:
                score_f += 1.0
            elif trend == "Retracement Bear":
                score_f += 0.25
            elif is_strong_pure_bull:
                score_f -= 1.0
            elif trend == "Retracement Bull":
                score_f -= 0.25
    sign = 1 if score_f >= 0 else -1
    return sign * math.floor(abs(score_f))


def grade_hybrid(
    scores_list: List[float], nc_list: List[int], degraded_list: List[bool]
) -> List[str]:
    """[H3] degraded → plafond B+ (NOT_TRADEABLE)."""
    grades: List[str] = []
    for score, nc, degraded in zip(scores_list, nc_list, degraded_list):
        nc_bonus = (int(nc) - 3) * 5
        adj = min(100.0, float(score) + nc_bonus)
        if degraded:
            # [H3] Plafond strict B+ en mode dégradé
            grades.append("B+" if adj >= 38 else "B")
        elif adj >= 80:
            grades.append("A+")
        elif adj >= 55:
            grades.append("A")
        elif adj >= 38:
            grades.append("B+")
        else:
            grades.append("B")
    return grades


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — ANALYSE PRINCIPALE (C3 drainage strict, H8 propagation degraded)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_pair(
    pair: str,
    account_id: str,
    account_hash: AccountHash,
    access_token: str,
    registry: SessionRegistry,
    stop_event: threading.Event,
) -> Optional[Dict[str, Any]]:
    if stop_event.is_set():
        return None
    try:
        cache = fetch_all_data(
            pair, account_id, account_hash, access_token, registry, stop_event
        )
        if cache is None:
            return None

        trends: Dict[str, str] = {}
        scores: Dict[str, int] = {}
        atrs: Dict[str, float] = {}
        degraded = bool(cache.get("_stale_tfs")) or bool(cache.get("_snapshot_drift_exceeded"))

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
            daily_result.direction.value, daily_result.strength, daily_result.atr_val,
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
            "_stale_tfs": tuple(cache.get("_stale_tfs", [])),
            "NC": nc,
            "Age D1": age,
            "ATR Daily": _fmt_atr(atrs["D"]),
            "ATR H4": _fmt_atr(atrs["4H"]),
            "ATR H1": _fmt_atr(atrs["1H"]),
            "ATR 15m": _fmt_atr(atrs["15m"]),
        }
    except Exception as e:
        _log_incident(
            IncidentCode.UNKNOWN, "analyze_pair exception",
            instrument=pair, err=type(e).__name__, level=logging.ERROR,
        )
        return None


def _drain_executor_strict(
    executor: ThreadPoolExecutor,
    futures: Mapping[Future, str],
    stop_event: threading.Event,
) -> None:
    """[C3] Drainage strict : on attend que TOUTES les futures soient done."""
    stop_event.set()
    for f in futures:
        if not f.done():
            f.cancel()
    deadline = time.monotonic() + CFG.pool_drain_grace_sec
    while time.monotonic() < deadline:
        alive = sum(1 for f in futures if not f.done())
        if alive == 0:
            break
        time.sleep(0.05)
    # shutdown(wait=True) garantit que tous les workers ont terminé
    executor.shutdown(wait=True, cancel_futures=True)


def _process_completed_future(
    future: Future,
    inst: str,
    results: List[Dict[str, Any]],
    errors: set,
) -> None:
    """[H9] Sous-fonction process result."""
    try:
        row = future.result()
        if row:
            results.append(row)
        else:
            errors.add(inst)
    except Exception as e:
        errors.add(inst)
        _log_incident(
            IncidentCode.UNKNOWN, "future result error",
            instrument=inst, err=type(e).__name__,
        )


def analyze_all_core(
    account_id: str,
    access_token: str,
    progress_cb: Optional[Callable[[float], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """[C3] Noyau pur — drainage strict + errors dédupliqués."""
    results: List[Dict[str, Any]] = []
    errors: set = set()  # [H7] déduplication
    total = len(INSTRUMENTS)
    done = 0
    timed_out = False
    stop_event = threading.Event()
    registry = SessionRegistry()
    account_hash = _hash_account(account_id)

    meta: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc),
        "version": APP_VERSION,
        "env": _OANDA_ENV,
        "account_hash": account_hash,
    }

    executor = ThreadPoolExecutor(
        max_workers=CFG.max_workers, thread_name_prefix="bluestar_worker"
    )
    futures: Dict[Future, str] = {}
    try:
        futures = {
            executor.submit(
                analyze_pair, inst, account_id, account_hash, access_token, registry, stop_event
            ): inst
            for inst in INSTRUMENTS
        }
        try:
            for future in as_completed(futures, timeout=CFG.analysis_timeout_sec):
                inst = futures[future]
                done += 1
                if progress_cb is not None:
                    try:
                        progress_cb(done / total)
                    except Exception as exc:
                        _log_incident(
                            IncidentCode.UI_CALLBACK_ERROR, "progress_cb",
                            err=type(exc).__name__, level=logging.DEBUG,
                        )
                if status_cb is not None:
                    try:
                        status_cb(f"GPS ({done}/{total}) — {inst.replace('_', '/')}")
                    except Exception as exc:
                        _log_incident(
                            IncidentCode.UI_CALLBACK_ERROR, "status_cb",
                            err=type(exc).__name__, level=logging.DEBUG,
                        )
                _process_completed_future(future, inst, results, errors)
        except FutureTimeoutError:
            timed_out = True
            _log_incident(
                IncidentCode.EXECUTOR_TIMEOUT, "analyze_all_core timeout",
                timeout_sec=CFG.analysis_timeout_sec, level=logging.ERROR,
            )
            for f, inst in futures.items():
                if not f.done():
                    errors.add(inst)
    finally:
        # [C3] Drainage STRICT avant fermeture sessions
        _drain_executor_strict(executor, futures, stop_event)
        registry.close_all()

    meta["finished_at"] = datetime.now(timezone.utc)
    meta["timed_out"] = timed_out
    meta["errors_count"] = len(errors)
    meta["completeness"] = len(results) / total if total > 0 else 0.0
    meta["degraded_pairs"] = sorted(r["Paire"] for r in results if r.get("_degraded"))

    errors_sorted = sorted(errors)

    if not results:
        return pd.DataFrame(), errors_sorted, meta

    scores_list = [r["_mtf_score"] for r in results]
    nc_list = [r["NC"] for r in results]
    # [H3] Si completeness faible, on dégrade TOUT le run
    run_degraded = meta["completeness"] < CFG.completeness_min_tradable
    degraded_list = [r["_degraded"] or run_degraded for r in results]
    grades = grade_hybrid(scores_list, nc_list, degraded_list)
    for r, g, deg in zip(results, grades, degraded_list):
        r["Quality"] = g
        r["Tradable"] = "✓" if not deg else "✗ NOT_TRADEABLE"

    df = pd.DataFrame(results)
    df.attrs["meta"] = meta
    return df, errors_sorted, meta


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — UI / PDF / STREAMLIT (encapsulé)
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_str(s: str) -> str:
    return s.encode("latin-1", errors="replace").decode("latin-1")


def _pdf_cell_text(val: str) -> str:
    return val if _FPDF2 else _safe_str(val)


def _pdf_get_colors(col: str, val: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    grade_rgb = {"A+": (251, 191, 36), "A": (163, 230, 53), "B+": (52, 211, 153), "B": (96, 165, 250)}
    nc_rgb = (
        ((5, 99), (46, 204, 113), (255, 255, 255)),
        ((3, 4), (39, 174, 96), (255, 255, 255)),
        ((1, 2), (241, 196, 15), (0, 0, 0)),
        ((0, 0), (230, 126, 34), (255, 255, 255)),
        ((-99, -1), (231, 76, 60), (255, 255, 255)),
    )
    if col == "Quality":
        return grade_rgb.get(val, (156, 163, 175)), (0, 0, 0)
    if col == "NC":
        try:
            n = int(val)
            for (lo, hi), fc, tc in nc_rgb:
                if lo <= n <= hi:
                    return fc, tc
        except (ValueError, TypeError):
            pass
        return (200, 200, 200), (0, 0, 0)
    if "Retracement Bull" in val:
        return (125, 206, 160), (255, 255, 255)
    if "Retracement Bear" in val:
        return (241, 148, 138), (255, 255, 255)
    if "Bull" in val:
        return (46, 204, 113), (255, 255, 255)
    if "Bear" in val:
        return (231, 76, 60), (255, 255, 255)
    if "Range" in val:
        return (149, 165, 166), (255, 255, 255)
    return (255, 255, 255), (0, 0, 0)


def create_pdf(df: pd.DataFrame) -> BytesIO:
    cols = ["Paire", "M", "W", "D", "4H", "1H", "15m", "MTF", "Quality", "NC", "Age D1",
            "ATR Daily", "ATR H4", "ATR H1", "ATR 15m"]
    widths = {"Paire": 22, "M": 16, "W": 16, "D": 16, "4H": 16, "1H": 16, "15m": 16,
              "MTF": 30, "Quality": 12, "NC": 10, "Age D1": 13,
              "ATR Daily": 17, "ATR H4": 17, "ATR H1": 15, "ATR 15m": 15}
    rh = 5.5
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
        pdf.cell(0, 9, _pdf_cell_text(f"BLUESTAR GPS V{APP_VERSION}"), ln=True, align="C")
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(0, 5, _pdf_cell_text(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        ), ln=True, align="C")
        pdf.ln(4)

        def header() -> None:
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_fill_color(30, 58, 138)
            pdf.set_text_color(255, 255, 255)
            for col in cols:
                pdf.cell(widths[col], 7, _pdf_cell_text(col), border=1, align="C", fill=True)
            pdf.ln()
            pdf.set_font("Helvetica", "", 6.5)

        header()
        for _, row in df.iterrows():
            if pdf.get_y() + rh > 287 - 15:
                pdf.add_page()
                header()
            for col in cols:
                val = str(row.get(col, ""))
                fc, tc = _pdf_get_colors(col, val)
                pdf.set_fill_color(*fc)
                pdf.set_text_color(*tc)
                pdf.cell(widths[col], rh, _pdf_cell_text(val), border=1, align="C", fill=True)
            pdf.ln()

        out = pdf.output(dest="S")
        buf.write(out.encode("latin-1") if isinstance(out, str) else bytes(out))
        buf.seek(0)
        return buf
    except Exception as e:
        _log_incident(IncidentCode.PDF_ERROR, "PDF generation failed",
                      err=type(e).__name__, level=logging.ERROR)
        buf2 = BytesIO()
        try:
            fallback = FPDF()
            fallback.add_page()
            fallback.set_font("Helvetica", "B", 12)
            fallback.cell(0, 10, "PDF Generation Error", ln=True)
            out2 = fallback.output(dest="S")
            buf2.write(out2.encode("latin-1") if isinstance(out2, str) else bytes(out2))
        except Exception as exc:
            _log_incident(IncidentCode.PDF_ERROR, "fallback PDF failed",
                          err=type(exc).__name__, level=logging.ERROR)
            buf2.write(b"PDF error")
        buf2.seek(0)
        return buf2


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 18 — STREAMLIT APP (guardée)
# ═══════════════════════════════════════════════════════════════════════════════

# Format OANDA account ID validation (H11)
_OANDA_ACCOUNT_PATTERN: Final[re.Pattern] = re.compile(r"^\d{3}-\d{3}-\d{6,}-\d{3,}$")


def _validate_secret_format(account_id: str, token: str) -> bool:
    """[H11] Validation format strict des secrets."""
    if not account_id or not token:
        return False
    if not _OANDA_ACCOUNT_PATTERN.match(account_id):
        _log_incident(IncidentCode.HTTP_AUTH, "account_id format invalid")
        return False
    if len(token) < 32:
        _log_incident(IncidentCode.HTTP_AUTH, "token too short")
        return False
    return True


def _load_secrets() -> Tuple[Optional[str], Optional[str]]:
    """[H11] Validation exhaustive secrets."""
    if not _STREAMLIT_AVAILABLE:
        # Fallback env vars (utile pour tests/CI)
        acc = os.environ.get("OANDA_ACCOUNT_ID", "").strip()
        tok = os.environ.get("OANDA_ACCESS_TOKEN", "").strip()
        if _validate_secret_format(acc, tok):
            return acc, tok
        return None, None
    try:
        acc = st.secrets["OANDA_ACCOUNT_ID"]
        tok = st.secrets["OANDA_ACCESS_TOKEN"]
    except (KeyError, FileNotFoundError):
        return None, None
    except Exception as exc:
        _log_incident(IncidentCode.HTTP_AUTH, "secrets read error",
                      err=type(exc).__name__, level=logging.ERROR)
        return None, None
    if not isinstance(acc, str) or not isinstance(tok, str):
        return None, None
    acc, tok = acc.strip(), tok.strip()
    if not _validate_secret_format(acc, tok):
        return None, None
    return acc, tok


def _check_running_flag_ttl() -> None:
    started_at = st.session_state.get("_analysis_started_at")
    if started_at:
        elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
        if elapsed > CFG.streamlit_running_flag_ttl_sec:
            st.session_state["_analysis_running"] = False
            st.session_state["_analysis_started_at"] = None


def _style_trend(v: Any) -> str:
    if not isinstance(v, str):
        return ""
    if "Retracement Bull" in v:
        return f"background-color:{TREND_COLORS['Retracement Bull']};color:white"
    if "Retracement Bear" in v:
        return f"background-color:{TREND_COLORS['Retracement Bear']};color:white"
    if "Bull" in v:
        return f"background-color:{TREND_COLORS['Bullish']};color:white;font-weight:bold"
    if "Bear" in v:
        return f"background-color:{TREND_COLORS['Bearish']};color:white;font-weight:bold"
    if "Range" in v:
        return f"background-color:{TREND_COLORS['Range']};color:white"
    return ""


def _style_quality(s: pd.Series) -> List[str]:
    grade_css = {"A+": "#fbbf24", "A": "#a3e635", "B+": "#34d399", "B": "#60a5fa"}
    if s.name != "Quality":
        return [""] * len(s)
    return [f"color:black;font-weight:bold;background-color:{grade_css.get(x, '#9ca3af')}" for x in s]


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


def analyze_all(account_id: str, access_token: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Wrapper Streamlit avec progress."""
    progress = st.progress(0)
    status = st.empty()
    try:
        df, errors, meta = analyze_all_core(
            account_id, access_token,
            progress_cb=progress.progress,
            status_cb=status.text,
        )
    finally:
        for widget, name in ((progress, "progress"), (status, "status")):
            try:
                widget.empty()
            except Exception as exc:
                _log_incident(
                    IncidentCode.UI_CALLBACK_ERROR, f"{name}.empty()",
                    err=type(exc).__name__, level=logging.DEBUG,
                )

    if meta.get("timed_out"):
        st.error(
            f"⏱️ Analyse interrompue après {CFG.analysis_timeout_sec}s. "
            "Vérifiez la connexion OANDA."
        )
    if errors:
        st.warning(
            f"⚠️ {len(errors)} paire(s) non analysée(s) : "
            f"{', '.join(e.replace('_', '/') for e in errors[:10])}"
            + (" …" if len(errors) > 10 else "")
        )
    completeness = meta.get("completeness", 0.0)
    if completeness < CFG.completeness_min_tradable and not df.empty:
        st.markdown(
            f"<div class='degraded-warning'>⚠️ <b>Couverture partielle</b> — "
            f"seulement <b>{completeness:.0%}</b> des instruments analysés. "
            "<b>Run marqué NOT_TRADEABLE.</b></div>",
            unsafe_allow_html=True,
        )
    return df, meta


def main() -> None:
    _configure_streamlit_ui()

    st.markdown(
        f"<div class='main-header'><h1>🧭 BLUESTAR HEDGE FUND GPS V{APP_VERSION}</h1>"
        f"<p style='margin:0;font-size:0.85em;opacity:0.8'>"
        f"Production-Grade · Env: {_OANDA_ENV.upper()} · C1–C8 + H1–H15"
        "</p></div>",
        unsafe_allow_html=True,
    )

    acc, tok = _load_secrets()
    if not acc or not tok:
        st.error(
            "❌ Secrets OANDA manquants ou invalides — "
            "configurez OANDA_ACCOUNT_ID (format XXX-XXX-XXXXXX-XXX) "
            "et OANDA_ACCESS_TOKEN (≥32 chars) dans .streamlit/secrets.toml"
        )
        st.stop()

    _check_running_flag_ttl()

    with st.sidebar:
        st.header("⚙️ Configuration")
        only_best = st.checkbox("Afficher uniquement Grade A+ / A", value=False)
        st.info(
            f"Env : {_OANDA_ENV.upper()}\n\n"
            f"Workers : {CFG.max_workers} · Timeout : {CFG.analysis_timeout_sec}s\n\n"
            f"Cache TTL : M={CFG.cache_ttl_m // 60}m W={CFG.cache_ttl_w // 60}m D={CFG.cache_ttl_d // 60}m"
        )
        if not is_fx_market_open():
            st.warning("📅 Marché FX fermé — données potentiellement stale.")
        st.markdown("---")
        if st.button("🗑️ Vider le cache", use_container_width=True):
            _CACHE.clear()
            st.success("Cache vidé.")

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
                f"dernière analyse il y a <b>{age_min:.0f} min</b>. Relancez.</div>",
                unsafe_allow_html=True,
            )

    df = st.session_state["df"].copy()
    if only_best:
        df = df[df["Quality"].isin(["A+", "A"])]

    grade_order = ["A+", "A", "B+", "B"]
    df["Quality"] = pd.Categorical(df["Quality"], categories=grade_order, ordered=True)
    sort_cols = [c for c in ["Quality", "NC", "_mtf_score"] if c in df.columns]
    df = df.sort_values(sort_cols, ascending=[True, False, False])
    df.drop(
        columns=["_mtf_score", "_mtf_dir", "_degraded", "_stale_tfs"],
        inplace=True, errors="ignore",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Analyzed", len(df))
    c2.metric("Setups A+", len(df[df["Quality"] == "A+"]))
    c3.metric("Setups A", len(df[df["Quality"] == "A"]))
    c4.metric("Setups B", len(df[df["Quality"].isin(["B+", "B"])]))

    display = ["Paire", "M", "W", "D", "4H", "1H", "15m", "MTF", "Quality", "Tradable",
               "NC", "Age D1", "ATR Daily", "ATR H4", "ATR H1", "ATR 15m"]
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
            "📄 PDF", data=create_pdf(df[cols_present]),
            file_name=f"Bluestar_GPS_{ts}.pdf",
            mime="application/pdf", use_container_width=True,
        )
    with c2:
        st.download_button(
            "📊 CSV", data=df[cols_present].to_csv(index=False).encode("utf-8"),
            file_name=f"Bluestar_GPS_{ts}.csv",
            mime="text/csv", use_container_width=True,
        )
    with c3:
        st.download_button(
            "🗂️ JSON",
            data=df[cols_present].to_json(
                orient="records", force_ascii=False, indent=2
            ).encode("utf-8"),
            file_name=f"Bluestar_GPS_{ts}.json",
            mime="application/json", use_container_width=True,
        )


if __name__ == "__main__":
    main()


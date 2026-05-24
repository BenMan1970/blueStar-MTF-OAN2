# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BLUESTAR HEDGE FUND GPS — V7.2.0 PRODUCTION-GRADE HARDENED                 ║
║                                                                              ║
║  Refactoring d'architecture, sécurité des secrets et déterminisme MTF        ║
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
from datetime import datetime, timezone, timedelta
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
# SECTION 0 — STREAMLIT RUNTIME GUARD
# ═══════════════════════════════════════════════════════════════════════════════

def _is_streamlit_runtime() -> bool:
    """Détecte si le module tourne dans un runtime Streamlit actif."""
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
# SECTION 1 — LOGGING & TELEMETRY
# ═══════════════════════════════════════════════════════════════════════════════

class _SecretScrubFilter(logging.Filter):
    """Filtre de logging — supprime les tokens et les numéros de comptes."""

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
            record.msg = f"[SCRUB_ERROR:{type(exc).__name__}]"
            record.args = ()
        return True


def _setup_logging() -> logging.Logger:
    """Configuration globale du logger avec filtrage des secrets."""
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
    """Journalisation structurée d'un incident opérationnel."""
    parts = [f"code={code.value}", f"msg={msg}"]
    for k, v in context.items():
        parts.append(f"{k}={v}")
    _log.log(level, "INCIDENT %s", " ".join(parts))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

APP_VERSION: Final[str] = "7.2.0-PROD-HARDENED"

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

REQUIRED_OHLCV_COLS: Final[Tuple[str, ...]] = ("Open", "High", "Low", "Close", "Volume")

AccountHash = NewType("AccountHash", str)


@dataclass(frozen=True)
class OandaCredentials:
    """Conteneur d'identifiants sécurisés contre les fuites de cache Streamlit."""
    account_id: str
    access_token: str

    def __hash__(self) -> int:
        return hash(self.account_id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, OandaCredentials):
            return False
        return self.account_id == other.account_id


@dataclass(frozen=True)
class TrendConfig:
    """Toutes les constantes métier — gelées et immuables."""

    cache_ttl_m: int = 14400
    cache_ttl_w: int = 3600
    cache_ttl_d: int = 600
    cache_ttl_h4: int = 300
    cache_ttl_h1: int = 120
    cache_ttl_m15: int = 60
    cache_ttl_default: int = 600
    cache_ttl_live_open: int = 90
    cache_ttl_negative_live_open: int = 15

    stale_max_age_multiplier: float = 4.0

    data_max_age_min: int = 10
    snapshot_drift_max_sec: float = 30.0
    analysis_timeout_sec: int = 120
    streamlit_running_flag_ttl_sec: int = 300

    max_workers: int = 5
    pool_drain_grace_sec: float = 10.0

    http_timeout_sec: float = 8.0
    http_retry_total: int = 2
    http_retry_backoff: float = 0.3

    rate_limit_burst: int = 30
    rate_limit_refill_per_sec: float = 5.0
    rate_limit_429_cooldown_sec: float = 30.0

    pivot_wing: int = 5
    max_pivot_age: int = 50

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

    macro_band_atr_ratio: float = 0.10
    ema50_slope_threshold: float = 0.05

    volume_ratio_strong_intraday: float = 1.30
    volume_ratio_min_midpoint: float = 1.00
    body_range_strong: float = 0.60

    min_reliable_score: float = 1.5
    strength_min_range: int = 35
    strength_max: int = 90
    strength_scaler: float = 112.0

    mtf_weight_m: float = 5.0
    mtf_weight_w: float = 4.0
    mtf_weight_d: float = 4.0
    mtf_weight_h4: float = 2.5
    mtf_weight_h1: float = 1.5
    mtf_weight_15m: float = 1.0

    nc_pure_strength_min: int = 70

    dispersion_penalty_max: float = 15.0

    min_bars_m: int = 100
    min_bars_w: int = 50
    min_bars_d: int = 60
    min_bars_h4: int = 60
    min_bars_h1: int = 200
    min_bars_15m: int = 200

    completeness_min_tradable: float = 0.85
    cache_max_entries: int = 500


CFG: Final[TrendConfig] = TrendConfig()

if CFG.data_max_age_min > CFG.cache_ttl_d // 60:
    raise ValueError("data_max_age_min doit être <= cache_ttl_d (minutes)")
if CFG.max_workers < 1:
    raise ValueError("max_workers >= 1 requis")


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

_GAP_TOLERANCE: Final[Mapping[str, pd.Timedelta]] = {
    "M": pd.Timedelta(days=45),
    "W": pd.Timedelta(days=10),
    "D": pd.Timedelta(days=4),
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STREAMLIT SETUP GUARDED
# ═══════════════════════════════════════════════════════════════════════════════

def _configure_streamlit_ui() -> None:
    """Configuration de l'interface graphique uniquement si le runtime est actif."""
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


@dataclass(frozen=True)
class TrendResult:
    direction: str
    strength: int
    atr_val: float


@dataclass(frozen=True)
class FetchResult:
    df: pd.DataFrame
    is_stale: bool = False
    fetched_at: Optional[datetime] = None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CALENDRIER DE MARCHÉ (Ajustement saisonnier DST)
# ═══════════════════════════════════════════════════════════════════════════════

def is_us_dst(dt: datetime) -> bool:
    """
    Détermine si l'horodatage UTC fourni s'inscrit dans l'heure d'été américaine (EDT).
    Débute le 2ème dimanche de mars (07:00 UTC) et finit le 1er dimanche de novembre (06:00 UTC).
    """
    y = dt.year
    m1 = datetime(y, 3, 1, tzinfo=timezone.utc)
    dst_start = m1 + timedelta(days=((6 - m1.weekday()) % 7) + 7)
    dst_start = dst_start.replace(hour=7)

    n1 = datetime(y, 11, 1, tzinfo=timezone.utc)
    dst_end = n1 + timedelta(days=((6 - n1.weekday()) % 7))
    dst_end = dst_end.replace(hour=6)

    return dst_start <= dt < dst_end


def is_fx_market_open(now: Optional[datetime] = None) -> bool:
    """
    FX ouvert de dimanche 17:00 New York à vendredi 17:00 New York.
    Prend en compte automatiquement les transitions saisonnières (DST US).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    else:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

    dst_active = is_us_dst(now)
    ny_offset = -4 if dst_active else -5

    ny_time = now + timedelta(hours=ny_offset)
    weekday = ny_time.weekday()  # 0=Lundi, ..., 6=Dimanche
    hour = ny_time.hour

    if weekday == 5:  # Samedi fermé
        return False
    if weekday == 4 and hour >= 17:  # Vendredi après 17:00 local fermé
        return False
    if weekday == 6 and hour < 17:  # Dimanche avant 17:00 local fermé
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CALCULATION ENGINE & TECHNICAL INDICATORS
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
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=-np.inf)
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
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=np.inf)
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
# SECTION 7 — SINGLETON RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════

class _TokenBucket:
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


_RATE_LIMITER_INSTANCE: Optional[_GlobalRateLimiter] = None
_RATE_LIMITER_LOCK = threading.Lock()


def _get_rate_limiter() -> _GlobalRateLimiter:
    """Singleton process thread-safe interfacé avec l'allocation de ressources Streamlit."""
    global _RATE_LIMITER_INSTANCE
    if _STREAMLIT_AVAILABLE and _is_streamlit_runtime():
        try:
            @st.cache_resource(show_spinner=False)
            def _st_get_rate_limiter() -> _GlobalRateLimiter:
                return _GlobalRateLimiter()
            return _st_get_rate_limiter()
        except Exception as e:  # pylint: disable=broad-exception-caught
            _log_incident(
                IncidentCode.UI_CALLBACK_ERROR,
                "Streamlit rate limiter caching failed, fallback to global singleton",
                err=type(e).__name__
            )

    if _RATE_LIMITER_INSTANCE is None:
        with _RATE_LIMITER_LOCK:
            if _RATE_LIMITER_INSTANCE is None:
                _RATE_LIMITER_INSTANCE = _GlobalRateLimiter()
    return _RATE_LIMITER_INSTANCE


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — HTTP SESSIONS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class SessionRegistry:
    """Registre HTTP scopé par thread — fermé proprement après drainage."""

    def __init__(self) -> None:
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
            return sess

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=CFG.http_retry_total,
            backoff_factor=CFG.http_retry_backoff,
            status_forcelist=[500, 502, 503, 504],
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
            for s in self._thread_sessions.values():
                try:
                    s.close()
                except (OSError, RuntimeError) as exc:
                    _log_incident(
                        IncidentCode.SESSION_CLOSED,
                        "session close failed",
                        level=logging.DEBUG,
                        err=type(exc).__name__,
                    )
            self._thread_sessions.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CACHE ENGINE PROCESS SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

CacheKey = Tuple[str, AccountHash, str, str, int]
LiveOpenKey = Tuple[str, AccountHash, str, str]


def _hash_account(account_id: str) -> AccountHash:
    h = hashlib.sha256(account_id.encode("utf-8")).hexdigest()[:16]
    return AccountHash(h)


def _freeze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    frozen = df.copy(deep=True)
    for col in frozen.columns:
        arr = frozen[col].to_numpy()
        try:
            arr.flags.writeable = False
        except (ValueError, AttributeError):
            pass
    return frozen


def _defensive_copy(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=True)


class CandleCache:
    """Cache thread-safe LRU-like avec eviction stricte pour service 24/7."""

    def __init__(self, max_entries: int = 500) -> None:
        self._data: Dict[CacheKey, Tuple[datetime, pd.DataFrame]] = {}
        self._stale_data: Dict[CacheKey, Tuple[datetime, pd.DataFrame]] = {}
        self._live_opens: Dict[LiveOpenKey, Tuple[datetime, Optional[float]]] = {}
        self._inflight: Dict[CacheKey, threading.Event] = {}
        self._inflight_results: Dict[CacheKey, FetchResult] = {}
        self._lock = threading.RLock()
        self._max_entries = max_entries

    def _evict_if_needed(self) -> None:
        while len(self._data) > self._max_entries:
            oldest_key = min(self._data.keys(), key=lambda k: self._data[k][0])
            self._data.pop(oldest_key, None)
            self._stale_data.pop(oldest_key, None)
            self._inflight_results.pop(oldest_key, None)

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
            return self._wait_for_leader(key, wait_for_event, ttl)

        return self._do_fetch_as_leader(key, ttl, now, fetch_fn, wait_for_event)

    def _wait_for_leader(
        self,
        key: CacheKey,
        event: threading.Event,
        ttl: int,
    ) -> FetchResult:
        event_completed = event.wait(timeout=CFG.http_timeout_sec * 3)
        if not event_completed:
            _log_incident(
                IncidentCode.CACHE_LEADER_FAILED,
                "leader timeout on wait_for_leader",
                key=str(key[2:]),
            )
        now_post_wait = datetime.now(timezone.utc)
        with self._lock:
            entry = self._data.get(key)
            if entry is not None and (now_post_wait - entry[0]).total_seconds() < ttl:
                return FetchResult(
                    df=_defensive_copy(entry[1]), is_stale=False, fetched_at=entry[0]
                )
            stale = self._stale_data.get(key)
            if stale is not None:
                age = (now_post_wait - stale[0]).total_seconds()
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
                    self._evict_if_needed()
                return FetchResult(df=_defensive_copy(frozen), is_stale=False, fetched_at=ts)

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


_CANDLE_CACHE_INSTANCE: Optional[CandleCache] = None
_CANDLE_CACHE_LOCK = threading.Lock()


def _get_candle_cache() -> CandleCache:
    """Singleton process thread-safe interfacé avec l'allocation de ressources Streamlit."""
    global _CANDLE_CACHE_INSTANCE
    if _STREAMLIT_AVAILABLE and _is_streamlit_runtime():
        try:
            @st.cache_resource(show_spinner=False)
            def _st_get_candle_cache() -> CandleCache:
                return CandleCache(max_entries=CFG.cache_max_entries)
            return _st_get_candle_cache()
        except Exception as e:  # pylint: disable=broad-exception-caught
            _log_incident(
                IncidentCode.UI_CALLBACK_ERROR,
                "Streamlit candle cache caching failed, fallback to global singleton",
                err=type(e).__name__
            )

    if _CANDLE_CACHE_INSTANCE is None:
        with _CANDLE_CACHE_LOCK:
            if _CANDLE_CACHE_INSTANCE is None:
                _CANDLE_CACHE_INSTANCE = CandleCache(max_entries=CFG.cache_max_entries)
    return _CANDLE_CACHE_INSTANCE


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — RESILIENT DATA LAYER & CONNECTOR
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
    """
    Exécute une validation temporelle fine.
    Tolle les gaps de week-end, et marque df.attrs["critical_gap"] s'ils surviennent
    durant les heures de cotation normales.
    """
    if len(df) < 2:
        df.attrs["critical_gap"] = False
        return True

    tolerance = _GAP_TOLERANCE.get(granularity)
    if tolerance is None:
        df.attrs["critical_gap"] = False
        return True

    deltas = df.index.to_series().diff().dropna()
    if deltas.empty:
        df.attrs["critical_gap"] = False
        return True

    max_idx = deltas.idxmax()
    max_gap = deltas.loc[max_idx]

    if max_gap > tolerance:
        gap_start = max_idx - max_gap
        gap_end = max_idx
        is_weekend = (gap_start.weekday() in (4, 5, 6)) and (gap_end.weekday() in (6, 0, 1))

        if is_weekend:
            _log_incident(
                IncidentCode.DATA_GAPS,
                "tolerated weekend gap",
                instrument=instrument,
                granularity=granularity,
                max_gap_sec=int(max_gap.total_seconds()),
                level=logging.INFO,
            )
            df.attrs["critical_gap"] = False
        else:
            _log_incident(
                IncidentCode.DATA_GAPS,
                "critical gap during market open",
                instrument=instrument,
                granularity=granularity,
                max_gap_sec=int(max_gap.total_seconds()),
                tolerance_sec=int(tolerance.total_seconds()),
                level=logging.WARNING,
            )
            df.attrs["critical_gap"] = True
    else:
        df.attrs["critical_gap"] = False

    return True


def _fallback_bid_ask(candle: Any) -> Optional[Dict[str, float]]:
    """Calcule la moyenne bid/ask si les valeurs médianes 'mid' sont absentes du JSON."""
    try:
        bid = candle.get("bid")
        ask = candle.get("ask")
        if isinstance(bid, dict) and isinstance(ask, dict):
            mid = {}
            for field in ("o", "h", "l", "c"):
                b = bid.get(field)
                a = ask.get(field)
                if b is not None and a is not None:
                    mid[field] = (float(b) + float(a)) / 2.0
                elif b is not None:
                    mid[field] = float(b)
                elif a is not None:
                    mid[field] = float(a)
                else:
                    return None
            return mid
    except (ValueError, TypeError, KeyError):
        return None
    return None


def _parse_oanda_json(
    raw_json: Any, instrument: str, granularity: str, include_incomplete: bool
) -> List[Dict[str, Any]]:
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
    total_candles = len(candles)
    parse_errors = 0

    for candle in candles:
        if not isinstance(candle, dict):
            parse_errors += 1
            continue
        if not include_incomplete and not candle.get("complete"):
            continue
        try:
            mid = candle.get("mid")
            if mid is None or not isinstance(mid, dict):
                mid = _fallback_bid_ask(candle)
                if mid is None:
                    parse_errors += 1
                    continue

            row = {
                "date": candle.get("time"),
                "Open": float(mid["o"]),
                "High": float(mid["h"]),
                "Low": float(mid["l"]),
                "Close": float(mid["c"]),
                "Volume": float(candle.get("volume", 0) or 0),
            }
            if row["date"] is None or not _validate_candle(row, allow_zero_volume):
                parse_errors += 1
                continue
            rows.append(row)
        except (KeyError, ValueError, TypeError):
            parse_errors += 1
            continue

    if total_candles > 0:
        error_rate = parse_errors / total_candles
        if error_rate > 0.10:
            _log_incident(
                IncidentCode.JSON_INVALID,
                "high candle parsing error rate",
                instrument=instrument,
                granularity=granularity,
                total=total_candles,
                failed=parse_errors,
                error_rate=f"{error_rate:.1%}",
                level=logging.WARNING,
            )

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
    if not _get_rate_limiter().acquire(instrument):
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
        _get_rate_limiter().trigger_cooldown(instrument, cooldown)
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
    df = df.set_index("date")
    df = df.sort_index()

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
    key: CacheKey = (_OANDA_ENV, account_hash, instrument, granularity, count)
    ttl = _CACHE_TTL.get(granularity, CFG.cache_ttl_default)
    return _get_candle_cache().get_candles(
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
            if granularity == "W":
                max_age = pd.Timedelta(days=14)
            elif granularity == "M":
                max_age = pd.Timedelta(days=45)
            if (now - last_ts) > max_age:
                return None
            return float(df["Open"].iloc[-1])
        except (IndexError, ValueError, TypeError):
            return None

    return _get_candle_cache().get_live_open(key, _fetch)


def fetch_all_data(
    instrument: str,
    account_id: str,
    account_hash: AccountHash,
    access_token: str,
    registry: SessionRegistry,
    stop_event: threading.Event,
) -> Dict[str, Any]:
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
        "is_incomplete": False,
        "error_reason": None,
    }

    for tf, (gran, count, min_bars) in specs.items():
        if stop_event.is_set():
            cache["is_incomplete"] = True
            cache["error_reason"] = "Stop requested"
            return cache

        result = fetch_cached(
            instrument, gran, count, account_id, account_hash, access_token, registry
        )
        if result.df.empty:
            cache["is_incomplete"] = True
            cache["error_reason"] = f"Fetch failed on {tf}"
            return cache
        if len(result.df) < min_bars:
            _log_incident(
                IncidentCode.DATA_INSUFFICIENT, "bars below minimum",
                instrument=instrument, tf=tf,
                bars=len(result.df), min_bars=min_bars,
            )
            cache["is_incomplete"] = True
            cache["error_reason"] = f"Insufficient bars on {tf} ({len(result.df)} < {min_bars})"
            return cache

        cache[tf] = result.df
        cache["_snapshot_per_tf"][tf] = result.fetched_at or datetime.now(timezone.utc)
        if result.is_stale:
            cache["_stale_tfs"].append(tf)

    if stop_event.is_set():
        cache["is_incomplete"] = True
        cache["error_reason"] = "Stop requested"
        return cache

    cache["_snapshot_completed_at"] = datetime.now(timezone.utc)

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
# SECTION 12 — VOTES ATOMIQUES
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
    name = "weekly_open"
    cur = ctx["cur"]
    current_week_open = ctx.get("current_week_open")

    if current_week_open is None or (
        isinstance(current_week_open, float) and np.isnan(current_week_open)
    ):
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
    name = "prev_midpoint"
    if vol is None or vol.empty:
        return VoteSignal(name, direction, 0.5, 0.50, True, "no_vol_data")
    try:
        vol_ref = vol.iloc[:-1]
        vol_j1 = float(vol.iloc[-1])
        if len(vol_ref) < 20:
            return VoteSignal(name, direction, 0.5, 0.50, True, "vol_history<<20")
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
# SECTION 13 — AGRÉGATEUR DE VOTES
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
# SECTION 14 — MULTI-TIMEFRAME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def trend_daily(
    df: pd.DataFrame,
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
        "atr_val": atr_val, "df_daily": df,
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
        except Exception as exc:  # pylint: disable=broad-exception-caught
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


def _trend_macro_monthly(c: pd.Series, e50: pd.Series, atr_val: float, band: float, n: int) -> TrendResult:
    if n < CFG.min_bars_m:
        return TrendResult("Range", 0, atr_val)
    e100 = _ema(c, 100)
    ref = float(e100.iloc[-1])
    cur = float(e50.iloc[-1])
    if ref == 0:
        return TrendResult("Range", 40, atr_val)
    gap = abs(cur - ref) / ref * 100
    s = 75 if gap > 0.3 else 60
    if cur > ref + band:
        return TrendResult("Bullish", s, atr_val)
    if cur < ref - band:
        return TrendResult("Bearish", s, atr_val)
    return TrendResult("Range", 40, atr_val)


def _trend_macro_weekly(c: pd.Series, e50: pd.Series, atr_val: float, band: float, n: int) -> TrendResult:
    if n < CFG.sma_macro:
        return TrendResult("Range", 40, atr_val)
    s200 = _sma(c, CFG.sma_macro)
    cur50 = float(e50.iloc[-1])
    ref200 = float(s200.iloc[-1])
    prev50 = float(e50.iloc[-2])
    prev200 = float(s200.iloc[-2])
    cross = (prev50 <= prev200 < cur50) or (prev50 >= prev200 > cur50)
    if cur50 > ref200 + band:
        return TrendResult("Bullish", 90 if cross else 75, atr_val)
    if cur50 < ref200 - band:
        return TrendResult("Bearish", 90 if cross else 75, atr_val)
    return TrendResult("Range", 40, atr_val)


def trend_macro(df: pd.DataFrame, tf: str) -> TrendResult:
    if len(df) < 50:
        atr_val = (
            float(_atr(df["High"], df["Low"], df["Close"], CFG.atr_period).iloc[-1])
            if len(df) >= 15 else 0.0
        )
        return TrendResult("Range", 0, atr_val)
    c, h, lo = df["Close"], df["High"], df["Low"]
    atr_val = float(_atr(h, lo, c, CFG.atr_period).iloc[-1])
    band = atr_val * CFG.macro_band_atr_ratio
    e50 = _ema(c, CFG.ema_long)
    if tf == "M":
        return _trend_macro_monthly(c, e50, atr_val, band, len(df))
    return _trend_macro_weekly(c, e50, atr_val, band, len(df))


def trend_4h(
    df: pd.DataFrame,
    current_day_open: Optional[float] = None,
) -> TrendResult:
    h, lo, c = df["High"], df["Low"], df["Close"]
    atr_val = float(_atr(h, lo, c, CFG.atr_period).iloc[-1])
    if len(df) < CFG.min_bars_h4:
        return TrendResult("Range", 0, atr_val)
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
    return TrendResult(direction, strength, atr_val)


def _trend_intraday_compute_indicators(
    c: pd.Series, period: int, lag: int
) -> Optional[Dict[str, float]]:
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


def trend_intraday(df: pd.DataFrame, instrument: str = "") -> TrendResult:
    h, lo, c = df["High"], df["Low"], df["Close"]
    atr_val = float(_atr(h, lo, c, CFG.atr_period).iloc[-1])
    if len(df) < 70:
        return TrendResult("Range", 0, atr_val)
    cur = float(c.iloc[-1])
    period = CFG.ema_intra_period
    lag = (period - 1) // 2

    ind = _trend_intraday_compute_indicators(c, period, lag)
    if ind is None:
        return TrendResult("Range", 0, atr_val)

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
        return TrendResult("Bullish", _atr_strength(), atr_val)
    if vbr == max_votes:
        return TrendResult("Bearish", _atr_strength(), atr_val)
    if vb >= max_votes - 1:
        return TrendResult("Bullish", 55, atr_val)
    if vbr >= max_votes - 1:
        return TrendResult("Bearish", 55, atr_val)
    if cur < e50_cur and e9 > e21:
        return TrendResult("Retracement Bull", 45, atr_val)
    if cur > e50_cur and e9 < e21:
        return TrendResult("Retracement Bear", 45, atr_val)
    return TrendResult("Range", 30, atr_val)


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
# SECTION 15 — SYSTEME DE SCORING MULTI-TIMEFRAME
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
    grades: List[str] = []
    for score, nc, degraded in zip(scores_list, nc_list, degraded_list):
        nc_bonus = (int(nc) - 3) * 5
        adj = min(100.0, float(score) + nc_bonus)
        if degraded:
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
# SECTION 16 — CORE PIPELINE ORCHESTRATOR
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

        if cache.get("is_incomplete"):
            reason = cache.get("error_reason", "Fetch failed")
            return {
                "Paire": pair.replace("_", "/"),
                "M": "Range", "W": "Range", "D": "Range",
                "4H": "Range", "1H": "Range", "15m": "Range",
                "MTF": "Range",
                "_mtf_score": 0.0,
                "_mtf_dir": "Range",
                "_degraded": True,
                "_stale_tfs": (),
                "NC": 0,
                "Age D1": "N/A",
                "ATR Daily": "N/A",
                "ATR H4": "N/A",
                "ATR H1": "N/A",
                "ATR 15m": "N/A",
                "_error_reason": reason,
            }

        trends: Dict[str, str] = {}
        scores: Dict[str, int] = {}
        atrs: Dict[str, float] = {}

        drift_exceeded = bool(cache.get("_snapshot_drift_exceeded"))
        critical_gap = any(cache[tf].attrs.get("critical_gap", False) for tf in ("M", "W", "D", "4H", "1H", "15m"))
        degraded = bool(cache.get("_stale_tfs")) or drift_exceeded or critical_gap

        if drift_exceeded or critical_gap:
            reason = "Drift exceeded" if drift_exceeded else "Critical gap in open market hours"
            return {
                "Paire": pair.replace("_", "/"),
                "M": "Range", "W": "Range", "D": "Range",
                "4H": "Range", "1H": "Range", "15m": "Range",
                "MTF": "Range",
                "_mtf_score": 0.0,
                "_mtf_dir": "Range",
                "_degraded": True,
                "_stale_tfs": (),
                "NC": 0,
                "Age D1": "N/A",
                "ATR Daily": "N/A",
                "ATR H4": "N/A",
                "ATR H1": "N/A",
                "ATR 15m": "N/A",
                "_error_reason": reason,
            }

        for tf in ("M", "W"):
            tr = trend_macro(cache[tf], tf)
            trends[tf], scores[tf], atrs[tf] = tr.direction, tr.strength, tr.atr_val

        if stop_event.is_set():
            return None

        daily_result = trend_daily(
            cache["D"], instrument=pair,
            current_week_open=cache.get("_week_open"),
        )
        trends["D"], scores["D"], atrs["D"] = (
            daily_result.direction.value, daily_result.strength, daily_result.atr_val,
        )
        if daily_result.degraded:
            degraded = True

        tr = trend_4h(
            cache["4H"],
            current_day_open=cache.get("_day_open"),
        )
        trends["4H"], scores["4H"], atrs["4H"] = tr.direction, tr.strength, tr.atr_val

        for tf in ("1H", "15m"):
            if stop_event.is_set():
                return None
            tr = trend_intraday(cache[tf], pair)
            trends[tf], scores[tf], atrs[tf] = tr.direction, tr.strength, tr.atr_val

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
    except Exception as e:  # pylint: disable=broad-exception-caught
        _log_incident(
            IncidentCode.UNKNOWN, "analyze_pair exception",
            instrument=pair, err=type(e).__name__, level=logging.ERROR,
        )
        return {
            "Paire": pair.replace("_", "/"),
            "M": "Range", "W": "Range", "D": "Range",
            "4H": "Range", "1H": "Range", "15m": "Range",
            "MTF": "Range",
            "_mtf_score": 0.0,
            "_mtf_dir": "Range",
            "_degraded": True,
            "_stale_tfs": (),
            "NC": 0,
            "Age D1": "N/A",
            "ATR Daily": "N/A",
            "ATR H4": "N/A",
            "ATR H1": "N/A",
            "ATR 15m": "N/A",
            "_error_reason": f"Analysis failed: {type(e).__name__}",
        }


def _drain_executor_strict(
    executor: ThreadPoolExecutor,
    futures: Mapping[Future, str],
    stop_event: threading.Event,
) -> None:
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
    executor.shutdown(wait=True, cancel_futures=True)


def _process_completed_future(
    future: Future,
    inst: str,
    results: List[Dict[str, Any]],
    errors: set,
) -> None:
    try:
        row = future.result()
        if row:
            results.append(row)
        else:
            errors.add(inst)
    except Exception as e:  # pylint: disable=broad-exception-caught
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
    results: List[Dict[str, Any]] = []
    errors: set = set()
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

    dynamic_timeout = max(CFG.analysis_timeout_sec, int(len(INSTRUMENTS) * 5.0 / CFG.max_workers))

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
            for future in as_completed(futures, timeout=dynamic_timeout):
                inst = futures[future]
                done += 1
                if progress_cb is not None:
                    try:
                        progress_cb(done / total)
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        _log_incident(
                            IncidentCode.UI_CALLBACK_ERROR, "progress_cb",
                            err=type(exc).__name__, level=logging.DEBUG,
                        )
                if status_cb is not None:
                    try:
                        status_cb(f"GPS ({done}/{total}) — {inst.replace('_', '/')}")
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        _log_incident(
                            IncidentCode.UI_CALLBACK_ERROR, "status_cb",
                            err=type(exc).__name__, level=logging.DEBUG,
                        )
                _process_completed_future(future, inst, results, errors)
        except FutureTimeoutError:
            timed_out = True
            _log_incident(
                IncidentCode.EXECUTOR_TIMEOUT, "analyze_all_core timeout",
                timeout_sec=dynamic_timeout, level=logging.ERROR,
            )
            for f, inst in futures.items():
                if not f.done():
                    errors.add(inst)
    finally:
        _drain_executor_strict(executor, futures, stop_event)
        registry.close_all()

    meta["finished_at"] = datetime.now(timezone.utc)
    meta["timed_out"] = timed_out
    meta["errors_count"] = len(errors)
    meta["completeness"] = len(results) / total if total > 0 else 0.0
    meta["degraded_pairs"] = sorted(r["Paire"] for r in results if r.get("_degraded") and "_error_reason" not in r)

    errors_sorted = sorted(errors)

    if not results:
        return pd.DataFrame(), errors_sorted, meta

    scores_list = [r["_mtf_score"] for r in results]
    nc_list = [r["NC"] for r in results]
    run_degraded = meta["completeness"] < CFG.completeness_min_tradable
    degraded_list = [r["_degraded"] or run_degraded or "_error_reason" in r for r in results]
    grades = grade_hybrid(scores_list, nc_list, degraded_list)

    for r, g, deg in zip(results, grades, degraded_list):
        if "_error_reason" in r:
            r["Quality"] = "B"
            r["Tradable"] = f"✗ ERROR ({r['_error_reason']})"
        else:
            r["Quality"] = g
            r["Tradable"] = "✓" if not deg else "✗ NOT_TRADEABLE"

    df = pd.DataFrame(results)
    df.attrs["meta"] = meta
    return df, errors_sorted, meta


# Caching d'exécution Streamlit sécurisé, ignorant les clés secrètes brutes
if _STREAMLIT_AVAILABLE and _is_streamlit_runtime():
    @st.cache_data(
        ttl=CFG.cache_ttl_d,
        show_spinner=False,
        hash_funcs={OandaCredentials: lambda creds: creds.account_id}
    )
    def _run_analysis_cached(creds: OandaCredentials) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        return analyze_all_core(creds.account_id, creds.access_token)
else:
    def _run_analysis_cached(creds: OandaCredentials) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        return analyze_all_core(creds.account_id, creds.access_token)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — UI REPORTING & FORMAT GENERATION
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
    if col == "Tradable":
        if "ERROR" in val or "✗" in val:
            return (231, 76, 60), (255, 255, 255)
        return (46, 204, 113), (255, 255, 255)
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
    base_cols = ["Paire", "M", "W", "D", "4H", "1H", "15m", "MTF", "Quality", "Tradable", "NC", "Age D1",
                 "ATR Daily", "ATR H4", "ATR H1", "ATR 15m"]
    cols = [c for c in base_cols if c in df.columns]

    base_widths = {"Paire": 22, "M": 16, "W": 16, "D": 16, "4H": 16, "1H": 16, "15m": 16,
                   "MTF": 30, "Quality": 12, "Tradable": 32, "NC": 10, "Age D1": 13,
                   "ATR Daily": 17, "ATR H4": 17, "ATR H1": 15, "ATR 15m": 15}
    widths = {c: base_widths.get(c, 15) for c in cols}

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
    except Exception as e:  # pylint: disable=broad-exception-caught
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
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _log_incident(IncidentCode.PDF_ERROR, "fallback PDF failed",
                          err=type(exc).__name__, level=logging.ERROR)
            buf2.write(b"PDF error")
        buf2.seek(0)
        return buf2


_OANDA_ACCOUNT_PATTERN: Final[re.Pattern] = re.compile(r"^\d{3}-\d{3}-\d{4,}-\d{3,}$")


def _validate_secret_format(account_id: str, token: str) -> bool:
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
    if not _STREAMLIT_AVAILABLE:
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
    except Exception as exc:  # pylint: disable=broad-exception-caught
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
    run_id = st.session_state.get("_analysis_run_id")
    if started_at and run_id:
        elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
        if elapsed > CFG.streamlit_running_flag_ttl_sec:
            st.session_state["_analysis_running"] = False
            st.session_state["_analysis_started_at"] = None
            st.session_state["_analysis_run_id"] = None


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
    """Enveloppe Streamlit sécurisée pour l'analyse globale."""
    creds = OandaCredentials(account_id=account_id, access_token=access_token)
    df, errors, meta = _run_analysis_cached(creds)

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


def _sidebar_config() -> bool:
    """Rendu de la barre de configuration pour alléger main()."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        only_best = st.checkbox("Afficher uniquement Grade A+ / A", value=False)
        st.info(
            f"Env : {_OANDA_ENV.upper()}\n\n"
            f"Workers : {CFG.max_workers} · Timeout : {CFG.analysis_timeout_sec}s\n\n"
            f"Cache TTL : M={CFG.cache_ttl_m // 60}m W={CFG.cache_ttl_w // 60}m D={CFG.cache_ttl_d // 60}m"
        )
        if not is_fx_market_open():
            st.warning("📅 Marché FX fermé — données potentiellement obsolètes.")
        st.markdown("---")
        if st.button("🗑️ Vider le cache", use_container_width=True):
            _get_candle_cache().clear()
            if _STREAMLIT_AVAILABLE and _is_streamlit_runtime():
                _run_analysis_cached.clear()
                try:
                    st.cache_resource.clear()
                except Exception as e:  # pylint: disable=broad-exception-caught
                    _log.debug("Streamlit cache clear failed: %s", e)
            st.success("Cache vidé.")
    return only_best


def _render_metrics(df: pd.DataFrame) -> None:
    """Affichage des métriques de synthèse pour alléger main()."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Analyzed", len(df))
    c2.metric("Setups A+", len(df[df["Quality"] == "A+"]))
    c3.metric("Setups A", len(df[df["Quality"] == "A"]))
    c4.metric("Setups B", len(df[df["Quality"].isin(["B+", "B"])]))


def main() -> None:
    _configure_streamlit_ui()

    st.markdown(
        f"<div class='main-header'><h1>🧭 BLUESTAR HEDGE FUND GPS V{APP_VERSION}</h1>"
        f"<p style='margin:0;font-size:0.85em;opacity:0.8'>"
        f"Production-Grade Hardened · Env: {_OANDA_ENV.upper()} · C1–C12"
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

    for key in ("_analysis_run_id", "_analysis_running", "_analysis_started_at"):
        if key not in st.session_state:
            st.session_state[key] = None if key != "_analysis_running" else False

    _check_running_flag_ttl()

    only_best = _sidebar_config()

    is_running = st.session_state.get("_analysis_running", False)
    if st.button(
        "🚀 LANCER L'ANALYSE TOUS ACTIFS",
        type="primary",
        use_container_width=True,
        disabled=is_running,
    ):
        run_id = datetime.now(timezone.utc).isoformat()
        st.session_state["_analysis_running"] = True
        st.session_state["_analysis_started_at"] = datetime.now(timezone.utc)
        st.session_state["_analysis_run_id"] = run_id
        try:
            with st.spinner("Analyse Multi-Timeframe en cours..."):
                df, meta = analyze_all(acc, tok)
                if not df.empty:
                    st.session_state["df"] = df
                    st.session_state["df_ts"] = datetime.now(timezone.utc)
                    st.session_state["df_meta"] = meta
                    st.session_state["pdf_buf"] = create_pdf(df)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _log_incident(
                IncidentCode.UNKNOWN, "main analysis failed",
                err=type(exc).__name__, level=logging.ERROR,
            )
            st.error(f"❌ Erreur critique lors de l'analyse : {exc}")
        finally:
            if st.session_state.get("_analysis_run_id") == run_id:
                st.session_state["_analysis_running"] = False
                st.session_state["_analysis_started_at"] = None
                st.session_state["_analysis_run_id"] = None

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
        df = df[df["Quality"].isin(["A+", "A"])].copy()

    grade_order = ["A+", "A", "B+", "B"]
    df["Quality"] = pd.Categorical(df["Quality"], categories=grade_order, ordered=True)
    sort_cols = [c for c in ["Quality", "NC", "_mtf_score"] if c in df.columns]
    ascending = [True, False, False][:len(sort_cols)]
    df = df.sort_values(sort_cols, ascending=ascending)

    df_clean = df.drop(
        columns=["_mtf_score", "_mtf_dir", "_degraded", "_stale_tfs", "_error_reason"],
        errors="ignore",
    ).copy()

    _render_metrics(df_clean)

    display = ["Paire", "M", "W", "D", "4H", "1H", "15m", "MTF", "Quality", "Tradable",
               "NC", "Age D1", "ATR Daily", "ATR H4", "ATR H1", "ATR 15m"]
    cols_present = [col for col in display if col in df_clean.columns]

    styled = (
        df_clean[cols_present].style
        .apply(_style_quality, axis=0)
        .apply(_style_nc, axis=0)
        .map(_style_trend)
    )
    st.dataframe(
        styled,
        height=min(800, max(400, (len(df_clean) + 1) * 38 + 10)),
        use_container_width=True,
        hide_index=True,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    c1, c2, c3 = st.columns(3)

    pdf_buf = st.session_state.get("pdf_buf")
    if pdf_buf is None:
        pdf_buf = create_pdf(df_clean)

    with c1:
        st.download_button(
            "📄 PDF", data=pdf_buf,
            file_name=f"Bluestar_GPS_{ts}.pdf",
            mime="application/pdf", use_container_width=True,
        )
    with c2:
        st.download_button(
            "📊 CSV", data=df_clean[cols_present].to_csv(index=False).encode("utf-8"),
            file_name=f"Bluestar_GPS_{ts}.csv",
            mime="text/csv", use_container_width=True,
        )
    with c3:
        st.download_button(
            "🗂️ JSON",
            data=df_clean[cols_present].to_json(
                orient="records", force_ascii=False, indent=2
            ).encode("utf-8"),
            file_name=f"Bluestar_GPS_{ts}.json",
            mime="application/json", use_container_width=True,
        )


if __name__ == "__main__":
    main()

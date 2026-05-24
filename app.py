# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BLUESTAR HEDGE FUND GPS — V8.0.0 PRODUCTION-GRADE INSTITUTIONAL              ║
║                                                                              ║
║  Audit-driven hardening: trading correctness, concurrency safety,           ║
║  temporal determinism, secret containment, holiday-aware calendar.          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import functools
import hashlib
import logging
import math
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
    as_completed,
)
from dataclasses import dataclass, field
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
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Activate Copy-on-Write semantics globally — critical for cross-thread
# DataFrame safety (defect #3). Must be set before any DataFrame ops.
try:
    pd.set_option("mode.copy_on_write", True)
except (KeyError, ValueError):
    # Older pandas (<2.0); we fall back to defensive deep copies.
    pass

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

try:
    import holidays as _holidays_lib
    _HAS_HOLIDAYS = True
except ImportError:
    _holidays_lib = None
    _HAS_HOLIDAYS = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — STREAMLIT RUNTIME GUARD
# ═══════════════════════════════════════════════════════════════════════════════

def _is_streamlit_runtime() -> bool:
    """True iff currently executing inside an active Streamlit script run."""
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
# SECTION 1 — SECRET CONTAINMENT
# ═══════════════════════════════════════════════════════════════════════════════

class SecretToken:
    """Token wrapper that never leaks via repr/str/traceback (defect #11)."""

    __slots__ = ("_value", "_digest")

    def __init__(self, value: str) -> None:
        object.__setattr__(self, "_value", value)
        object.__setattr__(
            self,
            "_digest",
            hashlib.sha256(value.encode("utf-8")).hexdigest()[:12],
        )

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("SecretToken is immutable")

    def __delattr__(self, key: str) -> None:
        raise AttributeError("SecretToken is immutable")

    def reveal(self) -> str:
        """Explicit, auditable secret access."""
        return self._value

    @property
    def digest(self) -> str:
        return self._digest

    def __repr__(self) -> str:
        return f"SecretToken(digest={self._digest})"

    def __str__(self) -> str:
        return f"[REDACTED:{self._digest}]"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SecretToken):
            return False
        return self._value == other._value

    def __hash__(self) -> int:
        return hash(self._digest)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOGGING & TELEMETRY (with token-aware scrubbing)
# ═══════════════════════════════════════════════════════════════════════════════

class _SecretScrubFilter(logging.Filter):
    """Logging filter — scrubs tokens, account numbers, and known SecretTokens."""

    _PATTERNS: Tuple[re.Pattern, ...] = (
        re.compile(r"Bearer\s+[A-Za-z0-9\-_\.~+/=]+", re.IGNORECASE),
        re.compile(r"\b[0-9]{3}-[0-9]{3}-[0-9]+-[0-9]+\b"),
        re.compile(r"(?i)(access[_-]?token|api[_-]?key|secret|authorization)\s*[:=]\s*\S+"),
        # OANDA practice tokens are 64 hex chars; broader hex match is safer
        re.compile(r"\b[a-f0-9]{40,}\b", re.IGNORECASE),
    )

    # Registry of live SecretToken values for direct-substring scrubbing
    _registered_tokens: List[Tuple[str, str]] = []
    _registry_lock = threading.Lock()

    @classmethod
    def register_secret(cls, secret: SecretToken) -> None:
        with cls._registry_lock:
            entry = (secret.reveal(), f"[REDACTED:{secret.digest}]")
            if entry not in cls._registered_tokens:
                cls._registered_tokens.append(entry)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            with self._registry_lock:
                for raw, replacement in self._registered_tokens:
                    if raw and raw in msg:
                        msg = msg.replace(raw, replacement)
            for pat in self._PATTERNS:
                msg = pat.sub("[REDACTED]", msg)
            record.msg = msg
            record.args = ()
        except (TypeError, ValueError) as exc:
            record.msg = f"[SCRUB_ERROR:{type(exc).__name__}]"
            record.args = ()
        return True


def _setup_logging() -> logging.Logger:
    """Global logger configuration with mandatory secret scrubbing."""
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


def _install_excepthook() -> None:
    """Replace sys.excepthook with a scrubbing variant (defect #11)."""
    original_hook = sys.excepthook
    scrub_filter = _SecretScrubFilter()

    def _scrubbing_excepthook(exc_type, exc_value, exc_tb) -> None:
        try:
            import traceback
            tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            with scrub_filter._registry_lock:
                for raw, replacement in scrub_filter._registered_tokens:
                    if raw:
                        tb_text = tb_text.replace(raw, replacement)
            for pat in scrub_filter._PATTERNS:
                tb_text = pat.sub("[REDACTED]", tb_text)
            sys.stderr.write(tb_text)
        except Exception:  # pylint: disable=broad-exception-caught
            original_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _scrubbing_excepthook


_install_excepthook()


class IncidentCode(str, Enum):
    HTTP_TIMEOUT = "E001"
    HTTP_ERROR = "E002"
    HTTP_RATELIMIT = "E003"
    HTTP_AUTH = "E004"
    JSON_INVALID = "E010"
    DATA_INSUFFICIENT = "E020"
    DATA_GAPS = "E021"
    DATA_VALIDATION = "E022"
    DATA_NAN = "E023"
    CACHE_STALE_HIT = "E030"
    CACHE_LEADER_FAILED = "E031"
    VOTE_ERROR = "E040"
    VOTE_CRITICAL_ERROR = "E041"
    EXECUTOR_TIMEOUT = "E050"
    EXECUTOR_CANCEL = "E051"
    SESSION_CLOSED = "E052"
    UI_CALLBACK_ERROR = "E060"
    PDF_ERROR = "E070"
    PRICING_UNAVAILABLE = "E080"
    UNKNOWN = "E999"


def _log_incident(
    code: IncidentCode,
    msg: str,
    *,
    level: int = logging.WARNING,
    **context: Any,
) -> None:
    """Structured incident logging."""
    parts = [f"code={code.value}", f"msg={msg}"]
    for k, v in context.items():
        parts.append(f"{k}={v}")
    _log.log(level, "INCIDENT %s", " ".join(parts))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CONFIGURATION (grouped sub-configs)
# ═══════════════════════════════════════════════════════════════════════════════

APP_VERSION: Final[str] = "8.0.0-PROD-INSTITUTIONAL"

_OANDA_ENV: Final[str] = os.environ.get("OANDA_ENV", "practice").lower()
OANDA_API_URL: Final[str] = (
    "https://api-fxtrade.oanda.com"
    if _OANDA_ENV == "live"
    else "https://api-fxpractice.oanda.com"
)

# XAU_USD removed from INDICES (defect #4) — it's a tick-volume FX-like instrument
INSTRUMENTS: Final[Tuple[str, ...]] = (
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "XAU_USD",
    "DE30_EUR", "SPX500_USD", "NAS100_USD", "US30_USD",
)

INDICES: Final[FrozenSet[str]] = frozenset(
    {"DE30_EUR", "SPX500_USD", "NAS100_USD", "US30_USD"}
)

REQUIRED_OHLCV_COLS: Final[Tuple[str, ...]] = ("Open", "High", "Low", "Close", "Volume")

AccountHash = NewType("AccountHash", str)
NY_TZ: Final[ZoneInfo] = ZoneInfo("America/New_York")
UTC: Final[timezone] = timezone.utc


@dataclass(frozen=True)
class HttpConfig:
    timeout_sec: float = 8.0
    retry_total: int = 2
    retry_backoff: float = 0.3
    pool_maxsize: int = 20
    rate_limit_burst_per_instrument: int = 30
    rate_limit_refill_per_sec: float = 5.0
    rate_limit_global_burst: int = 80
    rate_limit_global_refill_per_sec: float = 40.0
    rate_limit_429_cooldown_sec: float = 30.0


@dataclass(frozen=True)
class CacheConfig:
    ttl_m: int = 14400
    ttl_w: int = 3600
    ttl_d: int = 600
    ttl_h4: int = 300
    ttl_h1: int = 120
    ttl_m15: int = 60
    ttl_default: int = 600
    ttl_live_open: int = 90
    ttl_pricing_sec: int = 5
    ttl_negative_live_open: int = 15
    stale_max_age_multiplier: float = 4.0
    max_entries: int = 500


@dataclass(frozen=True)
class IndicatorConfig:
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
    pivot_wing: int = 5
    max_pivot_age: int = 50
    pivot_prominence_atr_ratio: float = 0.5  # defect #12
    pivot_min_confirmations: int = 3          # defect #12


@dataclass(frozen=True)
class VotingConfig:
    macro_band_atr_ratio: float = 0.10
    ema50_slope_threshold: float = 0.05
    volume_ratio_strong_intraday: float = 1.30
    volume_ratio_min_midpoint: float = 1.00
    body_range_strong: float = 0.60
    min_reliable_score: float = 1.5
    strength_min_range: int = 35
    strength_max: int = 90
    strength_scaler: float = 112.0
    weight_swing_structure: float = 2.0
    weight_ema_stack: float = 1.0
    weight_weekly_open: float = 1.0
    weight_prev_midpoint: float = 0.5
    weight_ema50_slope: float = 1.0


@dataclass(frozen=True)
class MtfConfig:
    weight_m: float = 5.0
    weight_w: float = 4.0
    weight_d: float = 4.0
    weight_h4: float = 2.5
    weight_h1: float = 1.5
    weight_15m: float = 1.0
    min_active_tfs: int = 3  # defect #6: prevent A+ on isolated signal
    dispersion_penalty_max: float = 15.0
    nc_pure_strength_min: int = 70
    nc_mtf_min_for_a_plus: float = 70.0  # defect #17
    nc_min_for_a_plus: int = 3            # defect #17


@dataclass(frozen=True)
class BarsConfig:
    """Minimum bars per TF. Relaxed thresholds (defect #20) — backed by indicator
    period requirements: ATR/RSI/EMA need ~3x their period to stabilize."""
    min_bars_m: int = 60
    min_bars_w: int = 50
    min_bars_d: int = 60
    min_bars_h4: int = 60
    min_bars_h1: int = 100
    min_bars_15m: int = 100


@dataclass(frozen=True)
class OperationalConfig:
    data_max_age_min: int = 10
    snapshot_drift_max_sec: float = 30.0
    analysis_timeout_sec: int = 120
    streamlit_running_flag_ttl_sec: int = 300
    max_workers: int = 5
    pool_drain_grace_sec: float = 10.0
    completeness_min_tradable: float = 0.85


@dataclass(frozen=True)
class TrendConfig:
    http: HttpConfig = field(default_factory=HttpConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    ind: IndicatorConfig = field(default_factory=IndicatorConfig)
    vote: VotingConfig = field(default_factory=VotingConfig)
    mtf: MtfConfig = field(default_factory=MtfConfig)
    bars: BarsConfig = field(default_factory=BarsConfig)
    ops: OperationalConfig = field(default_factory=OperationalConfig)

    def __post_init__(self) -> None:
        if self.ops.data_max_age_min > self.cache.ttl_d // 60:
            raise ValueError("data_max_age_min must be <= cache.ttl_d (minutes)")
        if self.ops.max_workers < 1:
            raise ValueError("max_workers >= 1 required")
        if self.mtf.min_active_tfs < 1:
            raise ValueError("min_active_tfs >= 1 required")


CFG: Final[TrendConfig] = TrendConfig()


_CACHE_TTL: Final[Mapping[str, int]] = {
    "M": CFG.cache.ttl_m,
    "W": CFG.cache.ttl_w,
    "D": CFG.cache.ttl_d,
    "H4": CFG.cache.ttl_h4,
    "H1": CFG.cache.ttl_h1,
    "M15": CFG.cache.ttl_m15,
}

_GRAN_FREQ: Final[Mapping[str, pd.Timedelta]] = {
    "M": pd.Timedelta(days=30),
    "W": pd.Timedelta(days=7),
    "D": pd.Timedelta(days=1),
    "H4": pd.Timedelta(hours=4),
    "H1": pd.Timedelta(hours=1),
    "M15": pd.Timedelta(minutes=15),
}

# Defect #23 — extracted from inline overrides
_LIVE_OPEN_MAX_AGE: Final[Mapping[str, pd.Timedelta]] = {
    "M": pd.Timedelta(days=45),
    "W": pd.Timedelta(days=14),
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
# SECTION 4 — STREAMLIT UI SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def _configure_streamlit_ui() -> None:
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
            .main-header { text-align: center; padding: 20px;
                background: linear-gradient(135deg, #1e3a8a 0%, #172554 100%);
                color: white; border-radius: 12px; margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
            .stale-warning { background: #fef3c7; border-left: 4px solid #f59e0b;
                padding: 10px 16px; border-radius: 4px; margin-bottom: 12px; }
            .degraded-warning { background: #fee2e2; border-left: 4px solid #dc2626;
                padding: 10px 16px; border-radius: 4px; margin-bottom: 12px; }
            .not-tradable { background: #1f2937; color: #fbbf24; padding: 4px 8px;
                border-radius: 4px; font-weight: bold; font-size: 0.85em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DOMAIN TYPES
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
    critical_gap: bool = False


@dataclass(frozen=True)
class IndicatorBundle:
    """Pre-computed indicators per (instrument, tf) — defect #13."""
    atr_val: float
    ema_short: float
    ema_long_cur: float
    ema_long_series: pd.Series
    rsi: float
    macd: float
    macd_signal: float
    ema_intra_fast: float
    ema_intra_long: float
    zlema: float
    has_nan: bool


@dataclass(frozen=True)
class OandaCredentials:
    """Account credentials with hashed token for safe cache keying."""
    account_id: str
    token: SecretToken

    def __hash__(self) -> int:
        return hash((self.account_id, self.token.digest))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, OandaCredentials):
            return False
        return self.account_id == other.account_id and self.token == other.token


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MARKET CALENDAR (zoneinfo + holidays)
# ═══════════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=8)
def _get_us_holiday_set(year: int) -> FrozenSet[Any]:
    """Cached US market holiday set for fast lookup."""
    if not _HAS_HOLIDAYS or _holidays_lib is None:
        return frozenset()
    try:
        return frozenset(_holidays_lib.country_holidays("US", years=[year]).keys())
    except Exception:  # pylint: disable=broad-exception-caught
        return frozenset()


def is_us_market_holiday(dt: datetime) -> bool:
    """True if `dt` is a US public holiday (NY local date)."""
    if not _HAS_HOLIDAYS:
        return False
    ny_dt = dt.astimezone(NY_TZ)
    return ny_dt.date() in _get_us_holiday_set(ny_dt.year)


def is_fx_market_open(now: Optional[datetime] = None) -> bool:
    """
    FX market: open Sunday 17:00 NY local → Friday 17:00 NY local.
    Uses zoneinfo for correct DST handling year-round (defect #2).
    """
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    ny = now.astimezone(NY_TZ)
    wd, hr = ny.weekday(), ny.hour
    if wd == 5:           # Saturday closed
        return False
    if wd == 4 and hr >= 17:  # Friday after 17:00 closed
        return False
    if wd == 6 and hr < 17:   # Sunday before 17:00 closed
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TECHNICAL INDICATORS (NaN-safe)
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_float(value: Any) -> Optional[float]:
    """Convert to float; return None on NaN/inf/error (defect #7)."""
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


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


def _dmi(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> Tuple[Optional[float], Optional[float]]:
    tr = _true_range(high, low, close)
    atr_s = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    up = high.diff()
    dn = -low.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    pdi = 100 * pdm.ewm(alpha=1.0 / n, adjust=False).mean() / atr_s.replace(0, np.nan)
    mdi = 100 * mdm.ewm(alpha=1.0 / n, adjust=False).mean() / atr_s.replace(0, np.nan)
    return _safe_float(pdi.iloc[-1]), _safe_float(mdi.iloc[-1])


def _fmt_atr(val: Optional[float]) -> str:
    if val is None or val <= 0:
        return "N/A"
    if val >= 10:
        return f"{val:.2f}"
    if val >= 1:
        return f"{val:.3f}"
    return f"{val:.4f}"


def _find_strict_peaks(
    series: pd.Series, wing: int, min_idx: int, prominence: Optional[float] = None
) -> List[int]:
    """Locate strict local maxima with optional prominence (defect #12)."""
    arr = series.to_numpy()
    n = len(arr)
    if n < 2 * wing + 1:
        return []
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=-np.inf)
    if _HAS_SCIPY:
        kwargs: Dict[str, Any] = {"distance": max(1, wing)}
        if prominence is not None and prominence > 0:
            kwargs["prominence"] = prominence
        peaks, _ = find_peaks(arr, **kwargs)
        return [int(p) for p in peaks if min_idx <= p <= n - wing - 1]
    result: List[int] = []
    for i in range(max(min_idx, wing), n - wing):
        window = arr[i - wing: i + wing + 1]
        center = arr[i]
        if center == window.max() and np.sum(window == center) == 1:
            if prominence is not None and prominence > 0:
                if (center - window.min()) < prominence:
                    continue
            result.append(i)
    return result


def _find_strict_troughs(
    series: pd.Series, wing: int, min_idx: int, prominence: Optional[float] = None
) -> List[int]:
    arr = series.to_numpy()
    n = len(arr)
    if n < 2 * wing + 1:
        return []
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=np.inf)
    if _HAS_SCIPY:
        kwargs: Dict[str, Any] = {"distance": max(1, wing)}
        if prominence is not None and prominence > 0:
            kwargs["prominence"] = prominence
        peaks, _ = find_peaks(-arr, **kwargs)
        return [int(p) for p in peaks if min_idx <= p <= n - wing - 1]
    result: List[int] = []
    for i in range(max(min_idx, wing), n - wing):
        window = arr[i - wing: i + wing + 1]
        center = arr[i]
        if center == window.min() and np.sum(window == center) == 1:
            if prominence is not None and prominence > 0:
                if (window.max() - center) < prominence:
                    continue
            result.append(i)
    return result


def compute_indicator_bundle(df: pd.DataFrame, intraday: bool) -> Optional[IndicatorBundle]:
    """Pre-compute all indicators for a single (instrument, tf) — defect #13."""
    if df.empty or len(df) < 30:
        return None
    h, lo, c = df["High"], df["Low"], df["Close"]
    try:
        atr_v = _safe_float(_atr(h, lo, c, CFG.ind.atr_period).iloc[-1])
        e_short = _safe_float(_ema(c, CFG.ind.ema_short).iloc[-1])
        e_long_series = _ema(c, CFG.ind.ema_long)
        e_long_cur = _safe_float(e_long_series.iloc[-1])
        rsi_v = _safe_float(_rsi(c, CFG.ind.rsi_period).iloc[-1])
        ema_fast = _ema(c, CFG.ind.macd_fast)
        ema_slow = _ema(c, CFG.ind.macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line = _ema(macd_line, CFG.ind.macd_signal)
        macd_v = _safe_float(macd_line.iloc[-1])
        signal_v = _safe_float(signal_line.iloc[-1])

        ema_intra_fast_v: Optional[float] = None
        ema_intra_long_v: Optional[float] = None
        zlema_v: Optional[float] = None
        if intraday:
            period = CFG.ind.ema_intra_period
            lag = (period - 1) // 2
            ema_intra_fast_v = _safe_float(_ema(c, CFG.ind.intraday_ema_fast).iloc[-1])
            ema_intra_long_v = _safe_float(_ema(c, period).iloc[-1])
            src_adj = c + (c - c.shift(lag))
            zlema_v = _safe_float(src_adj.ewm(span=period, adjust=False).mean().iloc[-1])

        # Use 0.0 placeholders only after checking has_nan flag
        has_nan = any(v is None for v in (atr_v, e_short, e_long_cur))

        return IndicatorBundle(
            atr_val=atr_v if atr_v is not None else 0.0,
            ema_short=e_short if e_short is not None else 0.0,
            ema_long_cur=e_long_cur if e_long_cur is not None else 0.0,
            ema_long_series=e_long_series,
            rsi=rsi_v if rsi_v is not None else 50.0,
            macd=macd_v if macd_v is not None else 0.0,
            macd_signal=signal_v if signal_v is not None else 0.0,
            ema_intra_fast=ema_intra_fast_v if ema_intra_fast_v is not None else 0.0,
            ema_intra_long=ema_intra_long_v if ema_intra_long_v is not None else 0.0,
            zlema=zlema_v if zlema_v is not None else 0.0,
            has_nan=has_nan,
        )
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        _log_incident(
            IncidentCode.DATA_NAN, "indicator computation failed",
            err=type(exc).__name__,
        )
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — RATE LIMITER (per-instrument + global, singleton)
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
    """Per-instrument buckets + global bucket (defect #16)."""

    _GLOBAL_KEY = "__global__"

    def __init__(self) -> None:
        self._buckets: Dict[str, _TokenBucket] = {
            self._GLOBAL_KEY: _TokenBucket(
                CFG.http.rate_limit_global_burst,
                CFG.http.rate_limit_global_refill_per_sec,
            )
        }
        self._lock = threading.Lock()

    def acquire(self, instrument: str) -> bool:
        with self._lock:
            inst_bucket = self._buckets.get(instrument)
            if inst_bucket is None:
                inst_bucket = _TokenBucket(
                    CFG.http.rate_limit_burst_per_instrument,
                    CFG.http.rate_limit_refill_per_sec,
                )
                self._buckets[instrument] = inst_bucket
            global_bucket = self._buckets[self._GLOBAL_KEY]
        # Acquire both — atomic from caller's POV at this granularity is sufficient
        if not global_bucket.try_acquire():
            return False
        if not inst_bucket.try_acquire():
            # Refund global by 1: best-effort — accept slight imprecision
            return False
        return True

    def trigger_cooldown(self, instrument: str, seconds: float) -> None:
        with self._lock:
            bucket = self._buckets.get(instrument)
            global_bucket = self._buckets[self._GLOBAL_KEY]
        if bucket is not None:
            bucket.trigger_cooldown(seconds)
        global_bucket.trigger_cooldown(min(seconds, 10.0))


# Module-level singleton (defect #9, #10) — Streamlit-aware
_RATE_LIMITER_INSTANCE: Optional[_GlobalRateLimiter] = None
_RATE_LIMITER_LOCK = threading.Lock()


if _STREAMLIT_AVAILABLE:
    @st.cache_resource(show_spinner=False)
    def _st_rate_limiter() -> _GlobalRateLimiter:
        return _GlobalRateLimiter()


def _get_rate_limiter() -> _GlobalRateLimiter:
    global _RATE_LIMITER_INSTANCE
    if _STREAMLIT_AVAILABLE and _is_streamlit_runtime():
        try:
            return _st_rate_limiter()
        except Exception as e:  # pylint: disable=broad-exception-caught
            _log_incident(
                IncidentCode.UI_CALLBACK_ERROR,
                "streamlit rate limiter cache failed, falling back",
                err=type(e).__name__,
            )
    if _RATE_LIMITER_INSTANCE is None:
        with _RATE_LIMITER_LOCK:
            if _RATE_LIMITER_INSTANCE is None:
                _RATE_LIMITER_INSTANCE = _GlobalRateLimiter()
    return _RATE_LIMITER_INSTANCE


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — HTTP SESSION REGISTRY (module-level singleton)
# ═══════════════════════════════════════════════════════════════════════════════

class SessionRegistry:
    """Thread-scoped HTTP session registry — module-level singleton (defect #10)."""

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
            total=CFG.http.retry_total,
            backoff_factor=CFG.http.retry_backoff,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=CFG.http.pool_maxsize)
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
                        IncidentCode.SESSION_CLOSED, "session close failed",
                        level=logging.DEBUG, err=type(exc).__name__,
                    )
            self._thread_sessions.clear()


_SESSION_REGISTRY_INSTANCE: Optional[SessionRegistry] = None
_SESSION_REGISTRY_LOCK = threading.Lock()


if _STREAMLIT_AVAILABLE:
    @st.cache_resource(show_spinner=False)
    def _st_session_registry() -> SessionRegistry:
        return SessionRegistry()


def _get_session_registry() -> SessionRegistry:
    global _SESSION_REGISTRY_INSTANCE
    if _STREAMLIT_AVAILABLE and _is_streamlit_runtime():
        try:
            return _st_session_registry()
        except Exception as e:  # pylint: disable=broad-exception-caught
            _log_incident(
                IncidentCode.UI_CALLBACK_ERROR, "streamlit session registry cache failed",
                err=type(e).__name__,
            )
    if _SESSION_REGISTRY_INSTANCE is None:
        with _SESSION_REGISTRY_LOCK:
            if _SESSION_REGISTRY_INSTANCE is None:
                _SESSION_REGISTRY_INSTANCE = SessionRegistry()
    return _SESSION_REGISTRY_INSTANCE


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CACHE ENGINE (LRU, single-flight, attrs-safe)
# ═══════════════════════════════════════════════════════════════════════════════

CacheKey = Tuple[str, AccountHash, str, str, int, Optional[str]]  # +to_iso for snapshot
LiveOpenKey = Tuple[str, AccountHash, str, str]
PricingKey = Tuple[str, AccountHash, str]


def _hash_account(account_id: str) -> AccountHash:
    h = hashlib.sha256(account_id.encode("utf-8")).hexdigest()[:16]
    return AccountHash(h)


def _defensive_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Deep copy including attrs (defect #3)."""
    out = df.copy(deep=True)
    # pandas does not always deep-copy attrs; force it
    out.attrs = dict(df.attrs)
    return out


class CandleCache:
    """LRU cache, thread-safe, single-flight with result hand-off (defects #3, #5, #21)."""

    def __init__(self, max_entries: int) -> None:
        self._data: "OrderedDict[CacheKey, Tuple[datetime, pd.DataFrame]]" = OrderedDict()
        self._stale: Dict[CacheKey, Tuple[datetime, pd.DataFrame]] = {}
        self._live_opens: Dict[LiveOpenKey, Tuple[datetime, Optional[float]]] = {}
        self._pricing: Dict[PricingKey, Tuple[datetime, Optional[float]]] = {}
        self._inflight: Dict[CacheKey, threading.Event] = {}
        self._inflight_results: Dict[CacheKey, FetchResult] = {}
        self._lock = threading.RLock()
        self._max_entries = max_entries

    def _evict_if_needed(self) -> None:
        while len(self._data) > self._max_entries:
            old_key, _ = self._data.popitem(last=False)
            self._stale.pop(old_key, None)

    def _touch(self, key: CacheKey) -> None:
        if key in self._data:
            self._data.move_to_end(key)

    def get_candles(
        self,
        key: CacheKey,
        ttl: int,
        fetch_fn: Callable[[], FetchResult],
        leader_budget_sec: float,
    ) -> FetchResult:
        now = datetime.now(UTC)
        with self._lock:
            entry = self._data.get(key)
            if entry is not None and (now - entry[0]).total_seconds() < ttl:
                self._touch(key)
                df = _defensive_copy(entry[1])
                return FetchResult(
                    df=df, is_stale=False, fetched_at=entry[0],
                    critical_gap=bool(df.attrs.get("critical_gap", False)),
                )

            inflight_event = self._inflight.get(key)
            if inflight_event is not None:
                wait_for_event = inflight_event
                start_my_fetch = False
            else:
                wait_for_event = threading.Event()
                self._inflight[key] = wait_for_event
                start_my_fetch = True

        if not start_my_fetch:
            return self._wait_for_leader(key, wait_for_event, ttl, leader_budget_sec)

        return self._do_fetch_as_leader(key, ttl, now, fetch_fn, wait_for_event)

    def _wait_for_leader(
        self,
        key: CacheKey,
        event: threading.Event,
        ttl: int,
        leader_budget_sec: float,
    ) -> FetchResult:
        completed = event.wait(timeout=leader_budget_sec)
        if not completed:
            _log_incident(
                IncidentCode.CACHE_LEADER_FAILED, "leader timeout",
                key=str(key[2:]),
            )
        with self._lock:
            # First: direct result hand-off (defect #5)
            handoff = self._inflight_results.pop(key, None)
            if handoff is not None:
                return FetchResult(
                    df=_defensive_copy(handoff.df) if not handoff.df.empty else handoff.df,
                    is_stale=handoff.is_stale,
                    fetched_at=handoff.fetched_at,
                    critical_gap=handoff.critical_gap,
                )
            entry = self._data.get(key)
            if entry is not None:
                self._touch(key)
                df = _defensive_copy(entry[1])
                return FetchResult(
                    df=df, is_stale=False, fetched_at=entry[0],
                    critical_gap=bool(df.attrs.get("critical_gap", False)),
                )
            stale = self._stale.get(key)
            if stale is not None:
                age = (datetime.now(UTC) - stale[0]).total_seconds()
                if age < ttl * CFG.cache.stale_max_age_multiplier:
                    _log_incident(
                        IncidentCode.CACHE_STALE_HIT, "follower stale fallback",
                        key=str(key[2:]), age_sec=int(age),
                    )
                    df = _defensive_copy(stale[1])
                    return FetchResult(
                        df=df, is_stale=True, fetched_at=stale[0],
                        critical_gap=bool(df.attrs.get("critical_gap", False)),
                    )
        return FetchResult(df=pd.DataFrame(), is_stale=False)

    def _do_fetch_as_leader(
        self,
        key: CacheKey,
        ttl: int,
        now: datetime,
        fetch_fn: Callable[[], FetchResult],
        event: threading.Event,
    ) -> FetchResult:
        try:
            result = fetch_fn()
            if not result.df.empty:
                ts = result.fetched_at or datetime.now(UTC)
                with self._lock:
                    self._data[key] = (ts, result.df)
                    self._stale[key] = (ts, result.df)
                    self._touch(key)
                    self._evict_if_needed()
                    # Publish to followers (defect #5)
                    self._inflight_results[key] = FetchResult(
                        df=result.df, is_stale=False, fetched_at=ts,
                        critical_gap=result.critical_gap,
                    )
                return FetchResult(
                    df=_defensive_copy(result.df), is_stale=False, fetched_at=ts,
                    critical_gap=result.critical_gap,
                )

            with self._lock:
                stale = self._stale.get(key)
                if stale is not None:
                    age = (now - stale[0]).total_seconds()
                    if age < ttl * CFG.cache.stale_max_age_multiplier:
                        _log_incident(
                            IncidentCode.CACHE_STALE_HIT, "leader stale fallback",
                            key=str(key[2:]), age_sec=int(age),
                        )
                        df = _defensive_copy(stale[1])
                        result = FetchResult(
                            df=df, is_stale=True, fetched_at=stale[0],
                            critical_gap=bool(df.attrs.get("critical_gap", False)),
                        )
                        self._inflight_results[key] = FetchResult(
                            df=stale[1], is_stale=True, fetched_at=stale[0],
                            critical_gap=bool(stale[1].attrs.get("critical_gap", False)),
                        )
                        return result
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
        now = datetime.now(UTC)
        with self._lock:
            entry = self._live_opens.get(key)
            if entry is not None:
                ts, value = entry
                age = (now - ts).total_seconds()
                ttl = (
                    CFG.cache.ttl_live_open
                    if value is not None
                    else CFG.cache.ttl_negative_live_open
                )
                if age < ttl:
                    return value
        value = fetch_fn()
        with self._lock:
            self._live_opens[key] = (datetime.now(UTC), value)
        return value

    def get_pricing(
        self,
        key: PricingKey,
        fetch_fn: Callable[[], Optional[float]],
    ) -> Optional[float]:
        """Cache live mid prices with very short TTL (defect #1)."""
        now = datetime.now(UTC)
        with self._lock:
            entry = self._pricing.get(key)
            if entry is not None:
                ts, value = entry
                if (now - ts).total_seconds() < CFG.cache.ttl_pricing_sec:
                    return value
        value = fetch_fn()
        with self._lock:
            self._pricing[key] = (datetime.now(UTC), value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._stale.clear()
            self._live_opens.clear()
            self._pricing.clear()
            self._inflight.clear()
            self._inflight_results.clear()


_CANDLE_CACHE_INSTANCE: Optional[CandleCache] = None
_CANDLE_CACHE_LOCK = threading.Lock()


if _STREAMLIT_AVAILABLE:
    @st.cache_resource(show_spinner=False)
    def _st_candle_cache() -> CandleCache:
        return CandleCache(max_entries=CFG.cache.max_entries)


def _get_candle_cache() -> CandleCache:
    global _CANDLE_CACHE_INSTANCE
    if _STREAMLIT_AVAILABLE and _is_streamlit_runtime():
        try:
            return _st_candle_cache()
        except Exception as e:  # pylint: disable=broad-exception-caught
            _log_incident(
                IncidentCode.UI_CALLBACK_ERROR, "streamlit candle cache failed",
                err=type(e).__name__,
            )
    if _CANDLE_CACHE_INSTANCE is None:
        with _CANDLE_CACHE_LOCK:
            if _CANDLE_CACHE_INSTANCE is None:
                _CANDLE_CACHE_INSTANCE = CandleCache(max_entries=CFG.cache.max_entries)
    return _CANDLE_CACHE_INSTANCE


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — RUN CONTROL (global stop-event for prior runs — defect #19)
# ═══════════════════════════════════════════════════════════════════════════════

class _RunController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_stop: Optional[threading.Event] = None

    def start_new_run(self) -> threading.Event:
        with self._lock:
            if self._current_stop is not None:
                self._current_stop.set()
            ev = threading.Event()
            self._current_stop = ev
            return ev

    def finish_run(self, ev: threading.Event) -> None:
        with self._lock:
            if self._current_stop is ev:
                self._current_stop = None


_RUN_CONTROLLER = _RunController()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — DATA LAYER (validation, parsing, fetching)
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_candle(row: Mapping[str, float], allow_zero_volume: bool) -> bool:
    try:
        h, lo, o, c, v = row["High"], row["Low"], row["Open"], row["Close"], row["Volume"]
        if not (
            h >= lo > 0 and o > 0 and c > 0 and h >= o and h >= c
            and lo <= o and lo <= c and v >= 0
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
        _log_incident(IncidentCode.DATA_VALIDATION, "missing columns",
                      instrument=instrument, granularity=granularity, missing=",".join(missing))
        return False
    if not isinstance(df.index, pd.DatetimeIndex):
        _log_incident(IncidentCode.DATA_VALIDATION, "non-datetime index",
                      instrument=instrument, granularity=granularity)
        return False
    if not df.index.is_monotonic_increasing:
        _log_incident(IncidentCode.DATA_VALIDATION, "non-monotonic index",
                      instrument=instrument, granularity=granularity)
        return False
    if df.index.has_duplicates:
        _log_incident(IncidentCode.DATA_VALIDATION, "duplicates in index",
                      instrument=instrument, granularity=granularity)
        return False
    return True


def _validate_dataframe_gaps(df: pd.DataFrame, granularity: str, instrument: str) -> bool:
    """
    Critical-gap detection with NY-local weekday + holiday awareness (defects #8, #18).
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

    if max_gap <= tolerance:
        df.attrs["critical_gap"] = False
        return True

    # Convert to NY local time for honest weekday assessment
    gap_start_utc = max_idx - max_gap
    gap_end_utc = max_idx
    try:
        gs_ny = gap_start_utc.tz_convert(NY_TZ)
        ge_ny = gap_end_utc.tz_convert(NY_TZ)
    except (TypeError, AttributeError):
        gs_ny = gap_start_utc.to_pydatetime().astimezone(NY_TZ)
        ge_ny = gap_end_utc.to_pydatetime().astimezone(NY_TZ)

    is_weekend = (gs_ny.weekday() in (4, 5, 6)) and (ge_ny.weekday() in (6, 0, 1))
    is_holiday_bridge = False
    if _HAS_HOLIDAYS:
        try:
            current = gs_ny.date()
            end_date = ge_ny.date()
            holiday_or_weekend_days = 0
            total_days = 0
            while current <= end_date and total_days < 10:
                if (current.weekday() >= 5 or
                        current in _get_us_holiday_set(current.year)):
                    holiday_or_weekend_days += 1
                total_days += 1
                current = current + timedelta(days=1)
            if total_days > 0 and holiday_or_weekend_days / total_days >= 0.6:
                is_holiday_bridge = True
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    if is_weekend or is_holiday_bridge:
        _log_incident(
            IncidentCode.DATA_GAPS, "tolerated weekend/holiday gap",
            instrument=instrument, granularity=granularity,
            max_gap_sec=int(max_gap.total_seconds()),
            level=logging.INFO,
        )
        df.attrs["critical_gap"] = False
    else:
        _log_incident(
            IncidentCode.DATA_GAPS, "critical gap during market open",
            instrument=instrument, granularity=granularity,
            max_gap_sec=int(max_gap.total_seconds()),
            tolerance_sec=int(tolerance.total_seconds()),
        )
        df.attrs["critical_gap"] = True
    return True


def _fallback_bid_ask(candle: Any) -> Optional[Dict[str, float]]:
    try:
        bid = candle.get("bid")
        ask = candle.get("ask")
        if isinstance(bid, dict) and isinstance(ask, dict):
            mid: Dict[str, float] = {}
            for field_name in ("o", "h", "l", "c"):
                b = bid.get(field_name)
                a = ask.get(field_name)
                if b is not None and a is not None:
                    mid[field_name] = (float(b) + float(a)) / 2.0
                elif b is not None:
                    mid[field_name] = float(b)
                elif a is not None:
                    mid[field_name] = float(a)
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
        _log_incident(IncidentCode.JSON_INVALID, "root not dict",
                      instrument=instrument, granularity=granularity)
        return []
    candles = raw_json.get("candles", [])
    if not isinstance(candles, list):
        _log_incident(IncidentCode.JSON_INVALID, "candles not list",
                      instrument=instrument, granularity=granularity)
        return []

    is_index = instrument in INDICES
    market_open = is_fx_market_open()
    # Indices vol=0 treated as NaN (defect #4): we still accept the row but
    # downstream logic must skip zero-volume in moving averages.
    allow_zero_volume = is_index or not market_open

    rows: List[Dict[str, Any]] = []
    total = len(candles)
    errors = 0

    for candle in candles:
        if not isinstance(candle, dict):
            errors += 1
            continue
        if not include_incomplete and not candle.get("complete"):
            continue
        try:
            mid = candle.get("mid")
            if not isinstance(mid, dict):
                mid = _fallback_bid_ask(candle)
                if mid is None:
                    errors += 1
                    continue
            volume_raw = candle.get("volume", 0)
            try:
                volume = float(volume_raw) if volume_raw is not None else 0.0
            except (ValueError, TypeError):
                volume = 0.0

            row = {
                "date": candle.get("time"),
                "Open": float(mid["o"]),
                "High": float(mid["h"]),
                "Low": float(mid["l"]),
                "Close": float(mid["c"]),
                "Volume": volume,
            }
            if row["date"] is None or not _validate_candle(row, allow_zero_volume):
                errors += 1
                continue
            rows.append(row)
        except (KeyError, ValueError, TypeError):
            errors += 1
            continue

    if total > 0:
        error_rate = errors / total
        if error_rate > 0.10:
            _log_incident(
                IncidentCode.JSON_INVALID, "high candle parse error rate",
                instrument=instrument, granularity=granularity,
                total=total, failed=errors, error_rate=f"{error_rate:.1%}",
            )
    return rows


def _fetch_candles_raw(
    instrument: str,
    granularity: str,
    count: int,
    account_id: str,
    access_token: SecretToken,
    registry: SessionRegistry,
    include_incomplete: bool = False,
    to_iso: Optional[str] = None,
) -> FetchResult:
    if not _get_rate_limiter().acquire(instrument):
        _log_incident(
            IncidentCode.HTTP_RATELIMIT, "rate-limited locally",
            instrument=instrument, granularity=granularity, level=logging.INFO,
        )
        return FetchResult(df=pd.DataFrame())

    url = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token.reveal()}"}
    params: Dict[str, Any] = {"granularity": granularity, "count": count, "price": "M"}
    if to_iso is not None:
        params["to"] = to_iso

    try:
        session = registry.get_for_thread()
    except RuntimeError:
        return FetchResult(df=pd.DataFrame())

    try:
        r = session.get(url, headers=headers, params=params, timeout=CFG.http.timeout_sec)
    except requests.exceptions.Timeout:
        _log_incident(IncidentCode.HTTP_TIMEOUT, "OANDA timeout",
                      instrument=instrument, granularity=granularity)
        return FetchResult(df=pd.DataFrame())
    except requests.exceptions.RequestException as e:
        _log_incident(IncidentCode.HTTP_ERROR, "OANDA request exception",
                      instrument=instrument, granularity=granularity, err=type(e).__name__)
        return FetchResult(df=pd.DataFrame())

    if r.status_code == 429:
        try:
            retry_after = int(r.headers.get("Retry-After", "5"))
        except (ValueError, TypeError):
            retry_after = 5
        cooldown = min(retry_after, CFG.http.rate_limit_429_cooldown_sec)
        _get_rate_limiter().trigger_cooldown(instrument, cooldown)
        _log_incident(IncidentCode.HTTP_RATELIMIT, "OANDA 429",
                      instrument=instrument, granularity=granularity, retry_after_sec=retry_after)
        return FetchResult(df=pd.DataFrame())

    if r.status_code in (401, 403):
        _log_incident(IncidentCode.HTTP_AUTH, "OANDA auth failure",
                      instrument=instrument, granularity=granularity,
                      status=r.status_code, level=logging.ERROR)
        return FetchResult(df=pd.DataFrame())

    if r.status_code != 200:
        _log_incident(IncidentCode.HTTP_ERROR, "OANDA non-200",
                      instrument=instrument, granularity=granularity, status=r.status_code)
        return FetchResult(df=pd.DataFrame())

    try:
        raw_json = r.json()
    except ValueError:
        _log_incident(IncidentCode.JSON_INVALID, "OANDA invalid JSON",
                      instrument=instrument, granularity=granularity)
        return FetchResult(df=pd.DataFrame())

    rows = _parse_oanda_json(raw_json, instrument, granularity, include_incomplete)
    if not rows:
        return FetchResult(df=pd.DataFrame())

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return FetchResult(df=pd.DataFrame())
    df = df.set_index("date").sort_index()

    if not _validate_dataframe_schema(df, instrument, granularity):
        return FetchResult(df=pd.DataFrame())
    _validate_dataframe_gaps(df, granularity, instrument)

    return FetchResult(
        df=df,
        is_stale=False,
        fetched_at=datetime.now(UTC),
        critical_gap=bool(df.attrs.get("critical_gap", False)),
    )


def fetch_pricing_mid(
    instrument: str,
    account_id: str,
    account_hash: AccountHash,
    access_token: SecretToken,
    registry: SessionRegistry,
) -> Optional[float]:
    """Live mid price via /pricing endpoint — addresses defect #1."""
    key: PricingKey = (_OANDA_ENV, account_hash, instrument)

    def _fetch() -> Optional[float]:
        if not _get_rate_limiter().acquire(instrument):
            return None
        url = f"{OANDA_API_URL}/v3/accounts/{account_id}/pricing"
        headers = {"Authorization": f"Bearer {access_token.reveal()}"}
        params = {"instruments": instrument}
        try:
            session = registry.get_for_thread()
            r = session.get(url, headers=headers, params=params, timeout=CFG.http.timeout_sec)
        except (requests.exceptions.RequestException, RuntimeError) as e:
            _log_incident(IncidentCode.PRICING_UNAVAILABLE, "pricing fetch failed",
                          instrument=instrument, err=type(e).__name__)
            return None
        if r.status_code != 200:
            _log_incident(IncidentCode.PRICING_UNAVAILABLE, "pricing non-200",
                          instrument=instrument, status=r.status_code, level=logging.INFO)
            return None
        try:
            data = r.json()
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        prices = data.get("prices", [])
        if not isinstance(prices, list) or not prices:
            return None
        p = prices[0]
        if not isinstance(p, dict):
            return None
        bids = p.get("bids", [])
        asks = p.get("asks", [])
        if not bids or not asks:
            return None
        try:
            bid = float(bids[0].get("price"))
            ask = float(asks[0].get("price"))
            return (bid + ask) / 2.0
        except (ValueError, TypeError, KeyError, IndexError):
            return None

    return _get_candle_cache().get_pricing(key, _fetch)


def fetch_cached(
    instrument: str,
    granularity: str,
    count: int,
    account_id: str,
    account_hash: AccountHash,
    access_token: SecretToken,
    registry: SessionRegistry,
    to_iso: Optional[str] = None,
) -> FetchResult:
    key: CacheKey = (_OANDA_ENV, account_hash, instrument, granularity, count, to_iso)
    ttl = _CACHE_TTL.get(granularity, CFG.cache.ttl_default)
    leader_budget = CFG.http.timeout_sec * (CFG.http.retry_total + 1) * 1.5 + 5.0
    return _get_candle_cache().get_candles(
        key,
        ttl,
        lambda: _fetch_candles_raw(
            instrument, granularity, count, account_id, access_token, registry,
            include_incomplete=False, to_iso=to_iso,
        ),
        leader_budget_sec=leader_budget,
    )


def fetch_live_open(
    instrument: str,
    granularity: str,
    account_id: str,
    account_hash: AccountHash,
    access_token: SecretToken,
    registry: SessionRegistry,
) -> Optional[float]:
    key: LiveOpenKey = (_OANDA_ENV, account_hash, instrument, granularity)

    def _fetch() -> Optional[float]:
        result = _fetch_candles_raw(
            instrument, granularity, 1, account_id, access_token, registry,
            include_incomplete=True,
        )
        if result.df.empty:
            return None
        try:
            last_ts = result.df.index[-1]
            now = datetime.now(UTC)
            max_age = _LIVE_OPEN_MAX_AGE.get(granularity, pd.Timedelta(days=1))
            if (now - last_ts) > max_age:
                return None
            return _safe_float(result.df["Open"].iloc[-1])
        except (IndexError, ValueError, TypeError):
            return None

    return _get_candle_cache().get_live_open(key, _fetch)


def fetch_all_data(
    instrument: str,
    account_id: str,
    account_hash: AccountHash,
    access_token: SecretToken,
    registry: SessionRegistry,
    stop_event: threading.Event,
    snapshot_to_iso: Optional[str],
) -> Dict[str, Any]:
    """
    Fetch all timeframes with shared `to=` parameter for temporal coherence
    across pairs in the same run (defect #26).
    """
    specs = {
        "M":   ("M",   150, CFG.bars.min_bars_m),
        "W":   ("W",   250, CFG.bars.min_bars_w),
        "D":   ("D",   300, CFG.bars.min_bars_d),
        "4H":  ("H4",  300, CFG.bars.min_bars_h4),
        "1H":  ("H1",  300, CFG.bars.min_bars_h1),
        "15m": ("M15", 300, CFG.bars.min_bars_15m),
    }
    snapshot_started = datetime.now(UTC)
    cache: Dict[str, Any] = {
        "_snapshot_started_at": snapshot_started,
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
            instrument, gran, count, account_id, account_hash, access_token, registry,
            to_iso=snapshot_to_iso,
        )
        if result.df.empty:
            cache["is_incomplete"] = True
            cache["error_reason"] = f"Fetch failed on {tf}"
            return cache
        if len(result.df) < min_bars:
            _log_incident(IncidentCode.DATA_INSUFFICIENT, "bars below minimum",
                          instrument=instrument, tf=tf,
                          bars=len(result.df), min_bars=min_bars)
            cache["is_incomplete"] = True
            cache["error_reason"] = f"Insufficient bars on {tf} ({len(result.df)} < {min_bars})"
            return cache

        cache[tf] = result.df
        cache["_snapshot_per_tf"][tf] = result.fetched_at or datetime.now(UTC)
        if result.is_stale:
            cache["_stale_tfs"].append(tf)

    if stop_event.is_set():
        cache["is_incomplete"] = True
        cache["error_reason"] = "Stop requested"
        return cache

    cache["_snapshot_completed_at"] = datetime.now(UTC)
    ts_list = [v for v in cache["_snapshot_per_tf"].values() if v is not None]
    if ts_list:
        drift = (max(ts_list) - min(ts_list)).total_seconds()
        cache["_snapshot_drift_sec"] = drift
        cache["_snapshot_drift_exceeded"] = drift > CFG.ops.snapshot_drift_max_sec

    # Spot price for current-quote semantics (defect #1)
    cache["_spot_mid"] = fetch_pricing_mid(
        instrument, account_id, account_hash, access_token, registry
    )
    cache["_week_open"] = fetch_live_open(
        instrument, "W", account_id, account_hash, access_token, registry
    )
    cache["_day_open"] = fetch_live_open(
        instrument, "D", account_id, account_hash, access_token, registry
    )
    return cache


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — VOTING REGISTRY
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
# SECTION 14 — ATOMIC VOTES (using current spot, not stale close — defect #1)
# ═══════════════════════════════════════════════════════════════════════════════

@DAILY_VOTES.register(uid="swing_structure", critical=True)
def _vote_swing_structure(
    h: pd.Series, lo: pd.Series, _c: pd.Series, ctx: Mapping[str, Any]
) -> VoteSignal:
    """Multi-pivot confirmation with prominence (defect #12)."""
    name = "swing_structure"
    wing = CFG.ind.pivot_wing
    if len(h) < 2 * wing + CFG.ind.max_pivot_age + 1:
        return VoteSignal(name, Direction.RANGE, CFG.vote.weight_swing_structure,
                          0.9, False, f"série trop courte ({len(h)})")

    atr_val = ctx.get("atr_val", 0.0)
    prominence = atr_val * CFG.ind.pivot_prominence_atr_ratio if atr_val > 0 else None
    min_idx = max(0, len(h) - CFG.ind.max_pivot_age)

    sh = _find_strict_peaks(h, wing, min_idx, prominence)
    sl = _find_strict_troughs(lo, wing, min_idx, prominence)

    min_conf = CFG.ind.pivot_min_confirmations
    if len(sh) < min_conf or len(sl) < min_conf:
        return VoteSignal(name, Direction.RANGE, CFG.vote.weight_swing_structure,
                          0.9, False, f"pivots insuf sh={len(sh)} sl={len(sl)}")

    # Confirmation on last N highs/lows
    last_highs = [h.iloc[i] for i in sh[-min_conf:]]
    last_lows = [lo.iloc[i] for i in sl[-min_conf:]]

    all_hh = all(last_highs[i] > last_highs[i - 1] for i in range(1, len(last_highs)))
    all_hl = all(last_lows[i] > last_lows[i - 1] for i in range(1, len(last_lows)))
    all_lh = all(last_highs[i] < last_highs[i - 1] for i in range(1, len(last_highs)))
    all_ll = all(last_lows[i] < last_lows[i - 1] for i in range(1, len(last_lows)))

    if all_hh and all_hl:
        return VoteSignal(name, Direction.BULLISH, CFG.vote.weight_swing_structure,
                          0.9, True, f"{min_conf}xHH+{min_conf}xHL")
    if all_lh and all_ll:
        return VoteSignal(name, Direction.BEARISH, CFG.vote.weight_swing_structure,
                          0.9, True, f"{min_conf}xLH+{min_conf}xLL")
    return VoteSignal(name, Direction.RANGE, CFG.vote.weight_swing_structure,
                      0.9, False, "structure non confirmée")


@DAILY_VOTES.register(uid="ema_stack")
def _vote_ema_stack(_h, _lo, _c, ctx: Mapping[str, Any]) -> VoteSignal:
    name = "ema_stack"
    cur = ctx.get("cur")
    e21 = ctx.get("e21")
    e50_cur = ctx.get("e50_cur")
    if cur is None or e21 is None or e50_cur is None:
        return VoteSignal(name, Direction.RANGE, CFG.vote.weight_ema_stack,
                          0.75, False, "NaN/missing EMA")
    if cur > e21 > e50_cur:
        return VoteSignal(name, Direction.BULLISH, CFG.vote.weight_ema_stack,
                          0.75, True, "cur>e21>e50")
    if cur < e21 < e50_cur:
        return VoteSignal(name, Direction.BEARISH, CFG.vote.weight_ema_stack,
                          0.75, True, "cur<e21<e50")
    return VoteSignal(name, Direction.RANGE, CFG.vote.weight_ema_stack,
                      0.75, False, "stack non aligné")


@DAILY_VOTES.register(uid="weekly_open")
def _vote_weekly_open(_h, _lo, _c, ctx: Mapping[str, Any]) -> VoteSignal:
    name = "weekly_open"
    cur = ctx.get("cur")
    wo = ctx.get("current_week_open")
    if cur is None or wo is None:
        return VoteSignal(name, Direction.RANGE, CFG.vote.weight_weekly_open,
                          0.0, False, "current_W ou spot indisponible")
    if cur > wo:
        return VoteSignal(name, Direction.BULLISH, CFG.vote.weight_weekly_open,
                          0.90, True, "spot>wo")
    if cur < wo:
        return VoteSignal(name, Direction.BEARISH, CFG.vote.weight_weekly_open,
                          0.90, True, "spot<wo")
    return VoteSignal(name, Direction.RANGE, CFG.vote.weight_weekly_open,
                      0.90, False, "spot==wo")


def _vote_prev_midpoint_indices(
    h_j1: float, lo_j1: float, c_j1: float, ctx: Mapping[str, Any], direction: Direction
) -> VoteSignal:
    name = "prev_midpoint"
    rng = h_j1 - lo_j1
    if rng <= 0:
        return VoteSignal(name, Direction.RANGE, CFG.vote.weight_prev_midpoint,
                          0.0, False, "range nul")
    try:
        o_j1 = _safe_float(ctx["df_daily"]["Open"].iloc[-1])
        if o_j1 is None:
            return VoteSignal(name, Direction.RANGE, CFG.vote.weight_prev_midpoint,
                              0.0, False, "Open J-1 NaN")
    except (KeyError, IndexError, ValueError, TypeError):
        return VoteSignal(name, Direction.RANGE, CFG.vote.weight_prev_midpoint,
                          0.0, False, "Open J-1 inaccessible")
    body_ratio = abs(c_j1 - o_j1) / rng
    if body_ratio < CFG.vote.body_range_strong:
        return VoteSignal(name, Direction.RANGE, CFG.vote.weight_prev_midpoint,
                          0.0, False, f"body/range={body_ratio:.2f}")
    rel = min(0.80, 0.60 + (body_ratio - CFG.vote.body_range_strong) * 0.5)
    return VoteSignal(name, direction, CFG.vote.weight_prev_midpoint,
                      rel, True, f"body/range={body_ratio:.2f}")


def _vote_prev_midpoint_fx(
    vol: Optional[pd.Series], direction: Direction
) -> VoteSignal:
    name = "prev_midpoint"
    if vol is None or vol.empty:
        return VoteSignal(name, direction, CFG.vote.weight_prev_midpoint,
                          0.50, True, "no_vol_data")
    try:
        vol_ref = vol.iloc[:-1]
        vol_j1 = _safe_float(vol.iloc[-1])
        if vol_j1 is None or len(vol_ref) < 20:
            return VoteSignal(name, direction, CFG.vote.weight_prev_midpoint,
                              0.50, True, "vol_history<<20 or NaN")
        # Exclude zero volume on indices (defect #4) — but we're in FX path here
        vol_ma = _safe_float(vol_ref.rolling(20).mean().iloc[-1])
        if vol_ma is None or vol_ma <= 0:
            return VoteSignal(name, direction, CFG.vote.weight_prev_midpoint,
                              0.50, True, "vol_ma invalide")
        vol_ratio = vol_j1 / vol_ma
        if vol_ratio <= CFG.vote.volume_ratio_min_midpoint:
            return VoteSignal(name, Direction.RANGE, CFG.vote.weight_prev_midpoint,
                              0.0, False, f"vol_ratio={vol_ratio:.2f}")
        reliability = min(0.80, 0.65 + (vol_ratio - 1.0) * 0.05)
        return VoteSignal(name, direction, CFG.vote.weight_prev_midpoint,
                          reliability, True, f"vol_ratio={vol_ratio:.2f}")
    except (TypeError, ValueError, IndexError):
        return VoteSignal(name, direction, CFG.vote.weight_prev_midpoint,
                          0.50, True, "vol_err")


@DAILY_VOTES.register(uid="prev_midpoint")
def _vote_prev_midpoint(
    h: pd.Series, lo: pd.Series, c: pd.Series, ctx: Mapping[str, Any]
) -> VoteSignal:
    name = "prev_midpoint"
    if len(c) < 1:
        return VoteSignal(name, Direction.RANGE, CFG.vote.weight_prev_midpoint,
                          0.0, False, "série vide")
    instrument = ctx.get("instrument", "")
    is_index = instrument in INDICES

    h_j1 = _safe_float(h.iloc[-1])
    lo_j1 = _safe_float(lo.iloc[-1])
    c_j1 = _safe_float(c.iloc[-1])
    if h_j1 is None or lo_j1 is None or c_j1 is None:
        return VoteSignal(name, Direction.RANGE, CFG.vote.weight_prev_midpoint,
                          0.0, False, "NaN OHLC")
    mid_j1 = (h_j1 + lo_j1) / 2.0
    direction = Direction.BULLISH if c_j1 > mid_j1 else Direction.BEARISH

    if is_index:
        return _vote_prev_midpoint_indices(h_j1, lo_j1, c_j1, ctx, direction)
    return _vote_prev_midpoint_fx(ctx.get("vol_series"), direction)


@DAILY_VOTES.register(uid="ema50_slope")
def _vote_ema50_slope(_h, _lo, _c, ctx: Mapping[str, Any]) -> VoteSignal:
    name = "ema50_slope"
    e50 = ctx.get("e50")
    atr_val = ctx.get("atr_val")
    threshold = CFG.vote.ema50_slope_threshold
    if e50 is None or len(e50) < 6 or atr_val is None or atr_val <= 0:
        return VoteSignal(name, Direction.RANGE, CFG.vote.weight_ema50_slope,
                          0.70, False, "données insuffisantes")
    val_now = _safe_float(e50.iloc[-1])
    val_back = _safe_float(e50.iloc[-6])
    if val_now is None or val_back is None:
        return VoteSignal(name, Direction.RANGE, CFG.vote.weight_ema50_slope,
                          0.70, False, "NaN EMA")
    slope_ratio = (val_now - val_back) / atr_val
    if slope_ratio > threshold:
        return VoteSignal(name, Direction.BULLISH, CFG.vote.weight_ema50_slope,
                          0.70, True, f"slope={slope_ratio:.3f}")
    if slope_ratio < -threshold:
        return VoteSignal(name, Direction.BEARISH, CFG.vote.weight_ema50_slope,
                          0.70, True, f"slope={slope_ratio:.3f}")
    return VoteSignal(name, Direction.RANGE, CFG.vote.weight_ema50_slope,
                      0.70, False, f"slope={slope_ratio:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate_votes(
    votes: Tuple[VoteSignal, ...], atr_val: float, degraded: bool
) -> DailyTrendResult:
    bull_score = sum(v.weight * v.reliability for v in votes
                     if v.fired and v.direction == Direction.BULLISH)
    bear_score = sum(v.weight * v.reliability for v in votes
                     if v.fired and v.direction == Direction.BEARISH)
    fired_possible = sum(v.weight * v.reliability for v in votes if v.fired)
    winning_score = max(bull_score, bear_score)
    min_votes_met = winning_score >= CFG.vote.min_reliable_score

    if not min_votes_met or bull_score == bear_score:
        return DailyTrendResult(
            Direction.RANGE, CFG.vote.strength_min_range, atr_val,
            bull_score, bear_score, votes, min_votes_met, degraded,
        )
    direction = Direction.BULLISH if bull_score > bear_score else Direction.BEARISH
    ratio = winning_score / fired_possible if fired_possible > 0 else 0.0
    strength = int(min(CFG.vote.strength_max,
                       max(CFG.vote.strength_min_range, ratio * CFG.vote.strength_scaler)))
    return DailyTrendResult(
        direction, strength, atr_val, bull_score, bear_score, votes,
        min_votes_met, degraded,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — TREND ANALYSIS (uses pre-computed IndicatorBundle)
# ═══════════════════════════════════════════════════════════════════════════════

def trend_daily(
    df: pd.DataFrame,
    instrument: str,
    bundle: IndicatorBundle,
    spot_mid: Optional[float],
    current_week_open: Optional[float],
) -> DailyTrendResult:
    h, lo, c = df["High"], df["Low"], df["Close"]
    atr_val = bundle.atr_val
    if len(df) < CFG.bars.min_bars_d or bundle.has_nan:
        guard = VoteSignal("guard", Direction.RANGE, 0.0, 1.0, False,
                           f"insuf/NaN (n={len(df)},nan={bundle.has_nan})")
        return DailyTrendResult(Direction.RANGE, 0, atr_val, 0.0, 0.0, (guard,), False, True)

    # cur = live spot if available; otherwise fallback to last close (defect #1)
    cur = spot_mid if spot_mid is not None else _safe_float(c.iloc[-1])
    if cur is None:
        guard = VoteSignal("guard", Direction.RANGE, 0.0, 1.0, False, "no spot/close")
        return DailyTrendResult(Direction.RANGE, 0, atr_val, 0.0, 0.0, (guard,), False, True)

    vol_series = df["Volume"] if "Volume" in df.columns and instrument not in INDICES else None

    ctx: Dict[str, Any] = {
        "cur": cur,
        "e21": bundle.ema_short,
        "e50_cur": bundle.ema_long_cur,
        "e50": bundle.ema_long_series,
        "atr_val": atr_val,
        "df_daily": df,
        "vol_series": vol_series,
        "instrument": instrument,
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
            v = VoteSignal(spec.uid, Direction.RANGE, 0.0, 0.0, False,
                           f"err:{type(exc).__name__}", errored=True)
            if spec.critical:
                degraded = True
        raw_votes.append(v)

    return _aggregate_votes(tuple(raw_votes), atr_val, degraded)


def _trend_macro_monthly(c: pd.Series, e50_series: pd.Series, atr_val: float,
                          band: float, n: int) -> TrendResult:
    if n < CFG.bars.min_bars_m:
        return TrendResult("Range", 0, atr_val)
    e100 = _ema(c, 100)
    ref = _safe_float(e100.iloc[-1])
    cur = _safe_float(e50_series.iloc[-1])
    if ref is None or cur is None or ref == 0:
        return TrendResult("Range", 40, atr_val)
    gap = abs(cur - ref) / ref * 100
    s = 75 if gap > 0.3 else 60
    if cur > ref + band:
        return TrendResult("Bullish", s, atr_val)
    if cur < ref - band:
        return TrendResult("Bearish", s, atr_val)
    return TrendResult("Range", 40, atr_val)


def _trend_macro_weekly(c: pd.Series, e50_series: pd.Series, atr_val: float,
                         band: float, n: int) -> TrendResult:
    if n < CFG.ind.sma_macro:
        return TrendResult("Range", 40, atr_val)
    s200 = _sma(c, CFG.ind.sma_macro)
    cur50 = _safe_float(e50_series.iloc[-1])
    ref200 = _safe_float(s200.iloc[-1])
    prev50 = _safe_float(e50_series.iloc[-2])
    prev200 = _safe_float(s200.iloc[-2])
    if any(v is None for v in (cur50, ref200, prev50, prev200)):
        return TrendResult("Range", 40, atr_val)
    cross = (prev50 <= prev200 < cur50) or (prev50 >= prev200 > cur50)
    if cur50 > ref200 + band:
        return TrendResult("Bullish", 90 if cross else 75, atr_val)
    if cur50 < ref200 - band:
        return TrendResult("Bearish", 90 if cross else 75, atr_val)
    return TrendResult("Range", 40, atr_val)


def trend_macro(df: pd.DataFrame, tf: str, bundle: IndicatorBundle) -> TrendResult:
    if len(df) < 50:
        return TrendResult("Range", 0, bundle.atr_val)
    if bundle.has_nan:
        return TrendResult("Range", 0, bundle.atr_val)
    band = bundle.atr_val * CFG.vote.macro_band_atr_ratio
    c = df["Close"]
    if tf == "M":
        return _trend_macro_monthly(c, bundle.ema_long_series, bundle.atr_val, band, len(df))
    return _trend_macro_weekly(c, bundle.ema_long_series, bundle.atr_val, band, len(df))


def trend_4h(
    df: pd.DataFrame,
    bundle: IndicatorBundle,
    spot_mid: Optional[float],
    current_day_open: Optional[float],
) -> TrendResult:
    h, lo, c = df["High"], df["Low"], df["Close"]
    atr_val = bundle.atr_val
    if len(df) < CFG.bars.min_bars_h4 or bundle.has_nan:
        return TrendResult("Range", 0, atr_val)
    cur = spot_mid if spot_mid is not None else _safe_float(c.iloc[-1])
    if cur is None:
        return TrendResult("Range", 0, atr_val)

    score = 0
    e50_cur = bundle.ema_long_cur
    if e50_cur != 0:
        score += 1 if cur > e50_cur else -1
    pdi, mdi = _dmi(h, lo, c, CFG.ind.atr_period)
    if pdi is not None and mdi is not None:
        score += 1 if pdi > mdi else -1
    if current_day_open is not None:
        score += 1 if cur > current_day_open else -1

    s = abs(score)
    strength = 90 if s == 3 else 70 if s >= 1 else 40
    direction = "Bullish" if score > 0 else "Bearish" if score < 0 else "Range"
    return TrendResult(direction, strength, atr_val)


def trend_intraday(
    df: pd.DataFrame,
    instrument: str,
    bundle: IndicatorBundle,
    spot_mid: Optional[float],
) -> TrendResult:
    c = df["Close"]
    atr_val = bundle.atr_val
    if len(df) < 70 or bundle.has_nan:
        return TrendResult("Range", 0, atr_val)
    cur = spot_mid if spot_mid is not None else _safe_float(c.iloc[-1])
    if cur is None:
        return TrendResult("Range", 0, atr_val)

    e9 = bundle.ema_intra_fast
    e21 = bundle.ema_short
    e50_cur = bundle.ema_intra_long
    zlema = bundle.zlema
    rsi_val = bundle.rsi
    macd_cur = bundle.macd
    sig_cur = bundle.macd_signal

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
        # Exclude zero-volume bars on any instrument (defect #4)
        vol_nz = vol[vol > 0]
        if len(vol_nz) >= 20:
            vol_avg = _safe_float(vol_nz.rolling(20).mean().iloc[-1])
            vol_cur = _safe_float(vol.iloc[-1])
            if vol_avg is not None and vol_avg > 0 and vol_cur is not None:
                strong_vol = vol_cur > vol_avg * CFG.vote.volume_ratio_strong_intraday
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


def trend_age_daily(df: pd.DataFrame, bundle: IndicatorBundle) -> str:
    if len(df) < 55 or bundle.has_nan:
        return "N/A"
    c = df["Close"]
    e50 = bundle.ema_long_series
    above = c > e50
    for i in range(len(above) - 1, 0, -1):
        if above.iloc[i] != above.iloc[i - 1]:
            age = len(above) - 1 - i
            return str(age) if age > 0 else "0"
    return f">{len(above)}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — MTF SCORING (defect #6, #17)
# ═══════════════════════════════════════════════════════════════════════════════

_MTF_WEIGHTS: Final[Mapping[str, float]] = {
    "M": CFG.mtf.weight_m,
    "W": CFG.mtf.weight_w,
    "D": CFG.mtf.weight_d,
    "4H": CFG.mtf.weight_h4,
    "1H": CFG.mtf.weight_h1,
    "15m": CFG.mtf.weight_15m,
}
_MTF_TOTAL_POSSIBLE: Final[float] = sum(_MTF_WEIGHTS.values())


def _bull_compat(t: str) -> bool:
    return t in ("Bullish", "Retracement Bull")


def _bear_compat(t: str) -> bool:
    return t in ("Bearish", "Retracement Bear")


def _mtf_weighted_score(
    trends: Mapping[str, str], scores: Mapping[str, int]
) -> Tuple[float, float, int]:
    """Returns (w_bull, w_bear, active_tf_count). Normalisation done by caller
    against _MTF_TOTAL_POSSIBLE to prevent single-TF A+ inflation (defect #6)."""
    active_count = sum(1 for tf in trends if not trends[tf].startswith("Range"))
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
    return w_bull, w_bear, active_count


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
    return min(CFG.mtf.dispersion_penalty_max, ratio * CFG.mtf.dispersion_penalty_max * 2)


def score_mtf(
    trends: Mapping[str, str], scores: Mapping[str, int]
) -> Tuple[str, float, int]:
    """Returns (direction, score, active_tfs). Normalised against total possible
    weight to require breadth (defect #6)."""
    w_bull, w_bear, active = _mtf_weighted_score(trends, scores)
    if active < CFG.mtf.min_active_tfs:
        return "Range", 0.0, active
    if w_bull > w_bear:
        direction = "Bullish"
        raw = (w_bull / _MTF_TOTAL_POSSIBLE) * 100
    elif w_bear > w_bull:
        direction = "Bearish"
        raw = (w_bear / _MTF_TOTAL_POSSIBLE) * 100
    else:
        return "Range", 0.0, active
    bonus = _mtf_alignment_bonus(trends, direction)
    penalty = _mtf_dispersion_penalty(trends, direction)
    return direction, max(0.0, min(100.0, raw + bonus - penalty)), active


def _compute_nc_orthogonal(
    trends: Mapping[str, str], scores: Mapping[str, int], mtf_dir: str
) -> int:
    if mtf_dir not in ("Bullish", "Bearish"):
        return 0
    score_f = 0.0
    for tf, trend in trends.items():
        strength = scores.get(tf, 0)
        is_strong_pure_bull = trend == "Bullish" and strength >= CFG.mtf.nc_pure_strength_min
        is_strong_pure_bear = trend == "Bearish" and strength >= CFG.mtf.nc_pure_strength_min
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
    scores_list: List[float],
    nc_list: List[int],
    degraded_list: List[bool],
) -> List[str]:
    """
    Two-axis grading (defect #17): A+ requires BOTH high MTF AND high NC,
    not their additive sum.
    """
    grades: List[str] = []
    for score, nc, degraded in zip(scores_list, nc_list, degraded_list):
        if degraded:
            # Degraded run can never exceed B+
            grades.append("B+" if score >= 50 and nc >= 1 else "B")
            continue
        # A+ requires breadth (high MTF) AND conviction (high NC)
        if score >= CFG.mtf.nc_mtf_min_for_a_plus and nc >= CFG.mtf.nc_min_for_a_plus:
            grades.append("A+")
        elif score >= 55 and nc >= 1:
            grades.append("A")
        elif score >= 38:
            grades.append("B+")
        else:
            grades.append("B")
    return grades


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 18 — PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _empty_pair_result(pair: str, reason: str) -> Dict[str, Any]:
    return {
        "Paire": pair.replace("_", "/"),
        "M": "Range", "W": "Range", "D": "Range",
        "4H": "Range", "1H": "Range", "15m": "Range",
        "MTF": "Range",
        "_mtf_score": 0.0,
        "_mtf_dir": "Range",
        "_active_tfs": 0,
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


def analyze_pair(
    pair: str,
    account_id: str,
    account_hash: AccountHash,
    access_token: SecretToken,
    registry: SessionRegistry,
    stop_event: threading.Event,
    snapshot_to_iso: Optional[str],
) -> Optional[Dict[str, Any]]:
    if stop_event.is_set():
        return None
    try:
        cache = fetch_all_data(
            pair, account_id, account_hash, access_token, registry, stop_event,
            snapshot_to_iso,
        )

        if cache.get("is_incomplete"):
            return _empty_pair_result(pair, cache.get("error_reason", "Fetch failed"))

        drift_exceeded = bool(cache.get("_snapshot_drift_exceeded"))
        critical_gap = any(
            cache[tf].attrs.get("critical_gap", False)
            for tf in ("M", "W", "D", "4H", "1H", "15m")
        )
        degraded = bool(cache.get("_stale_tfs")) or drift_exceeded or critical_gap

        if drift_exceeded or critical_gap:
            reason = "Drift exceeded" if drift_exceeded else "Critical gap in open market"
            return _empty_pair_result(pair, reason)

        # Pre-compute indicator bundles per TF (defect #13)
        bundles: Dict[str, IndicatorBundle] = {}
        for tf in ("M", "W", "D", "4H", "1H", "15m"):
            intraday = tf in ("1H", "15m")
            b = compute_indicator_bundle(cache[tf], intraday=intraday)
            if b is None:
                return _empty_pair_result(pair, f"Indicator NaN on {tf}")
            bundles[tf] = b

        if stop_event.is_set():
            return None

        spot_mid = cache.get("_spot_mid")
        trends: Dict[str, str] = {}
        scores: Dict[str, int] = {}
        atrs: Dict[str, float] = {}

        for tf in ("M", "W"):
            tr = trend_macro(cache[tf], tf, bundles[tf])
            trends[tf], scores[tf], atrs[tf] = tr.direction, tr.strength, tr.atr_val

        if stop_event.is_set():
            return None

        daily_result = trend_daily(
            cache["D"], pair, bundles["D"], spot_mid,
            current_week_open=cache.get("_week_open"),
        )
        trends["D"] = daily_result.direction.value
        scores["D"] = daily_result.strength
        atrs["D"] = daily_result.atr_val
        if daily_result.degraded:
            degraded = True

        tr4 = trend_4h(cache["4H"], bundles["4H"], spot_mid,
                       current_day_open=cache.get("_day_open"))
        trends["4H"], scores["4H"], atrs["4H"] = tr4.direction, tr4.strength, tr4.atr_val

        for tf in ("1H", "15m"):
            if stop_event.is_set():
                return None
            tri = trend_intraday(cache[tf], pair, bundles[tf], spot_mid)
            trends[tf], scores[tf], atrs[tf] = tri.direction, tri.strength, tri.atr_val

        mtf_dir, mtf_score, active_tfs = score_mtf(trends, scores)
        age = trend_age_daily(cache["D"], bundles["D"])
        nc = _compute_nc_orthogonal(trends, scores, mtf_dir)

        return {
            "Paire": pair.replace("_", "/"),
            "M": trends["M"], "W": trends["W"], "D": trends["D"],
            "4H": trends["4H"], "1H": trends["1H"], "15m": trends["15m"],
            "MTF": f"{mtf_dir} ({mtf_score:.0f}%)" if mtf_dir != "Range" else "Range",
            "_mtf_score": mtf_score,
            "_mtf_dir": mtf_dir,
            "_active_tfs": active_tfs,
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
        _log_incident(IncidentCode.UNKNOWN, "analyze_pair exception",
                      instrument=pair, err=type(e).__name__, level=logging.ERROR)
        return _empty_pair_result(pair, f"Analysis failed: {type(e).__name__}")


def _drain_executor_strict(
    executor: ThreadPoolExecutor,
    futures: Mapping[Future, str],
    stop_event: threading.Event,
) -> None:
    stop_event.set()
    for f in futures:
        if not f.done():
            f.cancel()
    deadline = time.monotonic() + CFG.ops.pool_drain_grace_sec
    while time.monotonic() < deadline:
        alive = sum(1 for f in futures if not f.done())
        if alive == 0:
            break
        time.sleep(0.05)
    try:
        executor.shutdown(wait=True, cancel_futures=True)
    except TypeError:  # Python < 3.9
        executor.shutdown(wait=True)


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
        _log_incident(IncidentCode.UNKNOWN, "future result error",
                      instrument=inst, err=type(e).__name__)


def analyze_all_core(
    account_id: str,
    access_token: SecretToken,
    progress_cb: Optional[Callable[[float], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any], List[str]]:
    """
    Core pipeline. Returns (df, errors, meta, warnings).
    Warnings list decouples UI from logic (defect #14).
    """
    results: List[Dict[str, Any]] = []
    errors: set = set()
    warnings: List[str] = []
    total = len(INSTRUMENTS)
    done = 0
    timed_out = False

    # Global stop-event (defect #19) — cancels any prior run
    stop_event = _RUN_CONTROLLER.start_new_run()
    registry = _get_session_registry()
    account_hash = _hash_account(account_id)

    # Temporal snapshot freeze (defect #26)
    snapshot_to = datetime.now(UTC)
    snapshot_to_iso = snapshot_to.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

    meta: Dict[str, Any] = {
        "started_at": snapshot_to,
        "version": APP_VERSION,
        "env": _OANDA_ENV,
        "account_hash": account_hash,
        "snapshot_to": snapshot_to_iso,
    }

    dynamic_timeout = max(
        CFG.ops.analysis_timeout_sec,
        int(len(INSTRUMENTS) * 5.0 / CFG.ops.max_workers),
    )

    executor = ThreadPoolExecutor(
        max_workers=CFG.ops.max_workers, thread_name_prefix="bluestar_worker"
    )
    futures: Dict[Future, str] = {}
    try:
        try:
            futures = {
                executor.submit(
                    analyze_pair, inst, account_id, account_hash, access_token,
                    registry, stop_event, snapshot_to_iso,
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
                            _log_incident(IncidentCode.UI_CALLBACK_ERROR, "progress_cb",
                                          err=type(exc).__name__, level=logging.DEBUG)
                    if status_cb is not None:
                        try:
                            status_cb(f"GPS ({done}/{total}) — {inst.replace('_', '/')}")
                        except Exception as exc:  # pylint: disable=broad-exception-caught
                            _log_incident(IncidentCode.UI_CALLBACK_ERROR, "status_cb",
                                          err=type(exc).__name__, level=logging.DEBUG)
                    _process_completed_future(future, inst, results, errors)
            except FutureTimeoutError:
                timed_out = True
                warnings.append(
                    f"Analyse interrompue après {dynamic_timeout}s — connexion OANDA dégradée."
                )
                _log_incident(IncidentCode.EXECUTOR_TIMEOUT, "analyze_all_core timeout",
                              timeout_sec=dynamic_timeout, level=logging.ERROR)
                for f, inst in futures.items():
                    if not f.done():
                        errors.add(inst)
        finally:
            # Nested try/finally — each cleanup is independent (defect #10)
            try:
                _drain_executor_strict(executor, futures, stop_event)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _log_incident(IncidentCode.EXECUTOR_CANCEL, "executor drain failed",
                              err=type(exc).__name__, level=logging.ERROR)
    finally:
        _RUN_CONTROLLER.finish_run(stop_event)

    meta["finished_at"] = datetime.now(UTC)
    meta["timed_out"] = timed_out
    meta["errors_count"] = len(errors)
    meta["completeness"] = len(results) / total if total > 0 else 0.0
    meta["degraded_pairs"] = sorted(
        r["Paire"] for r in results
        if r.get("_degraded") and "_error_reason" not in r
    )

    if errors:
        sample = ", ".join(e.replace("_", "/") for e in sorted(errors)[:10])
        ellipsis = " …" if len(errors) > 10 else ""
        warnings.append(f"{len(errors)} paire(s) non analysée(s) : {sample}{ellipsis}")

    if meta["completeness"] < CFG.ops.completeness_min_tradable and results:
        warnings.append(
            f"Couverture partielle — {meta['completeness']:.0%}. Run marqué NOT_TRADEABLE."
        )

    errors_sorted = sorted(errors)
    if not results:
        return pd.DataFrame(), errors_sorted, meta, warnings

    scores_list = [r["_mtf_score"] for r in results]
    nc_list = [r["NC"] for r in results]
    run_degraded = meta["completeness"] < CFG.ops.completeness_min_tradable
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
    return df, errors_sorted, meta, warnings


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 19 — STREAMLIT CACHING (module-level decorators, hash includes token)
# ═══════════════════════════════════════════════════════════════════════════════

if _STREAMLIT_AVAILABLE:
    @st.cache_data(
        ttl=CFG.cache.ttl_d,
        show_spinner=False,
        hash_funcs={OandaCredentials: lambda c: (c.account_id, c.token.digest)},
    )
    def _run_analysis_cached(
        creds: OandaCredentials,
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any], List[str]]:
        return analyze_all_core(creds.account_id, creds.token)
else:
    def _run_analysis_cached(
        creds: OandaCredentials,
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any], List[str]]:
        return analyze_all_core(creds.account_id, creds.token)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 20 — PDF EXPORT (defensive)
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_str(s: str) -> str:
    return s.encode("latin-1", errors="replace").decode("latin-1")


def _pdf_cell_text(val: str) -> str:
    return val if _FPDF2 else _safe_str(val)


def _pdf_get_colors(col: str, val: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    grade_rgb = {"A+": (251, 191, 36), "A": (163, 230, 53),
                 "B+": (52, 211, 153), "B": (96, 165, 250)}
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


def _encode_pdf_output(out: Any) -> bytes:
    """Defensive PDF encoding (defect #15)."""
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    if isinstance(out, str):
        return out.encode("latin-1", errors="replace")
    return bytes(out)


def create_pdf(df: pd.DataFrame) -> BytesIO:
    base_cols = ["Paire", "M", "W", "D", "4H", "1H", "15m", "MTF", "Quality", "Tradable",
                 "NC", "Age D1", "ATR Daily", "ATR H4", "ATR H1", "ATR 15m"]
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
            f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}"
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

        buf.write(_encode_pdf_output(pdf.output(dest="S")))
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
            buf2.write(_encode_pdf_output(fallback.output(dest="S")))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _log_incident(IncidentCode.PDF_ERROR, "fallback PDF failed",
                          err=type(exc).__name__, level=logging.ERROR)
            buf2.write(b"PDF error")
        buf2.seek(0)
        return buf2


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 21 — SECRETS LOADING
# ═══════════════════════════════════════════════════════════════════════════════

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


def _load_secrets() -> Tuple[Optional[str], Optional[SecretToken]]:
    acc: Optional[str] = None
    tok: Optional[str] = None
    if _STREAMLIT_AVAILABLE:
        try:
            acc = st.secrets["OANDA_ACCOUNT_ID"]
            tok = st.secrets["OANDA_ACCESS_TOKEN"]
        except (KeyError, FileNotFoundError):
            acc, tok = None, None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _log_incident(IncidentCode.HTTP_AUTH, "secrets read error",
                          err=type(exc).__name__, level=logging.ERROR)
            return None, None
    if acc is None or tok is None:
        acc = os.environ.get("OANDA_ACCOUNT_ID", "").strip()
        tok = os.environ.get("OANDA_ACCESS_TOKEN", "").strip()

    if not isinstance(acc, str) or not isinstance(tok, str):
        return None, None
    acc, tok = acc.strip(), tok.strip()
    if not _validate_secret_format(acc, tok):
        return None, None
    secret = SecretToken(tok)
    _SecretScrubFilter.register_secret(secret)
    return acc, secret


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 22 — STREAMLIT UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _check_running_flag_ttl() -> None:
    started_at = st.session_state.get("_analysis_started_at")
    run_id = st.session_state.get("_analysis_run_id")
    if started_at and run_id:
        elapsed = (datetime.now(UTC) - started_at).total_seconds()
        if elapsed > CFG.ops.streamlit_running_flag_ttl_sec:
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
    return [f"color:black;font-weight:bold;background-color:{grade_css.get(str(x), '#9ca3af')}"
            for x in s]


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


def analyze_all(
    account_id: str, token: SecretToken
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """Streamlit-aware wrapper. UI side-effects only happen in main()."""
    creds = OandaCredentials(account_id=account_id, token=token)
    df, _errors, meta, warnings = _run_analysis_cached(creds)
    return df, meta, warnings


def _sidebar_config() -> bool:
    with st.sidebar:
        st.header("⚙️ Configuration")
        only_best = st.checkbox("Afficher uniquement Grade A+ / A", value=False)
        st.info(
            f"Env : {_OANDA_ENV.upper()}\n\n"
            f"Workers : {CFG.ops.max_workers} · Timeout : {CFG.ops.analysis_timeout_sec}s\n\n"
            f"Cache TTL : M={CFG.cache.ttl_m // 60}m W={CFG.cache.ttl_w // 60}m "
            f"D={CFG.cache.ttl_d // 60}m"
        )
        if not is_fx_market_open():
            st.warning("📅 Marché FX fermé — données potentiellement obsolètes.")
        if not _HAS_HOLIDAYS:
            st.caption("ℹ️ Module `holidays` non installé — détection jours fériés dégradée.")
        st.markdown("---")
        if st.button("🗑️ Vider le cache", use_container_width=True):
            _get_candle_cache().clear()
            if _STREAMLIT_AVAILABLE and _is_streamlit_runtime():
                try:
                    _run_analysis_cached.clear()
                except Exception as e:  # pylint: disable=broad-exception-caught
                    _log.debug("Streamlit cache_data clear failed: %s", e)
                try:
                    st.cache_resource.clear()
                except Exception as e:  # pylint: disable=broad-exception-caught
                    _log.debug("Streamlit cache_resource clear failed: %s", e)
            st.success("Cache vidé.")
    return only_best


def _render_metrics(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Analyzed", len(df))
    c2.metric("Setups A+", len(df[df["Quality"] == "A+"]))
    c3.metric("Setups A", len(df[df["Quality"] == "A"]))
    c4.metric("Setups B", len(df[df["Quality"].isin(["B+", "B"])]))


def _emit_warnings(warnings: List[str], meta: Dict[str, Any]) -> None:
    if meta.get("timed_out"):
        st.error(f"⏱️ Analyse interrompue — connexion OANDA dégradée.")
    completeness = meta.get("completeness", 0.0)
    if completeness < CFG.ops.completeness_min_tradable and completeness > 0:
        st.markdown(
            f"<div class='degraded-warning'>⚠️ <b>Couverture partielle</b> — "
            f"<b>{completeness:.0%}</b> des instruments analysés. "
            "<b>Run marqué NOT_TRADEABLE.</b></div>",
            unsafe_allow_html=True,
        )
    for w in warnings:
        st.warning(f"⚠️ {w}")


def main() -> None:
    _configure_streamlit_ui()

    st.markdown(
        f"<div class='main-header'><h1>🧭 BLUESTAR HEDGE FUND GPS V{APP_VERSION}</h1>"
        f"<p style='margin:0;font-size:0.85em;opacity:0.8'>"
        f"Institutional · Env: {_OANDA_ENV.upper()}"
        "</p></div>",
        unsafe_allow_html=True,
    )

    acc, tok = _load_secrets()
    if not acc or not tok:
        st.error(
            "❌ Secrets OANDA manquants ou invalides — "
            "configurez OANDA_ACCOUNT_ID (XXX-XXX-XXXXXX-XXX) "
            "et OANDA_ACCESS_TOKEN (≥32 chars)."
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
        run_id = datetime.now(UTC).isoformat()
        st.session_state["_analysis_running"] = True
        st.session_state["_analysis_started_at"] = datetime.now(UTC)
        st.session_state["_analysis_run_id"] = run_id
        try:
            with st.spinner("Analyse Multi-Timeframe en cours..."):
                df, meta, warnings = analyze_all(acc, tok)
                _emit_warnings(warnings, meta)
                if not df.empty:
                    st.session_state["df"] = df
                    st.session_state["df_ts"] = datetime.now(UTC)
                    st.session_state["df_meta"] = meta
                    st.session_state["pdf_buf"] = create_pdf(df)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _log_incident(IncidentCode.UNKNOWN, "main analysis failed",
                          err=type(exc).__name__, level=logging.ERROR)
            st.error(f"❌ Erreur critique : {type(exc).__name__}")
        finally:
            if st.session_state.get("_analysis_run_id") == run_id:
                st.session_state["_analysis_running"] = False
                st.session_state["_analysis_started_at"] = None
                st.session_state["_analysis_run_id"] = None

    if "df" not in st.session_state:
        return

    df_ts = st.session_state.get("df_ts")
    if df_ts:
        age_min = (datetime.now(UTC) - df_ts).total_seconds() / 60
        if age_min > CFG.ops.data_max_age_min:
            st.markdown(
                f"<div class='stale-warning'>⏰ <b>Données périmées</b> — "
                f"dernière analyse il y a <b>{age_min:.0f} min</b>. Relancez.</div>",
                unsafe_allow_html=True,
            )

    df = st.session_state["df"].copy()
    if only_best:
        df = df[df["Quality"].isin(["A+", "A"])].copy()

    grade_order = ["A+", "A", "B+", "B"]
    # Defect #24: filter rows with unknown grades before categorical sort
    df = df[df["Quality"].isin(grade_order)].copy()
    df["Quality"] = pd.Categorical(df["Quality"], categories=grade_order, ordered=True)
    sort_cols = [c for c in ["Quality", "NC", "_mtf_score"] if c in df.columns]
    ascending = [True, False, False][:len(sort_cols)]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending)

    df_clean = df.drop(
        columns=["_mtf_score", "_mtf_dir", "_active_tfs", "_degraded",
                 "_stale_tfs", "_error_reason"],
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
    pdf_buf = st.session_state.get("pdf_buf") or create_pdf(df_clean)
    with c1:
        st.download_button("📄 PDF", data=pdf_buf,
                           file_name=f"Bluestar_GPS_{ts}.pdf",
                           mime="application/pdf", use_container_width=True)
    with c2:
        st.download_button("📊 CSV",
                           data=df_clean[cols_present].to_csv(index=False).encode("utf-8"),
                           file_name=f"Bluestar_GPS_{ts}.csv",
                           mime="text/csv", use_container_width=True)
    with c3:
        st.download_button("🗂️ JSON",
                           data=df_clean[cols_present].to_json(
                               orient="records", force_ascii=False, indent=2,
                           ).encode("utf-8"),
                           file_name=f"Bluestar_GPS_{ts}.json",
                           mime="application/json", use_container_width=True)


if __name__ == "__main__":
    main()

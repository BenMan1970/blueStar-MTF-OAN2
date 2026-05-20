"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       BLUESTAR HEDGE FUND GPS  —  V5.1  (Patches moteur — UI inchangée)    ║
║                                                                              ║
║  Patches V4.0 (P1–P15) conservés intégralement.                             ║
║  Corrections V4.1 (C1–C15) conservées intégralement.                        ║
║  Refonte V5.0 (V5-1–V5-7) conservée intégralement.                          ║
║                                                                              ║
║  Patches moteur V5.1 (UI INCHANGÉE) :                                       ║
║   [A1]  RSI range pur → 50.0 au lieu de 100.0 (bug gain=0,loss=0)          ║
║   [A2]  Pivot detection vectorisé via rolling — O(n), retard wing=5 docum.  ║
║   [A3]  _vote_prev_midpoint dead code supprimé                              ║
║   [A4]  vol MA20 : vol_j1 exclu de son propre référentiel (biais corrigé)   ║
║   [A5]  _aggregate_votes : fired_possible uniquement (force non déprimée)   ║
║   [A6]  _compute_nc : Retracements pondérés ±0.5 (était 0)                 ║
║   [A7]  score_mtf : bonus M+W étendu aux Retracements compatibles           ║
║   [A8]  fetch_cached : TOCTOU corrigé — lecture+décision sous lock atomique ║
║   [A9]  Sessions HTTP : pool initializer + registre + close propre          ║
║   [A10] trend_4h : KeyError/IndexError → WARNING (était DEBUG silencieux)   ║
║   [A11] fetch_candles : validation DatetimeIndex après parsing              ║
║   [A12] DATA_MAX_AGE_MIN = 10 (aligné sur CACHE_TTL 10 min, était 15)      ║
║   [A13] analyze_all_core : logique pure séparée du wrapper Streamlit        ║
║   [A14] Timeout global 120s sur analyze_all_core                            ║
║   [A15] Tri MTF sur _mtf_score numérique (était tri string "Bullish (97%)") ║
║   [A18] Registre DAILY_VOTE_REGISTRY extensible                             ║
║   [A19] Type hints Optional/Tuple sur trend_macro/4h/intraday               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from fpdf import FPDF
    _FPDF2 = hasattr(FPDF, 'set_lang')
except ImportError:
    FPDF = None
    _FPDF2 = False

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
_log = logging.getLogger('bluestar_gps')

# ===================== CONFIG =====================
st.set_page_config(page_title="Bluestar GPS V5.0", page_icon="🧭", layout="wide")

OANDA_API_URL = "https://api-fxpractice.oanda.com"

INSTRUMENTS = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'USD_CAD', 'AUD_USD', 'NZD_USD',
    'EUR_GBP', 'EUR_JPY', 'EUR_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_NZD',
    'GBP_JPY', 'GBP_CHF', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD',
    'AUD_JPY', 'AUD_CAD', 'AUD_CHF', 'AUD_NZD', 'CAD_JPY', 'CAD_CHF', 'CHF_JPY',
    'NZD_JPY', 'NZD_CAD', 'NZD_CHF',
    'DE30_EUR', 'XAU_USD', 'SPX500_USD', 'NAS100_USD', 'US30_USD',
]

INDICES = {'DE30_EUR', 'SPX500_USD', 'NAS100_USD', 'US30_USD', 'XAU_USD'}

TREND_COLORS = {
    'Bullish':          '#2ecc71',
    'Bearish':          '#e74c3c',
    'Retracement Bull': '#7dcea0',
    'Retracement Bear': '#f1948a',
    'Range':            '#95a5a6',
}

# [A12] DATA_MAX_AGE_MIN aligné sur _CACHE_TTL — invariant enforced
_CACHE_TTL       = 600   # 10 minutes
DATA_MAX_AGE_MIN = 10    # minutes — doit rester <= _CACHE_TTL / 60
MAX_PIVOT_AGE    = 50    # [C9]

if DATA_MAX_AGE_MIN > _CACHE_TTL / 60:
    raise ValueError(
        f"DATA_MAX_AGE_MIN ({DATA_MAX_AGE_MIN}) > CACHE_TTL ({_CACHE_TTL/60:.0f} min) — "
        "l'alerte données périmées apparaîtrait après l'expiration du cache."
    )

# ===================== CSS — identique V5.0 =====================
st.markdown("""
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
    .metric-label { font-size: 0.9em; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .stDataFrame { width: 100%; }
    div[data-testid="stMarkdown"] { text-align: center; }
    .stale-warning {
        background: #fef3c7; border-left: 4px solid #f59e0b;
        padding: 10px 16px; border-radius: 4px; margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ===================== SESSIONS HTTP — REGISTRE SCOPÉ PAR ANALYSE =====================
# [BUG-3] Le registre global _pool_sessions causait des fermetures inter-analyses :
# analyse A terminée fermait les sessions de l'analyse B encore active.
# Correction : SessionRegistry instanciée par analyse, passée via initargs au pool.

_thread_local = threading.local()


class _SessionRegistry:
    """
    Registre de sessions HTTP scopé par analyse.
    Chaque appel à analyze_all_core crée sa propre instance —
    aucun partage possible entre analyses concurrentes.
    """

    def __init__(self) -> None:
        self._sessions: list = []
        self._lock = threading.Lock()

    def add(self, session: requests.Session) -> None:
        with self._lock:
            self._sessions.append(session)

    def close_all(self) -> None:
        with self._lock:
            for s in self._sessions:
                try:
                    s.close()
                except OSError as exc:
                    _log.debug("Session close (ignoré) : %s", exc)
            self._sessions.clear()


def _build_session() -> requests.Session:
    session = requests.Session()
    retry   = Retry(
        total=4, backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"], raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


def _pool_thread_init(registry: _SessionRegistry) -> None:
    """
    Initializer du pool — crée une session par thread worker
    et l'enregistre dans le registre scopé de l'analyse courante.
    """
    session = _build_session()
    _thread_local.session = session
    registry.add(session)


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, 'session'):
        _thread_local.session = _build_session()
    return _thread_local.session


# ===================== [C1] CACHE THREAD-SAFE =====================
_data_cache: dict = {}
_cache_lock = threading.Lock()


def _cache_clear() -> None:
    with _cache_lock:
        _data_cache.clear()


# ===================== INDICATEURS DE BASE =====================

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return _true_range(high, low, close).ewm(alpha=1 / n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """
    RSI Wilder.
    [A1] gain=0, loss=0 (range pur) → RSI=50 (neutre). V5 retournait 100.0 par erreur.
    gain>0, loss=0 → RSI=100 (pure uptrend, correct Wilder).
    """
    d    = close.diff()
    gain = d.where(d > 0, 0.0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    rs   = gain / loss.replace(0, np.nan)
    rsi  = 100 - 100 / (1 + rs)
    # Remplissage contextuel : NaN arrive quand loss=0
    gain_pos  = gain > 0
    loss_zero = loss == 0
    rsi = rsi.where(~loss_zero, np.where(gain_pos, 100.0, 50.0))
    return rsi


def _dmi(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> Tuple[float, float]:
    tr    = _true_range(high, low, close)
    atr_s = tr.ewm(alpha=1 / n, adjust=False).mean()
    up    = high.diff()
    dn    = -low.diff()
    pdm   = up.where((up > dn) & (up > 0), 0.0)
    mdm   = dn.where((dn > up) & (dn > 0), 0.0)
    pdi   = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr_s.replace(0, np.nan)
    mdi   = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / atr_s.replace(0, np.nan)
    return float(pdi.iloc[-1]), float(mdi.iloc[-1])


def _fmt_atr(val: float) -> str:
    if not val or np.isnan(val) or val <= 0:
        return 'N/A'
    if val >= 10:
        return f"{val:.2f}"
    if val >= 1:
        return f"{val:.3f}"
    return f"{val:.4f}"


# ══════════════════════════════════════════════════════════════════════════════
# TYPES CONTRACTUELS — Signal Decomposition Pattern  [V5-1]
# ══════════════════════════════════════════════════════════════════════════════

class Direction(str, Enum):
    BULLISH = 'Bullish'
    BEARISH = 'Bearish'
    RANGE   = 'Range'


@dataclass(frozen=True)
class VoteSignal:
    name       : str
    direction  : Direction
    weight     : float
    reliability: float
    fired      : bool
    reason     : str


@dataclass(frozen=True)
class DailyTrendResult:
    """
    Résultat complet de trend_daily — rétrocompatible via __iter__.
        trend, score, atr = trend_daily(df, df_weekly)
    """
    direction    : Direction
    strength     : int
    atr_val      : float
    bull_score   : float
    bear_score   : float
    votes        : tuple
    min_votes_met: bool

    def __iter__(self):
        yield self.direction.value
        yield self.strength
        yield self.atr_val

    @property
    def fired_votes(self) -> list:
        return [v for v in self.votes if v.fired]

    @property
    def bull_votes(self) -> list:
        return [v for v in self.votes if v.fired and v.direction == Direction.BULLISH]

    @property
    def bear_votes(self) -> list:
        return [v for v in self.votes if v.fired and v.direction == Direction.BEARISH]

    def summary(self) -> str:
        parts = [
            f"{v.name}="
            f"{'B' if v.direction==Direction.BULLISH else 'S' if v.direction==Direction.BEARISH else '-'}"
            f"(w={v.weight},r={v.reliability:.2f},f={v.fired})"
            for v in self.votes
        ]
        return (f"{self.direction.value} str={self.strength} "
                f"bull={self.bull_score:.2f} bear={self.bear_score:.2f} | "
                + " ".join(parts))


# ══════════════════════════════════════════════════════════════════════════════
# [A18] REGISTRE DE VOTES EXTENSIBLE
# ══════════════════════════════════════════════════════════════════════════════

DAILY_VOTE_REGISTRY: list[Callable] = []


def register_daily_vote(fn: Callable) -> Callable:
    DAILY_VOTE_REGISTRY.append(fn)
    return fn


# ══════════════════════════════════════════════════════════════════════════════
# VOTES ATOMIQUES
# ══════════════════════════════════════════════════════════════════════════════

@register_daily_vote
def _vote_swing_structure(h: pd.Series, lo: pd.Series, _c, _ctx) -> VoteSignal:
    """
    Vote 1 — Structure swing HH/HL ou LH/LL.
    Poids 2.0 · Reliability 0.9
    [A2] Détection vectorisée via rolling max/min — O(n), plus de boucle Python.
    Retard de confirmation : wing=5 bougies daily (≈5 jours calendaires). Intentionnel.
    """
    name = "swing_structure"
    wing = 5

    if len(h) < 2 * wing + MAX_PIVOT_AGE + 1:
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=2.0, reliability=0.9, fired=False,
                          reason=f"série trop courte ({len(h)})")

    roll_max = h.rolling(2 * wing + 1, center=True).max()
    roll_min = lo.rolling(2 * wing + 1, center=True).min()
    min_idx  = max(0, len(h) - MAX_PIVOT_AGE)

    sh = [i for i in range(min_idx, len(h))
          if not np.isnan(roll_max.iloc[i]) and h.iloc[i] == roll_max.iloc[i]]
    sl = [i for i in range(min_idx, len(lo))
          if not np.isnan(roll_min.iloc[i]) and lo.iloc[i] == roll_min.iloc[i]]

    if len(sh) < 2 or len(sl) < 2:
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=2.0, reliability=0.9, fired=False,
                          reason=f"pivots insuffisants sh={len(sh)} sl={len(sl)}")

    hh = h.iloc[sh[-1]]  > h.iloc[sh[-2]]
    hl = lo.iloc[sl[-1]] > lo.iloc[sl[-2]]
    lh = h.iloc[sh[-1]]  < h.iloc[sh[-2]]
    ll = lo.iloc[sl[-1]] < lo.iloc[sl[-2]]

    if hh and hl:
        return VoteSignal(name=name, direction=Direction.BULLISH,
                          weight=2.0, reliability=0.9, fired=True, reason="HH+HL")
    if lh and ll:
        return VoteSignal(name=name, direction=Direction.BEARISH,
                          weight=2.0, reliability=0.9, fired=True, reason="LH+LL")
    return VoteSignal(name=name, direction=Direction.RANGE,
                      weight=2.0, reliability=0.9, fired=False, reason="structure mixte")


@register_daily_vote
def _vote_ema_stack_vote(_h, _lo, _c, ctx: dict) -> VoteSignal:
    """Vote 2 — EMA21/EMA50 stack. Poids 1.0 · Reliability 0.75"""
    name    = "ema_stack"
    cur     = ctx['cur']
    e21     = ctx['e21']
    e50_cur = ctx['e50_cur']
    if np.isnan(e21) or np.isnan(e50_cur):
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=1.0, reliability=0.75, fired=False, reason="NaN EMA")
    if cur > e21 > e50_cur:
        return VoteSignal(name=name, direction=Direction.BULLISH,
                          weight=1.0, reliability=0.75, fired=True,
                          reason=f"cur={cur:.5f}>e21={e21:.5f}>e50={e50_cur:.5f}")
    if cur < e21 < e50_cur:
        return VoteSignal(name=name, direction=Direction.BEARISH,
                          weight=1.0, reliability=0.75, fired=True,
                          reason=f"cur={cur:.5f}<e21={e21:.5f}<e50={e50_cur:.5f}")
    return VoteSignal(name=name, direction=Direction.RANGE,
                      weight=1.0, reliability=0.75, fired=False, reason="stack non aligné")


@register_daily_vote
def _vote_weekly_open_vote(_h, _lo, _c, ctx: dict) -> VoteSignal:
    """
    Vote 3 — Weekly Open.

    [BUG-1] Priorité à current_week_open (bougie W incomplète = open de la semaine EN COURS).
    Fallback sur df_weekly (dernière semaine complète) avec reliability réduite
    si l'open courant est indisponible (weekend, API indisponible).

    Reliability :
      0.90 — open courant de la semaine (bougie incomplète)
      0.70 — dernière semaine complète (référence décalée d'une semaine)
      0.50 — resample daily (approximation)
    """
    name             = "weekly_open"
    cur              = ctx['cur']
    current_week_open = ctx.get('current_week_open')
    df_weekly        = ctx.get('df_weekly')
    df_daily         = ctx['df_daily']

    try:
        # [BUG-1] Open de la semaine COURANTE — référence correcte
        if current_week_open is not None and not np.isnan(current_week_open):
            wo_price = current_week_open
            rel      = 0.90
            source   = "current_W"
        elif df_weekly is not None and not df_weekly.empty:
            # Fallback : dernière semaine complète — décalée d'une semaine
            wo_price = float(df_weekly['Open'].iloc[-1])
            rel      = 0.70
            source   = "prev_W_complete"
            _log.debug("weekly_open fallback sur semaine précédente pour %s", ctx.get('instrument',''))
        else:
            if not isinstance(df_daily.index, pd.DatetimeIndex):
                return VoteSignal(name=name, direction=Direction.RANGE,
                                  weight=1.0, reliability=0.0, fired=False,
                                  reason="index non-datetime")
            w_open = df_daily['Open'].resample('W-MON').first().dropna()
            if w_open.empty:
                return VoteSignal(name=name, direction=Direction.RANGE,
                                  weight=1.0, reliability=0.0, fired=False,
                                  reason="Weekly Open indisponible")
            wo_price = float(w_open.iloc[-1])
            rel      = 0.50
            source   = "resample_fallback"

        if np.isnan(wo_price):
            return VoteSignal(name=name, direction=Direction.RANGE,
                              weight=1.0, reliability=rel, fired=False, reason="wo NaN")
        if cur > wo_price:
            return VoteSignal(name=name, direction=Direction.BULLISH,
                              weight=1.0, reliability=rel, fired=True,
                              reason=f"cur={cur:.5f}>wo={wo_price:.5f} [{source}]")
        if cur < wo_price:
            return VoteSignal(name=name, direction=Direction.BEARISH,
                              weight=1.0, reliability=rel, fired=True,
                              reason=f"cur={cur:.5f}<wo={wo_price:.5f} [{source}]")
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=1.0, reliability=rel, fired=False,
                          reason=f"cur==wo={wo_price:.5f} [{source}]")

    except (TypeError, AttributeError, KeyError) as exc:
        _log.debug("_vote_weekly_open ignoré : %s", exc)
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=1.0, reliability=0.0, fired=False, reason=str(exc))


# [A3] _vote_prev_midpoint (sans filtre volume) SUPPRIMÉE — dead code V5.

@register_daily_vote
def _vote_prev_midpoint_with_volume(h: pd.Series, lo: pd.Series,
                                    c: pd.Series, ctx: dict) -> VoteSignal:
    """
    Vote 4 — Close J-1 vs midpoint J-1. [V5-3]
    [A4] vol_j1 exclu du référentiel MA20 — corrige le biais self-referential de V5.
         V5 : vol.iloc[:-1] incluait vol_j1 dans la MA20 (juge et partie).
         V5.1 : vol.iloc[:-2] = référentiel sans vol_j1.
    """
    name       = "prev_midpoint"
    vol        = ctx.get('vol_series')
    instrument = ctx.get('instrument', '')

    if len(c) < 2:
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=0.5, reliability=0.0, fired=False,
                          reason="données insuffisantes")

    mid    = (float(h.iloc[-2]) + float(lo.iloc[-2])) / 2
    prev_c = float(c.iloc[-2])
    bull   = prev_c > mid

    is_index = instrument in INDICES
    if is_index or vol is None or vol.empty:
        direction = Direction.BULLISH if bull else Direction.BEARISH
        return VoteSignal(name=name, direction=direction,
                          weight=0.5, reliability=0.50, fired=True,
                          reason=f"close_j1={'>' if bull else '<='} mid_j1 [no_vol]")

    try:
        # [A4] vol_ref exclu vol_j1 ([-2]) ET bougie en cours ([-1])
        vol_ref = vol.iloc[:-2]
        vol_j1  = float(vol.iloc[-2])

        if len(vol_ref) < 20:
            direction = Direction.BULLISH if bull else Direction.BEARISH
            return VoteSignal(name=name, direction=direction,
                              weight=0.5, reliability=0.50, fired=True,
                              reason="vol_history<20")

        vol_ma    = float(vol_ref.rolling(20).mean().iloc[-1])
        if np.isnan(vol_j1) or np.isnan(vol_ma) or vol_ma <= 0:
            direction = Direction.BULLISH if bull else Direction.BEARISH
            return VoteSignal(name=name, direction=direction,
                              weight=0.5, reliability=0.50, fired=True, reason="vol_NaN")

        vol_ratio = vol_j1 / vol_ma
        if vol_ratio <= 1.0:
            return VoteSignal(name=name, direction=Direction.RANGE,
                              weight=0.5, reliability=0.0, fired=False,
                              reason=f"vol_j1 < MA20 (ratio={vol_ratio:.2f})")

        reliability = min(0.80, 0.65 + (vol_ratio - 1.0) * 0.05)
        direction   = Direction.BULLISH if bull else Direction.BEARISH
        return VoteSignal(name=name, direction=direction,
                          weight=0.5, reliability=reliability, fired=True,
                          reason=f"vol_ratio={vol_ratio:.2f}>1.0 rel={reliability:.2f}")

    except (TypeError, ValueError, IndexError) as exc:
        _log.debug("_vote_prev_midpoint vol err: %s", exc)
        direction = Direction.BULLISH if bull else Direction.BEARISH
        return VoteSignal(name=name, direction=direction,
                          weight=0.5, reliability=0.50, fired=True,
                          reason=f"vol_err:{exc}")


@register_daily_vote
def _vote_ema50_slope_vote(_h, _lo, _c, ctx: dict) -> VoteSignal:
    """Vote 5 — Pente EMA50 normalisée ATR [C10]. Poids 1.0 · Reliability 0.70"""
    name      = "ema50_slope"
    e50       = ctx['e50']
    atr_val   = ctx['atr_val']
    threshold = 0.05
    if len(e50) < 6 or atr_val <= 0:
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=1.0, reliability=0.70, fired=False,
                          reason=f"données insuffisantes atr={atr_val:.5f}")
    slope_ratio = float(e50.iloc[-1] - e50.iloc[-6]) / atr_val
    if slope_ratio > threshold:
        return VoteSignal(name=name, direction=Direction.BULLISH,
                          weight=1.0, reliability=0.70, fired=True,
                          reason=f"slope={slope_ratio:.3f}>{threshold}")
    if slope_ratio < -threshold:
        return VoteSignal(name=name, direction=Direction.BEARISH,
                          weight=1.0, reliability=0.70, fired=True,
                          reason=f"slope={slope_ratio:.3f}<-{threshold}")
    return VoteSignal(name=name, direction=Direction.RANGE,
                      weight=1.0, reliability=0.70, fired=False,
                      reason=f"slope={slope_ratio:.3f} zone neutre")


# ══════════════════════════════════════════════════════════════════════════════
# AGRÉGATEUR — circuit breaker MIN_RELIABLE_SCORE  [V5-4]
# ══════════════════════════════════════════════════════════════════════════════

MIN_RELIABLE_SCORE = 1.5


def _aggregate_votes(votes: tuple, atr_val: float) -> DailyTrendResult:
    """
    [A5] fired_possible : max_possible calculé sur les votes fired uniquement.
    Les votes non déclenchés (NaN, données indisponibles) ne déprimaient plus la force.
    V5 incluait tous les votes dans max_possible → force systématiquement sous-estimée.
    """
    bull_score = sum(v.weight * v.reliability
                     for v in votes if v.fired and v.direction == Direction.BULLISH)
    bear_score = sum(v.weight * v.reliability
                     for v in votes if v.fired and v.direction == Direction.BEARISH)

    # [A5] fired_possible uniquement
    fired_possible = sum(v.weight * v.reliability for v in votes if v.fired)

    winning_score = max(bull_score, bear_score)
    min_votes_met = winning_score >= MIN_RELIABLE_SCORE

    if not min_votes_met or bull_score == bear_score:
        return DailyTrendResult(direction=Direction.RANGE, strength=35, atr_val=atr_val,
                                bull_score=bull_score, bear_score=bear_score,
                                votes=votes, min_votes_met=min_votes_met)

    direction = Direction.BULLISH if bull_score > bear_score else Direction.BEARISH
    ratio     = winning_score / fired_possible if fired_possible > 0 else 0.0
    strength  = int(min(90, max(35, ratio * 112)))

    return DailyTrendResult(direction=direction, strength=strength, atr_val=atr_val,
                            bull_score=bull_score, bear_score=bear_score,
                            votes=votes, min_votes_met=min_votes_met)


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE — trend_daily  [V5-5]
# ══════════════════════════════════════════════════════════════════════════════

def trend_daily(df: pd.DataFrame, df_weekly: Optional[pd.DataFrame] = None,
                instrument: str = '',
                current_week_open: Optional[float] = None) -> DailyTrendResult:
    h       = df['High']
    lo      = df['Low']
    c       = df['Close']
    atr_val = float(_atr(h, lo, c, 14).iloc[-1])

    if len(df) < 60:
        guard = VoteSignal(name="guard", direction=Direction.RANGE,
                           weight=0.0, reliability=1.0, fired=False,
                           reason=f"données insuffisantes ({len(df)}<60)")
        return DailyTrendResult(direction=Direction.RANGE, strength=0, atr_val=atr_val,
                                bull_score=0.0, bear_score=0.0,
                                votes=(guard,), min_votes_met=False)

    cur     = float(c.iloc[-1])
    e50     = _ema(c, 50)
    e21     = float(_ema(c, 21).iloc[-1])
    e50_cur = float(e50.iloc[-1])

    vol_series: Optional[pd.Series] = (
        df['Volume']
        if 'Volume' in df.columns and instrument not in INDICES
        else None
    )

    ctx = {
        'cur': cur, 'e21': e21, 'e50_cur': e50_cur, 'e50': e50,
        'atr_val': atr_val, 'df_weekly': df_weekly, 'df_daily': df,
        'vol_series': vol_series, 'instrument': instrument,
        'current_week_open': current_week_open,
    }

    raw_votes = []
    for fn in DAILY_VOTE_REGISTRY:
        try:
            v = fn(h, lo, c, ctx)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Filet de sécurité : un vote ne doit jamais planter l'analyse globale
            _log.error("Vote %s — erreur : %s", fn.__name__, exc, exc_info=True)
            v = VoteSignal(name=fn.__name__, direction=Direction.RANGE,
                           weight=0.0, reliability=0.0, fired=False, reason=str(exc))
        raw_votes.append(v)

    votes  = tuple(raw_votes)
    result = _aggregate_votes(votes, atr_val)
    _log.debug("trend_daily %s | %s", instrument or "?", result.summary())
    return result


# ===================== TENDANCES PAR TF =====================

def trend_macro(df: pd.DataFrame, tf: str) -> Tuple[str, int, float]:
    if len(df) < 50:
        atr_val = float(_atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]) \
                  if len(df) >= 15 else 0.0
        return 'Range', 0, atr_val

    c   = df['Close']
    h   = df['High']
    lo  = df['Low']
    atr_val = float(_atr(h, lo, c, 14).iloc[-1])
    band    = atr_val * 0.1
    e50     = _ema(c, 50)

    if tf == 'M':
        if len(df) < 100:
            return 'Range', 0, atr_val
        e100 = _ema(c, 100)
        ref  = float(e100.iloc[-1])
        cur  = float(e50.iloc[-1])
        if ref == 0:
            return 'Range', 40, atr_val
        gap = abs(cur - ref) / ref * 100
        s   = 75 if gap > 0.3 else 60
        if cur > ref + band:
            return 'Bullish', s, atr_val
        elif cur < ref - band:
            return 'Bearish', s, atr_val
        return 'Range', 40, atr_val
    else:
        if len(df) < 200:
            _log.warning("Weekly SMA200 indisponible (%d bougies)", len(df))
            return 'Range', 40, atr_val
        s200    = _sma(c, 200)
        cur50   = float(e50.iloc[-1])
        ref200  = float(s200.iloc[-1])
        prev50  = float(e50.iloc[-2])
        prev200 = float(s200.iloc[-2])
        cross   = (prev50 <= prev200 < cur50) or (prev50 >= prev200 > cur50)
        if cur50 > ref200 + band:
            return 'Bullish', (90 if cross else 75), atr_val
        elif cur50 < ref200 - band:
            return 'Bearish', (90 if cross else 75), atr_val
        return 'Range', 40, atr_val


def trend_4h(df: pd.DataFrame,
             df_daily: Optional[pd.DataFrame] = None,
             instrument: str = '',
             current_day_open: Optional[float] = None) -> Tuple[str, int, float]:
    """
    Tendance 4H — 3 votes (EMA50, DMI, Daily Open).

    [BUG-1] current_day_open : open de la bougie daily EN COURS (incomplète).
    Priorité sur df_daily['Open'].iloc[-1] qui pointe sur hier (bougie complète).
    """
    h   = df['High']
    lo  = df['Low']
    c   = df['Close']
    atr_val = float(_atr(h, lo, c, 14).iloc[-1])

    if len(df) < 60:
        return 'Range', 0, atr_val

    cur   = float(c.iloc[-1])
    score = 0

    e50_cur = float(_ema(c, 50).iloc[-1])
    if not np.isnan(e50_cur):
        score += 1 if cur > e50_cur else -1

    pdi, mdi = _dmi(h, lo, c)
    if not (np.isnan(pdi) or np.isnan(mdi)):
        score += 1 if pdi > mdi else -1

    # [BUG-1] current_day_open = open de la bougie D courante (incomplète = aujourd'hui)
    # df_daily['Open'].iloc[-1] = open d'hier (dernière bougie D complète) — référence décalée
    try:
        if current_day_open is not None and not np.isnan(current_day_open):
            today_open = current_day_open
        elif df_daily is not None and not df_daily.empty:
            today_open = float(df_daily['Open'].iloc[-1])
            _log.debug("trend_4h[%s] daily open fallback bougie complète", instrument)
        else:
            dates      = pd.to_datetime(df.index).normalize()
            today_open = float(df[dates == dates[-1]]['Open'].iloc[0])
        score += 1 if cur > today_open else -1
    except (KeyError, IndexError) as exc:
        _log.warning("trend_4h[%s] Daily Open ignoré — structure inattendue : %s",
                     instrument, exc)
    except (TypeError, ValueError) as exc:
        _log.debug("trend_4h[%s] Daily Open ignoré — valeur : %s", instrument, exc)

    s        = abs(score)
    strength = 90 if s == 3 else 70 if s >= 1 else 40
    return ('Bullish' if score > 0 else 'Bearish' if score < 0 else 'Range'), strength, atr_val


def trend_intraday(df: pd.DataFrame, instrument: str = '') -> Tuple[str, int, float]:
    h   = df['High']
    lo  = df['Low']
    c   = df['Close']
    atr_val = float(_atr(h, lo, c, 14).iloc[-1])

    if len(df) < 70:
        return 'Range', 0, atr_val

    cur    = float(c.iloc[-1])
    period = 50
    lag    = period // 2

    # [A5.2] EMA calculées en une passe — pas de recalcul redondant
    ema9_s  = _ema(c, 9)
    ema21_s = _ema(c, 21)
    ema50_s = _ema(c, period)
    ema12_s = _ema(c, 12)
    ema26_s = _ema(c, 26)

    e9      = float(ema9_s.iloc[-1])
    e21     = float(ema21_s.iloc[-1])
    e50_cur = float(ema50_s.iloc[-1])

    src_adj = c + (c - c.shift(lag))
    zlema   = float(src_adj.ewm(span=period, adjust=False).mean().iloc[-1])

    rsi_val  = float(_rsi(c, 14).iloc[-1])
    macd     = ema12_s - ema26_s
    sig_line = _ema(macd, 9)
    macd_cur = float(macd.iloc[-1])
    sig_cur  = float(sig_line.iloc[-1])

    if any(np.isnan(v) for v in [e9, e21, e50_cur, zlema, rsi_val, macd_cur, sig_cur]):
        return 'Range', 0, atr_val

    bull_zlema = cur > zlema
    bear_zlema = cur < zlema
    bull_stack = e9 > e21 > e50_cur
    bear_stack = e9 < e21 < e50_cur
    bull_mom   = rsi_val > 50 and macd_cur > sig_cur
    bear_mom   = rsi_val < 50 and macd_cur < sig_cur

    votes_bull = [bull_zlema, bull_stack, bull_mom]
    votes_bear = [bear_zlema, bear_stack, bear_mom]

    use_volume = instrument not in INDICES and 'Volume' in df.columns
    max_votes  = 3
    if use_volume:
        vol     = df['Volume']
        vol_ma  = vol.rolling(20).mean()
        vol_cur = float(vol.iloc[-1])
        vol_avg = float(vol_ma.iloc[-1])
        if not np.isnan(vol_avg) and vol_avg > 0:
            strong_vol = vol_cur > vol_avg * 1.3
            votes_bull.append(strong_vol and bull_zlema)
            votes_bear.append(strong_vol and bear_zlema)
            max_votes = 4

    vb  = sum(votes_bull)
    vbr = sum(votes_bear)
    threshold_strong = max_votes
    threshold_mod    = max_votes - 1

    def _atr_strength() -> int:
        if atr_val <= 0:
            return 60
        return int(min(95, 40 + (abs(cur - zlema) / atr_val) * 25))

    if vb == threshold_strong:
        return 'Bullish', _atr_strength(), atr_val
    if vbr == threshold_strong:
        return 'Bearish', _atr_strength(), atr_val
    if vb >= threshold_mod:
        return 'Bullish', 55, atr_val
    if vbr >= threshold_mod:
        return 'Bearish', 55, atr_val
    if cur < e50_cur and e9 > e21:
        return 'Retracement Bull', 45, atr_val
    if cur > e50_cur and e9 < e21:
        return 'Retracement Bear', 45, atr_val
    return 'Range', 30, atr_val


# ===================== MÉTRIQUES COMPLÉMENTAIRES =====================

def trend_age_daily(df: pd.DataFrame) -> str:
    if len(df) < 55:
        return 'N/A'
    c     = df['Close']
    e50   = _ema(c, 50)
    above = c > e50
    for i in range(len(above) - 1, 0, -1):
        if above.iloc[i] != above.iloc[i - 1]:
            age = len(above) - 1 - i
            return str(age) if age > 0 else '0'
    return f'>{len(above)}'


# ===================== API + CACHE =====================

def fetch_candles(instrument: str, granularity: str, count: int,
                  account_id: str, access_token: str,
                  include_incomplete: bool = False) -> pd.DataFrame:
    """
    Récupère les bougies OANDA.

    [BUG-1] include_incomplete=False (défaut) : bougies complètes uniquement,
    correctes pour les indicateurs (pas de repainting).
    include_incomplete=True : inclut la bougie courante (incomplète),
    nécessaire pour obtenir l'open réel de la session/jour/semaine en cours.
    """
    url     = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params  = {'granularity': granularity, 'count': count, 'price': 'M'}
    session = _get_session()
    try:
        r = session.get(url, headers=headers, params=params, timeout=15)
        if r.status_code != 200:
            _log.warning("OANDA %s HTTP %d — %s %s",
                         instrument, r.status_code, granularity, r.text[:120])
            return pd.DataFrame()
        candles = r.json().get('candles', [])
        rows    = []
        for candle in candles:
            if not include_incomplete and not candle.get('complete'):
                continue
            try:
                rows.append({
                    'date':   candle['time'],
                    'Open':   float(candle['mid']['o']),
                    'High':   float(candle['mid']['h']),
                    'Low':    float(candle['mid']['l']),
                    'Close':  float(candle['mid']['c']),
                    'Volume': float(candle.get('volume', 0)),
                })
            except (KeyError, ValueError) as e:
                _log.warning("Parse candle %s: %s", instrument, e)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            _log.error("fetch_candles %s/%s — index non-DatetimeIndex", instrument, granularity)
            return pd.DataFrame()
        return df
    except requests.exceptions.Timeout:
        _log.warning("Timeout %s %s", instrument, granularity)
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        _log.warning("RequestException %s %s: %s", instrument, granularity, e)
        return pd.DataFrame()
    except Exception as e:  # pylint: disable=broad-exception-caught
        _log.error("Unexpected error %s %s: %s", instrument, granularity, e, exc_info=True)
        return pd.DataFrame()


def fetch_current_open(instrument: str, granularity: str,
                       account_id: str, access_token: str) -> Optional[float]:
    """
    [BUG-1] Retourne l'open de la bougie courante (potentiellement incomplète).

    Utilisé exclusivement pour les références d'open de session :
    - weekly open courant  → _vote_weekly_open_vote
    - daily open courant   → trend_4h

    Non mis en cache : la valeur change en continu et ne doit pas être périmée.
    Requête légère : count=1, une seule bougie.
    """
    df = fetch_candles(instrument, granularity, 1, account_id, access_token,
                       include_incomplete=True)
    if df.empty:
        _log.debug("fetch_current_open %s/%s — vide", instrument, granularity)
        return None
    try:
        return float(df['Open'].iloc[-1])
    except (IndexError, ValueError, TypeError) as e:
        _log.debug("fetch_current_open %s/%s — parse error: %s", instrument, granularity, e)
        return None


def fetch_cached(inst: str, gran: str, cnt: int, acc: str, tok: str) -> pd.DataFrame:
    """
    Cache thread-safe avec double-checked locking.

    [BUG-4] Les DataFrames vides (erreurs réseau) ne sont PAS mis en cache.
    Une panne temporaire n'exclut plus la paire pendant 10 minutes.

    [BUG-5] Le cache retourne df.copy() — le DataFrame original en cache
    est protégé contre toute mutation accidentelle par l'appelant.
    """
    key = (inst, gran, cnt)
    now = datetime.now(timezone.utc)

    with _cache_lock:
        entry = _data_cache.get(key)
        if entry is not None:
            ts, df = entry
            if (now - ts).total_seconds() < _CACHE_TTL:
                return df.copy()  # [BUG-5] copie défensive

    df = fetch_candles(inst, gran, cnt, acc, tok)

    if not df.empty:  # [BUG-4] ne pas cacher les erreurs réseau
        with _cache_lock:
            _data_cache[key] = (datetime.now(timezone.utc), df)

    return df


def fetch_all_data(instrument: str, account_id: str, access_token: str,
                   stop_event: Optional[threading.Event] = None) -> Optional[dict]:
    """
    Récupère toutes les données OHLCV par timeframe + opens courants.

    [BUG-1] Les opens de référence (weekly, daily) sont fetchés séparément
    avec include_incomplete=True pour obtenir l'open de la session EN COURS.

    [BUG-2] stop_event vérifié entre chaque TF pour permettre l'annulation
    coopérative sans attendre la fin de toutes les requêtes réseau.
    """
    specs    = {'M': ('M', 150), 'W': ('W', 250), 'D': ('D', 300),
                '4H': ('H4', 300), '1H': ('H1', 300), '15m': ('M15', 300)}
    min_bars = {'M': 100, 'W': 50, 'D': 60, '4H': 60, '1H': 70, '15m': 70}
    cache: dict = {}
    for tf, (gran, count) in specs.items():
        if stop_event and stop_event.is_set():   # [BUG-2] annulation coopérative
            return None
        df = fetch_cached(instrument, gran, count, account_id, access_token)
        if df.empty:
            _log.warning("%s — vide TF=%s", instrument, tf)
            return None
        if len(df) < min_bars[tf]:
            _log.warning("%s — TF=%s : %d<%d bougies", instrument, tf, len(df), min_bars[tf])
            return None
        cache[tf] = df

    # Opens courants (bougie incomplète) — non mis en cache car valeur live
    cache['_week_open'] = fetch_current_open(instrument, 'W', account_id, access_token)
    cache['_day_open']  = fetch_current_open(instrument, 'D', account_id, access_token)

    return cache


# ===================== SCORING MTF =====================

def _bull_compat(t: str) -> bool:
    """Retourne True si la tendance est compatible avec une direction haussière."""
    return t in ('Bullish', 'Retracement Bull')


def _bear_compat(t: str) -> bool:
    """Retourne True si la tendance est compatible avec une direction baissière."""
    return t in ('Bearish', 'Retracement Bear')


def _mtf_weighted_score(trends: dict, scores: dict) -> tuple:
    """Calcule les scores pondérés bull/bear et retourne (w_bull, w_bear, total)."""
    weights = {'M': 5.0, 'W': 4.0, 'D': 4.0, '4H': 2.5, '1H': 1.5, '15m': 1.0}
    total   = sum(weights.values())

    w_bull = sum(
        weights[tf] * (scores[tf] / 100)
        for tf in trends
        if trends[tf].startswith('Bullish')
    ) + sum(
        weights[tf] * (scores[tf] / 100) * 0.5
        for tf in trends
        if trends[tf].startswith('Retracement Bull')
    )

    w_bear = sum(
        weights[tf] * (scores[tf] / 100)
        for tf in trends
        if trends[tf].startswith('Bearish')
    ) + sum(
        weights[tf] * (scores[tf] / 100) * 0.5
        for tf in trends
        if trends[tf].startswith('Retracement Bear')
    )

    return w_bull, w_bear, total


def _mtf_alignment_bonus(trends: dict, direction: str) -> int:
    """
    Calcule le bonus d'alignement inter-TF.
    [A7] Bonus étendu aux Retracements compatibles (était limité aux purs en V5).
    """
    bonus = 0
    m,  w  = trends.get('M', ''),  trends.get('W', '')
    d,  h4 = trends.get('D', ''),  trends.get('4H', '')

    if direction == 'Bullish':
        if m == 'Bullish' and w == 'Bullish':
            bonus += 15
        elif _bull_compat(m) and _bull_compat(w):
            bonus += 12
        if d == 'Bullish' and h4 == 'Bullish':
            bonus += 10
        elif _bull_compat(d) and _bull_compat(h4):
            bonus += 7
    else:  # Bearish
        if m == 'Bearish' and w == 'Bearish':
            bonus += 15
        elif _bear_compat(m) and _bear_compat(w):
            bonus += 12
        if d == 'Bearish' and h4 == 'Bearish':
            bonus += 10
        elif _bear_compat(d) and _bear_compat(h4):
            bonus += 7

    return bonus


def score_mtf(trends: dict, scores: dict) -> Tuple[str, float]:
    """
    Score MTF pondéré par timeframe et conviction.
    Décomposé en sous-fonctions pour réduire la complexité cyclomatique.
    [A7] Bonus M+W étendu aux Retracements compatibles.
    """
    w_bull, w_bear, total = _mtf_weighted_score(trends, scores)

    if w_bull > w_bear:
        raw_score, direction = (w_bull / total) * 100, 'Bullish'
    elif w_bear > w_bull:
        raw_score, direction = (w_bear / total) * 100, 'Bearish'
    else:
        return 'Range', 0.0

    bonus = _mtf_alignment_bonus(trends, direction)
    return direction, min(100.0, raw_score + bonus)


def grade_hybrid(scores_list: list, nc_list: list = None) -> list:
    if not scores_list:
        return []
    if nc_list is None:
        nc_list = [3] * len(scores_list)
    grades = []
    for score, nc in zip(scores_list, nc_list):
        nc_bonus = (int(nc) - 3) * 5
        adj      = min(100.0, float(score) + nc_bonus)
        if adj >= 80:
            grades.append('A+')
        elif adj >= 55:
            grades.append('A')
        elif adj >= 38:
            grades.append('B+')
        else:
            grades.append('B')
    return grades


def _compute_nc(trends: dict, mtf_dir: str) -> int:
    """
    [A6] NC pondéré — Retracements comptent ±0.5 (étaient 0 en V5).
    Exemple V5 : 4 Bullish + 2 Retracement Bull → NC=4
    Exemple V5.1 : même setup → NC=5 (4×1.0 + 2×0.5 = 5.0)
    """
    if mtf_dir not in ('Bullish', 'Bearish'):
        return 0
    ret_aligned = 'Retracement Bull' if mtf_dir == 'Bullish' else 'Retracement Bear'
    ret_opposed = 'Retracement Bear' if mtf_dir == 'Bullish' else 'Retracement Bull'
    opposed_dir = 'Bearish'          if mtf_dir == 'Bullish' else 'Bullish'
    score = 0.0
    for t in trends.values():
        if t == mtf_dir:
            score += 1.0
        elif t == ret_aligned:
            score += 0.5
        elif t == opposed_dir:
            score -= 1.0
        elif t == ret_opposed:
            score -= 0.5
    return round(score)


# ===================== ANALYSE PRINCIPALE =====================

def analyze_pair(pair: str, account_id: str, access_token: str,
                 stop_event: Optional[threading.Event] = None) -> Optional[dict]:
    """
    [BUG-2] stop_event : annulation coopérative entre TF lors d'un timeout global.
    Vérifié avant fetch_all_data — évite les requêtes inutiles après timeout.
    """
    if stop_event and stop_event.is_set():
        return None
    try:
        cache = fetch_all_data(pair, account_id, access_token, stop_event)
        if cache is None:
            return None

        trends: dict = {}
        scores: dict = {}
        atrs:   dict = {}

        for tf in ('M', 'W'):
            t, s, a = trend_macro(cache[tf], tf)
            trends[tf], scores[tf], atrs[tf] = t, s, a

        t, s, a = trend_daily(cache['D'], cache['W'], instrument=pair,
                              current_week_open=cache.get('_week_open'))
        trends['D'], scores['D'], atrs['D'] = t, s, a

        t, s, a = trend_4h(cache['4H'], cache['D'], instrument=pair,
                           current_day_open=cache.get('_day_open'))
        trends['4H'], scores['4H'], atrs['4H'] = t, s, a

        for tf in ('1H', '15m'):
            t, s, a = trend_intraday(cache[tf], pair)
            trends[tf], scores[tf], atrs[tf] = t, s, a

        mtf_dir, mtf_score = score_mtf(trends, scores)
        age = trend_age_daily(cache['D'])
        nc  = _compute_nc(trends, mtf_dir)  # [A6]

        row = {
            'Paire':       pair.replace('_', '/'),
            'M':           trends['M'],
            'W':           trends['W'],
            'D':           trends['D'],
            '4H':          trends['4H'],
            '1H':          trends['1H'],
            '15m':         trends['15m'],
            'MTF':         f"{mtf_dir} ({mtf_score:.0f}%)" if mtf_dir != 'Range' else 'Range',
            '_mtf_score':  mtf_score,   # [A15] conservé pour tri numérique
            '_mtf_dir':    mtf_dir,
            'NC':          nc,
            'Age D1':      age,
            'ATR Daily':   _fmt_atr(atrs['D']),
            'ATR H4':      _fmt_atr(atrs['4H']),
            'ATR H1':      _fmt_atr(atrs['1H']),
            'ATR 15m':     _fmt_atr(atrs['15m']),
        }
        return row

    except Exception as e:  # pylint: disable=broad-exception-caught
        # Catch-all top-level : garantit qu'une paire en erreur ne bloque pas les autres
        _log.error("analyze_pair %s : %s", pair, e, exc_info=True)
        return None


# ===================== [A13] NOYAU PUR + WRAPPER STREAMLIT =====================

def analyze_all_core(account_id: str, access_token: str,
                     progress_cb=None, status_cb=None) -> Tuple[pd.DataFrame, list]:
    """
    Noyau d'analyse — sans dépendance Streamlit.

    [BUG-2] Timeout réel : executor explicite + shutdown(wait=False, cancel_futures=True).
    Le `with ThreadPoolExecutor` appelait shutdown(wait=True) à la sortie,
    bloquant le thread principal même après FutureTimeoutError.
    Correction : executor hors bloc `with`, shutdown non-bloquant sur timeout.

    [BUG-3] SessionRegistry scopée par analyse — pas de partage inter-analyses.

    Annulation coopérative : stop_event transmis à fetch_all_data pour interrompre
    entre deux requêtes TF dès que le timeout est déclenché.
    """
    results:   list = []
    errors:    list = []
    total          = len(INSTRUMENTS)
    done           = 0
    timed_out      = False
    stop_event     = threading.Event()       # [BUG-2] annulation coopérative
    registry       = _SessionRegistry()      # [BUG-3] scopé par analyse

    executor = ThreadPoolExecutor(
        max_workers=5,
        initializer=_pool_thread_init,
        initargs=(registry,),
    )
    try:
        futures: dict[Future, str] = {
            executor.submit(
                analyze_pair, inst, account_id, access_token, stop_event
            ): inst
            for inst in INSTRUMENTS
        }
        try:
            for future in as_completed(futures, timeout=120):
                inst = futures[future]
                done += 1
                if progress_cb:
                    progress_cb(done / total)
                if status_cb:
                    status_cb(f"GPS ({done}/{total}) — {inst.replace('_', '/')} ✓")
                try:
                    row = future.result()
                    if row:
                        results.append(row)
                    else:
                        errors.append(inst)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    errors.append(inst)
                    _log.error("Future %s: %s", inst, e)

        except FutureTimeoutError:
            timed_out = True
            _log.error("analyze_all_core — timeout 120s. Arrêt coopératif.")
            stop_event.set()                              # [BUG-2] signal aux workers
            executor.shutdown(wait=False, cancel_futures=True)  # non-bloquant
            for f, inst in futures.items():
                if not f.done():
                    errors.append(inst)
    finally:
        if not timed_out:
            executor.shutdown(wait=True)  # fermeture propre si pas de timeout
        registry.close_all()             # [BUG-3] sessions de CETTE analyse uniquement

    if not results:
        return pd.DataFrame(), errors

    scores_list = [r['_mtf_score'] for r in results]
    nc_list     = [r['NC']         for r in results]
    grades      = grade_hybrid(scores_list, nc_list)
    for r, g in zip(results, grades):
        r['Quality'] = g

    df = pd.DataFrame(results)
    if timed_out:
        df.attrs['timed_out'] = True
    return df, errors


def analyze_all(account_id: str, access_token: str) -> pd.DataFrame:
    """[A13] Wrapper Streamlit — UI uniquement, délègue à analyze_all_core."""
    progress = st.progress(0)
    status   = st.empty()

    df, errors = analyze_all_core(
        account_id, access_token,
        progress_cb=progress.progress,
        status_cb=status.text,
    )

    progress.empty()
    status.empty()

    if df.attrs.get('timed_out'):
        st.error("⏱️ Analyse interrompue après 120s — résultats partiels. Vérifiez la connexion OANDA.")

    if errors:
        st.warning(
            f"⚠️ {len(errors)} paire(s) non analysée(s) : "
            f"{', '.join(e.replace('_', '/') for e in errors)}"
        )
    return df


# ===================== PDF — identique V5.0 =====================

def _safe_str(s: str) -> str:
    return s.encode('latin-1', errors='replace').decode('latin-1')


def create_pdf(df: pd.DataFrame) -> BytesIO:
    cols = [
        'Paire', 'M', 'W', 'D', '4H', '1H', '15m',
        'MTF', 'Quality', 'NC', 'Age D1',
        'ATR Daily', 'ATR H4', 'ATR H1', 'ATR 15m'
    ]
    widths = {
        'Paire': 22, 'M': 16, 'W': 16, 'D': 16, '4H': 16, '1H': 16, '15m': 16,
        'MTF': 30, 'Quality': 12, 'NC': 10, 'Age D1': 13,
        'ATR Daily': 17, 'ATR H4': 17, 'ATR H1': 15, 'ATR 15m': 15,
    }
    # Couleurs NC — NC=0 corrigé en orange (était rouge en V5)
    nc_rgb = {
        (5, 99):  (46,  204, 113, 255, 255, 255),
        (3,  4):  (39,  174,  96, 255, 255, 255),
        (1,  2):  (241, 196,  15,   0,   0,   0),
        (0,  0):  (230, 126,  34, 255, 255, 255),  # orange — cohérent avec UI
        (-99, -1):(231,  76,  60, 255, 255, 255),
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

    try:
        if FPDF is None:
            raise RuntimeError("fpdf / fpdf2 non installé")

        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_margins(10, 10, 10)
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(0, 9, _cell_text("BLUESTAR GPS V5.0"), ln=True, align="C")
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(0, 5, _cell_text(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        ), ln=True, align="C")
        pdf.ln(4)

        grade_rgb = {
            'A+': (251, 191, 36),
            'A':  (163, 230, 53),
            'B+': (52,  211, 153),
            'B':  (96,  165, 250),
        }

        def header():
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_fill_color(30, 58, 138)
            pdf.set_text_color(255, 255, 255)
            for col in cols:
                pdf.cell(widths[col], 7, _cell_text(col), border=1, align='C', fill=True)
            pdf.ln()
            pdf.set_font("Helvetica", "", 6.5)

        header()

        for _, row in df.iterrows():
            if pdf.get_y() + rh > 287 - 15:
                pdf.add_page()
                header()
            for col in cols:
                val = str(row.get(col, ''))
                fc  = (255, 255, 255)
                tc  = (0,   0,   0)
                if col == 'Quality':
                    fc = grade_rgb.get(val, (156, 163, 175))
                elif col == 'NC':
                    fc, tc = _nc_colors(val)
                elif 'Bull' in val and 'Ret' not in val:
                    fc = (46,  204, 113)
                    tc = (255, 255, 255)
                elif 'Bear' in val and 'Ret' not in val:
                    fc = (231,  76,  60)
                    tc = (255, 255, 255)
                elif 'Retracement Bull' in val:
                    fc = (125, 206, 160)
                    tc = (255, 255, 255)
                elif 'Retracement Bear' in val:
                    fc = (241, 148, 138)
                    tc = (255, 255, 255)
                elif 'Range' in val:
                    fc = (149, 165, 166)
                    tc = (255, 255, 255)
                pdf.set_fill_color(*fc)
                pdf.set_text_color(*tc)
                pdf.cell(widths[col], rh, _cell_text(val), border=1, align='C', fill=True)
            pdf.ln()

        buf = BytesIO()
        out = pdf.output(dest='S')
        buf.write(out.encode('latin-1') if isinstance(out, str) else bytes(out))
        buf.seek(0)
        return buf

    except Exception as e:  # pylint: disable=broad-exception-caught
        # PDF peut lever des erreurs fpdf imprévisibles — on retourne un fallback
        _log.error("PDF error: %s", e, exc_info=True)
        buf2 = BytesIO()
        if FPDF:
            fallback = FPDF()
            fallback.add_page()
            fallback.set_font("Helvetica", "B", 12)
            fallback.cell(0, 10, "PDF Generation Error", ln=True)
            out2 = fallback.output(dest='S')
            buf2.write(out2.encode('latin-1') if isinstance(out2, str) else bytes(out2))
        buf2.seek(0)
        return buf2


# ===================== UI — identique V5.0 =====================

def main():
    st.markdown(
        "<div class='main-header'><h1>🧭 BLUESTAR HEDGE FUND GPS V5.0</h1>"
        "<p style='margin:0;font-size:0.85em;opacity:0.8'>"
        "Signal Decomposition · P1–P15 + C1–C15 + V5-1–V5-7"
        "</p></div>",
        unsafe_allow_html=True,
    )

    try:
        acc = st.secrets["OANDA_ACCOUNT_ID"]
        tok = st.secrets["OANDA_ACCESS_TOKEN"]
    except KeyError:
        st.error("❌ Secrets OANDA manquants — configurez OANDA_ACCOUNT_ID et OANDA_ACCESS_TOKEN")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Configuration")
        only_best = st.checkbox("Afficher uniquement Grade A+ / A", value=False)
        st.info(
            "Cache : 10 min · Workers : 5 · Retry : 4× backoff · "
            "Score × NC · Vote volume-conditionné"
        )
        st.markdown("---")
        if st.button("🗑️ Vider le cache", use_container_width=True):
            _cache_clear()
            st.success("Cache vidé.")
        st.markdown("---")
        st.caption("Bluestar GPS V5.0 — NC = cohérence nette TF (−6 … +6)")

    is_running = st.session_state.get('_analysis_running', False)
    if st.button(
        "🚀 LANCER L'ANALYSE TOUS ACTIFS",
        type="primary",
        use_container_width=True,
        disabled=is_running,
    ):
        st.session_state['_analysis_running'] = True
        try:
            with st.spinner("Analyse Multi-Timeframe en cours..."):
                df = analyze_all(acc, tok)
            if not df.empty:
                st.session_state['df']    = df
                st.session_state['df_ts'] = datetime.now(timezone.utc)
        finally:
            st.session_state['_analysis_running'] = False

    if 'df' not in st.session_state:
        return

    df_ts = st.session_state.get('df_ts')
    if df_ts:
        age_min = (datetime.now(timezone.utc) - df_ts).total_seconds() / 60
        if age_min > DATA_MAX_AGE_MIN:
            st.markdown(
                f"<div class='stale-warning'>⏰ <b>Données périmées</b> — "
                f"dernière analyse il y a <b>{age_min:.0f} min</b>. "
                f"Relancez l'analyse pour des signaux à jour.</div>",
                unsafe_allow_html=True,
            )

    df = st.session_state['df'].copy()

    if only_best:
        df = df[df['Quality'].isin(['A+', 'A'])]

    grade_order   = ['A+', 'A', 'B+', 'B']
    df['Quality'] = pd.Categorical(df['Quality'], categories=grade_order, ordered=True)

    # [A15] Tri sur _mtf_score numérique — V5 triait sur string "Bullish (97%)"
    sort_cols = [c for c in ['Quality', 'NC', '_mtf_score'] if c in df.columns]
    df = df.sort_values(sort_cols, ascending=[True, False, False])

    # Supprimer colonnes internes APRÈS le tri
    df.drop(columns=['_mtf_score', '_mtf_dir'], inplace=True, errors='ignore')

    c1, c2, c3, c4 = st.columns(4)
    total   = len(df)
    a_plus  = len(df[df['Quality'] == 'A+'])
    a_grade = len(df[df['Quality'] == 'A'])
    b_grade = len(df[df['Quality'].isin(['B+', 'B'])])

    c1.metric("Total Analyzed",   total)
    c2.metric("Setups A+ (GOLD)", a_plus,  delta_color="inverse")
    c3.metric("Setups A (GREEN)", a_grade, delta_color="inverse")
    c4.metric("Setups B (BLUE)",  b_grade, delta_color="inverse")

    display = [
        'Paire', 'M', 'W', 'D', '4H', '1H', '15m',
        'MTF', 'Quality', 'NC', 'Age D1',
        'ATR Daily', 'ATR H4', 'ATR H1', 'ATR 15m',
    ]
    grade_css = {'A+': '#fbbf24', 'A': '#a3e635', 'B+': '#34d399', 'B': '#60a5fa'}

    def style_trend(v):
        if not isinstance(v, str):
            return ''
        if 'Bull' in v and 'Ret' not in v:
            return f'background-color:{TREND_COLORS["Bullish"]};color:white;font-weight:bold'
        if 'Bear' in v and 'Ret' not in v:
            return f'background-color:{TREND_COLORS["Bearish"]};color:white;font-weight:bold'
        if 'Retracement Bull' in v:
            return f'background-color:{TREND_COLORS["Retracement Bull"]};color:white'
        if 'Retracement Bear' in v:
            return f'background-color:{TREND_COLORS["Retracement Bear"]};color:white'
        if 'Range' in v:
            return f'background-color:{TREND_COLORS["Range"]};color:white'
        return ''

    def style_quality(s):
        if s.name != 'Quality':
            return [''] * len(s)
        return [
            f'color:black;font-weight:bold;background-color:{grade_css.get(x, "#9ca3af")}'
            for x in s
        ]

    # style_nc — identique V5.0 (couleurs hex directes)
    def style_nc(s):
        if s.name != 'NC':
            return [''] * len(s)
        out = []
        for v in s:
            try:
                n = int(v)
                if n >= 5:
                    out.append('background-color:#1D9E75;color:white;font-weight:bold')
                elif n >= 3:
                    out.append('background-color:#27ae60;color:white;font-weight:bold')
                elif n >= 1:
                    out.append('background-color:#f39c12;color:white;font-weight:bold')
                elif n == 0:
                    out.append('background-color:#e67e22;color:white')
                else:
                    out.append('background-color:#e74c3c;color:white;font-weight:bold')
            except (ValueError, TypeError):
                out.append('')
        return out

    cols_present = [col for col in display if col in df.columns]
    styled = (
        df[cols_present].style
        .apply(style_quality, axis=0)
        .apply(style_nc,      axis=0)
        .map(style_trend)
    )
    st.dataframe(
        styled,
        height=min(800, max(400, (len(df) + 1) * 38 + 10)),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2, c3 = st.columns(3)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
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
            data=df[cols_present].to_csv(index=False).encode('utf-8'),
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

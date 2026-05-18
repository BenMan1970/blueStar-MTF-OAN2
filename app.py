"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       BLUESTAR HEDGE FUND GPS  —  V5.0  (Architecture Signal Decomposition) ║
║                                                                              ║
║  Patches V4.0 (P1–P15) conservés intégralement.                             ║
║  Corrections V4.1 (C1–C15) conservées intégralement.                        ║
║                                                                              ║
║  Refonte V5.0 :                                                              ║
║   [V5-1] trend_daily : Signal Decomposition Pattern                          ║
║          Chaque vote = fonction pure testable (VoteSignal typé)              ║
║   [V5-2] Contribution réelle = weight × reliability (fini les votes égaux)  ║
║   [V5-3] _vote_prev_midpoint conditionné au volume J-1 (> MA20)             ║
║          weight 0.5, reliability 0.65 si volume confirmé, 0.0 sinon          ║
║   [V5-4] Circuit breaker MIN_RELIABLE_SCORE = 1.5                           ║
║   [V5-5] DailyTrendResult : rétrocompatible + détail complet votes          ║
║   [V5-6] Import order PEP 8 (stdlib avant third-party)                       ║
║   [V5-7] Zéro variable ambiguë (l → lo partout)                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── Retry imports (P1) ────────────────────────────────────────────────────────
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── PDF : fpdf2 préféré, fallback fpdf legacy (P6) ───────────────────────────
try:
    from fpdf import FPDF
    _FPDF2 = hasattr(FPDF, 'set_lang')
except ImportError:
    FPDF = None
    _FPDF2 = False

# ── Logging structuré (P11) ───────────────────────────────────────────────────
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

DATA_MAX_AGE_MIN = 15
MAX_PIVOT_AGE    = 50   # [C9]

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

# ===================== [P7] SESSION HTTP THREAD-SAFE =====================
_thread_local = threading.local()

def _get_session() -> requests.Session:
    """Session requests propre à chaque thread (threading.local)."""
    if not hasattr(_thread_local, 'session'):
        session = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        _thread_local.session = session
    return _thread_local.session

# ===================== [C1] CACHE THREAD-SAFE CUSTOM =====================
_data_cache: dict = {}
_cache_lock = threading.Lock()
_CACHE_TTL  = 600

def _cache_clear() -> None:
    with _cache_lock:
        _data_cache.clear()

# ===================== [P13] TRUE RANGE FACTORISÉ =====================
def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

# ===================== INDICATEURS DE BASE =====================

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d    = close.diff()
    gain = d.where(d > 0, 0.0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    rs   = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)

def _dmi(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
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
# [V5-1] TYPES CONTRACTUELS — Signal Decomposition Pattern
# ══════════════════════════════════════════════════════════════════════════════

class Direction(str, Enum):
    BULLISH = 'Bullish'
    BEARISH = 'Bearish'
    RANGE   = 'Range'


@dataclass(frozen=True)
class VoteSignal:
    """
    Résultat atomique d'un vote individuel.

    weight      : contribution structurelle (ex. 2.0 pour swing).
    reliability : confiance intrinsèque [0.0–1.0].
    fired       : True si le vote a produit un signal directionnel.
    reason      : trace d'audit — jamais vide.

    Contribution effective = weight × reliability.
    Un vote fired=False contribue 0, quelle que soit sa pondération.
    """
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

    Déstructuration existante conservée :
        trend, score, atr = trend_daily(df, df_weekly)

    Nouvelles propriétés disponibles :
        result.votes        — tous les VoteSignal
        result.bull_votes   — votes haussiers déclenchés
        result.bear_votes   — votes baissiers déclenchés
        result.summary()    — ligne d'audit structurée
    """
    direction    : Direction
    strength     : int
    atr_val      : float
    bull_score   : float
    bear_score   : float
    votes        : tuple
    min_votes_met: bool

    def __iter__(self):
        """Rétrocompat : trend, score, atr = trend_daily(df)"""
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
            f"{'B' if v.direction == Direction.BULLISH else 'S' if v.direction == Direction.BEARISH else '-'}"
            f"(w={v.weight},r={v.reliability:.2f},fired={v.fired})"
            for v in self.votes
        ]
        return (
            f"{self.direction.value} str={self.strength} "
            f"bull={self.bull_score:.2f} bear={self.bear_score:.2f} | "
            + " ".join(parts)
        )


# ══════════════════════════════════════════════════════════════════════════════
# [V5-2/V5-3] VOTES ATOMIQUES — chacun testable en isolation
# ══════════════════════════════════════════════════════════════════════════════

def _vote_swing_structure(h: pd.Series, lo: pd.Series) -> VoteSignal:
    """
    Vote 1 — Structure swing HH/HL ou LH/LL.
    Poids 2.0 · Reliability 0.9
    Recency filter [C9] : pivots > MAX_PIVOT_AGE bougies exclus.
    """
    name    = "swing_structure"
    wing    = 5
    min_idx = max(0, len(h) - MAX_PIVOT_AGE)
    sh, sl  = [], []

    for i in range(wing, len(h) - wing):
        if h.iloc[i] == h.iloc[i - wing:i + wing + 1].max():
            sh.append(i)
        if lo.iloc[i] == lo.iloc[i - wing:i + wing + 1].min():
            sl.append(i)

    sh = [i for i in sh if i >= min_idx]
    sl = [i for i in sl if i >= min_idx]

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
                          weight=2.0, reliability=0.9, fired=True,
                          reason="HH+HL confirmés")
    if lh and ll:
        return VoteSignal(name=name, direction=Direction.BEARISH,
                          weight=2.0, reliability=0.9, fired=True,
                          reason="LH+LL confirmés")
    return VoteSignal(name=name, direction=Direction.RANGE,
                      weight=2.0, reliability=0.9, fired=False,
                      reason="structure mixte")


def _vote_ema_stack(cur: float, e21: float, e50_cur: float) -> VoteSignal:
    """
    Vote 2 — EMA21/EMA50 stack.
    Poids 1.0 · Reliability 0.75
    """
    name = "ema_stack"
    if np.isnan(e21) or np.isnan(e50_cur):
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=1.0, reliability=0.75, fired=False,
                          reason="NaN sur EMA21 ou EMA50")
    if cur > e21 > e50_cur:
        return VoteSignal(name=name, direction=Direction.BULLISH,
                          weight=1.0, reliability=0.75, fired=True,
                          reason=f"cur={cur:.5f} > e21={e21:.5f} > e50={e50_cur:.5f}")
    if cur < e21 < e50_cur:
        return VoteSignal(name=name, direction=Direction.BEARISH,
                          weight=1.0, reliability=0.75, fired=True,
                          reason=f"cur={cur:.5f} < e21={e21:.5f} < e50={e50_cur:.5f}")
    return VoteSignal(name=name, direction=Direction.RANGE,
                      weight=1.0, reliability=0.75, fired=False,
                      reason="stack EMA non aligné")


def _vote_weekly_open(
    cur: float,
    df_weekly: Optional[pd.DataFrame],
    df_daily: pd.DataFrame,
) -> VoteSignal:
    """
    Vote 3 — Weekly Open [C6].
    Reliability 0.80 (cache W natif) ou 0.60 (resample fallback).
    """
    name = "weekly_open"
    try:
        if df_weekly is not None and not df_weekly.empty:
            wo_price = float(df_weekly['Open'].iloc[-1])
            source, rel = "cache_W", 0.80
        else:
            w_open = df_daily['Open'].resample('W-MON').first().dropna()
            if w_open.empty:
                return VoteSignal(name=name, direction=Direction.RANGE,
                                  weight=1.0, reliability=0.60, fired=False,
                                  reason="Weekly Open indisponible")
            wo_price = float(w_open.iloc[-1])
            source, rel = "resample_fallback", 0.60

        if np.isnan(wo_price):
            return VoteSignal(name=name, direction=Direction.RANGE,
                              weight=1.0, reliability=rel, fired=False,
                              reason="Weekly Open NaN")
        if cur > wo_price:
            return VoteSignal(name=name, direction=Direction.BULLISH,
                              weight=1.0, reliability=rel, fired=True,
                              reason=f"cur={cur:.5f} > wo={wo_price:.5f} [{source}]")
        if cur < wo_price:
            return VoteSignal(name=name, direction=Direction.BEARISH,
                              weight=1.0, reliability=rel, fired=True,
                              reason=f"cur={cur:.5f} < wo={wo_price:.5f} [{source}]")
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=1.0, reliability=rel, fired=False,
                          reason=f"cur == wo={wo_price:.5f} [{source}]")

    except (TypeError, AttributeError, KeyError) as exc:
        _log.debug("_vote_weekly_open ignoré : %s", exc)
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=1.0, reliability=0.0, fired=False,
                          reason=f"exception: {exc}")


def _vote_prev_midpoint(
    h: pd.Series,
    lo: pd.Series,
    c: pd.Series,
    instrument: str = '',
) -> VoteSignal:
    """
    Vote 4 — Close J-1 vs midpoint J-1.  [V5-3]

    Changements V5.0 vs V4 :
      · Poids réduit à 0.5 (vs 1.0) — vote binaire faible.
      · Conditionné au volume J-1 > MA20 quand disponible.
        - Si volume confirmé   → reliability 0.65, fired selon direction.
        - Si volume < MA20     → fired=False, reliability 0.0.
          Le vote ne se prononce pas sans conviction de marché.
        - Si volume indisponible (indices ou colonne absente)
          → reliability 0.50, vote binaire comme V4 (dégradé accepté).

    Raisonnement institutionnel :
      Un close au-dessus du midpoint sans volume = bruit de fin de séance.
      Conditionner au volume filtre les faux signaux de faible liquidité
      (asia session, jours fériés, gaps d'ouverture).
    """
    name = "prev_midpoint"

    if len(c) < 2:
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=0.5, reliability=0.0, fired=False,
                          reason="données insuffisantes (<2 bougies)")

    mid    = (float(h.iloc[-2]) + float(lo.iloc[-2])) / 2
    prev_c = float(c.iloc[-2])
    bull   = prev_c > mid

    # ── Tentative de conditionnement au volume ────────────────────────────────
    is_index       = instrument in INDICES
    has_vol_col    = 'Volume' in c.index.names or hasattr(c, 'name')   # sera testé via df
    # On reçoit une pd.Series : le volume est dans le DataFrame parent.
    # On accède au volume via un attribut injecté si disponible (voir trend_daily).
    # Convention : on passe `vol_series` séparément — voir signature trend_daily.
    # Ce vote reçoit `lo` qui est df['Low'] ; le volume n'est pas directement accessible
    # depuis ici sans refactoring du caller. Solution propre : paramètre explicite.
    # → voir l'appel dans trend_daily ci-dessous.

    direction = Direction.BULLISH if bull else Direction.BEARISH
    reason    = f"close_j1={prev_c:.5f} {'>' if bull else '<='} mid_j1={mid:.5f} [no_vol_filter]"
    return VoteSignal(name=name, direction=direction,
                      weight=0.5, reliability=0.50, fired=True,
                      reason=reason)


def _vote_prev_midpoint_with_volume(
    h: pd.Series,
    lo: pd.Series,
    c: pd.Series,
    vol: Optional[pd.Series],
    instrument: str = '',
) -> VoteSignal:
    """
    [V5-3] Version complète avec conditionnement volume.
    Appelée par trend_daily quand le DataFrame contient 'Volume'.

    Logique :
      vol_j1  = volume de la bougie J-1
      vol_ma  = moyenne mobile 20 périodes du volume (hors dernière bougie)
      Condition : vol_j1 > vol_ma  → vote actif
                  vol_j1 <= vol_ma → vote silencieux (fired=False)
    """
    name = "prev_midpoint"

    if len(c) < 2:
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=0.5, reliability=0.0, fired=False,
                          reason="données insuffisantes (<2 bougies)")

    mid    = (float(h.iloc[-2]) + float(lo.iloc[-2])) / 2
    prev_c = float(c.iloc[-2])
    bull   = prev_c > mid

    # ── Pas de volume fiable (indices ou colonne vide) ────────────────────────
    is_index = instrument in INDICES
    if is_index or vol is None or vol.empty:
        direction = Direction.BULLISH if bull else Direction.BEARISH
        return VoteSignal(
            name=name, direction=direction,
            weight=0.5, reliability=0.50, fired=True,
            reason=(
                f"close_j1={prev_c:.5f} {'>' if bull else '<='} mid_j1={mid:.5f} "
                f"[no_vol — {'index' if is_index else 'unavailable'}]"
            ),
        )

    # ── Conditionnement volume ────────────────────────────────────────────────
    try:
        # MA20 calculée sur les bougies fermées (on exclut la dernière — incomplète)
        vol_series  = vol.iloc[:-1]
        if len(vol_series) < 20:
            # Pas assez d'historique pour la MA20 → dégradé sans filtre
            direction = Direction.BULLISH if bull else Direction.BEARISH
            return VoteSignal(
                name=name, direction=direction,
                weight=0.5, reliability=0.50, fired=True,
                reason=f"close_j1={prev_c:.5f} [vol_history<20, no_filter]",
            )

        vol_j1  = float(vol.iloc[-2])
        vol_ma  = float(vol_series.rolling(20).mean().iloc[-1])

        if np.isnan(vol_j1) or np.isnan(vol_ma) or vol_ma <= 0:
            direction = Direction.BULLISH if bull else Direction.BEARISH
            return VoteSignal(
                name=name, direction=direction,
                weight=0.5, reliability=0.50, fired=True,
                reason=f"close_j1={prev_c:.5f} [vol_NaN, no_filter]",
            )

        vol_ratio = vol_j1 / vol_ma

        if vol_ratio <= 1.0:
            # Volume insuffisant → vote silencieux
            return VoteSignal(
                name=name, direction=Direction.RANGE,
                weight=0.5, reliability=0.0, fired=False,
                reason=(
                    f"volume J-1 < MA20 (ratio={vol_ratio:.2f}) — "
                    f"close_j1={prev_c:.5f} ignoré (bruit faible liquidité)"
                ),
            )

        # Volume confirmé → vote actif avec reliability boostée
        # Plus le volume est fort, plus on est confiant (plafonné à 0.80)
        reliability = min(0.80, 0.65 + (vol_ratio - 1.0) * 0.05)
        direction   = Direction.BULLISH if bull else Direction.BEARISH
        return VoteSignal(
            name=name, direction=direction,
            weight=0.5, reliability=reliability, fired=True,
            reason=(
                f"close_j1={prev_c:.5f} {'>' if bull else '<='} mid_j1={mid:.5f} "
                f"[vol_ratio={vol_ratio:.2f} > 1.0, rel={reliability:.2f}]"
            ),
        )

    except (TypeError, ValueError, IndexError) as exc:
        _log.debug("_vote_prev_midpoint_with_volume erreur volume : %s", exc)
        direction = Direction.BULLISH if bull else Direction.BEARISH
        return VoteSignal(
            name=name, direction=direction,
            weight=0.5, reliability=0.50, fired=True,
            reason=f"close_j1={prev_c:.5f} [vol_error: {exc}]",
        )


def _vote_ema50_slope(e50: pd.Series, atr_val: float, threshold: float = 0.05) -> VoteSignal:
    """
    Vote 5 — Pente EMA50 normalisée par ATR [C10].
    Poids 1.0 · Reliability 0.70
    """
    name = "ema50_slope"
    if len(e50) < 6 or atr_val <= 0:
        return VoteSignal(name=name, direction=Direction.RANGE,
                          weight=1.0, reliability=0.70, fired=False,
                          reason=f"données insuffisantes (len={len(e50)}, atr={atr_val:.5f})")
    slope_ratio = float(e50.iloc[-1] - e50.iloc[-6]) / atr_val
    if slope_ratio > threshold:
        return VoteSignal(name=name, direction=Direction.BULLISH,
                          weight=1.0, reliability=0.70, fired=True,
                          reason=f"slope_ratio={slope_ratio:.3f} > {threshold}")
    if slope_ratio < -threshold:
        return VoteSignal(name=name, direction=Direction.BEARISH,
                          weight=1.0, reliability=0.70, fired=True,
                          reason=f"slope_ratio={slope_ratio:.3f} < -{threshold}")
    return VoteSignal(name=name, direction=Direction.RANGE,
                      weight=1.0, reliability=0.70, fired=False,
                      reason=f"slope_ratio={slope_ratio:.3f} dans zone neutre ±{threshold}")


# ══════════════════════════════════════════════════════════════════════════════
# [V5-4] AGRÉGATEUR — séparation collecte / décision + circuit breaker
# ══════════════════════════════════════════════════════════════════════════════

MIN_RELIABLE_SCORE = 1.5  # score pondéré minimum pour décision directionnelle


def _aggregate_votes(votes: tuple, atr_val: float) -> DailyTrendResult:
    """
    Contribution effective = weight × reliability (seulement si fired=True).

    Circuit breaker : si max(bull, bear) < MIN_RELIABLE_SCORE → Range forcé.
    Évite toute décision sur un unique vote faible ou non confirmé.

    Force (strength) : ratio contribution / max_théorique → échelle 35–90.
    Plus conservateur que V4 : un seul vote fort ne peut pas atteindre 90.
    """
    bull_score = sum(v.weight * v.reliability
                     for v in votes if v.fired and v.direction == Direction.BULLISH)
    bear_score = sum(v.weight * v.reliability
                     for v in votes if v.fired and v.direction == Direction.BEARISH)
    max_possible   = sum(v.weight * v.reliability for v in votes)
    winning_score  = max(bull_score, bear_score)
    min_votes_met  = winning_score >= MIN_RELIABLE_SCORE

    if not min_votes_met or bull_score == bear_score:
        return DailyTrendResult(
            direction=Direction.RANGE, strength=35, atr_val=atr_val,
            bull_score=bull_score, bear_score=bear_score,
            votes=votes, min_votes_met=min_votes_met,
        )

    direction = Direction.BULLISH if bull_score > bear_score else Direction.BEARISH
    ratio     = winning_score / max_possible if max_possible > 0 else 0.0
    strength  = int(min(90, max(35, ratio * 112)))

    return DailyTrendResult(
        direction=direction, strength=strength, atr_val=atr_val,
        bull_score=bull_score, bear_score=bear_score,
        votes=votes, min_votes_met=min_votes_met,
    )


# ══════════════════════════════════════════════════════════════════════════════
# [V5-5] POINT D'ENTRÉE — trend_daily API rétrocompatible
# ══════════════════════════════════════════════════════════════════════════════

def trend_daily(
    df: pd.DataFrame,
    df_weekly: Optional[pd.DataFrame] = None,
    instrument: str = '',
) -> DailyTrendResult:
    """
    Analyse directionnelle Daily — 5 votes indépendants.

    Rétrocompat :
        trend, score, atr = trend_daily(df, df_weekly)

    Nouveau :
        result = trend_daily(df, df_weekly, instrument)
        result.summary()   — audit trail complet
        result.bull_votes  — votes haussiers déclenchés
    """
    h   = df['High']
    lo  = df['Low']
    c   = df['Close']
    atr_val = float(_atr(h, lo, c, 14).iloc[-1])

    if len(df) < 60:
        guard = VoteSignal(
            name="guard_min_bars", direction=Direction.RANGE,
            weight=0.0, reliability=1.0, fired=False,
            reason=f"données insuffisantes ({len(df)} bougies < 60)",
        )
        return DailyTrendResult(
            direction=Direction.RANGE, strength=0, atr_val=atr_val,
            bull_score=0.0, bear_score=0.0,
            votes=(guard,), min_votes_met=False,
        )

    cur     = float(c.iloc[-1])
    e50     = _ema(c, 50)
    e21     = float(_ema(c, 21).iloc[-1])
    e50_cur = float(e50.iloc[-1])

    # Volume disponible ? (pas sur les indices)
    vol_series: Optional[pd.Series] = (
        df['Volume']
        if 'Volume' in df.columns and instrument not in INDICES
        else None
    )

    votes = tuple([
        _vote_swing_structure(h, lo),
        _vote_ema_stack(cur, e21, e50_cur),
        _vote_weekly_open(cur, df_weekly, df),
        _vote_prev_midpoint_with_volume(h, lo, c, vol_series, instrument),  # [V5-3]
        _vote_ema50_slope(e50, atr_val),
    ])

    result = _aggregate_votes(votes, atr_val)
    _log.debug("trend_daily %s | %s", instrument or "?", result.summary())
    return result


# ===================== TENDANCES PAR TF (inchangées) =====================

def trend_macro(df: pd.DataFrame, tf: str):
    if len(df) < 50:
        atr_val = float(_atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]) if len(df) >= 15 else 0.0
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

    else:  # Weekly
        if len(df) < 200:
            _log.warning("Weekly SMA200 indisponible (%d bougies) — signal neutralisé", len(df))
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


def trend_4h(df: pd.DataFrame, df_daily: pd.DataFrame = None):
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

    try:
        if df_daily is not None and not df_daily.empty:
            today_open = float(df_daily['Open'].iloc[-1])
        else:
            dates      = pd.to_datetime(df.index).normalize()
            today_open = float(df[dates == dates[-1]]['Open'].iloc[0])
        score += 1 if cur > today_open else -1
    except Exception as exc:
        _log.debug("Daily Open Vote ignoré : %s", exc)

    s        = abs(score)
    strength = 90 if s == 3 else 70 if s >= 1 else 40
    return ('Bullish' if score > 0 else 'Bearish' if score < 0 else 'Range'), strength, atr_val


def trend_intraday(df: pd.DataFrame, instrument: str = ''):
    h   = df['High']
    lo  = df['Low']
    c   = df['Close']
    atr_val = float(_atr(h, lo, c, 14).iloc[-1])

    if len(df) < 70:
        return 'Range', 0, atr_val

    cur    = float(c.iloc[-1])
    period = 50
    lag    = period // 2

    e9      = float(_ema(c, 9).iloc[-1])
    e21     = float(_ema(c, 21).iloc[-1])
    e50     = _ema(c, period)
    e50_cur = float(e50.iloc[-1])

    src_adj = c + (c - c.shift(lag))
    zlema   = float(src_adj.ewm(span=period, adjust=False).mean().iloc[-1])

    rsi_val  = float(_rsi(c, 14).iloc[-1])
    macd     = _ema(c, 12) - _ema(c, 26)
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
    c    = df['Close']
    e50  = _ema(c, 50)
    above = c > e50
    for i in range(len(above) - 1, 0, -1):
        if above.iloc[i] != above.iloc[i - 1]:
            age = len(above) - 1 - i
            return str(age) if age > 0 else '0'
    return f'>{len(above)}'


# ===================== [C1] API + CACHE THREAD-SAFE =====================

def fetch_candles(instrument: str, granularity: str, count: int,
                  account_id: str, access_token: str) -> pd.DataFrame:
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
        rows = []
        for c in candles:
            if not c.get('complete'):
                continue
            try:
                rows.append({
                    'date':   c['time'],
                    'Open':   float(c['mid']['o']),
                    'High':   float(c['mid']['h']),
                    'Low':    float(c['mid']['l']),
                    'Close':  float(c['mid']['c']),
                    'Volume': float(c.get('volume', 0)),
                })
            except (KeyError, ValueError) as parse_err:
                _log.warning("Parse candle error %s: %s", instrument, parse_err)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        return df

    except requests.exceptions.Timeout:
        _log.warning("Timeout %s %s", instrument, granularity)
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        _log.warning("Request error %s %s: %s", instrument, granularity, e)
        return pd.DataFrame()
    except Exception as e:
        _log.error("Unexpected error fetch_candles %s %s: %s", instrument, granularity, e)
        return pd.DataFrame()


def fetch_cached(inst: str, gran: str, cnt: int, acc: str, tok: str) -> pd.DataFrame:
    key = (inst, gran, cnt)
    now = datetime.now(timezone.utc)
    entry = _data_cache.get(key)
    if entry is not None:
        ts, df = entry
        if (now - ts).total_seconds() < _CACHE_TTL:
            return df
    df = fetch_candles(inst, gran, cnt, acc, tok)
    with _cache_lock:
        _data_cache[key] = (datetime.now(timezone.utc), df)
    return df


def fetch_all_data(instrument: str, account_id: str, access_token: str):
    specs = {
        'M':   ('M',   150),
        'W':   ('W',   250),
        'D':   ('D',   300),
        '4H':  ('H4',  300),
        '1H':  ('H1',  300),
        '15m': ('M15', 300),
    }
    min_bars = {'M': 100, 'W': 50, 'D': 60, '4H': 60, '1H': 70, '15m': 70}
    cache = {}
    for tf, (gran, count) in specs.items():
        df = fetch_cached(instrument, gran, count, account_id, access_token)
        if df.empty:
            _log.warning("%s — données vides TF=%s (%s)", instrument, tf, gran)
            return None
        if len(df) < min_bars[tf]:
            _log.warning("%s — TF=%s : %d bougies < minimum %d",
                         instrument, tf, len(df), min_bars[tf])
            return None
        cache[tf] = df
    return cache


# ===================== SCORING MTF =====================

def score_mtf(trends: dict, scores: dict):
    weights = {'M': 5.0, 'W': 4.0, 'D': 4.0, '4H': 2.5, '1H': 1.5, '15m': 1.0}
    total   = sum(weights.values())

    def weighted(direction: str) -> float:
        return sum(
            weights[tf] * (scores[tf] / 100)
            for tf in trends
            if trends[tf].startswith(direction)
        )

    w_bull = weighted('Bullish') + weighted('Retracement Bull') * 0.5
    w_bear = weighted('Bearish') + weighted('Retracement Bear') * 0.5

    if w_bull > w_bear:
        raw_score, direction = (w_bull / total) * 100, 'Bullish'
    elif w_bear > w_bull:
        raw_score, direction = (w_bear / total) * 100, 'Bearish'
    else:
        return 'Range', 0

    bonus = 0
    if (trends.get('M') == trends.get('W') == 'Bullish' or
            trends.get('M') == trends.get('W') == 'Bearish'):
        bonus += 15
    if (trends.get('D') == trends.get('4H') == 'Bullish' or
            trends.get('D') == trends.get('4H') == 'Bearish'):
        bonus += 10

    return direction, min(100, raw_score + bonus)


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


# ===================== ANALYSE PRINCIPALE =====================

def analyze_pair(pair: str, account_id: str, access_token: str):
    try:
        cache = fetch_all_data(pair, account_id, access_token)
        if cache is None:
            return None

        trends, scores, atrs = {}, {}, {}

        for tf in ('M', 'W'):
            t, s, a = trend_macro(cache[tf], tf)
            trends[tf], scores[tf], atrs[tf] = t, s, a

        # [V5-5] trend_daily reçoit maintenant l'instrument pour le filtre volume
        t, s, a = trend_daily(cache['D'], cache['W'], instrument=pair)
        trends['D'], scores['D'], atrs['D'] = t, s, a

        t, s, a = trend_4h(cache['4H'], cache['D'])
        trends['4H'], scores['4H'], atrs['4H'] = t, s, a

        for tf in ('1H', '15m'):
            t, s, a = trend_intraday(cache[tf], pair)
            trends[tf], scores[tf], atrs[tf] = t, s, a

        mtf_dir, mtf_score = score_mtf(trends, scores)
        age = trend_age_daily(cache['D'])

        if mtf_dir in ('Bullish', 'Bearish'):
            oppose  = 'Bearish' if mtf_dir == 'Bullish' else 'Bullish'
            aligned = sum(1 for t in trends.values() if t == mtf_dir)
            opposed = sum(1 for t in trends.values() if t == oppose)
            nc      = aligned - opposed
        else:
            nc = 0

        row = {
            'Paire':      pair.replace('_', '/'),
            'M':          trends['M'],
            'W':          trends['W'],
            'D':          trends['D'],
            '4H':         trends['4H'],
            '1H':         trends['1H'],
            '15m':        trends['15m'],
            'MTF':        f"{mtf_dir} ({mtf_score:.0f}%)" if mtf_dir != 'Range' else 'Range',
            '_mtf_score': mtf_score,
            '_mtf_dir':   mtf_dir,
            'NC':         nc,
            'Age D1':     age,
            'ATR Daily':  _fmt_atr(atrs['D']),
            'ATR H4':     _fmt_atr(atrs['4H']),
            'ATR H1':     _fmt_atr(atrs['1H']),
            'ATR 15m':    _fmt_atr(atrs['15m']),
        }
        return row

    except Exception as e:
        _log.error("analyze_pair %s — exception inattendue : %s", pair, e, exc_info=True)
        return None


def analyze_all(account_id: str, access_token: str):
    results, errors = [], []
    progress = st.progress(0)
    status   = st.empty()
    total    = len(INSTRUMENTS)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(analyze_pair, inst, account_id, access_token): inst
            for inst in INSTRUMENTS
        }
        done = 0
        for future in as_completed(futures):
            inst = futures[future]
            done += 1
            progress.progress(done / total)
            status.text(f"GPS ({done}/{total}) — {inst.replace('_', '/')} ✓")
            try:
                row = future.result()
                if row:
                    results.append(row)
                else:
                    errors.append(inst)
            except Exception as e:
                errors.append(inst)
                _log.error("Future error %s: %s", inst, e)

    progress.empty()
    status.empty()

    if errors:
        st.warning(
            f"⚠️ {len(errors)} paire(s) non analysée(s) "
            f"(données insuffisantes ou erreur API) : "
            f"{', '.join(e.replace('_', '/') for e in errors)}"
        )

    if not results:
        return pd.DataFrame()

    scores_list = [r['_mtf_score'] for r in results]
    nc_list     = [r['NC']         for r in results]
    grades      = grade_hybrid(scores_list, nc_list)
    for r, g in zip(results, grades):
        r['Quality'] = g

    df = pd.DataFrame(results)
    df.drop(columns=['_mtf_score', '_mtf_dir'], inplace=True)
    return df


# ===================== PDF =====================

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
    nc_rgb = {
        (5, 99):  (46,  204, 113, 255, 255, 255),
        (3,  4):  (39,  174,  96, 255, 255, 255),
        (1,  2):  (241, 196,  15,   0,   0,   0),
        (-99, 0): (231,  76,  60, 255, 255, 255),
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
                    fc = (231, 76,  60)
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

    except Exception as e:
        _log.error("PDF generation error: %s", e, exc_info=True)
        buf2 = BytesIO()
        if FPDF:
            fallback = FPDF()
            fallback.add_page()
            fallback.set_font("Helvetica", "B", 12)
            fallback.cell(0, 10, "PDF Generation Error — check logs", ln=True)
            out2 = fallback.output(dest='S')
            buf2.write(out2.encode('latin-1') if isinstance(out2, str) else bytes(out2))
        buf2.seek(0)
        return buf2


# ===================== UI =====================

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
    except (KeyError, Exception):
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

    grade_order  = ['A+', 'A', 'B+', 'B']
    df['Quality'] = pd.Categorical(df['Quality'], categories=grade_order, ordered=True)
    df = df.sort_values(['Quality', 'NC', 'MTF'], ascending=[True, False, False])

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

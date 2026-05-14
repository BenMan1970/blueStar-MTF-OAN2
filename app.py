"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          BLUESTAR HEDGE FUND GPS  —  V4.1  (Signal-Accurate Edition)       ║
║                                                                              ║
║  Patches V4.0 (P1–P15) conservés intégralement.                            ║
║                                                                              ║
║  Corrections V4.1 :                                                         ║
║   [C1]  Cache thread-safe custom (Lock+TTL) — remplace @st.cache_data      ║
║   [C2]  trend_age_daily off-by-one corrigé + retour '>N' si aucun flip     ║
║   [C3]  RSI fillna(100.0) — marché purement directionnel (loss=0)           ║
║   [C4]  Vote 3 typed except + NaN guard explicite sur wo_price              ║
║   [C5]  Daily Open 4H depuis cache D OANDA (était normalize() UTC)          ║
║   [C6]  Weekly Open depuis cache W natif (était resample sur Daily)         ║
║   [C7]  SMA200 fallback Weekly → Range si <200 bougies (était Bearish)      ║
║   [C8]  Grades absolus reproductibles (remplace percentiles relatifs)        ║
║   [C9]  Recency filter pivots swing D1 (MAX_PIVOT_AGE = 50 bougies)        ║
║   [C10] Strength intraday normalisé ATR (était /cur*1000 quasi-constant)    ║
║   [C11] Zone neutre ±0.1×ATR sur comparaisons EMA Macro (anti flip-flop)   ║
║   [C12] Bonus MTF sur tendances pures uniquement (pas Retracement)          ║
║   [C13] Protection double-clic bouton analyse (session_state lock)          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import threading
import logging
from datetime import datetime, timezone
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

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
st.set_page_config(page_title="Bluestar GPS V4.1", page_icon="🧭", layout="wide")

OANDA_API_URL = "https://api-fxpractice.oanda.com"

INSTRUMENTS = [
    # 28 paires Forex
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'USD_CAD', 'AUD_USD', 'NZD_USD',
    'EUR_GBP', 'EUR_JPY', 'EUR_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_NZD',
    'GBP_JPY', 'GBP_CHF', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD',
    'AUD_JPY', 'AUD_CAD', 'AUD_CHF', 'AUD_NZD', 'CAD_JPY', 'CAD_CHF', 'CHF_JPY',
    'NZD_JPY', 'NZD_CAD', 'NZD_CHF',
    # 4 indices + 1 métal
    'DE30_EUR', 'XAU_USD', 'SPX500_USD', 'NAS100_USD', 'US30_USD',
]

# Instruments sans volume fiable sur OANDA
INDICES = {'DE30_EUR', 'SPX500_USD', 'NAS100_USD', 'US30_USD', 'XAU_USD'}

TREND_COLORS = {
    'Bullish':          '#2ecc71',
    'Bearish':          '#e74c3c',
    'Retracement Bull': '#7dcea0',
    'Retracement Bear': '#f1948a',
    'Range':            '#95a5a6',
}

# Délai max données (minutes) avant alerte fraîcheur (P12)
DATA_MAX_AGE_MIN = 15

# [C9] Recency filter — max bougies D1 de lookback pour pivots swing
MAX_PIVOT_AGE = 50

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
# Remplace @st.cache_data (non thread-safe en écriture concurrente).
# Pattern : lecture rapide hors lock → fetch hors lock → écriture sous lock.
_data_cache: dict = {}
_cache_lock  = threading.Lock()
_CACHE_TTL   = 600  # secondes (10 min, identique à V4.0)

def _cache_clear() -> None:
    """Vide intégralement le cache de données (bouton sidebar)."""
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
    """ATR Wilder via EWM alpha=1/n (P13 : _true_range factorisé)."""
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """
    RSI Wilder.
    [C3] fillna(100.0) : si loss=0 (marché purement haussier sur n périodes),
         le RSI vaut 100 — plus de NaN silencieux qui force un faux 'Range'.
    """
    d    = close.diff()
    gain = d.where(d > 0, 0.0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    rs   = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)

def _dmi(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    """DI+, DI- Wilder (P13 : _true_range factorisé)."""
    tr    = _true_range(high, low, close)
    atr_s = tr.ewm(alpha=1 / n, adjust=False).mean()
    up    = high.diff()
    dn    = -low.diff()
    pdm   = up.where((up > dn) & (up > 0), 0.0)
    mdm   = dn.where((dn > up) & (dn > 0), 0.0)
    pdi   = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr_s.replace(0, np.nan)
    mdi   = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / atr_s.replace(0, np.nan)
    return float(pdi.iloc[-1]), float(mdi.iloc[-1])

# ===================== [P15] FORMAT ATR — 3 paliers =====================
def _fmt_atr(val: float) -> str:
    """
    [P15] Affichage adaptatif.
    [C3 addon] Retourne 'N/A' si val ≤ 0 ou NaN (données manquantes).
    """
    if not val or np.isnan(val) or val <= 0:
        return 'N/A'
    if val >= 10: return f"{val:.2f}"
    if val >= 1:  return f"{val:.3f}"
    return f"{val:.4f}"

# ===================== TENDANCES PAR TF =====================

def trend_macro(df: pd.DataFrame, tf: str):
    """
    Monthly / Weekly — EMA50 vs EMA100(M) ou EMA50 vs SMA200(W).

    [P3]  Guard EMA100 : min 100 bougies pour Monthly.
    [C7]  SMA200 indisponible Weekly → Range (supprime le bug Bearish systématique).
    [C11] Zone neutre ±0.1×ATR sur toutes les comparaisons EMA (anti flip-flop
          au tick près entre deux analyses rapprochées).
    """
    if len(df) < 50:
        atr_val = float(_atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]) if len(df) >= 15 else 0.0
        return 'Range', 0, atr_val

    c       = df['Close']
    h, l    = df['High'], df['Low']
    atr_val = float(_atr(h, l, c, 14).iloc[-1])
    band    = atr_val * 0.1   # [C11] zone neutre = 10 % de l'ATR du TF

    e50 = _ema(c, 50)

    if tf == 'M':
        if len(df) < 100:    # [P3]
            return 'Range', 0, atr_val
        e100 = _ema(c, 100)
        ref  = float(e100.iloc[-1])
        cur  = float(e50.iloc[-1])
        if ref == 0:
            return 'Range', 40, atr_val
        gap = abs(cur - ref) / ref * 100
        s   = 75 if gap > 0.3 else 60
        # [C11]
        if   cur > ref + band: return 'Bullish', s, atr_val
        elif cur < ref - band: return 'Bearish', s, atr_val
        return 'Range', 40, atr_val

    else:  # Weekly
        # [C7] SMA200 requiert ≥ 200 bougies — sinon neutraliser le signal
        if len(df) < 200:
            _log.warning("Weekly SMA200 indisponible (%d bougies) — signal neutralisé", len(df))
            return 'Range', 40, atr_val

        s200    = _sma(c, 200)
        cur50   = float(e50.iloc[-1])
        ref200  = float(s200.iloc[-1])
        prev50  = float(e50.iloc[-2])
        prev200 = float(s200.iloc[-2])
        cross   = (prev50 <= prev200 < cur50) or (prev50 >= prev200 > cur50)
        # [C11]
        if   cur50 > ref200 + band: return 'Bullish', (90 if cross else 75), atr_val
        elif cur50 < ref200 - band: return 'Bearish', (90 if cross else 75), atr_val
        return 'Range', 40, atr_val


def trend_daily(df: pd.DataFrame, df_weekly: pd.DataFrame = None):
    """
    5 votes indépendants — max 6 points (swing HH/HL compte double).

    [P4]  Tie-breaker : égalité → Range.
    [P9]  Weekly Open — [C6] remplacé par cache W natif OANDA
          (df_weekly['Open'].iloc[-1]) pour fiabilité maximale.
          Fallback resample si df_weekly indisponible.
    [P10] NaN guards sur toutes les comparaisons flottantes.
    [C4]  Vote 3 : typed except + NaN guard explicite sur wo_price.
    [C9]  Vote 1 : recency filter sur pivots (MAX_PIVOT_AGE = 50 bougies).
    """
    h, l, c = df['High'], df['Low'], df['Close']
    atr_val = float(_atr(h, l, c, 14).iloc[-1])

    if len(df) < 60:
        return 'Range', 0, atr_val

    cur = float(c.iloc[-1])
    vb = vbr = 0

    # Vote 1 — structure swing (double poids)
    wing = 5
    sh, sl = [], []
    for i in range(wing, len(h) - wing):
        if h.iloc[i] == h.iloc[i - wing:i + wing + 1].max(): sh.append(i)
        if l.iloc[i] == l.iloc[i - wing:i + wing + 1].min(): sl.append(i)

    # [C9] Recency filter : exclure les pivots trop anciens
    min_idx = max(0, len(h) - MAX_PIVOT_AGE)
    sh = [i for i in sh if i >= min_idx]
    sl = [i for i in sl if i >= min_idx]

    if len(sh) >= 2 and len(sl) >= 2:
        hh = h.iloc[sh[-1]] > h.iloc[sh[-2]]
        hl = l.iloc[sl[-1]] > l.iloc[sl[-2]]
        lh = h.iloc[sh[-1]] < h.iloc[sh[-2]]
        ll = l.iloc[sl[-1]] < l.iloc[sl[-2]]
        if hh and hl:   vb  += 2
        elif lh and ll: vbr += 2

    # Vote 2 — EMA21 / EMA50 stack
    e21     = float(_ema(c, 21).iloc[-1])
    e50     = _ema(c, 50)
    e50_cur = float(e50.iloc[-1])
    if not (np.isnan(e21) or np.isnan(e50_cur)):
        if   cur > e21 > e50_cur: vb  += 1
        elif cur < e21 < e50_cur: vbr += 1

    # Vote 3 — [C6] Weekly Open natif depuis cache W (plus fiable que resample Daily)
    try:
        if df_weekly is not None and not df_weekly.empty:
            wo_price = float(df_weekly['Open'].iloc[-1])
        else:
            # Fallback : resample sur données Daily (moins précis, week-ends / fériés)
            w_open   = df['Open'].resample('W-MON').first().dropna()
            wo_price = float(w_open.iloc[-1]) if not w_open.empty else float('nan')
        # [C4] NaN guard explicite avant comparaison
        if not np.isnan(wo_price):
            if   cur > wo_price: vb  += 1
            elif cur < wo_price: vbr += 1
    except (TypeError, AttributeError, KeyError) as e:
        _log.debug("Vote Weekly Open ignoré : %s", e)

    # Vote 4 — close J-1 vs midpoint J-1
    if len(df) >= 2:
        mid = (float(h.iloc[-2]) + float(l.iloc[-2])) / 2
        if float(c.iloc[-2]) > mid: vb  += 1
        else:                       vbr += 1

    # Vote 5 — pente EMA50 normalisée par ATR
    if len(e50) >= 6 and atr_val > 0:
        slope = float(e50.iloc[-1] - e50.iloc[-6])
        if   slope / atr_val >  0.05: vb  += 1
        elif slope / atr_val < -0.05: vbr += 1

    # [P4] Résolution sans biais
    if vb > vbr:
        if vb >= 5: return 'Bullish', 90, atr_val
        if vb >= 3: return 'Bullish', 70, atr_val
    elif vbr > vb:
        if vbr >= 5: return 'Bearish', 90, atr_val
        if vbr >= 3: return 'Bearish', 70, atr_val
    return 'Range', 35, atr_val


def trend_4h(df: pd.DataFrame, df_daily: pd.DataFrame = None):
    """
    3 votes : EMA50 · DMI · Daily Open.

    [P10] NaN guards.
    [C5]  Vote 3 : Daily Open depuis cache D OANDA (session 22h UTC).
          Remplace normalize() UTC qui pointait vers minuit et non la vraie
          ouverture de session OANDA — particulièrement faux en London session.
    """
    h, l, c = df['High'], df['Low'], df['Close']
    atr_val = float(_atr(h, l, c, 14).iloc[-1])

    if len(df) < 60:
        return 'Range', 0, atr_val

    cur   = float(c.iloc[-1])
    score = 0

    # Vote 1 — EMA50
    e50_cur = float(_ema(c, 50).iloc[-1])
    if not np.isnan(e50_cur):
        score += 1 if cur > e50_cur else -1

    # Vote 2 — DMI
    pdi, mdi = _dmi(h, l, c)
    if not (np.isnan(pdi) or np.isnan(mdi)):
        score += 1 if pdi > mdi else -1

    # Vote 3 — [C5] Daily Open OANDA depuis cache D
    try:
        if df_daily is not None and not df_daily.empty:
            # Vraie ouverture de session OANDA (bougie Daily la plus récente)
            today_open = float(df_daily['Open'].iloc[-1])
        else:
            # Fallback : premier Open du jour UTC (moins précis)
            dates      = pd.to_datetime(df.index).normalize()
            today_open = float(df[dates == dates[-1]]['Open'].iloc[0])
        score += 1 if cur > today_open else -1
    except Exception:
        pass

    s = abs(score)
    strength = 90 if s == 3 else 70 if s >= 1 else 40
    return ('Bullish' if score > 0 else 'Bearish' if score < 0 else 'Range'), strength, atr_val


def trend_intraday(df: pd.DataFrame, instrument: str = ''):
    """
    Intraday H1/M15 — ZLEMA · EMA stack · momentum RSI/MACD · volume (forex).

    [P14] ZLEMA lag = period//2 = 25.
    [P10] NaN guards.
    [C10] Strength normalisé par ATR : distance ZLEMA en multiples d'ATR.
          Remplace /cur*1000 qui produisait ~40 constant quelle que soit la paire.
    """
    h, l, c = df['High'], df['Low'], df['Close']
    atr_val = float(_atr(h, l, c, 14).iloc[-1])

    if len(df) < 70:
        return 'Range', 0, atr_val

    cur     = float(c.iloc[-1])
    period  = 50
    lag     = period // 2        # [P14] 25

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

    # [P10] NaN guards
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

    # [C10] Strength normalisé par ATR (1 ATR ≈ 65 de force, 2 ATR ≈ 90)
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
    """
    Nombre de bougies D1 depuis le dernier croisement Close/EMA50.

    [C2] Corrections :
      - range(len-1, 0, -1) au lieu de (len-2, 0, -1) — détecte le flip le plus récent.
      - Retourne '>N' (borne inférieure) si aucun croisement trouvé dans l'historique,
        plus trompeur que '300' brut.
    """
    if len(df) < 55:
        return 'N/A'
    c     = df['Close']
    e50   = _ema(c, 50)
    above = c > e50
    for i in range(len(above) - 1, 0, -1):    # [C2] -1 au lieu de -2
        if above.iloc[i] != above.iloc[i - 1]:
            age = len(above) - 1 - i
            return str(age) if age > 0 else '0'
    return f'>{len(above)}'    # [C2] borne inférieure explicite

# ===================== [C1] API + CACHE THREAD-SAFE =====================

def fetch_candles(instrument: str, granularity: str, count: int,
                  account_id: str, access_token: str) -> pd.DataFrame:
    """
    Récupère les bougies OANDA.
    [P1]  Session retry par thread (anti rate-limit 429).
    [P11] Log les erreurs avec niveau WARNING.
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
    """
    [C1] Cache thread-safe : remplace @st.cache_data (non atomique en cache miss concurrent).

    Stratégie :
      1. Lecture rapide hors lock (fast path — la plupart des appels).
      2. Cache miss → fetch_candles hors lock (bloquant, évite de pénaliser les autres).
      3. Écriture sous lock (atomique).

    Clé = (inst, gran, cnt) — tok exclu (sécurité + inutile pour la clé de cache).
    """
    key = (inst, gran, cnt)
    now = datetime.now(timezone.utc)

    # --- Lecture rapide (hors lock) ---
    entry = _data_cache.get(key)
    if entry is not None:
        ts, df = entry
        if (now - ts).total_seconds() < _CACHE_TTL:
            return df

    # --- Cache miss : fetch HTTP hors lock ---
    df = fetch_candles(inst, gran, cnt, acc, tok)

    # --- Écriture sous lock ---
    with _cache_lock:
        _data_cache[key] = (datetime.now(timezone.utc), df)

    return df


def fetch_all_data(instrument: str, account_id: str, access_token: str):
    """
    [P2] Granularités NATIVES OANDA pour W et M.
    Minimum de bougies garanti avant analyse (P3 pour Monthly).
    """
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
    """
    Score pondéré avec bonus additifs d'alignement.
    Poids : M=5 W=4 D=4 4H=2.5 1H=1.5 15m=1 → total=18.

    [C12] Bonus uniquement si M==W ou D==4H sont des tendances PURES
          (Bullish ou Bearish) — les Retracements alignés ne méritent pas
          un bonus de confirmation directionnelle forte.
    """
    WEIGHTS = {'M': 5.0, 'W': 4.0, 'D': 4.0, '4H': 2.5, '1H': 1.5, '15m': 1.0}
    TOTAL   = sum(WEIGHTS.values())   # 18.0

    def weighted(direction: str) -> float:
        return sum(
            WEIGHTS[tf] * (scores[tf] / 100)
            for tf in trends
            if trends[tf].startswith(direction)
        )

    w_bull = weighted('Bullish') + weighted('Retracement Bull') * 0.5
    w_bear = weighted('Bearish') + weighted('Retracement Bear') * 0.5

    if w_bull > w_bear:
        raw_score, direction = (w_bull / TOTAL) * 100, 'Bullish'
    elif w_bear > w_bull:
        raw_score, direction = (w_bear / TOTAL) * 100, 'Bearish'
    else:
        return 'Range', 0

    # [C12] Bonus sur tendances pures uniquement
    bonus = 0
    if (trends.get('M') == trends.get('W') == 'Bullish' or
            trends.get('M') == trends.get('W') == 'Bearish'):
        bonus += 15
    if (trends.get('D') == trends.get('4H') == 'Bullish' or
            trends.get('D') == trends.get('4H') == 'Bearish'):
        bonus += 10

    return direction, min(100, raw_score + bonus)


def grade_hybrid(scores_list: list) -> list:
    """
    [C8] Grading par seuils ABSOLUS — reproductible entre sessions.

    Remplace les percentiles relatifs qui rendaient les grades instables :
    un même score pouvait être A+ ou B+ selon combien de paires avaient
    échoué à l'API (population variable).

    Seuils :
      A+ : ≥ 68 (marché plat std<10) ou ≥ 55 (marché normal)
      A  : ≥ 48
      B+ : ≥ 35
      B  : < 35

    Le filtre std<10 (marché globalement plat) conserve l'exigence renforcée
    pour éviter les A+ en contexte de Range généralisé.
    """
    if not scores_list:
        return []

    arr     = np.array(scores_list, dtype=float)
    std_dev = float(np.std(arr))

    min_aplus = 68 if std_dev < 10 else 55

    grades = []
    for s in arr:
        if   s >= min_aplus: grades.append('A+')
        elif s >= 48:        grades.append('A')
        elif s >= 35:        grades.append('B+')
        else:                grades.append('B')
    return grades

# ===================== ANALYSE PRINCIPALE =====================

def analyze_pair(pair: str, account_id: str, access_token: str):
    """
    Analyse complète d'un instrument.

    [C5] trend_4h  reçoit cache['D'] → Daily Open OANDA (session 22h UTC).
    [C6] trend_daily reçoit cache['W'] → Weekly Open natif.
    [P8] ATR dédupliqué (retourné par chaque fonction tendance).
    """
    try:
        cache = fetch_all_data(pair, account_id, access_token)
        if cache is None:
            return None

        trends, scores, atrs = {}, {}, {}

        # Macro (M et W)
        for tf in ('M', 'W'):
            t, s, a = trend_macro(cache[tf], tf)
            trends[tf], scores[tf], atrs[tf] = t, s, a

        # Daily — [C6] Weekly Open natif
        t, s, a = trend_daily(cache['D'], cache['W'])
        trends['D'], scores['D'], atrs['D'] = t, s, a

        # 4H — [C5] Daily Open OANDA
        t, s, a = trend_4h(cache['4H'], cache['D'])
        trends['4H'], scores['4H'], atrs['4H'] = t, s, a

        # Intraday H1 et 15m
        for tf in ('1H', '15m'):
            t, s, a = trend_intraday(cache[tf], pair)
            trends[tf], scores[tf], atrs[tf] = t, s, a

        mtf_dir, mtf_score = score_mtf(trends, scores)
        age = trend_age_daily(cache['D'])

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
    """
    Parallélisme limité à 5 workers.
    [P11] Compteur d'erreurs visible dans l'UI.
    """
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
    grades      = grade_hybrid(scores_list)
    for r, g in zip(results, grades):
        r['Quality'] = g

    df = pd.DataFrame(results)
    df.drop(columns=['_mtf_score', '_mtf_dir'], inplace=True)
    return df

# ===================== PDF (P6) =====================

def _safe_str(s: str) -> str:
    return s.encode('latin-1', errors='replace').decode('latin-1')


def create_pdf(df: pd.DataFrame) -> BytesIO:
    COLS = [
        'Paire', 'M', 'W', 'D', '4H', '1H', '15m',
        'MTF', 'Quality', 'Age D1',
        'ATR Daily', 'ATR H4', 'ATR H1', 'ATR 15m'
    ]
    WIDTHS = {
        'Paire': 22, 'M': 18, 'W': 18, 'D': 18, '4H': 18, '1H': 18, '15m': 18,
        'MTF': 32, 'Quality': 13, 'Age D1': 14,
        'ATR Daily': 18, 'ATR H4': 18, 'ATR H1': 16, 'ATR 15m': 16,
    }
    RH = 5.5

    def _cell_text(val: str) -> str:
        return val if _FPDF2 else _safe_str(val)

    try:
        if FPDF is None:
            raise RuntimeError("fpdf / fpdf2 non installé")

        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_margins(10, 10, 10)

        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(0, 9, _cell_text("BLUESTAR GPS V4.1"), ln=True, align="C")
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(0, 5, _cell_text(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        ), ln=True, align="C")
        pdf.ln(4)

        GRADE_RGB = {
            'A+': (251, 191, 36),
            'A':  (163, 230, 53),
            'B+': (52,  211, 153),
            'B':  (96,  165, 250),
        }

        def header():
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_fill_color(30, 58, 138)
            pdf.set_text_color(255, 255, 255)
            for col in COLS:
                pdf.cell(WIDTHS[col], 7, _cell_text(col), border=1, align='C', fill=True)
            pdf.ln()
            pdf.set_font("Helvetica", "", 6.5)

        header()

        for _, row in df.iterrows():
            if pdf.get_y() + RH > 287 - 15:
                pdf.add_page()
                header()
            for col in COLS:
                val = str(row.get(col, ''))
                fc  = (255, 255, 255)
                tc  = (0,   0,   0)
                if col == 'Quality':
                    fc = GRADE_RGB.get(val, (156, 163, 175))
                elif 'Bull' in val and 'Ret' not in val:
                    fc = (46,  204, 113); tc = (255, 255, 255)
                elif 'Bear' in val and 'Ret' not in val:
                    fc = (231, 76,  60);  tc = (255, 255, 255)
                elif 'Retracement Bull' in val:
                    fc = (125, 206, 160); tc = (255, 255, 255)
                elif 'Retracement Bear' in val:
                    fc = (241, 148, 138); tc = (255, 255, 255)
                elif 'Range' in val:
                    fc = (149, 165, 166); tc = (255, 255, 255)
                pdf.set_fill_color(*fc)
                pdf.set_text_color(*tc)
                pdf.cell(WIDTHS[col], RH, _cell_text(val), border=1, align='C', fill=True)
            pdf.ln()

        buf = BytesIO()
        out = pdf.output(dest='S')
        buf.write(out.encode('latin-1') if isinstance(out, str) else bytes(out))
        buf.seek(0)
        return buf

    except Exception as e:
        _log.error("PDF generation error: %s", e, exc_info=True)
        fallback = FPDF() if FPDF else None
        buf2 = BytesIO()
        if fallback:
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
        "<div class='main-header'><h1>🧭 BLUESTAR HEDGE FUND GPS V4.1</h1>"
        "<p style='margin:0;font-size:0.85em;opacity:0.8'>"
        "Signal-Accurate Edition · P1–P15 + 13 corrections V4.1"
        "</p></div>",
        unsafe_allow_html=True,
    )

    try:
        acc = st.secrets["OANDA_ACCOUNT_ID"]
        tok = st.secrets["OANDA_ACCESS_TOKEN"]
    except Exception:
        st.error("❌ Secrets OANDA manquants — configurez OANDA_ACCOUNT_ID et OANDA_ACCESS_TOKEN")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Configuration")
        only_best = st.checkbox("Afficher uniquement Grade A+ / A", value=False)
        st.info(
            "Cache : 10 min (thread-safe) · Workers : 5 · "
            "Retry : 4× backoff · Grades absolus"
        )
        st.markdown("---")
        if st.button("🗑️ Vider le cache", use_container_width=True):
            _cache_clear()
            st.success("Cache vidé — prochain lancement refetchera toutes les données.")
        st.markdown("---")
        st.caption("Bluestar GPS V4.1 — 15 patches P + 13 corrections C")

    # [C13] Protection double-clic : bouton désactivé pendant l'analyse
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

    # [P12] Alerte fraîcheur des données
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

    GRADE_ORDER = ['A+', 'A', 'B+', 'B']
    df['Quality'] = pd.Categorical(df['Quality'], categories=GRADE_ORDER, ordered=True)
    df = df.sort_values(['Quality', 'MTF'], ascending=[True, False])

    c1, c2, c3, c4 = st.columns(4)
    total   = len(df)
    a_plus  = len(df[df['Quality'] == 'A+'])
    a_grade = len(df[df['Quality'] == 'A'])
    b_grade = len(df[df['Quality'].isin(['B+', 'B'])])

    c1.metric("Total Analyzed",   total)
    c2.metric("Setups A+ (GOLD)", a_plus,  delta_color="inverse")
    c3.metric("Setups A (GREEN)", a_grade, delta_color="inverse")
    c4.metric("Setups B (BLUE)",  b_grade, delta_color="inverse")

    DISPLAY = [
        'Paire', 'M', 'W', 'D', '4H', '1H', '15m',
        'MTF', 'Quality', 'Age D1',
        'ATR Daily', 'ATR H4', 'ATR H1', 'ATR 15m',
    ]
    GRADE_CSS = {'A+': '#fbbf24', 'A': '#a3e635', 'B+': '#34d399', 'B': '#60a5fa'}

    def style_trend(v):
        if not isinstance(v, str): return ''
        if 'Bull' in v and 'Ret' not in v:  return f'background-color:{TREND_COLORS["Bullish"]};color:white;font-weight:bold'
        if 'Bear' in v and 'Ret' not in v:  return f'background-color:{TREND_COLORS["Bearish"]};color:white;font-weight:bold'
        if 'Retracement Bull' in v:          return f'background-color:{TREND_COLORS["Retracement Bull"]};color:white'
        if 'Retracement Bear' in v:          return f'background-color:{TREND_COLORS["Retracement Bear"]};color:white'
        if 'Range' in v:                     return f'background-color:{TREND_COLORS["Range"]};color:white'
        return ''

    def style_quality(s):
        if s.name != 'Quality': return [''] * len(s)
        return [
            f'color:black;font-weight:bold;background-color:{GRADE_CSS.get(x, "#9ca3af")}'
            for x in s
        ]

    cols_present = [col for col in DISPLAY if col in df.columns]
    styled = df[cols_present].style.apply(style_quality, axis=0).map(style_trend)
    st.dataframe(
        styled,
        height=min(800, max(400, (len(df) + 1) * 38 + 10)),
        use_container_width=True,
        hide_index=True,
    )

    # ── Téléchargements ──────────────────────────────────────────────────────
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

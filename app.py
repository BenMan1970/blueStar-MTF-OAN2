# app.py — BLUESTAR GPS V3.0
# Refonte propre : parallélisation · grades percentiles · Conflit · Âge D1 · Divergence RSI
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from fpdf import FPDF
except ImportError:
    st.warning("fpdf manquant — pip install fpdf")

# ===================== CONFIG =====================
st.set_page_config(page_title="Bluestar GPS V3.1", page_icon="🧭", layout="wide")

OANDA_API_URL = "https://api-fxpractice.oanda.com"

INSTRUMENTS = [
    # 28 paires Forex
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'USD_CAD', 'AUD_USD', 'NZD_USD',
    'EUR_GBP', 'EUR_JPY', 'EUR_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_NZD',
    'GBP_JPY', 'GBP_CHF', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD',
    'AUD_JPY', 'AUD_CAD', 'AUD_CHF', 'AUD_NZD', 'CAD_JPY', 'CAD_CHF', 'CHF_JPY',
    'NZD_JPY', 'NZD_CAD', 'NZD_CHF',
    # 6 indices et métaux
    'DE30_EUR', 'XAU_USD', 'XAG_USD', 'SPX500_USD', 'NAS100_USD', 'US30_USD',
]

# Instruments sans volume fiable sur OANDA — volume ignoré en intraday
INDICES = {'DE30_EUR', 'SPX500_USD', 'NAS100_USD', 'US30_USD',
           'XAU_USD', 'XAG_USD'}

TREND_COLORS = {
    'Bullish':          '#2ecc71',
    'Bearish':          '#e74c3c',
    'Retracement Bull': '#7dcea0',
    'Retracement Bear': '#f1948a',
    'Range':            '#95a5a6',
}

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
</style>
""", unsafe_allow_html=True)

# ===================== INDICATEURS DE BASE =====================

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def _sma(s, n):
    return s.rolling(n).mean()

def _atr(high, low, close, n=14):
    """ATR Wilder — ewm(alpha=1/n)."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _rsi(close, n=14):
    """RSI Wilder."""
    d    = close.diff()
    gain = d.where(d > 0, 0.0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(alpha=1/n, adjust=False).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _dmi(high, low, close, n=14):
    """DI+, DI- Wilder."""
    tr    = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1/n, adjust=False).mean()
    up    = high.diff()
    dn    = -low.diff()
    pdm   = up.where((up > dn) & (up > 0), 0.0)
    mdm   = dn.where((dn > up) & (dn > 0), 0.0)
    pdi   = 100 * pdm.ewm(alpha=1/n, adjust=False).mean() / atr_s.replace(0, np.nan)
    mdi   = 100 * mdm.ewm(alpha=1/n, adjust=False).mean() / atr_s.replace(0, np.nan)
    return pdi.iloc[-1], mdi.iloc[-1]

# ===================== TENDANCE PAR TF =====================

def trend_macro(df, tf):
    """Monthly / Weekly — EMA50 vs SMA200 (W) ou EMA50 vs EMA100 (M)."""
    if len(df) < 50:
        return 'Range', 0
    c    = df['Close']
    e50  = _ema(c, 50)
    if tf == 'M':
        e100 = _ema(c, 100)
        ref  = e100.iloc[-1]
        cur  = e50.iloc[-1]
        gap  = abs(cur - ref) / ref * 100 if ref != 0 else 0
        s    = 75 if gap > 0.3 else 60
        if   cur > ref: return 'Bullish', s
        elif cur < ref: return 'Bearish', s
        return 'Range', 40
    else:
        s200 = _sma(c, 200) if len(df) >= 200 else e50
        cur50, ref200 = e50.iloc[-1], s200.iloc[-1]
        prev50, prev200 = e50.iloc[-2], s200.iloc[-2]
        cross = (prev50 <= prev200 and cur50 > ref200) or (prev50 >= prev200 and cur50 < ref200)
        if   cur50 > ref200: return 'Bullish', 90 if cross else 75
        elif cur50 < ref200: return 'Bearish', 90 if cross else 75
        return 'Range', 40


def trend_daily(df):
    """
    5 votes indépendants — max 6 points (swing HH/HL compte double).
    Seuils : ≥5 → fort, ≥3 → modéré, sinon Range.
    """
    if len(df) < 60:
        return 'Range', 0

    c, h, l = df['Close'], df['High'], df['Low']
    cur = float(c.iloc[-1])
    vb = vbr = 0

    # Vote 1 — structure swing (wing=5, double poids)
    wing = 5
    sh, sl = [], []
    for i in range(wing, len(h) - wing):
        if h.iloc[i] == h.iloc[i-wing:i+wing+1].max(): sh.append(i)
        if l.iloc[i] == l.iloc[i-wing:i+wing+1].min(): sl.append(i)
    if len(sh) >= 2 and len(sl) >= 2:
        hh = h.iloc[sh[-1]] > h.iloc[sh[-2]]
        hl = l.iloc[sl[-1]] > l.iloc[sl[-2]]
        lh = h.iloc[sh[-1]] < h.iloc[sh[-2]]
        ll = l.iloc[sl[-1]] < l.iloc[sl[-2]]
        if hh and hl:   vb  += 2
        elif lh and ll: vbr += 2

    # Vote 2 — EMA21 / EMA50 stack
    e21 = _ema(c, 21).iloc[-1]
    e50 = _ema(c, 50)
    e50_cur = e50.iloc[-1]
    if   cur > e21 > e50_cur: vb  += 1
    elif cur < e21 < e50_cur: vbr += 1

    # Vote 3 — Weekly Open vs prix courant
    try:
        mon = pd.to_datetime(df.index).dayofweek.isin([0, 6])
        wo  = df[mon]
        if not wo.empty:
            wo_price = float(wo['Open'].iloc[-1])
            if   cur > wo_price: vb  += 1
            elif cur < wo_price: vbr += 1
    except Exception:
        pass

    # Vote 4 — close J-1 vs midpoint J-1
    if len(df) >= 2:
        mid = (float(h.iloc[-2]) + float(l.iloc[-2])) / 2
        if float(c.iloc[-2]) > mid: vb  += 1
        else:                       vbr += 1

    # Vote 5 — pente EMA50 normalisée par ATR
    if len(e50) >= 6:
        atr_val = float(_atr(h, l, c, 14).iloc[-1])
        slope   = float(e50.iloc[-1] - e50.iloc[-6])
        if atr_val > 0:
            if   slope / atr_val >  0.05: vb  += 1
            elif slope / atr_val < -0.05: vbr += 1

    if   vb  >= 5: return 'Bullish', 90
    elif vb  >= 3: return 'Bullish', 70
    elif vbr >= 5: return 'Bearish', 90
    elif vbr >= 3: return 'Bearish', 70
    return 'Range', 35


def trend_4h(df):
    """3 votes : EMA50 · DMI · Daily Open."""
    if len(df) < 60:
        return 'Range', 0
    c, h, l = df['Close'], df['High'], df['Low']
    cur = float(c.iloc[-1])
    score = 0

    score += 1 if cur > _ema(c, 50).iloc[-1] else -1

    pdi, mdi = _dmi(h, l, c)
    if not (np.isnan(pdi) or np.isnan(mdi)):
        score += 1 if pdi > mdi else -1

    try:
        dates = pd.to_datetime(df.index).normalize()
        today_open = float(df[dates == dates[-1]]['Open'].iloc[0])
        score += 1 if cur > today_open else -1
    except Exception:
        pass

    s = abs(score)
    strength = 90 if s == 3 else 70 if s >= 1 else 40
    return ('Bullish' if score > 0 else 'Bearish' if score < 0 else 'Range'), strength


def trend_intraday(df, instrument=''):
    """
    Intraday H1/M15 — 3 votes de base : ZLEMA · EMA stack · momentum RSI/MACD.
    4ème vote (volume) activé uniquement sur forex — ignoré sur indices/métaux
    où OANDA ne fournit pas de volume fiable.
    Signal net : 3/3 (ou 4/4 sur forex) → fort
    2/3 (ou 3/4 sur forex) → modéré
    Retracement sinon.
    """
    if len(df) < 70:
        return 'Range', 0

    c   = df['Close']
    cur = float(c.iloc[-1])

    e9      = _ema(c, 9).iloc[-1]
    e21     = _ema(c, 21).iloc[-1]
    e50     = _ema(c, 50)
    e50_cur = e50.iloc[-1]

    lag     = (50 - 1) // 2
    src_adj = c + (c - c.shift(lag))
    zlema   = src_adj.ewm(span=50, adjust=False).mean().iloc[-1]

    rsi_val  = _rsi(c, 14).iloc[-1]
    macd     = _ema(c, 12) - _ema(c, 26)
    sig_line = _ema(macd, 9)
    macd_cur = macd.iloc[-1]
    sig_cur  = sig_line.iloc[-1]

    bull_zlema = cur > zlema
    bear_zlema = cur < zlema
    bull_stack = e9 > e21 > e50_cur
    bear_stack = e9 < e21 < e50_cur
    bull_mom   = rsi_val > 50 and macd_cur > sig_cur
    bear_mom   = rsi_val < 50 and macd_cur < sig_cur

    votes_bull = [bull_zlema, bull_stack, bull_mom]
    votes_bear = [bear_zlema, bear_stack, bear_mom]

    # Vote volume — forex uniquement
    use_volume = instrument not in INDICES and 'Volume' in df.columns
    max_votes  = 3
    if use_volume:
        vol    = df['Volume']
        vol_ma = vol.rolling(20).mean()
        strong_vol = float(vol.iloc[-1]) > float(vol_ma.iloc[-1]) * 1.3
        votes_bull.append(strong_vol and bull_zlema)
        votes_bear.append(strong_vol and bear_zlema)
        max_votes = 4

    vb  = sum(votes_bull)
    vbr = sum(votes_bear)
    threshold_strong = max_votes
    threshold_mod    = max_votes - 1

    if vb == threshold_strong:
        strength = min(80, abs(cur - zlema) / cur * 1000 + 40)
        return 'Bullish', int(strength)
    if vbr == threshold_strong:
        strength = min(80, abs(cur - zlema) / cur * 1000 + 40)
        return 'Bearish', int(strength)
    if vb >= threshold_mod:
        return 'Bullish', 55
    if vbr >= threshold_mod:
        return 'Bearish', 55

    if cur < e50_cur and e9 > e21:
        return 'Retracement Bull', 45
    if cur > e50_cur and e9 < e21:
        return 'Retracement Bear', 45

    return 'Range', 30

# ===================== MÉTRIQUES COMPLÉMENTAIRES =====================

def trend_age_daily(df):
    """
    Nombre de bougies D1 depuis que le close a croisé l'EMA50.
    Plus réactif que EMA21/50 — reflète le moment où le prix
    a réellement changé de camp par rapport à sa moyenne clé.
    """
    if len(df) < 55:
        return 'N/A'
    c   = df['Close']
    e50 = _ema(c, 50)
    above = c > e50
    for i in range(len(above) - 2, 0, -1):
        if above.iloc[i] != above.iloc[i - 1]:
            age = len(above) - 1 - i
            return str(age)
    return str(len(above))


def rsi_divergence_daily(df):
    """
    Divergence cachée simple sur D1 :
    Prix HH mais RSI LH → divergence baissière
    Prix LL mais RSI HL → divergence haussière
    Fenêtre : 20 dernières bougies.
    """
    if len(df) < 25:
        return ''
    c    = df['Close'].iloc[-20:]
    rsi_ = _rsi(df['Close'], 14).iloc[-20:]
    ph   = df['High'].iloc[-20:]

    # Comparer les 5 premières vs 5 dernières bougies de la fenêtre
    price_hi_early = ph.iloc[:5].max()
    price_hi_late  = ph.iloc[-5:].max()
    rsi_hi_early   = rsi_.iloc[:5].max()
    rsi_hi_late    = rsi_.iloc[-5:].max()

    price_lo_early = c.iloc[:5].min()
    price_lo_late  = c.iloc[-5:].min()
    rsi_lo_early   = rsi_.iloc[:5].min()
    rsi_lo_late    = rsi_.iloc[-5:].min()

    if price_hi_late > price_hi_early and rsi_hi_late < rsi_hi_early:
        return '⚠️ Div Bear'
    if price_lo_late < price_lo_early and rsi_lo_late > rsi_lo_early:
        return '⚠️ Div Bull'
    return ''

# ===================== API =====================

def fetch_candles(instrument, granularity, count, account_id, access_token):
    url     = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params  = {'granularity': granularity, 'count': count, 'price': 'M'}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()
        candles = r.json().get('candles', [])
        df = pd.DataFrame([{
            'date':   c['time'],
            'Open':   float(c['mid']['o']),
            'High':   float(c['mid']['h']),
            'Low':    float(c['mid']['l']),
            'Close':  float(c['mid']['c']),
            'Volume': float(c.get('volume', 0))
        } for c in candles if c.get('complete')])
        if df.empty:
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def fetch_cached(inst, gran, cnt, acc, tok):
    return fetch_candles(inst, gran, cnt, acc, tok)

def fetch_all_data(instrument, account_id, access_token):
    """Récupère toutes les granularités pour un instrument en une fois."""
    specs = {
        'M':   ('D',   4500),
        'W':   ('D',   2000),
        'D':   ('D',   300),
        '4H':  ('H4',  300),
        '1H':  ('H1',  300),
        '15m': ('M15', 300),
    }
    cache = {}
    for tf, (gran, count) in specs.items():
        df = fetch_cached(instrument, gran, count, account_id, access_token)
        if df.empty:
            return None
        if tf == 'M':
            df = df.resample('ME').agg(
                {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}
            ).dropna()
            if len(df) < 50:
                return None
        elif tf == 'W':
            df = df.resample('W-FRI').agg(
                {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}
            ).dropna()
        cache[tf] = df
    return cache

# ===================== SCORING MTF =====================

def score_mtf(trends, scores):
    """
    Score pondéré avec bonus additifs d'alignement (non multiplicatifs).
    Poids de base : M=5 W=4 D=4 4H=2.5 1H=1.5 15m=1 → total=18
    Bonus additifs (points directs, plafond 100) :
      +15 si M et W alignés dans le même sens
      +10 si D et 4H alignés dans le même sens
    """
    WEIGHTS = {'M': 5.0, 'W': 4.0, 'D': 4.0, '4H': 2.5, '1H': 1.5, '15m': 1.0}
    TOTAL   = sum(WEIGHTS.values())  # 18.0

    def weighted(direction):
        return sum(
            WEIGHTS[tf] * (scores[tf] / 100)
            for tf in trends
            if trends[tf].startswith(direction)
        )

    w_bull = weighted('Bullish') + weighted('Retracement Bull') * 0.5
    w_bear = weighted('Bearish') + weighted('Retracement Bear') * 0.5

    if w_bull > w_bear:
        raw_score = (w_bull / TOTAL) * 100
        direction = 'Bullish'
    elif w_bear > w_bull:
        raw_score = (w_bear / TOTAL) * 100
        direction = 'Bearish'
    else:
        return 'Range', 0

    # Bonus additifs d'alignement — prévisibles et auditables
    bonus = 0
    if (trends.get('M','') == trends.get('W','') and
            trends.get('M','') not in ('Range', '')):
        bonus += 15
    if (trends.get('D','') == trends.get('4H','') and
            trends.get('D','') not in ('Range', '')):
        bonus += 10

    return direction, min(100, raw_score + bonus)


def conflict_flag(trends):
    """
    Détecte les conflits macro vs intraday sans écraser les signaux.
    M/W/D = biais macro, 1H/15m = intraday.
    """
    macro_votes = [trends.get(tf) for tf in ('M', 'W', 'D')]
    bull_macro  = macro_votes.count('Bullish')
    bear_macro  = macro_votes.count('Bearish')

    intra = [trends.get(tf) for tf in ('1H', '15m')]
    bull_intra = sum(1 for t in intra if t and t.startswith('Bullish'))
    bear_intra = sum(1 for t in intra if t and t.startswith('Bearish'))

    if bull_macro >= 2 and bear_intra >= 1:
        return 'Macro↑/Intra↓'
    if bear_macro >= 2 and bull_intra >= 1:
        return 'Macro↓/Intra↑'
    return ''


def grade_hybrid(scores_list):
    """
    Grading hybride : percentile relatif + plancher absolu.
    Si score brut < 25 → B peu importe le rang (marché plat = pas de A+).
    Sinon :
      A+ = top 15%  ·  A = percentile 65–85
      B+ = percentile 40–65  ·  B = reste
    """
    if not scores_list:
        return []
    arr  = np.array(scores_list, dtype=float)
    p85  = np.percentile(arr, 85)
    p65  = np.percentile(arr, 65)
    p40  = np.percentile(arr, 40)
    grades = []
    for s in arr:
        if s < 25:
            grades.append('B')
        elif s >= p85:
            grades.append('A+')
        elif s >= p65:
            grades.append('A')
        elif s >= p40:
            grades.append('B+')
        else:
            grades.append('B')
    return grades

# ===================== ANALYSE PRINCIPALE =====================

def analyze_pair(pair, account_id, access_token):
    """Analyse complète d'un instrument — appelé en parallèle."""
    cache = fetch_all_data(pair, account_id, access_token)
    if cache is None:
        return None

    trends, scores = {}, {}
    for tf in ('M', 'W'):
        t, s = trend_macro(cache[tf], tf)
        trends[tf], scores[tf] = t, s
    trends['D'],   scores['D']   = trend_daily(cache['D'])
    trends['4H'],  scores['4H']  = trend_4h(cache['4H'])
    trends['1H'],  scores['1H']  = trend_intraday(cache['1H'],  pair)
    trends['15m'], scores['15m'] = trend_intraday(cache['15m'], pair)

    mtf_dir, mtf_score = score_mtf(trends, scores)
    conflict            = conflict_flag(trends)
    age                 = trend_age_daily(cache['D'])
    divergence          = rsi_divergence_daily(cache['D'])

    atr_vals = {}
    for tf_key, col in [('D', 'ATR Daily'), ('1H', 'ATR H1'), ('15m', 'ATR 15m')]:
        v = float(_atr(cache[tf_key]['High'], cache[tf_key]['Low'], cache[tf_key]['Close'], 14).iloc[-1])
        if v >= 10:
            atr_vals[col] = f"{v:.2f}"       # indices, or, argent
        elif v >= 0.1:
            atr_vals[col] = f"{v:.4f}"       # JPY pairs, métaux légers
        else:
            atr_vals[col] = f"{v:.5f}"       # paires forex standard

    row = {
        'Paire':     pair.replace('_', '/'),
        'M':         trends['M'],
        'W':         trends['W'],
        'D':         trends['D'],
        '4H':        trends['4H'],
        '1H':        trends['1H'],
        '15m':       trends['15m'],
        'MTF':       f"{mtf_dir} ({mtf_score:.0f}%)" if mtf_dir != 'Range' else 'Range',
        '_mtf_score': mtf_score,
        '_mtf_dir':   mtf_dir,
        'Conflit':   conflict,
        'Âge D1':   age,
        'Div RSI':   divergence,
    }
    row.update(atr_vals)
    return row


def analyze_all(account_id, access_token):
    results = []
    progress = st.progress(0)
    status   = st.empty()
    total    = len(INSTRUMENTS)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(analyze_pair, inst, account_id, access_token): inst
            for inst in INSTRUMENTS
        }
        done = 0
        for future in as_completed(futures):
            inst = futures[future]
            done += 1
            progress.progress(done / total)
            status.text(f"GPS ({done}/{total}) — {inst.replace('_', '/')}")
            try:
                row = future.result()
                if row:
                    results.append(row)
            except Exception:
                pass

    progress.empty()
    status.empty()

    if not results:
        return pd.DataFrame()

    # Grading hybride (percentile relatif + plancher absolu)
    scores_list = [r['_mtf_score'] for r in results]
    grades      = grade_hybrid(scores_list)
    for r, g in zip(results, grades):
        r['Quality'] = g

    df = pd.DataFrame(results)
    df.drop(columns=['_mtf_score', '_mtf_dir'], inplace=True)
    return df

# ===================== PDF =====================

def create_pdf(df):
    try:
        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_margins(10, 10, 10)

        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(0, 9, "BLUESTAR GPS V3.1", ln=True, align="C")
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}", ln=True, align="C")
        pdf.ln(4)

        COLS = ['Paire','M','W','D','4H','1H','15m','MTF','Quality','Conflit','Âge D1','Div RSI','ATR Daily','ATR H1','ATR 15m']
        WIDTHS = {'Paire':20,'M':18,'W':18,'D':18,'4H':18,'1H':18,'15m':18,'MTF':30,'Quality':13,
                  'Conflit':22,'Âge D1':14,'Div RSI':18,'ATR Daily':18,'ATR H1':16,'ATR 15m':16}
        RH = 5.5

        def header():
            pdf.set_font("Helvetica","B",7)
            pdf.set_fill_color(30,58,138); pdf.set_text_color(255,255,255)
            for c in COLS:
                pdf.cell(WIDTHS[c], 7, c, border=1, align='C', fill=True)
            pdf.ln(); pdf.set_font("Helvetica","",6.5)

        header()
        GRADE_RGB = {'A+':(251,191,36),'A':(163,230,53),'B+':(52,211,153),'B':(96,165,250)}

        for _, row in df.iterrows():
            if pdf.get_y() + RH > 287 - 15:
                pdf.add_page(); header()
            for c in COLS:
                val = str(row.get(c,''))
                fc  = (255,255,255); tc = (0,0,0)
                if c == 'Quality':
                    fc = GRADE_RGB.get(val,(156,163,175)); tc=(0,0,0)
                elif 'Bull' in val and 'Ret' not in val:
                    fc=(46,204,113); tc=(255,255,255)
                elif 'Bear' in val and 'Ret' not in val:
                    fc=(231,76,60);  tc=(255,255,255)
                elif 'Retracement Bull' in val:
                    fc=(125,206,160); tc=(255,255,255)
                elif 'Retracement Bear' in val:
                    fc=(241,148,138); tc=(255,255,255)
                elif 'Range' in val:
                    fc=(149,165,166); tc=(255,255,255)
                elif 'Macro' in val:
                    fc=(254,243,199); tc=(0,0,0)
                pdf.set_fill_color(*fc); pdf.set_text_color(*tc)
                pdf.cell(WIDTHS[c], RH, val, border=1, align='C', fill=True)
            pdf.ln()

        buf = BytesIO()
        out = pdf.output(dest='S')
        buf.write(out.encode('latin-1') if isinstance(out, str) else out)
        buf.seek(0)
        return buf
    except Exception as e:
        # fallback minimal
        pdf2 = FPDF(); pdf2.add_page()
        pdf2.set_font("Helvetica","B",12)
        pdf2.cell(0,10,"PDF Generation Error",ln=True)
        buf2 = BytesIO()
        out2 = pdf2.output(dest='S')
        buf2.write(out2.encode('latin-1') if isinstance(out2, str) else out2)
        buf2.seek(0)
        return buf2

# ===================== UI =====================

def main():
    st.markdown("<div class='main-header'><h1>🧭 BLUESTAR HEDGE FUND GPS V3.1</h1></div>", unsafe_allow_html=True)

    try:
        acc = st.secrets["OANDA_ACCOUNT_ID"]
        tok = st.secrets["OANDA_ACCESS_TOKEN"]
    except Exception:
        st.error("❌ Secrets OANDA manquants"); st.stop()

    with st.sidebar:
        st.header("⚙️ Configuration")
        only_best = st.checkbox("Afficher uniquement Grade A+ / A", value=False)
        hide_conflict = st.checkbox("Masquer les conflits macro/intra", value=False)
        st.info("Le cache dure 10 minutes pour optimiser les performances API.")

    if st.button("🚀 LANCER L'ANALYSE TOUS ACTIFS", type="primary", use_container_width=True):
        with st.spinner("Analyse Multi-Timeframe en cours..."):
            df = analyze_all(acc, tok)
        if not df.empty:
            st.session_state.df = df

    if "df" not in st.session_state:
        return

    df = st.session_state.df.copy()

    if only_best:
        df = df[df['Quality'].isin(['A+','A'])]
    if hide_conflict:
        df = df[df['Conflit'] == '']

    GRADE_ORDER = ['A+','A','B+','B']
    df['Quality'] = pd.Categorical(df['Quality'], categories=GRADE_ORDER, ordered=True)
    df = df.sort_values(['Quality','MTF'], ascending=[True, False])

    # Métriques — style V2.1
    c1, c2, c3, c4, c5 = st.columns(5)
    total      = len(df)
    a_plus     = len(df[df['Quality'] == 'A+'])
    a_grade    = len(df[df['Quality'] == 'A'])
    b_grade    = len(df[df['Quality'].isin(['B+','B'])])
    conflits   = len(df[df['Conflit'] != ''])

    c1.metric("Total Analyzed",    total)
    c2.metric("Setups A+ (GOLD)",  a_plus,  delta_color="inverse")
    c3.metric("Setups A (GREEN)",  a_grade, delta_color="inverse")
    c4.metric("Setups B (BLUE)",   b_grade, delta_color="inverse")
    c5.metric("Conflits MTF",      conflits)

    DISPLAY = ['Paire','M','W','D','4H','1H','15m','MTF','Quality','Conflit','Âge D1','Div RSI','ATR Daily','ATR H1','ATR 15m']
    GRADE_CSS = {'A+':'#fbbf24','A':'#a3e635','B+':'#34d399','B':'#60a5fa'}

    def style_trend(v):
        if not isinstance(v, str): return ''
        if 'Bull' in v and 'Ret' not in v:  return f'background-color:{TREND_COLORS["Bullish"]};color:white;font-weight:bold'
        if 'Bear' in v and 'Ret' not in v:  return f'background-color:{TREND_COLORS["Bearish"]};color:white;font-weight:bold'
        if 'Retracement Bull' in v:          return f'background-color:{TREND_COLORS["Retracement Bull"]};color:white'
        if 'Retracement Bear' in v:          return f'background-color:{TREND_COLORS["Retracement Bear"]};color:white'
        if 'Range' in v:                     return f'background-color:{TREND_COLORS["Range"]};color:white'
        if 'Macro' in v:                     return 'background-color:#fef3c7;color:#92400e;font-weight:bold'
        return ''

    def style_quality(s):
        if s.name != 'Quality': return [''] * len(s)
        return [f'color:black;font-weight:bold;background-color:{GRADE_CSS.get(x,"#9ca3af")}' for x in s]

    cols_present = [c for c in DISPLAY if c in df.columns]
    styled = df[cols_present].style.apply(style_quality, axis=0).map(style_trend)
    st.dataframe(
        styled,
        height=min(800, max(400, (len(df) + 1) * 38 + 10)),
        use_container_width=True
    )

    c1, c2 = st.columns(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    with c1:
        st.download_button(
            "📄 Télécharger PDF",
            data=create_pdf(df[cols_present]),
            file_name=f"Bluestar_GPS_{ts}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with c2:
        st.download_button(
            "📊 Télécharger CSV",
            data=df[cols_present].to_csv(index=False).encode(),
            file_name=f"Bluestar_GPS_{ts}.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()

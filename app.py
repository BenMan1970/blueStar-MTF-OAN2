import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import logging
from datetime import datetime
from io import BytesIO
import pytz

# ==========================================
# 1. IMPORTS ET GESTION D'ERREURS PDF (SAFE MODE)
# ==========================================

# Détection de la version de fpdf pour compatibilité
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    FPDF = None
    PDF_AVAILABLE = False

# ==========================================
# 2. CONFIGURATION & DESIGN
# ==========================================
st.set_page_config(
    page_title="Bluestar GPS V2.3 Ultimate",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Palette de couleurs
TREND_COLORS = {
    'Bullish': '#2ecc71',
    'Bearish': '#e74c3c',
    'Retracement Bull': '#7dcea0',
    'Retracement Bear': '#f1948a',
    'Range': '#95a5a6'
}

GRADE_COLORS = {
    'A+': '#fbbf24',
    'A':  '#a3e635',
    'B':  '#60a5fa',
    'B-': '#3b82f6',
    'C':  '#9ca3af'
}

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e6edf3; }
    .main-header {
        text-align:center; padding:20px; background: linear-gradient(135deg, #1e3a8a 0%, #172554 100%);
        color:white; border-radius:12px; margin-bottom:20px; box-shadow: 0 10px 20px rgba(0,0,0,0.4);
    }
    .section-title {
        color: #3b82f6; font-weight: 700; text-transform: uppercase; font-size: 1.2rem; 
        margin-top: 30px; border-bottom: 2px solid #1f2937; padding-bottom: 10px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #f1f5f9; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.9rem; }
    iframe { width: 100% !important; border: none; }
    div[data-testid="stMarkdown"] { text-align: center; }
</style>
""", unsafe_allow_html=True)

# Initialisation de l'état de session
if 'df' not in st.session_state:
    st.session_state['df'] = None

# ==========================================
# 3. MOTEUR TECHNIQUE
# ==========================================
def sma(series, length):
    return series.rolling(window=length).mean()

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==========================================
# 4. LOGIQUE MTF INSTITUTIONNELLE
# ==========================================
def calc_institutional_trend_macro(df):
    if len(df) < 50: return 'Range', 0
    close = df['Close']
    curr_price = close.iloc[-1]
    has_200 = len(df) >= 200
    sma200 = sma(close, 200) if has_200 else sma(close, 50)
    ema50 = ema(close, 50)
    curr_sma200 = sma200.iloc[-1]
    curr_ema50 = ema50.iloc[-1]
    above_sma200 = curr_price > curr_sma200
    below_sma200 = curr_price < curr_sma200
    ema50_above_sma = curr_ema50 > curr_sma200
    ema50_below_sma = curr_ema50 < curr_sma200
    
    if above_sma200 and ema50_above_sma: return "Bullish", 85
    elif below_sma200 and ema50_below_sma: return "Bearish", 85
    elif above_sma200: return "Bullish", 65
    elif below_sma200: return "Bearish", 65
    else: return "Range", 40

def calc_institutional_trend_daily(df):
    if len(df) < 200: return 'Range', 0
    close = df['Close']
    curr_price = close.iloc[-1]
    sma200 = sma(close, 200)
    ema50 = ema(close, 50)
    ema21 = ema(close, 21)
    curr_sma200 = sma200.iloc[-1]
    curr_ema50 = ema50.iloc[-1]
    curr_ema21 = ema21.iloc[-1]
    above_sma200 = curr_price > curr_sma200
    below_sma200 = curr_price < curr_sma200
    ema50_above_sma = curr_ema50 > curr_sma200
    ema50_below_sma = curr_ema50 < curr_sma200
    ema21_above_50 = curr_ema21 > curr_ema50
    ema21_below_50 = curr_ema21 < curr_ema50
    price_above_21 = curr_price > curr_ema21
    price_below_21 = curr_price < curr_ema21
    
    if above_sma200 and ema50_above_sma and ema21_above_50 and price_above_21: return "Bullish", 90
    if below_sma200 and ema50_below_sma and ema21_below_50 and price_below_21: return "Bearish", 90
    if above_sma200 and ema50_above_sma and (ema21_above_50 or price_above_21): return "Bullish", 70
    if below_sma200 and ema50_below_sma and (ema21_below_50 or price_below_21): return "Bearish", 70
    if not above_sma200 and ema50_above_sma: return "Retracement Bull", 55
    if above_sma200 and ema50_below_sma: return "Retracement Bear", 55
    if above_sma200: return "Bullish", 50
    if below_sma200: return "Bearish", 50
    return "Range", 35

def calc_institutional_trend_4h(df):
    if len(df) < 200: return 'Range', 0
    close = df['Close']
    curr_price = close.iloc[-1]
    sma200 = sma(close, 200)
    ema50 = ema(close, 50)
    ema21 = ema(close, 21)
    curr_sma200 = sma200.iloc[-1]
    curr_ema50 = ema50.iloc[-1]
    curr_ema21 = ema21.iloc[-1]
    above_sma200 = curr_price > curr_sma200
    below_sma200 = curr_price < curr_sma200
    ema21_above_50 = curr_ema21 > curr_ema50
    ema21_below_50 = curr_ema21 < curr_ema50
    ema50_above_sma = curr_ema50 > curr_sma200
    price_above_21 = curr_price > curr_ema21
    price_below_21 = curr_price < curr_ema21
    
    if above_sma200 and ema21_above_50 and ema50_above_sma and price_above_21: return "Bullish", 80
    if below_sma200 and ema21_below_50 and curr_ema50 < curr_sma200 and price_below_21: return "Bearish", 80
    if above_sma200 and price_above_21: return "Bullish", 60
    if below_sma200 and price_below_21: return "Bearish", 60
    if not above_sma200 and ema50_above_sma: return "Retracement Bull", 50
    if above_sma200 and curr_ema50 < curr_sma200: return "Retracement Bear", 50
    return "Range", 40

def calc_institutional_trend_intraday(df, macro_trend=None):
    if len(df) < 50: return 'Range', 0
    close = df['Close']
    curr_price = close.iloc[-1]
    ema50 = ema(close, 50)
    ema21 = ema(close, 21)
    ema9 = ema(close, 9)
    curr_ema50 = ema50.iloc[-1]
    curr_ema21 = ema21.iloc[-1]
    curr_ema9 = ema9.iloc[-1]
    lag = 17
    src_adj = close + (close - close.shift(lag))
    zlema_val = src_adj.ewm(span=50, adjust=False).mean()
    curr_zlema = zlema_val.iloc[-1]
    has_baseline = len(df) >= 200
    baseline = sma(close, 200) if has_baseline else curr_ema50
    curr_baseline = baseline.iloc[-1] if has_baseline else curr_ema50
    rsi_val = rsi(close, 14).iloc[-1]
    macd_line = ema(close, 12) - ema(close, 26)
    signal_line = ema(macd_line, 9)
    curr_macd = macd_line.iloc[-1]
    curr_signal = signal_line.iloc[-1]
    vol = df['Volume'].iloc[-1]
    vol_ma = df['Volume'].rolling(20).mean().iloc[-1]
    strong_vol = vol > vol_ma * 1.3
    ema_bull_align = curr_ema9 > curr_ema21 and curr_ema21 > curr_ema50
    ema_bear_align = curr_ema9 < curr_ema21 and curr_ema21 < curr_ema50
    momentum_bull = rsi_val > 50 and curr_macd > curr_signal
    momentum_bear = rsi_val < 50 and curr_macd < curr_signal
    bullish = curr_price > curr_zlema and ema_bull_align and momentum_bull
    bearish = curr_price < curr_zlema and ema_bear_align and momentum_bear
    
    if bullish:
        base_strength = min(75, abs(curr_price - curr_zlema) / curr_price * 1000)
        momentum_bonus = 15 if strong_vol else 0
        score = min(75, base_strength + momentum_bonus)
        return "Bullish", score
    if bearish:
        base_strength = min(75, abs(curr_price - curr_zlema) / curr_price * 1000)
        momentum_bonus = 15 if strong_vol else 0
        score = min(75, base_strength + momentum_bonus)
        return "Bearish", score
    if has_baseline:
        baseline_trend = "Bullish" if curr_ema50 > curr_baseline else "Bearish"
        if curr_price < curr_baseline and baseline_trend == "Bullish": return "Retracement Bull", 45
        if curr_price > curr_baseline and baseline_trend == "Bearish": return "Retracement Bear", 45
    return "Range", 30

# ==========================================
# 5. DATA FETCHING (CORRIGÉ)
# ==========================================
def get_oanda_data(client, instrument, granularity, count):
    try:
        params = {"count": count, "granularity": granularity, "price": "M"}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        client.request(r)
        data = []
        for c in r.response['candles']:
            if c['complete']:
                data.append({
                    'date': c['time'],
                    'Open': float(c['mid']['o']),
                    'High': float(c['mid']['h']),
                    'Low': float(c['mid']['l']),
                    'Close': float(c['mid']['c']),
                    'Volume': float(c.get('volume', 0))
                })
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_oanda_data(access_token, environment, inst, gran, cnt):
    """Wrapper caché qui recrée le client à l'intérieur (Fix UnhashableParamError)"""
    client_instance = oandapyV20.API(access_token=access_token, environment=environment)
    return get_oanda_data(client_instance, inst, gran, cnt)

# ==========================================
# 6. LOGIQUE HEATMAP
# ==========================================
def normalize_score(rsi_value):
    return ((rsi_value - 50) / 50 + 1) * 5

def calculate_heatmap_data(access_token, environment, gran="H1"):
    forex_pairs = [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
        "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
        "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
        "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
        "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY"
    ]
    indices = {'US30_USD': 'DOW JONES', 'NAS100_USD': 'NASDAQ 100', 'SPX500_USD': 'S&P 500', 'DE30_EUR': 'DAX 40'}
    metaux = {'XAU_USD': 'GOLD', 'XAG_USD': 'SILVER', 'XPT_USD': 'PLATINUM'}
    special_assets = {**indices, **metaux}
    prices = {}
    pct_special = {}
    scores_special = {}

    # Fetch Forex
    for pair in forex_pairs:
        df = get_cached_oanda_data(access_token, environment, pair, gran, 100)
        if df is not None and not df.empty: prices[pair] = df['Close']

    # Fetch Indices & Metaux
    for symbol, name in special_assets.items():
        df = get_cached_oanda_data(access_token, environment, symbol, gran, 100)
        if df is not None:
            rsi_series = rsi(df['Close'], 14)
            scores_special[name] = (normalize_score(rsi_series.iloc[-1]), normalize_score(rsi_series.iloc[-2]))
            pct = df['Close'].pct_change().iloc[-1] * 100
            cat = "INDICES" if symbol in indices else "METAUX"
            pct_special[name] = {'pct': pct, 'cat': cat}

    # Calc Forces Devises
    if not prices: return None, None, None
    df_prices = pd.DataFrame(prices).fillna(method='ffill').fillna(method='bfill')
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    scores_forex = {}
    for curr in currencies:
        total_curr, total_prev, count = 0.0, 0.0, 0
        opponents = [c for c in currencies if c != curr]
        for opp in opponents:
            pair_d = f"{curr}_{opp}"
            pair_i = f"{opp}_{curr}"
            rsi_s = None
            if pair_d in df_prices.columns: rsi_s = rsi(df_prices[pair_d])
            elif pair_i in df_prices.columns: rsi_s = rsi(1/df_prices[pair_i])
            if rsi_s is not None:
                total_curr += normalize_score(rsi_s.iloc[-1])
                total_prev += normalize_score(rsi_s.iloc[-2])
                count += 1
        if count > 0: scores_forex[curr] = (total_curr / count, total_prev / count)
        else: scores_forex[curr] = (5.0, 5.0)

    return scores_forex, scores_special, df_prices, pct_special

def generate_exact_map_html(df_prices, pct_special):
    pct_changes = df_prices.pct_change().iloc[-1] * 100
    def get_bg_color(pct):
        if pct >= 0.15: return "#009900"
        if pct >= 0.01: return "#33cc33"
        if pct <= -0.15: return "#cc0000"
        if pct <= -0.01: return "#ff3300"
        return "#f0f0f0"
    def get_text_color(pct):
        if -0.01 < pct < 0.01: return "#333"
        return "white"
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    forex_data = {}
    for base in currencies:
        forex_data[base] = []
        for col in df_prices.columns:
            if base in col:
                val = pct_changes[col]
                if col.startswith(base):
                    quote, pct = col.split('_')[1], val
                else:
                    quote, pct = col.split('_')[0], -val
                forex_data[base].append({'pair': quote, 'pct': pct})
    scores = {curr: sum(i['pct'] for i in items) for curr, items in forex_data.items()}
    sorted_cols = sorted(scores, key=scores.get, reverse=True)
    html = """<!DOCTYPE html><html><head><style>body { font-family: 'Arial', sans-serif; background-color: transparent; margin: 0; padding: 0; }.section-header { color: #aaa; font-size: 14px; font-weight: bold; text-transform: uppercase; margin: 25px 0 10px 0; display: flex; align-items: center; gap: 5px; border-bottom: 2px solid #333; padding-bottom: 5px; }.matrix-row { display: flex; gap: 4px; overflow-x: auto; padding-bottom: 10px; }.currency-col { display: flex; flex-direction: column; min-width: 95px; gap: 1px; }.tile { display: flex; justify-content: space-between; align-items: center; padding: 3px 6px; font-size: 11px; font-weight: bold; box-shadow: 0 1px 2px rgba(0,0,0,0.2); }.sep { background: #eee; color: #000; font-weight: 900; padding: 5px; margin: 2px 0; font-size: 13px; text-transform: uppercase; border-left: 4px solid #333; }.grid-container { display: flex; flex-wrap: wrap; gap: 10px; }.big-box { width: 140px; height: 60px; display: flex; flex-direction: column; justify-content: center; align-items: center; color: white; border-radius: 4px; box-shadow: 0 3px 5px rgba(0,0,0,0.3); text-shadow: 1px 1px 2px rgba(0,0,0,0.5); }.box-name { font-size: 11px; font-weight: bold; margin-bottom: 2px; text-transform: uppercase; }.box-val { font-size: 14px; font-weight: 900; }</style></head><body>"""
    html += '<div class="section-header">💱 FOREX MAP</div><div class="matrix-row">'
    for curr in sorted_cols:
        items = forex_data[curr]
        winners = sorted([x for x in items if x['pct'] >= 0.01], key=lambda x: x['pct'], reverse=True)
        losers = sorted([x for x in items if x['pct'] < -0.01], key=lambda x: x['pct'], reverse=True)
        flat = [x for x in items if -0.01 <= x['pct'] < 0.01]
        html += '<div class="currency-col">'
        for x in winners:
            col, txt = get_bg_color(x['pct']), get_text_color(x['pct'])
            html += f'<div class="tile" style="background:{col}; color:{txt};"><span>{x["pair"]}</span><span>+{x["pct"]:.2f}%</span></div>'
        html += f'<div class="sep">{curr}</div>'
        for x in flat:
             html += f'<div class="tile" style="background:#f0f0f0; color:#333;"><span>{x["pair"]}</span><span>unch</span></div>'
        for x in losers:
            col, txt = get_bg_color(x['pct']), get_text_color(x['pct'])
            html += f'<div class="tile" style="background:{col}; color:{txt};"><span>{x["pair"]}</span><span>{x["pct"]:.2f}%</span></div>'
        html += '</div>'
    html += '</div>'
    html += '<div class="section-header">📊 INDICES</div><div class="grid-container">'
    indices_data = {k: v for k, v in pct_special.items() if v['cat'] == "INDICES"}
    for name, data in indices_data.items():
        pct = data['pct']; col = get_bg_color(pct)
        html += f'<div class="big-box" style="background:{col}"><span class="box-name">{name}</span><span class="box-val">{pct:+.2f}%</span></div>'
    html += '</div>'
    html += '<div class="section-header">🪙 METAUX</div><div class="grid-container">'
    metaux_data = {k: v for k, v in pct_special.items() if v['cat'] == "METAUX"}
    for name, data in metaux_data.items():
        pct = data['pct']; col = get_bg_color(pct)
        html += f'<div class="big-box" style="background:{col}"><span class="box-name">{name}</span><span class="box-val">{pct:+.2f}%</span></div>'
    html += '</div>'
    html += "</body></html>"
    return html

# ==========================================
# 7. CORE ANALYTICS (GPS)
# ==========================================
def analyze_market(access_token, environment):
    results = []
    FOREX_PAIRS_EXTENDED = [
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD', 'NZD_USD',
        'EUR_GBP', 'EUR_JPY', 'EUR_CHF', 'EUR_CAD', 'EUR_AUD', 'EUR_NZD',
        'GBP_JPY', 'GBP_CHF', 'GBP_CAD', 'GBP_AUD', 'GBP_NZD',
        'AUD_JPY', 'AUD_CAD', 'AUD_CHF', 'AUD_NZD',
        'CAD_JPY', 'CAD_CHF', 'NZD_JPY', 'NZD_CAD', 'NZD_CHF', 'CHF_JPY',
        'XAU_USD', 'XPT_USD', 'US30_USD', 'SPX500_USD', 'NAS100_USD'
    ]
    tf_config = {
        'M':   ('D', 4500, 'Macro'),
        'W':   ('D', 2000, 'Macro'),
        'D':   ('D', 300,  'Daily'),
        '4H':  ('H4', 300, '4H'),
        '1H':  ('H1', 300, 'Intra'),
        '15m': ('M15', 300,'Intra')
    }
    bar = st.progress(0)
    status = st.empty()
    
    for idx, pair in enumerate(FOREX_PAIRS_EXTENDED):
        display_name = pair.replace('_', '/')
        status.text(f"GPS Analyse : {display_name}...")
        row_data = {'Paire': display_name}
        trends_map = {}
        scores_map = {}
        valid_pair = True
        
        data_cache = {}
        for tf, (gran, count, _) in tf_config.items():
            df = get_cached_oanda_data(access_token, environment, pair, gran, count)
            if df.empty:
                valid_pair = False
                break
            if tf == 'M':
                df = df.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
                if len(df) < 50:
                    df_temp = get_cached_oanda_data(access_token, environment, pair, 'D', 2000)
                    if not df_temp.empty:
                        df = df_temp.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            elif tf == 'W':
                df = df.resample('W-FRI').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            data_cache[tf] = df

        if not valid_pair: continue
        for tf, (_, _, mode) in tf_config.items():
            df = data_cache[tf]
            if mode == 'Macro': t, s = calc_institutional_trend_macro(df)
            elif mode == 'Daily': t, s = calc_institutional_trend_daily(df)
            elif mode == '4H': t, s = calc_institutional_trend_4h(df)
            else: t, s = calc_institutional_trend_intraday(df)
            trends_map[tf] = t
            scores_map[tf] = s
            row_data[tf] = t

        df_daily = data_cache['D']
        atr_daily = atr(df_daily['High'], df_daily['Low'], df_daily['Close'], 14).iloc[-1]
        row_data['ATR_Daily'] = f"{atr_daily:.5f}" if atr_daily < 1 else f"{atr_daily:.2f}"
        df_h1 = data_cache['1H']
        atr_h1 = atr(df_h1['High'], df_h1['Low'], df_h1['Close'], 14).iloc[-1]
        row_data['ATR_H1'] = f"{atr_h1:.5f}" if atr_h1 < 1 else f"{atr_h1:.2f}"
        df_15m = data_cache['15m']
        atr_15m = atr(df_15m['High'], df_15m['Low'], df_15m['Close'], 14).iloc[-1]
        row_data['ATR_15m'] = f"{atr_15m:.5f}" if atr_15m < 1 else f"{atr_15m:.2f}"

        macro_trend = trends_map['M'] if trends_map['M'] != 'Range' else trends_map['W'] if trends_map['W'] != 'Range' else trends_map['D']
        if macro_trend == 'Bearish' and trends_map['1H'] == 'Bullish': trends_map['1H'] = 'Range'
        if macro_trend == 'Bullish' and trends_map['1H'] == 'Bearish': trends_map['1H'] = 'Range'
        if macro_trend == 'Bearish' and trends_map['15m'] == 'Bullish': trends_map['15m'] = 'Range'
        if macro_trend == 'Bullish' and trends_map['15m'] == 'Bearish': trends_map['15m'] = 'Range'
        row_data['1H'] = trends_map['1H']
        row_data['15m'] = trends_map['15m']

        MTF_WEIGHTS = {'M': 5.0, 'W': 4.0, 'D': 4.0, '4H': 2.5, '1H': 1.5, '15m': 1.0}
        TOTAL_WEIGHT = sum(MTF_WEIGHTS.values())
        w_bull = sum(MTF_WEIGHTS[tf] * (scores_map[tf]/100) for tf in trends_map if trends_map[tf] == 'Bullish')
        w_bear = sum(MTF_WEIGHTS[tf] * (scores_map[tf]/100) for tf in trends_map if trends_map[tf] == 'Bearish')
        w_bull += sum(MTF_WEIGHTS[tf] * 0.3 for tf in trends_map if trends_map[tf] == 'Retracement Bull')
        w_bear += sum(MTF_WEIGHTS[tf] * 0.3 for tf in trends_map if trends_map[tf] == 'Retracement Bear')
        high_tf_avg = (scores_map['M'] + scores_map['W'] + scores_map['D']) / 3
        quality = 'C'
        high_tf_clean = ('Retracement' not in trends_map['D'] and 'Retracement' not in trends_map['M'] and 'Retracement' not in trends_map['W'])
        if trends_map['D'] == trends_map['M'] == trends_map['W'] and high_tf_clean:
            if high_tf_avg >= 80: quality = 'A+'
            elif high_tf_avg >= 70: quality = 'A'
            else: quality = 'B'
        elif trends_map['D'] == trends_map['M'] and high_tf_clean:
            if high_tf_avg >= 75: quality = 'B+'
            else: quality = 'B'
        elif trends_map['D'] == trends_map['W'] and high_tf_clean:
            quality = 'B-'
        else: quality = 'C'

        if w_bull > w_bear:
            perc = (w_bull / TOTAL_WEIGHT) * 100
            final_trend = f"Bullish ({perc:.0f}%)"
        elif w_bear > w_bull:
            perc = (w_bear / TOTAL_WEIGHT) * 100
            final_trend = f"Bearish ({perc:.0f}%)"
        else: final_trend = "Range"
        row_data['MTF'] = final_trend
        row_data['Quality'] = quality
        results.append(row_data)
        bar.progress((idx + 1) / len(FOREX_PAIRS_EXTENDED))
    bar.empty()
    status.empty()
    return pd.DataFrame(results)

# ==========================================
# 8. EXPORT PDF (CORRIGÉ & FONCTIONNEL)
# ==========================================
def create_pdf(df, heatmap_data=None):
    """
    Génère un PDF compatible et fonctionnel avec Heatmap
    """
    try:
        from fpdf import FPDF
        
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 18)
                self.cell(0, 10, 'Bluestar GPS Report', 0, 1, 'C')
                self.set_font('Arial', 'I', 10)
                self.set_text_color(100, 100, 100)
                self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
                self.set_text_color(0, 0, 0)
                self.ln(3)
        
        pdf = PDF(orientation='L')  # 'L' = Landscape (Paysage)
        pdf.add_page()
        
        # ===== SECTION 1: HEATMAP FOREX =====
        if heatmap_data:
            s_forex, s_special, df_prices, pct_special = heatmap_data
            if s_forex and df_prices is not None:
                pdf.set_font("Arial", "B", 14)
                pdf.set_fill_color(30, 58, 138)
                pdf.set_text_color(255, 255, 255)
                pdf.cell(0, 10, ' FOREX STRENGTH HEATMAP', 0, 1, 'L', True)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(3)
                
                # Calculer les forces relatives
                currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
                pct_changes = df_prices.pct_change().iloc[-1] * 100
                forex_strength = {}
                
                for curr in currencies:
                    total_pct = 0
                    count = 0
                    for col in df_prices.columns:
                        if curr in col:
                            val = pct_changes[col]
                            if col.startswith(curr):
                                total_pct += val
                            else:
                                total_pct -= val
                            count += 1
                    if count > 0:
                        forex_strength[curr] = total_pct / count
                
                # Trier par force
                sorted_curr = sorted(forex_strength.items(), key=lambda x: x[1], reverse=True)
                
                # Tableau des forces
                pdf.set_font("Arial", "B", 9)
                col_w = 35
                for curr, strength in sorted_curr:
                    # Couleur selon force
                    if strength > 0.05:
                        pdf.set_fill_color(0, 153, 0)  # Vert foncé
                    elif strength > 0:
                        pdf.set_fill_color(51, 204, 51)  # Vert clair
                    elif strength < -0.05:
                        pdf.set_fill_color(204, 0, 0)  # Rouge foncé
                    elif strength < 0:
                        pdf.set_fill_color(255, 51, 0)  # Rouge clair
                    else:
                        pdf.set_fill_color(240, 240, 240)  # Neutre
                    
                    pdf.set_text_color(255, 255, 255) if abs(strength) > 0.01 else pdf.set_text_color(0, 0, 0)
                    pdf.cell(col_w, 8, f'{curr}: {strength:+.2f}%', 1, 0, 'C', True)
                
                pdf.set_text_color(0, 0, 0)
                pdf.ln(15)
                
                # Indices & Métaux
                if pct_special:
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(100, 8, 'INDICES', 0, 0, 'L')
                    pdf.cell(100, 8, 'METAUX', 0, 1, 'L')
                    pdf.set_font("Arial", "", 9)
                    
                    indices_list = [k for k, v in pct_special.items() if v['cat'] == 'INDICES']
                    metaux_list = [k for k, v in pct_special.items() if v['cat'] == 'METAUX']
                    
                    max_len = max(len(indices_list), len(metaux_list))
                    
                    for i in range(max_len):
                        # Indices
                        if i < len(indices_list):
                            name = indices_list[i]
                            pct = pct_special[name]['pct']
                            if pct > 0.15:
                                pdf.set_fill_color(0, 153, 0)
                            elif pct > 0:
                                pdf.set_fill_color(51, 204, 51)
                            elif pct < -0.15:
                                pdf.set_fill_color(204, 0, 0)
                            elif pct < 0:
                                pdf.set_fill_color(255, 51, 0)
                            else:
                                pdf.set_fill_color(240, 240, 240)
                            
                            pdf.set_text_color(255, 255, 255) if abs(pct) > 0.01 else pdf.set_text_color(0, 0, 0)
                            pdf.cell(100, 7, f'{name}: {pct:+.2f}%', 1, 0, 'C', True)
                        else:
                            pdf.cell(100, 7, '', 0, 0)
                        
                        # Métaux
                        if i < len(metaux_list):
                            name = metaux_list[i]
                            pct = pct_special[name]['pct']
                            if pct > 0.15:
                                pdf.set_fill_color(0, 153, 0)
                            elif pct > 0:
                                pdf.set_fill_color(51, 204, 51)
                            elif pct < -0.15:
                                pdf.set_fill_color(204, 0, 0)
                            elif pct < 0:
                                pdf.set_fill_color(255, 51, 0)
                            else:
                                pdf.set_fill_color(240, 240, 240)
                            
                            pdf.set_text_color(255, 255, 255) if abs(pct) > 0.01 else pdf.set_text_color(0, 0, 0)
                            pdf.cell(100, 7, f'{name}: {pct:+.2f}%', 1, 1, 'C', True)
                        else:
                            pdf.ln()
                
                pdf.set_text_color(0, 0, 0)
                pdf.ln(10)
        
        # ===== SECTION 2: TABLEAU GPS =====
        pdf.set_font("Arial", "B", 14)
        pdf.set_fill_color(30, 58, 138)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, ' INSTITUTIONAL GPS ANALYSIS', 0, 1, 'L', True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)
        
        # Configuration des colonnes (optimisée pour paysage)
        cols = ['Paire', 'M', 'W', 'D', '4H', '1H', '15m', 'MTF', 'Quality', 'ATR_Daily', 'ATR_H1', 'ATR_15m']
        col_widths = {
            'Paire': 22,
            'M': 19, 'W': 19, 'D': 19, '4H': 19, '1H': 19, '15m': 19,
            'MTF': 30,
            'Quality': 18,
            'ATR_Daily': 19, 'ATR_H1': 19, 'ATR_15m': 19
        }
        
        # En-têtes du tableau
        pdf.set_fill_color(30, 58, 138)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 9)  # Police légèrement plus grande en paysage
        for col in cols:
            pdf.cell(col_widths[col], 8, col.replace('_', ' '), 1, 0, 'C', True)
        pdf.ln()
        
        # Lignes de données
        pdf.set_font("Arial", "", 8)  # Police des données
        for idx, row in df.iterrows():
            for col in cols:
                val = str(row[col])
                
                # Couleurs selon le contenu
                if col in ['M', 'W', 'D', '4H', '1H', '15m', 'MTF']:
                    if "Bull" in val and "Retracement" not in val:
                        pdf.set_fill_color(46, 204, 113)  # Bullish
                        pdf.set_text_color(255, 255, 255)
                    elif "Bear" in val and "Retracement" not in val:
                        pdf.set_fill_color(231, 76, 60)  # Bearish
                        pdf.set_text_color(255, 255, 255)
                    elif "Retracement Bull" in val:
                        pdf.set_fill_color(125, 206, 160)
                        pdf.set_text_color(255, 255, 255)
                    elif "Retracement Bear" in val:
                        pdf.set_fill_color(241, 148, 138)
                        pdf.set_text_color(255, 255, 255)
                    elif "Range" in val:
                        pdf.set_fill_color(149, 165, 166)
                        pdf.set_text_color(255, 255, 255)
                    else:
                        pdf.set_fill_color(255, 255, 255)
                        pdf.set_text_color(0, 0, 0)
                elif col == 'Quality':
                    grade_colors = {
                        'A+': (251, 191, 36),
                        'A': (163, 230, 53),
                        'B+': (96, 165, 250),
                        'B': (59, 130, 246),
                        'B-': (59, 130, 246),
                        'C': (156, 163, 175)
                    }
                    color = grade_colors.get(val, (255, 255, 255))
                    pdf.set_fill_color(*color)
                    pdf.set_text_color(0, 0, 0)
                else:
                    pdf.set_fill_color(255, 255, 255)
                    pdf.set_text_color(0, 0, 0)
                
                # Tronquer le texte si trop long
                if len(val) > 15:
                    val = val[:13] + '..'
                
                pdf.cell(col_widths[col], 7, val, 1, 0, 'C', True)
            
            pdf.ln()
            pdf.set_text_color(0, 0, 0)
            
            # Nouvelle page si nécessaire
            if pdf.get_y() > 180:  # Limite ajustée pour paysage
                pdf.add_page()
                pdf.add_page()
                pdf.set_font("Arial", "B", 8)
                pdf.set_fill_color(30, 58, 138)
                pdf.set_text_color(255, 255, 255)
                for col in cols:
                    pdf.cell(col_widths[col], 8, col.replace('_', ' '), 1, 0, 'C', True)
                pdf.ln()
                pdf.set_font("Arial", "", 7)
        
        # Génération du buffer
        buffer = BytesIO()
        pdf_output = pdf.output(dest='S')
        
        # Gestion compatibilité fpdf v1 vs v2
        if isinstance(pdf_output, str):
            buffer.write(pdf_output.encode('latin-1'))
        else:
            buffer.write(pdf_output)
        
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        logging.error(f"Erreur critique PDF : {e}")
        # En cas d'erreur, créer un PDF minimal valide
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Erreur generation PDF: {str(e)}", 0, 1)
            buffer = BytesIO()
            pdf_output = pdf.output(dest='S')
            if isinstance(pdf_output, str):
                buffer.write(pdf_output.encode('latin-1'))
            else:
                buffer.write(pdf_output)
            buffer.seek(0)
            return buffer.getvalue()
        except:
            return b"PDF Error"

# ==========================================
# 9. UI PRINCIPALE
# ==========================================
def main():
    st.markdown("""
        <div class='main-header'>
            <h2 style='margin:0'>🗺️ BLUESTAR GPS V2.3 ULTIMATE</h2>
            <div style='font-size:0.9rem; color:#cbd5e1;'>Macro Heatmap & Institutional Grades</div>
        </div>
    """, unsafe_allow_html=True)

    # Connexion API
    try:
        acc = st.secrets["OANDA_ACCOUNT_ID"]
        tok = st.secrets["OANDA_ACCESS_TOKEN"]
        env = st.secrets.get("OANDA_ENVIRONMENT", "practice")
    except Exception:
        st.error("❌ Secrets OANDA manquants")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        granularity = st.selectbox("Timeframe Heatmap", ["M5", "M15", "M30", "H1", "H4", "D"], index=3)
        st.info("Sticky Results V2.3 Activé")

    # --- LOGIQUE D'ANALYSE ---
    if st.button("🚀 LANCER L'ANALYSE", type="primary", use_container_width=True, key="run_scan_button"):
        with st.spinner("⏳ Analyse GPS & Heatmap en cours..."):
            # 1. GPS
            df_gps = analyze_market(tok, env)
            # 2. Heatmap
            s_forex, s_special, df_prices, pct_special = calculate_heatmap_data(tok, env, gran=granularity)
        
        # MISE A JOUR IMMÉDIATE DE LA SESSION (Sticky Results)
        if not df_gps.empty:
            st.session_state['df'] = df_gps
            st.session_state['heatmap_data'] = (s_forex, s_special, df_prices, pct_special)
            st.success(f"Analyse terminée : {len(df_gps)} setups détectés")

    # --- AFFICHAGE DES RÉSULTATS (PERSISTANT) ---
    if st.session_state.get('df') is not None:
        df = st.session_state['df']
        data_heat = st.session_state.get('heatmap_data')
        
        # -- AFFICHAGE HEATMAP --
        st.markdown('<div class="section-title">⚡ MARKET HEATMAP (Momentum)</div>', unsafe_allow_html=True)
        if data_heat:
            s_forex, s_special, df_prices, pct_special = data_heat
            if s_forex and df_prices is not None:
                html_map = generate_exact_map_html(df_prices, pct_special)
                st.components.v1.html(html_map, height=600, scrolling=True)
        
        st.markdown("---")
        
        # -- AFFICHAGE GPS --
        st.markdown('<div class="section-title">🏛️ INSTITUTIONAL GPS (Structure)</div>', unsafe_allow_html=True)
        
        # Metrics
        total = len(df)
        a_plus = len(df[df['Quality'] == 'A+'])
        a_grade = len(df[df['Quality'] == 'A'])
        b_grade = len(df[df['Quality'].str.startswith('B')])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", total)
        c2.metric("Setups A+", a_plus)
        c3.metric("Setups A", a_grade)
        c4.metric("Setups B", b_grade)
        
        # Tableau
        cols_order = ['Paire', 'M', 'W', 'D', '4H', '1H', '15m', 'MTF', 'Quality', 'ATR_Daily', 'ATR_H1', 'ATR_15m']
        
        # Styles
        def style_map(v):
            if isinstance(v, str):
                if "Bull" in v and "Retracement" not in v: return f"background-color: {TREND_COLORS['Bullish']}; color:white; font-weight:bold"
                if "Bear" in v and "Retracement" not in v: return f"background-color: {TREND_COLORS['Bearish']}; color:white; font-weight:bold"
                if "Retracement Bull" in v: return f"background-color: {TREND_COLORS['Retracement Bull']}; color:white"
                if "Retracement Bear" in v: return f"background-color: {TREND_COLORS['Retracement Bear']}; color:white"
                if "Range" in v: return f"background-color: {TREND_COLORS['Range']}; color:white"
            return ""

        def quality_style(s):
            if s.name == 'Quality':
                return [f"color: black; font-weight:bold; background-color: {GRADE_COLORS.get(x, 'grey')}" for x in s]
            return [''] * len(s)

        # Calcul hauteur dynamique
        h = min(600, (len(df) + 1) * 35 + 3)
        
        st.dataframe(
            df[cols_order].style.apply(quality_style, axis=0).applymap(style_map), 
            height=h, 
            use_container_width=True
        )
        
        # -- EXPORTS --
        c1, c2 = st.columns(2)
        with c1:
            # Passer les données heatmap au PDF
            heatmap_data = st.session_state.get('heatmap_data')
            st.download_button(
                "📄 Télécharger PDF", 
                create_pdf(df[cols_order], heatmap_data),
                "Bluestar_GPS_Report.pdf", 
                "application/pdf",
                use_container_width=True,
                key="download_pdf_v23"
            )
        with c2:
            st.download_button(
                "📊 Télécharger CSV", 
                df[cols_order].to_csv(index=False).encode(), 
                "Bluestar_GPS.csv", 
                "text/csv",
                use_container_width=True,
                key="download_csv_v23"
            )

if __name__ == "__main__":
    main()

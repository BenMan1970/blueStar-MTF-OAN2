# app.py - Bluestar Hedge Fund GPS (Version avec ATR Daily, H1 et 15m)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Bluestar Hedge Fund GPS",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== PARAMÈTRES INSTITUTIONNELS ====================
OANDA_API_URL = "https://api-fxpractice.oanda.com"

FOREX_PAIRS_EXTENDED = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
    'EURGBP', 'EURJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD',
    'GBPJPY', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD',
    'AUDJPY', 'AUDCAD', 'AUDCHF', 'AUDNZD',
    'CADJPY', 'CADCHF', 'CHFJPY',
    'NZDJPY', 'NZDCAD', 'NZDCHF',
    'XAUUSD', 'XPTUSD', 'US30USD', 'SPX500USD', 'NAS100USD'
]

# Couleurs Visuelles
TREND_COLORS = {
    'Bullish': '#2ecc71',     # Vert Vif
    'Bearish': '#e74c3c',     # Rouge Vif
    'Retracement': '#f39c12', # Orange
    'Range': '#95a5a6'        # Gris
}

# Poids pour le Score MTF global
MTF_WEIGHTS = {'M': 3.0, 'W': 2.5, 'D': 4.0, '4H': 3.0, '1H': 1.5, '15m': 1.0}
TOTAL_WEIGHT = sum(MTF_WEIGHTS.values())

# ==================== INDICATEURS ====================
def sma(series, length):
    return series.rolling(window=length).mean()

def zlema(series, length):
    lag = int((length - 1) / 2)
    src_adj = series + (series - series.shift(lag))
    return src_adj.ewm(span=length, adjust=False).mean()

def adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    atr_s = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_s)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_s)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.ewm(span=period, adjust=False).mean(), plus_di, minus_di

def atr(high, low, close, period=14):
    """Calcule l'Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

# ==================== LOGIQUE "GPS" (MACRO) ====================
def calc_macro_trend(df):
    """Logique SMA 200 stricte pour Monthly/Weekly (Pas de Range sauf exception)"""
    if len(df) < 50: return 'Neutral', 0
    
    close = df['Close']
    curr_price = close.iloc[-1]
    
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    
    curr_sma50 = sma50.iloc[-1]
    has_200 = len(df) >= 200
    curr_sma200 = sma200.iloc[-1] if has_200 else None

    trend = "Neutral"
    score = 0

    if has_200:
        if curr_price > curr_sma200:
            trend = "Bullish"
            score = 60
            if curr_price > curr_sma50: score += 20
            if curr_sma50 > curr_sma200: score += 20
        else:
            trend = "Bearish"
            score = 60
            if curr_price < curr_sma50: score += 20
            if curr_sma50 < curr_sma200: score += 20
    else:
        if curr_price > curr_sma50:
            trend = "Bullish"; score = 50
        else:
            trend = "Bearish"; score = 50
            
    return trend, min(100, score)

# ==================== LOGIQUE "TIMING" (INTRADAY) ====================
def calc_intraday_trend(df):
    """Logique ZLEMA + Baseline pour Daily -> 15m"""
    if len(df) < 50: return 'Range', 0
    
    close = df['Close']
    zlema_val = zlema(close, 50)
    baseline = sma(close, 200)
    adx_val, _, _ = adx(df['High'], df['Low'], close, 14)
    
    curr_price = close.iloc[-1]
    curr_zlema = zlema_val.iloc[-1]
    curr_adx = adx_val.iloc[-1]
    
    has_base = len(df) >= 200
    curr_base = baseline.iloc[-1] if has_base else curr_zlema

    trend = "Range"
    
    # Logique de Retracement vs Tendance
    if curr_price > curr_zlema:
        if has_base and curr_price > curr_base: trend = "Bullish"
        elif has_base and curr_price < curr_base: trend = "Retracement" # Hausse sous la 200
        else: trend = "Bullish"
    elif curr_price < curr_zlema:
        if has_base and curr_price < curr_base: trend = "Bearish"
        elif has_base and curr_price > curr_base: trend = "Retracement" # Baisse au dessus de la 200
        else: trend = "Bearish"

    if curr_adx < 20 and trend == "Retracement": trend = "Range"
    
    score = curr_adx
    return trend, score

# ==================== DATA FETCHING ====================
def get_oanda_data(instrument, granularity, count, account_id, access_token):
    url = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {'granularity': granularity, 'count': count, 'price': 'M'}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code != 200: return pd.DataFrame()
        data = r.json()
        if 'candles' not in data: return pd.DataFrame()
        df = pd.DataFrame([{
            'date': c['time'], 'Open': float(c['mid']['o']),
            'High': float(c['mid']['h']), 'Low': float(c['mid']['l']),
            'Close': float(c['mid']['c'])
        } for c in data['candles'] if c.get('complete')])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_oanda_data(inst, gran, cnt, acc, tok):
    return get_oanda_data(inst, gran, cnt, acc, tok)

# ==================== CORE ANALYTICS ====================
def analyze_market(account_id, access_token):
    results = []
    tf_config = {
        'M':   ('D', 4500, 'Macro'),
        'W':   ('D', 2000, 'Macro'),
        'D':   ('D', 300,  'Intra'),
        '4H':  ('H4', 300, 'Intra'),
        '1H':  ('H1', 300, 'Intra'),
        '15m': ('M15', 300,'Intra')
    }
    
    bar = st.progress(0)
    status = st.empty()
    
    for idx, pair in enumerate(FOREX_PAIRS_EXTENDED):
        # Gestion des noms spéciaux pour les indices et métaux
        if pair in ['XAUUSD', 'XPTUSD']:
            inst = f"{pair[:3]}_{pair[3:]}"
            display_name = f"{pair[:3]}/USD"
        elif pair in ['US30USD', 'SPX500USD', 'NAS100USD']:
            inst = pair
            display_name = pair.replace('USD', '')
        else:
            inst = f"{pair[:3]}_{pair[3:]}"
            display_name = f"{pair[:3]}/{pair[3:]}"
        
        status.text(f"GPS Institutionnel : {display_name}...")
        
        row_data = {'Paire': display_name}
        trends_map = {}
        scores_map = {}
        valid_pair = True
        
        data_cache = {}
        for tf, (gran, count, _) in tf_config.items():
            df = get_cached_oanda_data(inst, gran, count, account_id, access_token)
            if df.empty: valid_pair = False; break
            
            if tf == 'M':
                df = df.resample('ME').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            elif tf == 'W':
                df = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            data_cache[tf] = df

        if not valid_pair: continue

        for tf, (_, _, mode) in tf_config.items():
            df = data_cache[tf]
            t, s = calc_macro_trend(df) if mode == 'Macro' else calc_intraday_trend(df)
            trends_map[tf] = t
            scores_map[tf] = s
            row_data[tf] = t

        # Calcul de l'ATR Daily, H1 et 15m
        df_daily = data_cache['D']
        atr_daily = atr(df_daily['High'], df_daily['Low'], df_daily['Close'], 14).iloc[-1]
        row_data['ATR_Daily'] = f"{atr_daily:.5f}" if atr_daily < 1 else f"{atr_daily:.2f}"
        
        df_h1 = data_cache['1H']
        atr_h1 = atr(df_h1['High'], df_h1['Low'], df_h1['Close'], 14).iloc[-1]
        row_data['ATR_H1'] = f"{atr_h1:.5f}" if atr_h1 < 1 else f"{atr_h1:.2f}"
        
        df_15m = data_cache['15m']
        atr_15m = atr(df_15m['High'], df_15m['Low'], df_15m['Close'], 14).iloc[-1]
        row_data['ATR_15m'] = f"{atr_15m:.5f}" if atr_15m < 1 else f"{atr_15m:.2f}"

        w_bull = sum(MTF_WEIGHTS[tf] for tf in trends_map if trends_map[tf] == 'Bullish')
        w_bear = sum(MTF_WEIGHTS[tf] for tf in trends_map if trends_map[tf] == 'Bearish')
        
        quality = 'C'
        if trends_map['D'] == trends_map['M']: quality = 'B'
        if trends_map['D'] == trends_map['M'] == trends_map['W']: quality = 'A'
        if quality == 'A' and scores_map['D'] > 25: quality = 'A+'

        if w_bull > w_bear:
            perc = (w_bull / TOTAL_WEIGHT) * 100
            final_trend = f"Bullish ({perc:.0f}%)"
        elif w_bear > w_bull:
            perc = (w_bear / TOTAL_WEIGHT) * 100
            final_trend = f"Bearish ({perc:.0f}%)"
        else:
            final_trend = "Range"

        row_data['MTF'] = final_trend
        row_data['Quality'] = quality
        results.append(row_data)
        bar.progress((idx + 1) / len(FOREX_PAIRS_EXTENDED))
        
    bar.empty(); status.empty()
    return pd.DataFrame(results)

# ==================== EXPORTS (FIX PDF) ====================
def create_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Bluestar GPS Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(5)
    
    cols = ['Paire', 'M', 'W', 'D', '4H', '1H', '15m', 'MTF', 'Quality', 'ATR_Daily', 'ATR_H1', 'ATR_15m']
    w = pdf.w / (len(cols)+1)
    
    pdf.set_font("Helvetica", "B", 6)
    for c in cols: 
        pdf.cell(w, 8, c.replace('_', ' '), border=1, align='C', new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln()
    
    pdf.set_font("Helvetica", "", 6)
    for _, row in df.iterrows():
        for c in cols:
            val = str(row[c])
            pdf.set_fill_color(255,255,255)
            if "Bull" in val: pdf.set_fill_color(46, 204, 113)
            elif "Bear" in val: pdf.set_fill_color(231, 76, 60)
            elif "Retr" in val: pdf.set_fill_color(243, 156, 18)
            elif "Range" in val: pdf.set_fill_color(149, 165, 166)
            
            pdf.cell(w, 8, val, border=1, align='C', fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln()
    
    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()

# ==================== MAIN UI ====================
def main():
    st.markdown("""
        <div style='text-align:center; padding:15px; background:#2c3e50; color:white; border-radius:10px; margin-bottom:15px'>
            <h2 style='margin:0'>🏛️ Bluestar Hedge Fund GPS</h2>
        </div>
    """, unsafe_allow_html=True)

    try:
        acc = st.secrets["OANDA_ACCOUNT_ID"]
        tok = st.secrets["OANDA_ACCESS_TOKEN"]
    except: st.error("Secrets OANDA manquants."); st.stop()

    if st.button("LANCER L'ANALYSE TOP-DOWN", type="primary", use_container_width=True):
        with st.spinner("Analyse SMA 200 & Structures de marché..."):
            df = analyze_market(acc, tok)
            if not df.empty:
                df = df.sort_values(by=['Quality', 'MTF'], ascending=[True, False]) 
                st.session_state.df = df
    
    if "df" in st.session_state:
        df = st.session_state.df
        cols_order = ['Paire', 'M', 'W', 'D', '4H', '1H', '15m', 'MTF', 'Quality', 'ATR_Daily', 'ATR_H1', 'ATR_15m']
        
        def style_map(v):
            if isinstance(v, str):
                if "Bull" in v: return f"background-color: {TREND_COLORS['Bullish']}; color:white; font-weight:bold"
                if "Bear" in v: return f"background-color: {TREND_COLORS['Bearish']}; color:white; font-weight:bold"
                if "Retr" in v: return f"background-color: {TREND_COLORS['Retracement']}; color:white"
                if "Range" in v: return f"background-color: {TREND_COLORS['Range']}; color:white"
            return ""

        h = (len(df) + 1) * 35 + 3
        st.dataframe(df[cols_order].style.map(style_map), height=h, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1: st.download_button("PDF Report", create_pdf(df[cols_order]), "Bluestar_GPS.pdf", "application/pdf")
        with c2: st.download_button("CSV Data", df[cols_order].to_csv(index=False).encode(), "Bluestar_GPS.csv", "text/csv")

if __name__ == "__main__":
    main()

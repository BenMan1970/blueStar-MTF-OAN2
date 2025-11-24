# app.py - Bluestar MTF Pro+ (Version Top-Down Institutionnelle)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ==================== CONFIGURATION GLOBALE ====================
st.set_page_config(
    page_title="Bluestar Institutional",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CONSTANTES & PARAMÈTRES ====================
OANDA_API_URL = "https://api-fxpractice.oanda.com"

# Liste des paires
FOREX_PAIRS_EXTENDED = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
    'EURGBP', 'EURJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD',
    'GBPJPY', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD',
    'AUDJPY', 'AUDCAD', 'AUDCHF', 'AUDNZD',
    'CADJPY', 'CADCHF', 'CHFJPY',
    'NZDJPY', 'NZDCAD', 'NZDCHF'
]

# Couleurs (Net & Pro)
TREND_COLORS_HEX = {
    'Bullish': '#2ecc71',   # Vert
    'Bearish': '#e74c3c',   # Rouge
    'Correction': '#f39c12',# Orange
    'Range': '#95a5a6'      # Gris
}

# Paramètres Techniques
LENGTH_ZLEMA = 50       
LENGTH_BASELINE = 200   
ADX_PERIOD = 14

# Pondération (Toujours focus H4/D pour l'action, mais M/W pour le biais)
MTF_WEIGHTS = {
    'M': 2.0, 'W': 3.0, 'D': 5.0, '4H': 4.0, '1H': 2.0, '15m': 1.0
}
TOTAL_WEIGHT = sum(MTF_WEIGHTS.values())

# ==================== INDICATEURS TECHNIQUES ====================
def zlema(series, length):
    lag = int((length - 1) / 2)
    src_adj = series + (series - series.shift(lag))
    return src_adj.ewm(span=length, adjust=False).mean()

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def atr(high, low, close, period):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

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

# ==================== LOGIQUE INSTITUTIONNELLE ====================
def calc_professional_trend(df, timeframe='D'):
    if df.empty or len(df) < LENGTH_ZLEMA + 5:
        return 'Range', 0, 'C'

    close = df['Close']
    high = df['High']
    low = df['Low']

    zlema_val = zlema(close, LENGTH_ZLEMA)
    adx_val, plus_di, minus_di = adx(high, low, close, ADX_PERIOD)
    
    has_baseline = len(df) >= LENGTH_BASELINE
    baseline = ema(close, LENGTH_BASELINE) if has_baseline else pd.Series(0, index=close.index)

    curr_price = close.iloc[-1]
    curr_zlema = zlema_val.iloc[-1]
    curr_adx = adx_val.iloc[-1]
    curr_di_plus = plus_di.iloc[-1]
    curr_di_minus = minus_di.iloc[-1]
    curr_baseline = baseline.iloc[-1] if has_baseline else 0

    trend = "Range"
    strength_score = 0
    
    if has_baseline:
        # BULLISH
        if curr_price > curr_zlema:
            if curr_price > curr_baseline:
                trend = "Bullish"
                strength_score = 60
                if curr_zlema > curr_baseline: strength_score += 20
            else:
                trend = "Correction" if curr_di_plus > curr_di_minus else "Range"
                strength_score = 40
        # BEARISH
        elif curr_price < curr_zlema:
            if curr_price < curr_baseline:
                trend = "Bearish"
                strength_score = 60
                if curr_zlema < curr_baseline: strength_score += 20
            else:
                trend = "Correction" if curr_di_minus > curr_di_plus else "Range"
                strength_score = 40
        
        # Filtre Squeeze
        dist_ma = abs(curr_zlema - curr_baseline) / curr_baseline * 100
        if dist_ma < 0.2 and curr_adx < 20: 
            trend = "Range"
    else:
        # Fallback si pas assez de data (ex: Monthly court)
        if curr_price > curr_zlema and curr_di_plus > curr_di_minus:
            trend = "Bullish"; strength_score = 50
        elif curr_price < curr_zlema and curr_di_minus > curr_di_plus:
            trend = "Bearish"; strength_score = 50
        else:
            trend = "Range"

    if curr_adx > 25: strength_score += 10
    if curr_adx > 40: strength_score += 10

    strength_score = min(100, strength_score)
    if strength_score >= 80: quality = 'A+'
    elif strength_score >= 65: quality = 'A'
    elif strength_score >= 50: quality = 'B'
    else: quality = 'C'

    if trend == "Correction":
        if has_baseline and curr_price > curr_baseline: trend = "Range"
        elif has_baseline and curr_price < curr_baseline: trend = "Range"
    
    return trend, strength_score, quality

# ==================== DATA OANDA ====================
def get_oanda_data(instrument, granularity, count, account_id, access_token):
    url = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {'granularity': granularity, 'count': count, 'price': 'M'}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code != 200: return pd.DataFrame()
        data = r.json()
        if 'candles' not in data: return pd.DataFrame()
        candles = [c for c in data['candles'] if c.get('complete')]
        if not candles: return pd.DataFrame()
        
        df = pd.DataFrame([{
            'date': c['time'], 'Open': float(c['mid']['o']),
            'High': float(c['mid']['h']), 'Low': float(c['mid']['l']),
            'Close': float(c['mid']['c'])
        } for c in candles])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_oanda_data(instrument, granularity, count, account_id, access_token):
    return get_oanda_data(instrument, granularity, count, account_id, access_token)

# ==================== MOTEUR ANALYSE ====================
def analyze_forex_pairs(account_id, access_token):
    results = []
    # Paramètres Top-Down
    tf_params = {
        'M': ('D', 2500), 'W': ('D', 1500), 'D': ('D', 400),
        '4H': ('H4', 400), '1H': ('H1', 400), '15m': ('M15', 400)
    }
    
    total = len(FOREX_PAIRS_EXTENDED)
    bar = st.progress(0)
    status = st.empty()

    for idx, pair in enumerate(FOREX_PAIRS_EXTENDED):
        instrument = f"{pair[:3]}_{pair[3:]}"
        status.text(f"Analyse Top-Down : {pair}...")
        data = {}
        ok = True

        for tf, (g, c) in tf_params.items():
            df = get_cached_oanda_data(instrument, g, c, account_id, access_token)
            if tf == 'W' and not df.empty:
                df = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            if tf == 'M' and not df.empty:
                df = df.resample('ME').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            
            if df.empty: ok = False; break
            data[tf] = df

        if not ok: continue

        trends = {tf: calc_professional_trend(data[tf], tf)[0] for tf in tf_params}
        quality_d = calc_professional_trend(data['D'], 'D')[2]
        
        score_bull = sum(MTF_WEIGHTS[tf] for tf in trends if trends[tf] == 'Bullish')
        score_bear = sum(MTF_WEIGHTS[tf] for tf in trends if trends[tf] == 'Bearish')
        
        if score_bull > score_bear and score_bull >= (TOTAL_WEIGHT * 0.4):
            mtf_trend = 'Bullish'; power = (score_bull / TOTAL_WEIGHT) * 100
        elif score_bear > score_bull and score_bear >= (TOTAL_WEIGHT * 0.4):
            mtf_trend = 'Bearish'; power = (score_bear / TOTAL_WEIGHT) * 100
        else:
            mtf_trend = 'Range'; power = 0

        results.append({
            'Paire': f"{pair[:3]}/{pair[3:]}",
            'M': trends['M'], 
            'W': trends['W'], 
            'D': trends['D'],
            '4H': trends['4H'], 
            '1H': trends['1H'], 
            '15m': trends['15m'],
            'MTF': f"{mtf_trend} ({power:.0f}%)",
            'Quality': quality_d
        })
        bar.progress((idx + 1) / total)

    bar.empty(); status.empty()
    if not results: return pd.DataFrame()
    
    df = pd.DataFrame(results)
    # Tri par défaut : Bullish Forts en haut
    return df.sort_values(by=['MTF', 'Quality'], ascending=[False, True])

# ==================== EXPORTS ====================
def create_pdf_report(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Bluestar Institutional - Top Down Analysis", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(5)
    
    # Ordre Colonnes PDF
    cols = ['Paire', 'M', 'W', 'D', '4H', '1H', '15m', 'MTF', 'Quality']
    col_w = pdf.w / (len(cols) + 1)
    
    pdf.set_font("Helvetica", "B", 8)
    for col in cols:
        pdf.cell(col_w, 8, col, border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln()
    
    pdf.set_font("Helvetica", "", 7)
    for _, row in df.iterrows():
        for col in cols:
            val_str = str(row[col])
            fill = False
            pdf.set_fill_color(255, 255, 255); pdf.set_text_color(0, 0, 0)
            if "Bullish" in val_str:
                pdf.set_fill_color(46, 204, 113); pdf.set_text_color(255,255,255); fill=True
            elif "Bearish" in val_str:
                pdf.set_fill_color(231, 76, 60); pdf.set_text_color(255,255,255); fill=True
            elif "Range" in val_str:
                pdf.set_fill_color(149, 165, 166); pdf.set_text_color(255,255,255); fill=True
            pdf.cell(col_w, 8, val_str, border=1, align="C", fill=fill, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln()
    
    buffer = BytesIO()
    pdf.output(buffer); buffer.seek(0)
    return buffer.getvalue()

def create_png_report(df):
    text = f"Bluestar Top-Down Feed - {datetime.now():%d/%m/%Y %H:%M}\n" + "="*90 + "\n"
    # Forcer l'ordre des colonnes pour le PNG
    cols = ['Paire', 'M', 'W', 'D', '4H', '1H', '15m', 'MTF', 'Quality']
    text += df[cols].to_string(index=False)
    
    font = ImageFont.load_default()
    lines = text.split('\n')
    h = len(lines) * 15 + 50
    img = Image.new('RGB', (1500, h), 'white')
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((20, 20 + i*15), line, fill='black', font=font)
    buf = BytesIO(); img.save(buf, 'PNG')
    return buf.getvalue()

# ==================== MAIN ====================
def main():
    st.markdown("""
    <div style="text-align:center;padding:15px;background-color:#2c3e50;color:white;border-radius:10px;margin-bottom:20px">
        <h2 style='margin:0'>🏛️ Bluestar Top-Down Dashboard</h2>
        <p style='margin:5px 0 0 0; font-size:0.9rem'>M > W > D > H4 > H1 > 15m • Flux Institutionnel</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
    except:
        st.error("Configurez les secrets OANDA.")
        st.stop()

    if st.button("🔄 SCANNER (M -> 15m)", type="primary", use_container_width=True):
        with st.spinner("Analyse des structures majeures..."):
            df = analyze_forex_pairs(account_id, access_token)
        if not df.empty:
            st.session_state.df = df
            st.success("Données actualisées.")

    if "df" in st.session_state:
        df = st.session_state.df
        
        # RÉORGANISATION DES COLONNES (L'ordre demandé)
        cols_ordered = ['Paire', 'M', 'W', 'D', '4H', '1H', '15m', 'MTF', 'Quality']
        df_display = df[cols_ordered]

        def color_cell(val):
            if isinstance(val, str):
                if "Bullish" in val: return f"background-color: {TREND_COLORS_HEX['Bullish']}; color: white; font-weight: bold"
                if "Bearish" in val: return f"background-color: {TREND_COLORS_HEX['Bearish']}; color: white; font-weight: bold"
                if "Range" in val: return f"background-color: {TREND_COLORS_HEX['Range']}; color: white"
            return ""
        
        # CALCUL HAUTEUR DYNAMIQUE (Pour enlever la scrollbar)
        # ~35px par ligne + header + padding buffer
        height_dynamic = (len(df_display) + 1) * 35 + 3

        st.dataframe(
            df_display.style.map(color_cell),
            height=height_dynamic, # Hauteur exacte du contenu
            use_container_width=True
        )

        c1, c2, c3 = st.columns(3)
        now = datetime.now().strftime("%H%M")
        with c1: st.download_button("PDF", create_pdf_report(df_display), f"TopDown_{now}.pdf", "application/pdf")
        with c2: st.download_button("PNG", create_png_report(df_display), f"TopDown_{now}.png", "image/png")
        with c3: st.download_button("CSV", df_display.to_csv(index=False).encode(), f"TopDown_{now}.csv", "text/csv")

if __name__ == "__main__":
    main()

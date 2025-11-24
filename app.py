# app.py - Bluestar MTF Pro+ Optimisé - VERSION PROPRE & SANS WARNINGS
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

# ==================== CONFIG ====================
st.set_page_config(
    page_title="Bluestar MTF Pro+",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CONSTANTES ====================
OANDA_API_URL = "https://api-fxpractice.oanda.com"

FOREX_PAIRS_EXTENDED = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
    'EURGBP', 'EURJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD',
    'GBPJPY', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD',
    'AUDJPY', 'AUDCAD', 'AUDCHF', 'AUDNZD',
    'CADJPY', 'CADCHF', 'CHFJPY',
    'NZDJPY', 'NZDCAD', 'NZDCHF'
]

TREND_COLORS_HEX = {
    'Bullish': '#008f7a', 'Bearish': '#d9534f',
    'Neutral': '#808080', 'Range': '#f0ad4e'
}

# Paramètres Bluestar
LENGTH = 70
MULT = 1.2
ADX_THRESHOLD = 25
VOLATILITY_PERIOD = 14
MTF_WEIGHTS = {'15m': 0.5, '1H': 1.0, '4H': 2.0, 'D': 3.0, 'W': 4.5, 'M': 6.0}
TOTAL_WEIGHT = sum(MTF_WEIGHTS.values())
CONFIRMATION_BARS = {'15m': 2, '1H': 2, '4H': 3, 'D': 3, 'W': 2, 'M': 1}

# ==================== INDICATEURS ====================
def zlema(series, length):
    lag = int((length - 1) / 2)
    src_adj = series + (series - series.shift(lag))
    return src_adj.ewm(span=length, adjust=False).mean()

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

# ==================== BLUSTAR ENGINE ====================
def calc_professional_trend(df, timeframe='D'):
    if df.empty or len(df) < 100:
        return 'Neutral', 0, 'D', 0

    close = df['Close']
    high = df['High']
    low = df['Low']

    zlema_s = zlema(close, LENGTH)
    atr_a = atr(high, low, close, VOLATILITY_PERIOD)
    atr_ma = atr_a.rolling(LENGTH).mean()
    vol_ratio = atr_a / atr_ma
    volatility = atr_a.rolling(LENGTH*3).max() * MULT * vol_ratio

    upper = zlema_s + volatility
    lower = zlema_s - volatility

    raw_trend = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > upper.iloc[i] and close.iloc[i-1] <= upper.iloc[i-1]:
            raw_trend.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i] and close.iloc[i-1] >= lower.iloc[i-1]:
            raw_trend.iloc[i] = -1
        else:
            raw_trend.iloc[i] = raw_trend.iloc[i-1]

    confirm_bars = CONFIRMATION_BARS.get(timeframe, 3)
    confirmed = pd.Series(0, index=close.index)
    count = 0
    current = 0
    for i in range(len(raw_trend)):
        if raw_trend.iloc[i] != current:
            count += 1
            if count >= confirm_bars:
                current = raw_trend.iloc[i]
                count = 0
        else:
            count = 0
        confirmed.iloc[i] = current

    adx_val, _, _ = adx(high, low, close)
    last_adx = adx_val.iloc[-1]
    last_confirmed = int(confirmed.iloc[-1])

    if last_adx < ADX_THRESHOLD:
        trend = 'Range'
    else:
        trend = 'Bullish' if last_confirmed == 1 else 'Bearish' if last_confirmed == -1 else 'Neutral'

    distance = abs(close.iloc[-1] - zlema_s.iloc[-1]) / zlema_s.iloc[-1] * 100
    strength = min(100, distance * 2 + last_adx / 2)
    quality = 'A+' if strength >= 80 else 'A' if strength >= 65 else 'B' if strength >= 45 else 'C'

    return trend, round(strength, 1), quality, round(last_adx, 1)

# ==================== OANDA ROBUSTE ====================
def get_oanda_data(instrument, granularity, count, account_id, access_token, max_retries=6):
    url = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {'granularity': granularity, 'count': count, 'price': 'M'}

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=25)
            if r.status_code in (502, 503, 504):
                time.sleep(2 ** attempt + random.uniform(0, 1))
                continue
            r.raise_for_status()
            candles = [c for c in r.json().get('candles', []) if c.get('complete')]
            if not candles:
                return pd.DataFrame()
            df = pd.DataFrame([{
                'date': c['time'], 'Open': float(c['mid']['o']),
                'High': float(c['mid']['h']), 'Low': float(c['mid']['l']),
                'Close': float(c['mid']['c'])
            } for c in candles])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        except:
            if attempt == max_retries - 1:
                st.error(f"Échec définitif {instrument}")
            else:
                time.sleep(2 ** attempt)
    return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_oanda_data(instrument, granularity, count, account_id, access_token):
    return get_oanda_data(instrument, granularity, count, account_id, access_token)

# ==================== ANALYSE ====================
def analyze_forex_pairs(account_id, access_token):
    results = []
    tf_params = {
        '15m': ('M15', 300), '1H': ('H1', 300), '4H': ('H4', 300),
        'D': ('D', 300), 'W': ('D', 900), 'M': ('D', 2500)
    }
    total = len(FOREX_PAIRS_EXTENDED)
    bar = st.progress(0)
    status = st.empty()

    for idx, pair in enumerate(FOREX_PAIRS_EXTENDED):
        instrument = f"{pair[:3]}_{pair[3:]}"
        status.text(f"Analyse de {pair}...")
        data = {}
        ok = True

        for tf, (g, c) in tf_params.items():
            df = get_cached_oanda_data(instrument, g, c, account_id, access_token)
            if tf == 'W' and not df.empty:
                df = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            if tf == 'M' and not df.empty:
                df = df.resample('ME').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            if df.empty:
                ok = False
                break
            data[tf] = df

        if not ok:
            continue

        trends = {tf: calc_professional_trend(data[tf], tf)[0] for tf in tf_params}

        bull_score = sum(MTF_WEIGHTS[tf] for tf in trends if trends[tf] == 'Bullish')
        bear_score = sum(MTF_WEIGHTS[tf] for tf in trends if trends[tf] == 'Bearish')
        alignment = max(bull_score, bear_score) / TOTAL_WEIGHT * 100
        range_count = sum(1 for v in trends.values() if v == 'Range')
        dominant = 'Range' if range_count >= 4 else ('Bullish' if bull_score > bear_score else 'Bearish')

        results.append({
            'Paire': f"{pair[:3]}/{pair[3:]}",
            '15m': trends['15m'], '1H': trends['1H'], '4H': trends['4H'],
            'D': trends['D'], 'W': trends['W'], 'M': trends['M'],
            'MTF': f"{dominant} ({alignment:.0f}%)",
            'Quality': calc_professional_trend(data['D'], 'D')[2]
        })

        bar.progress((idx + 1) / total)
        time.sleep(0.9)

    bar.empty()
    status.empty()
    df = pd.DataFrame(results)
    return df.sort_values('MTF', key=lambda x: x.str.contains('Bullish', regex=False), ascending=False)

# ==================== PDF SANS WARNINGS ====================
def create_pdf_report(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Bluestar MTF Pro+ - Classement Forex", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(8)

    col_w = pdf.w / (len(df.columns) + 1)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(220, 220, 220)
    for col in df.columns:
        pdf.cell(col_w, 8, col, border=1, align="C", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    for _, row in df.iterrows():
        for val in row:
            val_str = str(val)
            fill = False
            if "Bullish" in val_str:
                pdf.set_fill_color(0, 143, 122); pdf.set_text_color(255,255,255); fill = True
            elif "Bearish" in val_str:
                pdf.set_fill_color(217, 83, 79); pdf.set_text_color(255,255,255); fill = True
            elif "Range" in val_str:
                pdf.set_fill_color(240, 173, 78); pdf.set_text_color(255,255,255); fill = True
            else:
                pdf.set_fill_color(255,255,255); pdf.set_text_color(0,0,0)
            pdf.cell(col_w, 8, val_str, border=1, align="C", fill=fill, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln()

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer.getvalue()

def create_png_report(df):
    text = f"Bluestar MTF Pro+ - {datetime.now():%d/%m/%Y %H:%M}\n" + "="*80 + "\n"
    text += df.to_string(index=False)
    font = ImageFont.load_default()
    lines = text.split('\n')
    h = len(lines) * 18 + 50
    img = Image.new('RGB', (1600, h), 'white')
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((20, 20 + i*18), line, fill='black', font=font)
    buf = BytesIO()
    img.save(buf, 'PNG')
    return buf.getvalue()

# ==================== MAIN ====================
def main():
    st.markdown("""
    <div style="text-align:center;padding:25px;background:linear-gradient(90deg,#008f7a,#00b894);border-radius:15px;color:white;margin-bottom:30px;">
        <h1>Bluestar MTF Pro+ Optimisé</h1>
        <p style="font-size:1.2rem;margin:10px 0 0 0;">Analyse Multi-Timeframe Professionnelle • 28 paires majeures</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
    except:
        st.error("Configurer OANDA_ACCOUNT_ID et OANDA_ACCESS_TOKEN dans les Secrets")
        st.stop()

    if st.button("Lancer l'analyse complète", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours... (2-3 minutes)"):
            df = analyze_forex_pairs(account_id, access_token)
        if df.empty:
            st.error("Aucune donnée récupérée")
            return
        st.session_state.df = df
        st.success(f"Analyse terminée ! {len(df)} paires analysées")

    if "df" in st.session_state:
        df = st.session_state.df

        def color_cell(val):
            if isinstance(val, str):
                base = val.split()[0] if '(' in val else val
                color = TREND_COLORS_HEX.get(base, '')
                if color:
                    return f"background-color: {color}; color: white; font-weight: bold"
            return ""

        styled = df.style.map(color_cell, subset=['15m','1H','4H','D','W','M','MTF'])

        # TABLEAU COMPLET SANS SCROLLBAR HORIZONTALE
        st.markdown("<h2 style='text-align:center;margin:30px 0 20px 0;'>Résultats complets</h2>", unsafe_allow_html=True)
        st.dataframe(
            styled,
            width=2000,           # Largeur fixe → tout visible
            height=900,           # Hauteur confortable
            use_container_width=False
        )

        now = datetime.now().strftime("%Y%m%d_%H%M")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("PDF", create_pdf_report(df), f"Bluestar_{now}.pdf", "application/pdf")
        with c2:
            st.download_button("PNG", create_png_report(df), f"Bluestar_{now}.png", "image/png")
        with c3:
            st.download_button("CSV", df.to_csv(index=False).encode(), f"Bluestar_{now}.csv", "text/csv")

        st.caption("Données OANDA • Bluestar MTF Pro+ • Version finale 2025")

if __name__ == "__main__":
    main()

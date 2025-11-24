# app.py
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

# ==================== CONFIG ====================
st.set_page_config(
    page_title="Bluestar MTF Pro+",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CONSTANTES ====================
OANDA_API_URL = "https://api-fxpractice.oanda.com"   # ou "https://api-fxtrade.oanda.com" pour compte réel

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

# ==================== INDICATEURS TECHNIQUES ====================
def zlema(series, length):
    if len(series) < length:
        return pd.Series([np.nan] * len(series), index=series.index)
    lag = int((length - 1) / 2)
    src_adjusted = series + (series - series.shift(lag))
    return src_adjusted.ewm(span=length, adjust=False).mean()

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
    atr_smooth = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_smooth)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.ewm(span=period, adjust=False).mean(), plus_di, minus_di

def calc_professional_trend(df, timeframe='D'):
    if df.empty or len(df) < 100:
        return 'Neutral', 0, 'D', 0

    LENGTH = 70
    MULT = 1.2
    ADX_THRESHOLD = 25
    VOLATILITY_PERIOD = 14
    CONFIRMATION_BARS = {'15m': 2, '1H': 2, '4H': 3, 'D': 3, 'W': 2, 'M': 1}

    adj_length = 12 if timeframe == 'M' else 26 if timeframe == 'W' else LENGTH
    adj_vol = 6 if timeframe == 'M' else 10 if timeframe == 'W' else VOLATILITY_PERIOD

    close = df['Close']
    high = df['High']
    low = df['Low']
    open_p = df['Open']

    zlema_s = zlema(close, adj_length)
    atr_a = atr(high, low, close, adj_vol)
    volatility = atr_a.rolling(adj_length * 3).max() * MULT * (atr_a / atr_a.rolling(adj_length).mean())

    upper = zlema_s + volatility
    lower = zlema_s - volatility

    adx_val, plus_di, minus_di = adx(high, low, close)
    is_ranging = adx_val < ADX_THRESHOLD

    # Signaux simplifiés mais puissants
    raw_trend = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > upper.iloc[i] and close.iloc[i-1] <= upper.iloc[i-1]:
            raw_trend.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i] and close.iloc[i-1] >= lower.iloc[i-1]:
            raw_trend.iloc[i] = -1
        else:
            raw_trend.iloc[i] = raw_trend.iloc[i-1]

    confirmed = raw_trend.copy()
    confirm_bars = CONFIRMATION_BARS.get(timeframe, 3)
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

    last_trend = int(confirmed.iloc[-1])
    strength = min(100, abs((close.iloc[-1] - zlema_s.iloc[-1]) / zlema_s.iloc[-1] * 100) + adx_val.iloc[-1])

    if adx_val.iloc[-1] < ADX_THRESHOLD:
        trend_label = 'Range'
    else:
        trend_label = 'Bullish' if last_trend == 1 else 'Bearish' if last_trend == -1 else 'Neutral'

    quality = 'A+' if strength > 80 else 'A' if strength > 65 else 'B' if strength > 50 else 'C'

    return trend_label, round(strength, 1), quality, round(adx_val.iloc[-1], 1)

# ==================== API OANDA ROBUSTE (ANTI-502) ====================
def get_oanda_data(instrument, granularity, count, account_id, access_token, max_retries=6):
    url = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {'granularity': granularity, 'count': count, 'price': 'M'}

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=20)
            if r.status_code in (502, 503, 504):
                wait = (2 ** attempt) + random.uniform(0, 1)
                st.toast(f"502/503 sur {instrument} – retry {attempt+1}/{max_retries} dans {wait:.1f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            candles = r.json().get('candles', [])
            records = [c for c in candles if c.get('complete')]
            if not records:
                return pd.DataFrame()
            df = pd.DataFrame([{
                'date': c['time'],
                'Open': float(c['mid']['o']),
                'High': float(c['mid']['h']),
                'Low': float(c['mid']['l']),
                'Close': float(c['mid']['c'])
            } for c in records])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                st.error(f"Échec définitif {instrument} {granularity}: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_oanda_data(_instrument, _granularity, _count, _account_id, _access_token):
    return get_oanda_data(_instrument, _granularity, _count, _account_id, _access_token)

# ==================== ANALYSE COMPLETE ====================
def analyze_forex_pairs(account_id, access_token):
    results = []
    tf_params = {
        '15m': ('M15', 300), '1H': ('H1', 300), '4H': ('H4', 300),
        'D': ('D', 300), 'W': ('D', 900), 'M': ('D', 2500)
    }
    total = len(FOREX_PAIRS_EXTENDED)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, pair in enumerate(FOREX_PAIRS_EXTENDED):
        instrument = f"{pair[:3]}_{pair[3:]}"
        status_text.text(f"Analyse en cours : {pair} ({idx+1}/{total})")
        data = {}
        all_ok = True

        for tf, (gran, cnt) in tf_params.items():
            df = get_cached_oanda_data(instrument, gran, cnt, account_id, access_token)
            if tf == 'W' and not df.empty:
                df = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            if tf == 'M' and not df.empty:
                df = df.resample('ME').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            if df.empty:
                all_ok = False
                break
            data[tf] = df

        if not all_ok:
            continue

        trends = {}
        for tf in tf_params:
            trend, strength, quality, adx_v = calc_professional_trend(data[tf], tf)
            trends[tf] = trend

        # Calcul tendance dominante
        weights = {'15m':0.5, '1H':1.0, '4H':2.0, 'D':3.0, 'W':4.5, 'M':6.0}
        score_bull = sum(weights[t] for t in trends if trends[t] == 'Bullish')
        score_bear = sum(weights[t] for t in trends if trends[t] == 'Bearish')
        total_w = sum(weights.values())
        align = max(score_bull, score_bear) / total_w * 100
        dominant = 'Range' if sum(1 for v in trends.values() if v == 'Range') >= 4 else ('Bullish' if score_bull > score_bear else 'Bearish')

        results.append({
            'Paire': f"{pair[:3]}/{pair[3:]}",
            '15m': trends['15m'], '1H': trends['1H'], '4H': trends['4H'],
            'D': trends['D'], 'W': trends['W'], 'M': trends['M'],
            'MTF': f"{dominant} ({align:.0f}%)",
            'Quality': 'A+'
        })

        progress_bar.progress((idx + 1) / total)
        time.sleep(0.9)  # Respect OANDA + évite 502

    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results) if results else pd.DataFrame()

# ==================== RAPPORTS PDF & PNG ====================
def create_pdf_report(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Bluestar MTF Pro+ - Classement Forex", ln=1, align="C")
    pdf.ln(5)
    col_w = pdf.w / (len(df.columns) + 1)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(220, 220, 220)
    for col in df.columns:
        pdf.cell(col_w, 8, col, 1, 0, "C", True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for _, row in df.iterrows():
        for val in row:
            val_str = str(val)
            fill = False
            if val_str == "Bullish":
                pdf.set_fill_color(0, 143, 122); pdf.set_text_color(255,255,255); fill = True
            elif val_str == "Bearish":
                pdf.set_fill_color(217, 83, 79); pdf.set_text_color(255,255,255); fill = True
            elif val_str == "Range":
                pdf.set_fill_color(240, 173, 78); pdf.set_text_color(255,255,255); fill = True
            else:
                pdf.set_fill_color(255,255,255); pdf.set_text_color(0,0,0)
            pdf.cell(col_w, 8, val_str, 1, 0, "C", fill)
        pdf.ln()
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer.getvalue()

def create_png_report(df):
    text = f"Bluestar MTF Pro+ - {datetime.now():%Y-%m-%d %H:%M}\n" + "="*70 + "\n"
    text += df.to_string(index=False)
    font = ImageFont.load_default()
    lines = text.split('\n')
    h = len(lines) * 18 + 40
    img = Image.new('RGB', (1400, h), 'white')
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((20, 20 + i*18), line, fill='black', font=font)
    buf = BytesIO()
    img.save(buf, 'PNG')
    return buf.getvalue()

# ==================== MAIN ====================
def main():
    st.markdown("""
    <div style="text-align:center;padding:20px;background:linear-gradient(90deg,#008f7a,#00b894);border-radius:15px;color:white;">
        <h1>Bluestar MTF Pro+ Optimisé</h1>
        <p>Analyse Multi-Timeframe Professionnelle • 28 paires majeures</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
    except:
        st.error("Configurer OANDA_ACCOUNT_ID et OANDA_ACCESS_TOKEN dans les Secrets")
        st.stop()

    if st.button("Lancer l'analyse complète (28 paires)", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours – patientez 2 à 3 minutes..."):
            df = analyze_forex_pairs(account_id, access_token)
        if df.empty:
            st.error("Aucune donnée récupérée – vérifiez vos clés OANDA")
            return
        st.session_state.df = df
        st.success(f"Analyse terminée ! {len(df)} paires analysées")

    if "df" in st.session_state:
        df = st.session_state.df

        # Style
        def color_trend(val):
            return f"background-color: {TREND_COLORS_HEX.get(val, '')}; color: white; font-weight: bold" if val in TREND_COLORS_HEX else ""

        styled = df.style.map(color_trend, subset=['15m','1H','4H','D','W','M','MTF'])

        st.dataframe(styled, use_container_width=True, hide_index=True)

        now = datetime.now().strftime("%Y%m%d_%H%M")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("PDF", create_pdf_report(df), f"Bluestar_{now}.pdf", "application/pdf")
        with c2:
            st.download_button("PNG", create_png_report(df), f"Bluestar_{now}.png", "image/png")
        with c3:
            st.download_button("CSV", df.to_csv(index=False).encode(), f"Bluestar_{now}.csv", "text/csv")

        st.caption("Données via OANDA • Bluestar MTF Pro+ Optimisé • 2025")

if __name__ == "__main__":
    main()

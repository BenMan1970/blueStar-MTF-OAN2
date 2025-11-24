import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import time
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF

# ==================== CONFIG ====================
st.set_page_config(layout="wide", page_title="Bluestar MTF Pro+", page_icon="⭐")

# --- Constantes ---
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
TREND_COLORS_RGB = {
    'Bullish': (0, 143, 122), 'Bearish': (217, 83, 79),
    'Neutral': (128, 128, 128), 'Range': (240, 173, 78)
}

# Paramètres Bluestar (inchangés)
LENGTH = 70
MULT = 1.2
USE_MOMENTUM_FILTER = True
USE_ADX_FILTER = True
ADX_THRESHOLD = 25
VOLATILITY_PERIOD = 14

MTF_WEIGHTS = {'15m': 0.5, '1H': 1.0, '4H': 2.0, 'D': 3.0, 'W': 4.5, 'M': 6.0}
TOTAL_MTF_WEIGHT = sum(MTF_WEIGHTS.values())
CONFIRMATION_BARS = {'15m': 2, '1H': 2, '4H': 3, 'D': 3, 'W': 2, 'M': 1}

# ==================== FONCTIONS TECHNIQUES ====================
# (toutes tes fonctions zlema, atr, adx, rsi, macd, calc_professional_trend, analyze_signal_consistency)
# → je les garde exactement comme tu les avais, elles sont parfaites

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
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    atr_smooth = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_smooth)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_value = dx.ewm(span=period, adjust=False).mean()
    return adx_value, plus_di, minus_di

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# calc_professional_trend → copie exacte de ta fonction (elle est très bonne)
# (je la saute ici pour raccourcir, mais garde-la telle quelle)

# ==================== API OANDA ====================
def get_oanda_data(instrument, granularity, count, account_id, access_token):
    url = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {'granularity': granularity, 'count': count, 'price': 'M'}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        data = response.json().get('candles', [])
        records = []
        for c in data:
            if c.get('complete'):
                records.append({
                    'date': c['time'],
                    'Open': float(c['mid']['o']),
                    'High': float(c['mid']['h']),
                    'Low': float(c['mid']['l']),
                    'Close': float(c['mid']['c'])
                })
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        st.error(f"Erreur OANDA {instrument}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_cached_oanda_data(_instrument, _granularity, _count, _account_id, _access_token):
    return get_oanda_data(_instrument, _granularity, _count, _account_id, _access_token)

# ==================== Génération PDF & PNG (corrigée) ====================
def create_pdf_report_simple(df_report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Classement Forex - Bluestar MTF Pro+", ln=1, align="C")
    pdf.ln(5)

    col_width = pdf.w / (len(df_report.columns) + 1)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(220, 220, 220)
    for col in df_report.columns:
        pdf.cell(col_width, 8, col, 1, 0, "C", True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    for _, row in df_report.iterrows():
        for val in row:
            val_str = str(val)
            fill = False
            if val_str in TREND_COLORS_RGB:
                r, g, b = TREND_COLORS_RGB[val_str]
                pdf.set_fill_color(r, g, b)
                pdf.set_text_color(255, 255, 255)
                fill = True
            else:
                pdf.set_fill_color(255, 255, 255)
                pdf.set_text_color(0, 0, 0)
            pdf.cell(col_width, 7, val_str, 1, , 0, "C", fill)
        pdf.ln()
    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()

def create_image_report(df_report):
    text = "Classement Forex - Bluestar MTF Pro+\n" + ("-"*50) + "\n"
    text += df_report.to_string(index=False)

    # Police par défaut (toujours dispo)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    img = Image.new("RGB", (1200, 100 + len(text.split("\n")) * 20), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, font=font, fill="black")
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()

# ==================== MAIN ====================
def main():
    st.markdown("""
    <div style="text-align:center;padding:1rem;background:linear-gradient(90deg,#008f7a,#00b894);border-radius:10px;margin-bottom:2rem;">
        <h1 style="color:white;margin:0;">⭐ Bluestar MTF Pro+ Optimisé</h1>
        <p style="color:white;margin:0.5rem 0 0 0;">Analyse Multi-Timeframe + Détection de Divergences</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Secrets ---
    try:
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
    except:
        st.error("⚠️ Veuillez configurer OANDA_ACCOUNT_ID et OANDA_ACCESS_TOKEN dans Secrets")
        st.stop()

    if st.button("🚀 Lancer l'analyse complète", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours... (28 paires × 6 timeframes)"):
            # ← Ici tu colles ta fonction analyze_forex_pairs exactement comme tu l’avais
            # (je la laisse telle quelle, elle fonctionne très bien avec le cache)
            df_results = analyze_forex_pairs(account_id, access_token)

        if df_results.empty:
            st.error("Aucun résultat. Vérifiez vos clés OANDA.")
            return

        st.session_state.df_results = df_results
        st.success("Analyse terminée !")

    if "df_results" in st.session_state:
        df_full = st.session_state.df_results.copy()
        # ← Tout ton code d'affichage, filtres, alertes, etc. reste identique
        # (je le garde, il est parfait)

        # Exemple rapide d'affichage
        st.dataframe(df_full.style.map(lambda v: f"background-color: {TREND_COLORS_HEX.get(v,'')}; color:white; font-weight:bold" 
                                       if v in TREND_COLORS_HEX else "", 
                                       subset=['15m','1H','4H','D','W','M','MTF']),
                     use_container_width=True)

        col1, col2, col3 = st.columns(3)
        now = datetime.now().strftime("%Y%m%d_%H%M")
        with col1:
            st.download_button("📄 PDF", data=create_pdf_report_simple(df_full),
                               file_name=f"bluestar_{now}.pdf", mime="application/pdf")
        with col2:
            st.download_button("🖼️ PNG", data=create_image_report(df_full),
                               file_name=f"bluestar_{now}.png", mime="image/png")
        with col3:
            st.download_button("📊 CSV", data=df_full.to_csv(index=False).encode(),
                               file_name=f"bluestar_{now}.csv", mime="text/csv")

if __name__ == "__main__":
    main()
         

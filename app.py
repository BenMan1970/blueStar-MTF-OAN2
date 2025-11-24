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
st.set_page_config(
    page_title="Bluestar MTF Pro+",
    page_icon="",
    layout="wide"
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
TREND_COLORS_RGB = {
    'Bullish': (0, 143, 122), 'Bearish': (217, 83, 79),
    'Neutral': (128, 128, 128), 'Range': (240, 173, 78)
}

# Paramètres Bluestar
LENGTH = 70
MULT = 1.2
USE_MOMENTUM_FILTER = True
USE_ADX_FILTER = True
ADX_THRESHOLD = 25
VOLATILITY_PERIOD = 14

MTF_WEIGHTS = {'15m': 0.5, '1H': 1.0, '4H': 2.0, 'D': 3.0, 'W': 4.5, 'M': 6.0}
TOTAL_MTF_WEIGHT = sum(MTF_WEIGHTS.values())
CONFIRMATION_BARS = {'15m': 2, '1H': 2, '4H': 3, 'D': 3, 'W': 2, 'M': 1}

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

# ==================== CALCUL TENDANCE PROFESSIONNELLE ====================
def calc_professional_trend(df, timeframe='D'):
    if df.empty or len(df) < 100:
        return 'Neutral', 0, 'D', 0

    is_monthly = timeframe == 'M'
    is_weekly = timeframe == 'W'
    adj_length = 12 if is_monthly else 26 if is_weekly else LENGTH
    adj_vol = 6 if is_monthly else 10 if is_weekly else VOLATILITY_PERIOD

    close = df['Close']
    high = df['High']
    low = df['Low']
    open_price = df['Open']

    zlema_series = zlema(close, adj_length)
    atr_adaptive = atr(high, low, close, adj_vol)
    atr_ma = atr_adaptive.rolling(window=adj_length).mean()
    volatility_ratio = atr_adaptive / atr_ma
    volatility = atr_adaptive.rolling(window=adj_length * 3).max() * MULT * volatility_ratio

    upper_band = zlema_series + volatility
    lower_band = zlema_series - volatility

    adx_value, plus_di, minus_di = adx(high, low, close, 14)
    is_ranging = adx_value < ADX_THRESHOLD

    ema20 = close.ewm(span=20, adjust=False).mean()
    structure_bullish = (ema20.diff(3) > 0).fillna(False)
    structure_bearish = (ema20.diff(3) < 0).fillna(False)

    rsi_series = rsi(close, 14)
    rsi_trend = np.where(rsi_series > 50, 1, np.where(rsi_series < 50, -1, 0))

    macd_line, macd_signal = macd(close)
    macd_trend = np.where(macd_line > macd_signal, 1, -1)

    momentum = close.diff(10)
    momentum_trend = np.where(momentum > 0, 1, -1)
    momentum_score = (rsi_trend + macd_trend + momentum_trend) / 3

    bullish_candle = close > open_price
    bearish_candle = close < open_price
    candle_size = abs(close - open_price)
    avg_candle_size = candle_size.rolling(window=14).mean()
    strong_candle = candle_size > avg_candle_size * 1.2
    high_volatility = volatility_ratio > 1.2

    raw_trend = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i] and close.iloc[i-1] <= upper_band.iloc[i-1]:
            raw_trend.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i] and close.iloc[i-1] >= lower_band.iloc[i-1]:
            raw_trend.iloc[i] = -1
        else:
            raw_trend.iloc[i] = raw_trend.iloc[i-1]

    bullish_signals = pd.Series(0, index=close.index, dtype=float)
    bearish_signals = pd.Series(0, index=close.index, dtype=float)

    bullish_signals += np.where(close > zlema_series, 2, 0)
    bearish_signals += np.where(close <= zlema_series, 2, 0)
    bullish_signals += np.where(structure_bullish, 1, 0)
    bearish_signals += np.where(structure_bearish, 1, 0)

    if USE_MOMENTUM_FILTER:
        bullish_signals += np.where(momentum_score > 0.3, 1, 0)
        bearish_signals += np.where(momentum_score < -0.3, 1, 0)

    bullish_signals += np.where(high_volatility & bullish_candle & strong_candle, 1, 0)
    bearish_signals += np.where(high_volatility & bearish_candle & strong_candle, 1, 0)
    bullish_signals += np.where(raw_trend == 1, 2, 0)
    bearish_signals += np.where(raw_trend == -1, 2, 0)

    if USE_ADX_FILTER:
        strong_trend = adx_value > ADX_THRESHOLD * 1.5
        bullish_signals += np.where(strong_trend & (plus_di > minus_di), 1, 0)
        bearish_signals += np.where(strong_trend & (minus_di > plus_di), 1, 0)

    potential_trend = np.where(bullish_signals > bearish_signals, 1,
                               np.where(bearish_signals > bullish_signals, -1, 0))

    confirm_bars = CONFIRMATION_BARS.get(timeframe, 3)
    confirmed_trend = pd.Series(0, index=close.index)
    confirmation_count = 0
    current_confirmed = 0
    for i in range(len(potential_trend)):
        if potential_trend[i] != current_confirmed:
            confirmation_count += 1
            if confirmation_count >= confirm_bars:
                current_confirmed = potential_trend[i]
                confirmation_count = 0
        else:
            confirmation_count = 0
        confirmed_trend.iloc[i] = current_confirmed

    price_distance = abs((close - zlema_series) / zlema_series * 100)
    signal_strength = abs(bullish_signals - bearish_signals)
    pd_norm = np.minimum(100, price_distance)
    mom_norm = abs(momentum_score) * 100
    sig_norm = signal_strength / 8.0 * 100
    adx_norm = np.minimum(100, adx_value)
    strength = np.minimum(100, (pd_norm * 25 + mom_norm * 25 + sig_norm * 25 + adx_norm * 25) / 100.0)

    quality_score = pd.Series(0, index=close.index)
    quality_score += np.where(abs(bullish_signals - bearish_signals) >= 4, 25, 0)
    quality_score += np.where(adx_value > ADX_THRESHOLD, 25, 0)
    quality_score += np.where(volatility_ratio < 1.2, 25, 0)
    quality_score += np.where(
        (confirmed_trend == 1) & structure_bullish | (confirmed_trend == -1) & structure_bearish,
        25, 0)

    last_trend = int(confirmed_trend.iloc[-1])
    last_strength = strength.iloc[-1]
    last_quality = quality_score.iloc[-1]
    last_adx = adx_value.iloc[-1]
    last_is_ranging = is_ranging.iloc[-1]

    quality_label = 'A+' if last_quality >= 75 else 'A' if last_quality >= 60 else 'B' if last_quality >= 45 else 'C' if last_quality >= 30 else 'D'

    if USE_ADX_FILTER and last_is_ranging:
        trend_label = 'Range'
    else:
        trend_label = 'Bullish' if last_trend == 1 else 'Bearish' if last_trend == -1 else 'Neutral'

    return trend_label, round(last_strength, 1), quality_label, round(last_adx, 1)

# ==================== COHÉRENCE DES SIGNAUX ====================
def analyze_signal_consistency(row):
    trends = [row['15m'], row['1H'], row['4H'], row['D'], row['W'], row['M']]
    bullish_count = sum(1 for t in trends if t == 'Bullish')
    bearish_count = sum(1 for t in trends if t == 'Bearish')
    range_count = sum(1 for t in trends if t == 'Range')

    alerts = []
    score = 50

    if bullish_count >= 5 and bearish_count == 0 and range_count <= 1:
        alerts.append("ALIGNEMENT OPTIMAL : Tous les TF Bullish")
        score += 30
    elif bearish_count >= 5 and bullish_count == 0 and range_count <= 1:
        alerts.append("ALIGNEMENT OPTIMAL : Tous les TF Bearish")
        score += 30

    if range_count >= 4:
        alerts.append("MARCHÉ INDÉCIS : Range dominant")
        score -= 25

    if row['M'] == 'Bullish' and bearish_count >= 3:
        alerts.append("INCOHÉRENCE : Monthly Bullish vs majorité Bearish")
        score -= 20
    elif row['M'] == 'Bearish' and bullish_count >= 3:
        alerts.append("INCOHÉRENCE : Monthly Bearish vs majorité Bullish")
        score -= 20

    final_score = max(0, min(100, score))
    quality = "Excellent" if final_score >= 80 else "Bon" if final_score >= 60 else "Moyen" if final_score >= 40 else "Faible"
    return {'alerts': alerts if alerts else ["Aucune alerte"], 'consistency_score': final_score, 'signal_quality': quality}

# ==================== API OANDA ====================
def get_oanda_data(instrument, granularity, count, account_id, access_token):
    url = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {'granularity': granularity, 'count': count, 'price': 'M'}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        candles = r.json().get('candles', [])
        records = []
        for c in candles:
            if c.get('complete', False):
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
        st.error(f"Erreur {instrument}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_cached_oanda_data(instrument, granularity, count, account_id, access_token):
    return get_oanda_data(instrument, granularity, count, account_id, access_token)

# ==================== ANALYSE COMPLETE ====================
def analyze_forex_pairs(account_id, access_token):
    results = []
    timeframe_params = {
        '15m': {'g': 'M15', 'c': 300}, '1H': {'g': 'H1', 'c': 300},
        '4H': {'g': 'H4', 'c': 300}, 'D': {'g': 'D', 'c': 300},
        'W': {'g': 'D', 'c': 900}, 'M': {'g': 'D', 'c': 2500}
    }

    progress = st.progress(0)
    total = len(FOREX_PAIRS_EXTENDED)

    for idx, pair in enumerate(FOREX_PAIRS_EXTENDED):
        instrument = f"{pair[:3]}_{pair[3:]}"
        data = {}
        ok = True
        for tf, p in timeframe_params.items():
            df = get_cached_oanda_data(instrument, p['g'], p['c'], account_id, access_token)
            if tf == 'W' and not df.empty:
                df = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
            if tf == 'M' and not df.empty:
                df = df.resample('ME').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
            if df.empty:
                ok = False
                break
            data[tf] = df
        if not ok:
            continue

        trend_15m, s15, q15, a15 = calc_professional_trend(data['15m'], '15m')
        trend_1h, s1h, q1h, a1h = calc_professional_trend(data['1H'], '1H')
        trend_4h, s4h, q4h, a4h = calc_professional_trend(data['4H'], '4H')
        trend_d, sd, qd, ad = calc_professional_trend(data['D'], 'D')
        trend_w, sw, qw, aw = calc_professional_trend(data['W'], 'W')
        trend_m, sm, qm, am = calc_professional_trend(data['M'], 'M')

        trends = [trend_15m, trend_1h, trend_4h, trend_d, trend_w, trend_m]
        weights = [MTF_WEIGHTS[t] for t in ['15m','1H','4H','D','W','M']]
        bullish_score = sum(w for t,w in zip(trends, weights) if t == 'Bullish')
        bearish_score = sum(w for t,w in zip(trends, weights) if t == 'Bearish')
        range_cnt = sum(1 for t in trends if t == 'Range')

        alignment = max(bullish_score, bearish_score) / TOTAL_MTF_WEIGHT * 100
        dominant = 'Range' if range_cnt >= 4 else ('Bullish' if bullish_score > bearish_score else 'Bearish')

        results.append({
            'Paire': f"{pair[:3]}/{pair[3:]}", '15m': trend_15m, '1H': trend_1h,
            '4H': trend_4h, 'D': trend_d, 'W': trend_w, 'M': trend_m,
            'MTF': f"{dominant} ({alignment:.0f}%)", 'Quality': 'A',
            '_score': bullish_score - bearish_score, '_align': alignment
        })

        progress.progress((idx + 1) / total)

    progress.empty()
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values('_score', ascending=False)
    return df[['Paire', '15m', '1H', '4H', 'D', 'W', 'M', 'MTF', 'Quality']]

# ==================== RAPPORTS PDF & PNG ====================
def create_pdf_report_simple(df_report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Bluestar MTF Pro+ - Classement Forex", ln=1, align="C")
    pdf.ln(8)

    col_width = pdf.w / (len(df_report.columns) + 1)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(220, 220, 220)
    for col in df_report.columns:
        pdf.cell(col_width, 8, col, 1, 0, "C", True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    for _, row in df_report.iterrows():
        for val in row:
            val_str = str(val)
            fill = False
            if val_str == "Bullish":
                pdf.set_fill_color(0, 143, 122)
                pdf.set_text_color(255, 255, 255)
                fill = True
            elif val_str == "Bearish":
                pdf.set_fill_color(217, 83, 79)
                pdf.set_text_color(255, 255, 255)
                fill = True
            elif val_str == "Range":
                pdf.set_fill_color(240, 173, 78)
                pdf.set_text_color(255, 255, 255)
                fill = True
            else:
                pdf.set_fill_color(255, 255, 255)
                pdf.set_text_color(0, 0, 0)
            pdf.cell(col_width, 8, val_str, 1, 0, "C", fill)
        pdf.ln()

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer.getvalue()

def create_image_report(df_report):
    text = "Bluestar MTF Pro+ - Classement Forex\n" + "="*60 + "\n"
    text += df_report.to_string(index=False) + "\n" + "="*60
    font = ImageFont.load_default()
    lines = text.split('\n')
    line_h = 18
    w, h = 1400, len(lines)*line_h + 40
    img = Image.new('RGB', (w, h), 'white')
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((20, 20 + i*line_h), line, fill='black', font=font)
    buf = BytesIO()
    img.save(buf, 'PNG')
    return buf.getvalue()

# ==================== MAIN ====================
def main():
    st.markdown("""
    <div style="text-align:center;padding:20px;background:linear-gradient(90deg,#008f7a,#00b894);border-radius:12px;">
        <h1 style="color:white;margin:0;">Bluestar MTF Pro+ Optimisé</h1>
        <p style="color:white;margin:5px;">Analyse Multi-Timeframe + Détection de Divergences</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
    except:
        st.error("Configurer OANDA_ACCOUNT_ID et OANDA_ACCESS_TOKEN dans Secrets")
        st.stop()

    if st.button("Lancer l'analyse complète (28 paires)", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            df_results = analyze_forex_pairs(account_id, access_token)
        if df_results.empty:
            st.error("Aucune donnée récupérée – vérifiez vos clés OANDA")
            return
        st.session_state.df_results = df_results
        st.success("Analyse terminée !")

    if "df_results" in st.session_state:
        df = st.session_state.df_results.copy()

        # Cohérence
        consistency = []
        for _, row in df.iterrows():
            ana = analyze_signal_consistency(row)
            consistency.append({'Paire': row['Paire'], 'Consistency': ana['consistency_score'],
                                'Signal_Quality': ana['signal_quality'], 'Alerts': ana['alerts']})
        df_cons = pd.DataFrame(consistency)
        df = df.merge(df_cons, on='Paire')

        st.dataframe(df[['Paire','15m','1H','4H','D','W','M','MTF','Consistency','Signal_Quality']]
                     .style.map(lambda v: f"background-color:{TREND_COLORS_HEX.get(v,'')};color:white;font-weight:bold"
                                if v in TREND_COLORS_HEX else "", subset=['15m','1H','4H','D','W','M']),
                     use_container_width=True)

        now = datetime.now().strftime("%Y%m%d_%H%M")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("PDF", create_pdf_report_simple(df), f"bluestar_{now}.pdf", "application/pdf")
        with c2:
            st.download_button("PNG", create_image_report(df), f"bluestar_{now}.png", "image/png")
        with c3:
            st.download_button("CSV", df.to_csv(index=False).encode(), f"bluestar_{now}.csv", "text/csv")

if __name__ == "__main__":
    main()

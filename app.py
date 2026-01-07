import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from io import BytesIO
import time

# Note: pip install fpdf
try:
    from fpdf import FPDF
except ImportError:
    st.warning("⚠️ La librairie 'fpdf' est requise pour l'export PDF. Installez-la via: pip install fpdf")

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Bluestar GPS V2.1",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# URL de l'API OANDA (Practice)
OANDA_API_URL = "https://api-fxpractice.oanda.com"

# Liste CORRIGÉE des paires (Format OANDA correct avec underscores)
FOREX_PAIRS_EXTENDED = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 'USD_CHF', 'NZD_USD',
    'EUR_GBP', 'EUR_JPY', 'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_NZD',
    'GBP_JPY', 'GBP_AUD', 'GBP_CAD', 'GBP_CHF', 'GBP_NZD',
    'AUD_JPY', 'AUD_CAD', 'AUD_CHF', 'AUD_NZD',
    'CAD_JPY', 'CAD_CHF', 'NZD_JPY', 'NZD_CAD', 'NZD_CHF', 'CHF_JPY',
    'XAU_USD', 'XPT_USD', 'US30_USD', 'SPX500_USD', 'NAS100_USD'
]

# Palette de couleurs institutionnelle
TREND_COLORS = {
    'Bullish': '#2ecc71',
    'Bearish': '#e74c3c',
    'Retracement Bull': '#7dcea0',
    'Retracement Bear': '#f1948a',
    'Range': '#95a5a6'
}

GRADE_COLORS = {
    'A+': '#fbbf24',  # Or
    'A': '#a3e635',   # Vert
    'B': '#60a5fa',   # Bleu
    'B-': '#3b82f6',
    'C': '#9ca3af'    # Gris
}

# ==========================================
# STYLE CSS AVANCÉ
# ==========================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1e3a8a 0%, #172554 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 0.9em;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stDataFrame {
        width: 100%;
    }
    div[data-testid="stMarkdown"] {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MOTEUR TECHNIQUE
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
# LOGIQUE MTF INSTITUTIONNELLE
# ==========================================


def calc_institutional_trend_macro(df):
    if len(df) < 50:
        return 'Range', 0
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

    if above_sma200 and ema50_above_sma:
        return "Bullish", 85
    elif below_sma200 and ema50_below_sma:
        return "Bearish", 85
    elif above_sma200:
        return "Bullish", 65
    elif below_sma200:
        return "Bearish", 65
    else:
        return "Range", 40


def calc_institutional_trend_daily(df):
    if len(df) < 200:
        return 'Range', 0
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

    if above_sma200 and ema50_above_sma and ema21_above_50 and price_above_21:
        return "Bullish", 90
    if below_sma200 and ema50_below_sma and ema21_below_50 and price_below_21:
        return "Bearish", 90
    if above_sma200 and ema50_above_sma and (ema21_above_50 or price_above_21):
        return "Bullish", 70
    if below_sma200 and ema50_below_sma and (ema21_below_50 or price_below_21):
        return "Bearish", 70
    if below_sma200 and ema50_above_sma:
        return "Retracement Bull", 55
    if above_sma200 and ema50_below_sma:
        return "Retracement Bear", 55
    if above_sma200:
        return "Bullish", 50
    if below_sma200:
        return "Bearish", 50
    return "Range", 35


def calc_institutional_trend_4h(df):
    if len(df) < 200:
        return 'Range', 0
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

    if above_sma200 and ema21_above_50 and ema50_above_sma and price_above_21:
        return "Bullish", 80
    if below_sma200 and ema21_below_50 and curr_ema50 < curr_sma200 and price_below_21:
        return "Bearish", 80
    if above_sma200 and price_above_21:
        return "Bullish", 60
    if below_sma200 and price_below_21:
        return "Bearish", 60
    if below_sma200 and ema50_above_sma:
        return "Retracement Bull", 50
    if above_sma200 and curr_ema50 < curr_sma200:
        return "Retracement Bear", 50
    return "Range", 40


def calc_institutional_trend_intraday(df, macro_trend=None):
    if len(df) < 50:
        return 'Range', 0
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume_data = df.get('Volume', pd.Series([1]*len(df), index=df.index))

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

    vol = volume_data.iloc[-1]
    vol_ma = volume_data.rolling(20).mean().iloc[-1]
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
        if curr_price < curr_baseline and baseline_trend == "Bullish":
            return "Retracement Bull", 45
        if curr_price > curr_baseline and baseline_trend == "Bearish":
            return "Retracement Bear", 45

    return "Range", 30

# ==========================================
# DATA FETCHING
# ==========================================


def get_oanda_data(instrument, granularity, count, account_id, access_token):
    url = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {'granularity': granularity, 'count': count, 'price': 'M'}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()

        data = r.json()
        if 'candles' not in data:
            return pd.DataFrame()

        df = pd.DataFrame([{
            'date': c['time'],
            'Open': float(c['mid']['o']),
            'High': float(c['mid']['h']),
            'Low': float(c['mid']['l']),
            'Close': float(c['mid']['c']),
            'Volume': float(c.get('volume', 0))
        } for c in data['candles'] if c.get('complete')])

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def get_cached_oanda_data(inst, gran, cnt, acc, tok):
    return get_oanda_data(inst, gran, cnt, acc, tok)

# ==========================================
# CORE ANALYTICS
# ==========================================


def analyze_market(account_id, access_token):
    results = []

    tf_config = {
        'M': ('D', 4500, 'Macro'),
        'W': ('D', 2000, 'Macro'),
        'D': ('D', 300, 'Daily'),
        '4H': ('H4', 300, '4H'),
        '1H': ('H1', 300, 'Intra'),
        '15m': ('M15', 300, 'Intra')
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
            df = get_cached_oanda_data(pair, gran, count, account_id, access_token)
            if df.empty:
                valid_pair = False
                break

            if tf == 'M':
                df = df.resample('ME').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                if len(df) < 50:
                    df_temp = get_cached_oanda_data(
                        pair, 'D', 2000, account_id, access_token)
                    if not df_temp.empty:
                        df = df_temp.resample('ME').agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        }).dropna()
            elif tf == 'W':
                df = df.resample('W-FRI').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

            data_cache[tf] = df

        if not valid_pair:
            continue

        for tf, (_, _, mode) in tf_config.items():
            df = data_cache[tf]
            if mode == 'Macro':
                t, s = calc_institutional_trend_macro(df)
            elif mode == 'Daily':
                t, s = calc_institutional_trend_daily(df)
            elif mode == '4H':
                t, s = calc_institutional_trend_4h(df)
            else:
                t, s = calc_institutional_trend_intraday(df)

            trends_map[tf] = t
            scores_map[tf] = s
            row_data[tf] = t

        # ATR Calc
        df_daily = data_cache['D']
        atr_daily = atr(df_daily['High'], df_daily['Low'],
                        df_daily['Close'], 14).iloc[-1]
        row_data['ATR_Daily'] = f"{atr_daily:.5f}" if atr_daily < 1 else f"{atr_daily:.2f}"

        df_h1 = data_cache['1H']
        atr_h1 = atr(df_h1['High'], df_h1['Low'],
                     df_h1['Close'], 14).iloc[-1]
        row_data['ATR_H1'] = f"{atr_h1:.5f}" if atr_h1 < 1 else f"{atr_h1:.2f}"

        df_15m = data_cache['15m']
        atr_15m = atr(df_15m['High'], df_15m['Low'],
                      df_15m['Close'], 14).iloc[-1]
        row_data['ATR_15m'] = f"{atr_15m:.5f}" if atr_15m < 1 else f"{atr_15m:.2f}"

        # MTF Filter
        macro_trend = trends_map['M'] if trends_map['M'] != 'Range' else trends_map['W'] if trends_map['W'] != 'Range' else trends_map['D']

        if macro_trend == 'Bearish' and trends_map['1H'] == 'Bullish':
            trends_map['1H'] = 'Range'
        if macro_trend == 'Bullish' and trends_map['1H'] == 'Bearish':
            trends_map['1H'] = 'Range'
        if macro_trend == 'Bearish' and trends_map['15m'] == 'Bullish':
            trends_map['15m'] = 'Range'
        if macro_trend == 'Bullish' and trends_map['15m'] == 'Bearish':
            trends_map['15m'] = 'Range'

        row_data['1H'] = trends_map['1H']
        row_data['15m'] = trends_map['15m']

        MTF_WEIGHTS = {'M': 5.0, 'W': 4.0, 'D': 4.0,
                       '4H': 2.5, '1H': 1.5, '15m': 1.0}
        TOTAL_WEIGHT = sum(MTF_WEIGHTS.values())

        w_bull = sum(MTF_WEIGHTS[tf] * (scores_map[tf]/100)
                     for tf in trends_map if trends_map[tf] == 'Bullish')
        w_bear = sum(MTF_WEIGHTS[tf] * (scores_map[tf]/100)
                     for tf in trends_map if trends_map[tf] == 'Bearish')

        w_bull += sum(MTF_WEIGHTS[tf] * 0.3 for tf in trends_map if trends_map[tf] == 'Retracement Bull')
        w_bear += sum(MTF_WEIGHTS[tf] * 0.3 for tf in trends_map if trends_map[tf] == 'Retracement Bear')

        high_tf_avg = (scores_map['M'] + scores_map['W'] + scores_map['D']) / 3
        quality = 'C'
        high_tf_clean = ('Retracement' not in trends_map['D'] and 'Retracement' not in trends_map['M'] and 'Retracement' not in trends_map['W'])

        if trends_map['D'] == trends_map['M'] == trends_map['W'] and high_tf_clean:
            if high_tf_avg >= 80:
                quality = 'A+'
            elif high_tf_avg >= 70:
                quality = 'A'
            else:
                quality = 'B'
        elif trends_map['D'] == trends_map['M'] and high_tf_clean:
            if high_tf_avg >= 75:
                quality = 'B+'
            else:
                quality = 'B'
        elif trends_map['D'] == trends_map['W'] and high_tf_clean:
            quality = 'B-'
        else:
            quality = 'C'

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

    bar.empty()
    status.empty()
    return pd.DataFrame(results)

# ==========================================
# GÉNÉRATION PDF PROFESSIONNELLE
# ==========================================


def create_pdf(df):
    """
    Génère un PDF professionnel pour analystes avec encodage garanti
    et présentation optimisée
    """
    try:
        # Initialiser PDF en mode paysage
        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()

        # =================== EN-TÊTE ===================
        # Titre principal
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(30, 58, 138)  # Bleu institutionnel
        pdf.cell(0, 15, "BLUESTAR HEDGE FUND GPS V2.1", ln=True, align="C")

        # Sous-titre
        pdf.set_font("Helvetica", "I", 11)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, "Multi-Timeframe Institutional Scanner",
                 ln=True, align="C")

        # Date et heure
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
                 ln=True, align="C")

        # Ligne séparatrice
        pdf.set_draw_color(30, 58, 138)
        pdf.line(10, pdf.get_y() + 3, 287, pdf.get_y() + 3)
        pdf.ln(8)

        # =================== TABLEAU ===================
        # Colonnes avec largeurs optimisées
        col_widths = {
            'Paire': 18,
            'M': 16, 'W': 16, 'D': 16, '4H': 16, '1H': 16, '15m': 16,
            'MTF': 25,
            'Quality': 15,
            'ATR_Daily': 18, 'ATR_H1': 18, 'ATR_15m': 18
        }

        cols_order = ['Paire', 'M', 'W', 'D', '4H', '1H', '15m', 'MTF', 'Quality',
                      'ATR_Daily', 'ATR_H1', 'ATR_15m']

        # En-têtes du tableau
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(40, 68, 148)
        pdf.set_text_color(255, 255, 255)

        # Noms complets des colonnes pour plus de clarté
        headers = {
            'Paire': 'PAIR',
            'M': 'MONTHLY',
            'W': 'WEEKLY',
            'D': 'DAILY',
            '4H': '4H',
            '1H': '1H',
            '15m': '15M',
            'MTF': 'CONSENSUS',
            'Quality': 'GRADE',
            'ATR_Daily': 'ATR D',
            'ATR_H1': 'ATR 1H',
            'ATR_15m': 'ATR 15M'
        }

        for col in cols_order:
            pdf.cell(col_widths[col], 8, headers[col],
                     border=1, align='C', fill=True)
        pdf.ln()

        # Données du tableau
        pdf.set_font("Helvetica", "", 8)

        for _, row in df.iterrows():
            for col in cols_order:
                val = str(row[col])

                # FORMATAGE DES VALEURS POUR MEILLEURE LISIBILITÉ
                if col == 'Paire':
                    val = val.replace('/', '')
                elif col in ['M', 'W', 'D', '4H', '1H', '15m']:
                    # Simplifier les tendances pour économiser de l'espace
                    if "Bullish" in val:
                        val = "BULL"
                    elif "Bearish" in val:
                        val = "BEAR"
                    elif "Retracement Bull" in val:
                        val = "R-BULL"
                    elif "Retracement Bear" in val:
                        val = "R-BEAR"
                    elif "Range" in val:
                        val = "RANGE"

                # COULEURS DES CELLULES
                fill_color = (255, 255, 255)  # Blanc par défaut
                text_color = (0, 0, 0)       # Noir par défaut

                # Bullish
                if "Bullish" in val and "Retracement" not in val:
                    fill_color = (46, 204, 113)
                    text_color = (255, 255, 255)
                # Bearish
                elif "Bearish" in val and "Retracement" not in val:
                    fill_color = (231, 76, 60)
                    text_color = (255, 255, 255)
                # Retracement Bull
                elif "Retracement Bull" in val:
                    fill_color = (125, 206, 160)
                    text_color = (255, 255, 255)
                # Retracement Bear
                elif "Retracement Bear" in val:
                    fill_color = (241, 148, 138)
                    text_color = (255, 255, 255)
                # Range
                elif "Range" in val:
                    fill_color = (149, 165, 166)
                    text_color = (255, 255, 255)
                # Grades
                elif col == 'Quality':
                    if val == 'A+':
                        fill_color = (255, 215, 0)  # Or
                        text_color = (0, 0, 0)
                    elif val == 'A':
                        fill_color = (76, 175, 80)  # Vert
                        text_color = (255, 255, 255)
                    elif val.startswith('B'):
                        fill_color = (33, 150, 243)  # Bleu
                        text_color = (255, 255, 255)
                    else:  # C
                        fill_color = (158, 158, 158)  # Gris
                        text_color = (255, 255, 255)

                # Appliquer les couleurs
                pdf.set_fill_color(*fill_color)
                pdf.set_text_color(*text_color)

                # Tronquer si trop long
                if len(val) > 12:
                    val = val[:10] + '..'

                # Dessiner la cellule
                pdf.cell(col_widths[col], 7, val,
                         border=1, align='C', fill=True)

            pdf.ln()

        # Saut de ligne avant légende
        pdf.ln(5)

        # =================== LÉGENDE DÉTAILLÉE ===================
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 8, "COLOR LEGEND & SYMBOLS:", ln=True)

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(0, 0, 0)

        # Légende en deux colonnes
        y_start = pdf.get_y()
        x_left = 10
        x_right = 110

        # Colonne gauche
        pdf.set_xy(x_left, y_start)

        legend_items_left = [
            ("BULL", "Strong Bullish Trend", (46, 204, 113)),
            ("BEAR", "Strong Bearish Trend", (231, 76, 60)),
            ("R-BULL", "Bullish Retracement", (125, 206, 160)),
            ("R-BEAR", "Bearish Retracement", (241, 148, 138)),
            ("RANGE", "Sideways Market", (149, 165, 166))
        ]

        for symbol, desc, color in legend_items_left:
            # Carré de couleur
            pdf.set_fill_color(*color)
            pdf.rect(pdf.get_x(), pdf.get_y() + 1, 4, 4, 'F')

            # Texte
            pdf.set_xy(pdf.get_x() + 6, pdf.get_y())
            pdf.cell(20, 5, symbol)

            pdf.set_xy(pdf.get_x() + 20, pdf.get_y())
            pdf.cell(70, 5, desc)

            pdf.ln(6)

        # Colonne droite (Grades)
        pdf.set_xy(x_right, y_start)

        legend_items_right = [
            ("A+", "Premium Setup (Gold)", (255, 215, 0)),
            ("A", "High Quality (Green)", (76, 175, 80)),
            ("B / B+ / B-", "Good Setup (Blue)", (33, 150, 243)),
            ("C", "Neutral / Wait (Gray)", (158, 158, 158))
        ]

        for symbol, desc, color in legend_items_right:
            # Carré de couleur
            pdf.set_fill_color(*color)
            pdf.rect(pdf.get_x(), pdf.get_y() + 1, 4, 4, 'F')

            # Texte
            pdf.set_xy(pdf.get_x() + 6, pdf.get_y())
            pdf.cell(25, 5, symbol)

            pdf.set_xy(pdf.get_x() + 25, pdf.get_y())
            pdf.cell(70, 5, desc)

            pdf.ln(6)

        # =================== NOTES TECHNIQUES ===================
        pdf.ln(5)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)

        notes = [
            "• MTF Consensus: Weighted average across all timeframes (M5, W4, D4, 4H2.5, 1H1.5, 15M1)",
            "• ATR Values: Average True Range - higher values indicate greater volatility",
            "• Grade A+/A: Highest conviction setups with multi-timeframe alignment",
            "• Retracement: Counter-trend move within a larger established trend",
            f"• Total Instruments Analyzed: {len(df)} • Analysis Time: {datetime.now().strftime('%H:%M UTC')}"
        ]

        for note in notes:
            pdf.cell(0, 4, note, ln=True)

        # =================== PIED DE PAGE ===================
        pdf.set_y(-15)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 10, "Bluestar GPS v2.1 - Institutional Use Only - Confidential",
                 0, 0, 'C')

        # =================== GÉNÉRATION DU PDF ===================
        buffer = BytesIO()
        pdf_output = pdf.output(dest='S')

        # ENCODAGE LATIN-1 POUR COMPATIBILITÉ MAXIMALE
        if isinstance(pdf_output, str):
            # Convertir en latin-1 (standard PDF)
            buffer.write(pdf_output.encode('latin-1', errors='replace'))
        else:
            # Déjà en bytes
            buffer.write(pdf_output)

        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        # Fallback en cas d'erreur
        return create_fallback_pdf(str(e))


def create_fallback_pdf(error_msg=""):
    """PDF minimal en cas d'erreur"""
    pdf = FPDF(orientation='L')
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 20, "BLUESTAR GPS REPORT", ln=True, align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 10, "Report generation encountered an issue.")
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 8, f"Error: {error_msg[:200]}")
    pdf.ln(10)
    pdf.multi_cell(0, 8, "Please use CSV export or try again later.")

    buffer = BytesIO()
    buffer.write(pdf.output(dest='S').encode('latin-1'))
    buffer.seek(0)
    return buffer

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
    'A+': '#fbbf24', # Or
    'A':  '#a3e635', # Vert
    'B':  '#60a5fa', # Bleu
    'B-': '#3b82f6',
    'C':  '#9ca3af'  # Gris
}

# ==========================================
# STYLE CSS AVANCÉ
# ==========================================
st.markdown("""
<style>
    .main-header {
        text-align: center; padding: 20px; background: linear-gradient(135deg, #1e3a8a 0%, #172554 100%);
        color: white; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1); text-align: center;
    }
    .metric-value { font-size: 1.5em; font-weight: bold; margin-top: 5px; }
    .metric-label { font-size: 0.9em; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Table Styles */
    .stDataFrame { width: 100%; }
    div[data-testid="stMarkdown"] { text-align: center; }
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
# LOGIQUE MTF INSTITUTIONNELLE (INCHANGÉE)
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
    if len(df) < 50: return 'Range', 0
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
        if r.status_code != 200: return pd.DataFrame()
            
        data = r.json()
        if 'candles' not in data: return pd.DataFrame()
        
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
            df = get_cached_oanda_data(pair, gran, count, account_id, access_token)
            if df.empty:
                valid_pair = False
                break
            
            if tf == 'M':
                df = df.resample('ME').agg({
                    'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'
                }).dropna()
                if len(df) < 50:
                    df_temp = get_cached_oanda_data(pair, 'D', 2000, account_id, access_token)
                    if not df_temp.empty:
                        df = df_temp.resample('ME').agg({
                            'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'
                        }).dropna()
            elif tf == 'W':
                df = df.resample('W-FRI').agg({
                    'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'
                }).dropna()
            
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

        # ATR Calc
        df_daily = data_cache['D']
        atr_daily = atr(df_daily['High'], df_daily['Low'], df_daily['Close'], 14).iloc[-1]
        row_data['ATR_Daily'] = f"{atr_daily:.5f}" if atr_daily < 1 else f"{atr_daily:.2f}"
        
        df_h1 = data_cache['1H']
        atr_h1 = atr(df_h1['High'], df_h1['Low'], df_h1['Close'], 14).iloc[-1]
        row_data['ATR_H1'] = f"{atr_h1:.5f}" if atr_h1 < 1 else f"{atr_h1:.2f}"
        
        df_15m = data_cache['15m']
        atr_15m = atr(df_15m['High'], df_15m['Low'], df_15m['Close'], 14).iloc[-1]
        row_data['ATR_15m'] = f"{atr_15m:.5f}" if atr_15m < 1 else f"{atr_15m:.2f}"

        # MTF Filter
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
# GÉNÉRATION PDF OPTIMISÉE
# ==========================================

def create_pdf(df):
    """Génère un PDF optimisé en mode paysage avec pagination et texte complet"""
    try:
        pdf = FPDF(orientation='L', unit='mm', format='A4')  # MODE PAYSAGE
        pdf.add_page()
        
        # --- CONFIGURATION DES MARGES ET COULEURS ---
        margin_left = 10
        margin_top = 10
        pdf.set_margins(left=margin_left, top=margin_top, right=10)
        
        # En-tête document
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "BLUESTAR HEDGE FUND GPS V2.1", ln=True, align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}", ln=True, align="C")
        pdf.ln(5)
        
        # --- DÉFINITION DES COLONNES ET LARGEURS OPTIMISÉES (Total ~277mm utilisable) ---
        cols = ['Paire', 'M', 'W', 'D', '4H', '1H', '15m', 'MTF', 'Quality', 'ATR_Daily', 'ATR_H1', 'ATR_15m']
        
        # Largeurs ajustées pour éviter la troncature du texte
        col_widths = {
            'Paire': 24,       # Ticker
            'M': 22, 'W': 22, 'D': 22, '4H': 22, '1H': 22, '15m': 22, # Tendance (espace pour "Retracement Bear")
            'MTF': 35,         # Pourcentage long
            'Quality': 15,     # Grade court
            'ATR_Daily': 20, 'ATR_H1': 20, 'ATR_15m': 20  # Chiffres
        }
        # Total largeur : 24 + (22*6) + 35 + 15 + (20*3) = 261mm (Fait dans les 277mm dispo)
        
        row_height = 6  # Hauteur de ligne réduite pour tenir plus de lignes
        
        # --- FONCTION INTERNE POUR IMPRIMER L'EN-TÊTE DU TABLEAU ---
        def print_table_header():
            pdf.set_y(pdf.get_y() + 2)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(30, 58, 138)  # Bleu foncé
            pdf.set_text_color(255, 255, 255)
            
            for c in cols:
                pdf.cell(col_widths[c], 8, c.replace('_', ' '), border=1, align='C', fill=True)
            pdf.ln()
            pdf.set_font("Helvetica", "", 7) # Retour police normale pour les données

        # Impression du premier en-tête
        print_table_header()
        
        # --- IMPRESSION DES DONNÉES ---
        page_height = 297 # Hauteur page A4 Landscape
        bottom_margin = 20 # Espace en bas pour la légende
        
        for _, row in df.iterrows():
            # VÉRIFICATION PAGINATION
            # Si la position Y actuelle + hauteur ligne dépasse la limite
            if pdf.get_y() + row_height > page_height - bottom_margin:
                pdf.add_page()
                print_table_header() # On réimprime les titres sur la nouvelle page
            
            for c in cols:
                val = str(row[c])
                
                # --- GESTION DES COULEURS ---
                fill_color = (255, 255, 255)
                text_color = (0, 0, 0)
                
                if "Bull" in val and "Retracement" not in val:
                    fill_color = (46, 204, 113) # Vert
                    text_color = (255, 255, 255)
                elif "Bear" in val and "Retracement" not in val:
                    fill_color = (231, 76, 60) # Rouge
                    text_color = (255, 255, 255)
                elif "Retracement Bull" in val:
                    fill_color = (125, 206, 160) # Vert clair
                    text_color = (255, 255, 255)
                elif "Retracement Bear" in val:
                    fill_color = (241, 148, 138) # Rouge clair
                    text_color = (255, 255, 255)
                elif "Range" in val:
                    fill_color = (149, 165, 166) # Gris
                    text_color = (255, 255, 255)
                elif c == 'Quality':
                    if val == 'A+': fill_color = (251, 191, 36) # Or
                    elif val == 'A': fill_color = (163, 230, 53) # Vert
                    elif val.startswith('B'): fill_color = (96, 165, 250) # Bleu
                    else: fill_color = (156, 163, 175) # Gris
                    text_color = (0, 0, 0)
                
                pdf.set_fill_color(*fill_color)
                pdf.set_text_color(*text_color)
                
                # IMPRESSION CELLULE (SANS TRONCATURE)
                # Largeur colonne suffisante pour contenir le texte
                pdf.cell(col_widths[c], row_height, val, border=1, align='C', fill=True)
            
            pdf.ln()

        # --- LÉGENDE (En bas de la dernière page) ---
        pdf.ln(4)
        # Vérifier si la légende tient, sinon nouvelle page
        if pdf.get_y() > page_height - 40:
            pdf.add_page()
            
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 5, "LEGEND:", ln=True)
        pdf.set_font("Helvetica", "", 7)
        
        legends = [
            ("Bullish", 46, 204, 113),
            ("Bearish", 231, 76, 60),
            ("Retracement Bull", 125, 206, 160),
            ("Retracement Bear", 241, 148, 138),
            ("Range", 149, 165, 166),
            ("Quality A+", 251, 191, 36),
            ("Quality A", 163, 230, 53),
            ("Quality B", 96, 165, 250)
        ]
        
        x_start = 10
        y_pos = pdf.get_y()
        for legend, r, g, b in legends:
            pdf.set_fill_color(r, g, b)
            pdf.rect(x_start, y_pos, 4, 4, 'F')
            pdf.set_xy(x_start + 5, y_pos)
            pdf.set_text_color(0,0,0)
            pdf.cell(35, 4, legend)
            x_start += 40
            if x_start > 250: # Retour à la ligne si trop large
                x_start = 10
                y_pos += 5
                pdf.set_xy(x_start, y_pos)
        
        # --- GÉNÉRATION BUFFER ---
        buffer = BytesIO()
        pdf_output = pdf.output(dest='S')
        
        # Gestion compatibilité Python
        if isinstance(pdf_output, str):
            buffer.write(pdf_output.encode('latin-1'))
        else:
            buffer.write(pdf_output)
        
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        # En cas d'erreur critique, PDF d'erreur
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "PDF Generation Error", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 10, f"Error: {str(e)}")
        
        buffer = BytesIO()
        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, str):
            buffer.write(pdf_output.encode('latin-1'))
        else:
            buffer.write(pdf_output)
        buffer.seek(0)
        return buffer.getvalue()

# ==========================================
# UI PRINCIPALE
# ==========================================

def main():
    st.markdown("<div class='main-header'><h1>🧭 BLUESTAR HEDGE FUND GPS V2.1</h1></div>", unsafe_allow_html=True)

    try:
        acc = st.secrets["OANDA_ACCOUNT_ID"]
        tok = st.secrets["OANDA_ACCESS_TOKEN"]
    except Exception:
        st.error("❌ Secrets OANDA manquants")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Configuration")
        show_only_best = st.checkbox("Afficher uniquement Grade A+ / A", value=False)
        st.info("Le cache dure 10 minutes pour optimiser les performances API.")

    if st.button("🚀 LANCER L'ANALYSE TOUS ACTIFS", type="primary", use_container_width=True):
        with st.spinner("Analyse Multi-Timeframe en cours..."):
            df = analyze_market(acc, tok)
        
        if not df.empty:
            # Filtrage UX
            if show_only_best:
                df = df[df['Quality'].isin(['A+', 'A'])]
            
            # Tri
            df = df.sort_values(by=['Quality', 'MTF'], ascending=[True, False]) 
            st.session_state.df = df
    
    if "df" in st.session_state:
        df = st.session_state.df
        
        # --- COMMAND CENTER ---
        c1, c2, c3, c4 = st.columns(4)
        total = len(df)
        a_plus = len(df[df['Quality'] == 'A+'])
        a_grade = len(df[df['Quality'] == 'A'])
        b_grade = len(df[df['Quality'].str.startswith('B')])
        
        c1.metric("Total Analyzed", total)
        c2.metric("Setups A+ (GOLD)", a_plus, delta_color="inverse")
        c3.metric("Setups A (GREEN)", a_grade, delta_color="inverse")
        c4.metric("Setups B (BLUE)", b_grade, delta_color="inverse")
        
        # --- TABLEAU STYLE ---
        cols_order = ['Paire', 'M', 'W', 'D', '4H', '1H', '15m', 'MTF', 'Quality', 'ATR_Daily', 'ATR_H1', 'ATR_15m']
        
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

        st.dataframe(
            df[cols_order].style.apply(quality_style, axis=0).applymap(style_map), 
            height=min(600, (len(df)+1)*35 + 10), 
            use_container_width=True
        )
        
        # --- EXPORTS ---
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📄 Télécharger PDF", create_pdf(df[cols_order]), "Bluestar_GPS.pdf", "application/pdf", use_container_width=True)
        with c2:
            st.download_button("📊 Télécharger CSV", df[cols_order].to_csv(index=False).encode(), "Bluestar_GPS.csv", "text/csv", use_container_width=True)

if __name__ == "__main__":
    main()

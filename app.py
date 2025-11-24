# app.py - Bluestar Institutional GPS (SMA 200 Logic)
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
    'NZDJPY', 'NZDCAD', 'NZDCHF'
]

# Couleurs Visuelles
TREND_COLORS = {
    'Bullish': '#2ecc71',   # Vert Vif
    'Bearish': '#e74c3c',   # Rouge Vif
    'Correction': '#f39c12',# Orange
    'Range': '#95a5a6'      # Gris
}

# Poids pour le Score MTF global
MTF_WEIGHTS = {'M': 3.0, 'W': 2.5, 'D': 4.0, '4H': 3.0, '1H': 1.5, '15m': 1.0}
TOTAL_WEIGHT = sum(MTF_WEIGHTS.values())

# ==================== INDICATEURS ====================
def sma(series, length):
    """Simple Moving Average (L'outil des pros pour le long terme)"""
    return series.rolling(window=length).mean()

def zlema(series, length):
    """Zero-Lag EMA (Pour le timing court terme)"""
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

# ==================== LOGIQUE "GPS" (MACRO) ====================
def calc_macro_trend(df):
    """
    Logique Spéciale Monthly/Weekly.
    Priorité absolue à la SMA 200 et SMA 50.
    Pas de "Range" sauf cas exceptionnel.
    """
    # Il nous faut au moins la SMA 50, idéalement la 200
    if len(df) < 50: return 'Neutral', 0
    
    close = df['Close']
    curr_price = close.iloc[-1]
    
    # Calculs SMA
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    
    curr_sma50 = sma50.iloc[-1]
    
    # On vérifie si on a assez de data pour la SMA 200
    has_200 = len(df) >= 200
    curr_sma200 = sma200.iloc[-1] if has_200 else None

    trend = "Neutral"
    score = 0

    # --- LOGIQUE INSTITUTIONNELLE ---
    if has_200:
        # La SMA 200 est le JUGE DE PAIX
        if curr_price > curr_sma200:
            trend = "Bullish"
            score = 60
            if curr_price > curr_sma50: score += 20 # Momentum fort
            if curr_sma50 > curr_sma200: score += 20 # Golden Cross
        else:
            trend = "Bearish"
            score = 60
            if curr_price < curr_sma50: score += 20 # Momentum fort
            if curr_sma50 < curr_sma200: score += 20 # Death Cross
    else:
        # Fallback sur SMA 50 si historique trop court (ex: paire récente)
        if curr_price > curr_sma50:
            trend = "Bullish"; score = 50
        else:
            trend = "Bearish"; score = 50
            
    return trend, min(100, score)

# ==================== LOGIQUE "TIMING" (INTRADAY) ====================
def calc_intraday_trend(df):
    """
    Logique Daily/H4/H1/15m.
    Utilise ZLEMA et Baseline pour gérer les corrections/ranges.
    """
    if len(df) < 50: return 'Range', 0
    
    close = df['Close']
    zlema_val = zlema(close, 50)
    baseline = sma(close, 200) # On garde SMA 200 comme baseline ici aussi
    adx_val, p_di, m_di = adx(df['High'], df['Low'], close, 14)
    
    curr_price = close.iloc[-1]
    curr_zlema = zlema_val.iloc[-1]
    curr_adx = adx_val.iloc[-1]
    
    has_base = len(df) >= 200
    curr_base = baseline.iloc[-1] if has_base else curr_zlema

    trend = "Range"
    
    # Logique ZLEMA (Réactivité)
    if curr_price > curr_zlema:
        if has_base and curr_price > curr_base: trend = "Bullish"
        elif has_base and curr_price < curr_base: trend = "Correction" # Hausse sous la 200
        else: trend = "Bullish"
    elif curr_price < curr_zlema:
        if has_base and curr_price < curr_base: trend = "Bearish"
        elif has_base and curr_price > curr_base: trend = "Correction" # Baisse au dessus de la 200
        else: trend = "Bearish"

    # Filtre ADX pour le range pur (Uniquement intraday)
    if curr_adx < 20 and trend == "Correction": trend = "Range"
    
    # Nettoyage pour l'affichage : On simplifie Correction -> Range pour le tableau compact
    if trend == "Correction": 
        # Si on est en correction Bullish (Prix > 200 mais < ZLEMA), c'est une zone d'achat potentielle
        # Mais pour le tableau, on peut afficher Range ou garder la couleur Correction
        pass 

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
    # IMPORTANT : Pour le Monthly SMA 200, il faut ~4200 jours (200 mois * 21 jours)
    # On demande le maximum possible en Daily pour reconstruire le Monthly
    tf_config = {
        'M':   ('D', 4500, 'Macro'), # Resample Monthly
        'W':   ('D', 2000, 'Macro'), # Resample Weekly
        'D':   ('D', 300,  'Intra'),
        '4H':  ('H4', 300, 'Intra'),
        '1H':  ('H1', 300, 'Intra'),
        '15m': ('M15', 300,'Intra')
    }
    
    bar = st.progress(0)
    status = st.empty()
    
    for idx, pair in enumerate(FOREX_PAIRS_EXTENDED):
        inst = f"{pair[:3]}_{pair[3:]}"
        status.text(f"GPS Institutionnel : {pair}...")
        
        row_data = {'Paire': f"{pair[:3]}/{pair[3:]}"}
        trends_map = {}
        scores_map = {}
        
        # Récupération et Calcul
        valid_pair = True
        
        # 1. Récupération des données brutes
        data_cache = {}
        for tf, (gran, count, mode) in tf_config.items():
            # Optimisation: Si on a déjà chargé D (4500) pour M, on peut l'utiliser pour W et D
            # Mais pour simplifier le cache, on appelle l'API (le cache gérera)
            df = get_cached_oanda_data(inst, gran, count, account_id, access_token)
            
            if df.empty: 
                valid_pair = False; break
            
            # Resampling pour M et W
            if tf == 'M':
                df = df.resample('ME').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            elif tf == 'W':
                df = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            
            data_cache[tf] = df

        if not valid_pair: continue

        # 2. Calcul des Tendances
        for tf, (gran, count, mode) in tf_config.items():
            df = data_cache[tf]
            if mode == 'Macro':
                t, s = calc_macro_trend(df)
            else:
                t, s = calc_intraday_trend(df)
            
            trends_map[tf] = t
            scores_map[tf] = s
            row_data[tf] = t

        # 3. Calcul Global MTF
        w_bull = sum(MTF_WEIGHTS[tf] for tf in trends_map if trends_map[tf] == 'Bullish')
        w_bear = sum(MTF_WEIGHTS[tf] for tf in trends_map if trends_map[tf] == 'Bearish')
        
        # Quality Check (Daily Alignment)
        quality = 'C'
        if trends_map['D'] == trends_map['M']: quality = 'B' # Daily suit Monthly
        if trends_map['D'] == trends_map['M'] == trends_map['W']: quality = 'A' # Tiercé gagnant
        if quality == 'A' and scores_map['D'] > 25: quality = 'A+' # Avec momentum

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

# ==================== EXPORTS ====================
def create_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Bluestar GPS Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(5)
    cols = ['Paire', 'M', 'W', 'D', '4H', '1H', '15m', 'MTF', 'Quality']
    w = pdf.w / (len(cols)+1)
    pdf.set_font("Helvetica", "B", 7)
    for c in cols: pdf.cell(w, 8, c, 1, align='C', new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln()
    pdf.set_font("Helvetica", "", 7)
    for _, row in df.iterrows():
        for c in cols:
            val = str(row[c])
            pdf.set_fill_color(255,255,255)
            if "Bull" in val: pdf.set_fill_color(46, 204, 113)
            elif "Bear" in val: pdf.set_fill_color(231, 76, 60)
            elif "Corr" in val: pdf.set_fill_color(243, 156, 18)
            elif "Range" in val: pdf.set_fill_color(149, 165, 166)
            pdf.cell(w, 8, val, 1, align='C', fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

# ==================== MAIN UI ====================
def main():
    st.markdown("""
        <div style='text-align:center; padding:15px; background:#2c3e50; color:white; border-radius:10px; margin-bottom:15px'>
            <h2 style='margin:0'>🏛️ Bluestar Hedge Fund GPS</h2>
            <p style='margin:5px 0 0 0; font-size:0.8rem'>Monthly & Weekly = SMA 200 Logic (Pas de Range) • Daily -> 15m = Timing</p>
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
                # Tri: Qualité d'abord, puis % MTF
                df = df.sort_values(by=['Quality', 'MTF'], ascending=[True, False]) 
                st.session_state.df = df
    
    if "df" in st.session_state:
        df = st.session_state.df
        cols_order = ['Paire', 'M', 'W', 'D', '4H', '1H', '15m', 'MTF', 'Quality']
        
        def style_map(v):
            if isinstance(v, str):
                if "Bull" in v: return f"background-color: {TREND_COLORS['Bullish']}; color:white; font-weight:bold"
                if "Bear" in v: return f"background-color: {TREND_COLORS['Bearish']}; color:white; font-weight:bold"
                if "Corr" in v: return f"background-color: {TREND_COLORS['Correction']}; color:white"
                if "Range" in v: return f"background-color: {TREND_COLORS['Range']}; color:white"
            return ""

        # Hauteur dynamique
        h = (len(df) + 1) * 35 + 3
        st.dataframe(df[cols_order].style.map(style_map), height=h, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1: st.download_button("PDF Report", create_pdf(df[cols_order]), "Bluestar_GPS.pdf", "application/pdf")
        with c2: st.download_button("CSV Data", df[cols_order].to_csv(index=False).encode(), "Bluestar_GPS.csv", "text/csv")

if __name__ == "__main__":
    main()

# app.py - Bluestar MTF Pro+ (Version Institutionnelle & Stable)
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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CONSTANTES & PARAMÈTRES ====================
OANDA_API_URL = "https://api-fxpractice.oanda.com"

# Liste étendue des paires Forex majeures et mineures
FOREX_PAIRS_EXTENDED = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
    'EURGBP', 'EURJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD',
    'GBPJPY', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD',
    'AUDJPY', 'AUDCAD', 'AUDCHF', 'AUDNZD',
    'CADJPY', 'CADCHF', 'CHFJPY',
    'NZDJPY', 'NZDCAD', 'NZDCHF'
]

# Couleurs Institutionnelles (Plus nettes)
TREND_COLORS_HEX = {
    'Bullish': '#2ecc71', # Vert Emeraude (Achat fort)
    'Bearish': '#e74c3c', # Rouge Alizarin (Vente forte)
    'Correction': '#f39c12', # Orange (Attente / Repli)
    'Range': '#95a5a6'    # Gris (Indécision)
}

# Paramètres Techniques
LENGTH_ZLEMA = 50       # Tendance Intermédiaire
LENGTH_BASELINE = 200   # Tendance de Fond (Institutionnelle)
ADX_PERIOD = 14

# Pondération "Hedge Fund" : Focus sur le Daily et H4 pour l'exécution
# On réduit l'impact du Monthly qui est trop lent pour le trading actif
MTF_WEIGHTS = {
    '15m': 1.0, 
    '1H': 2.0, 
    '4H': 4.0,  # Structure de marché clé
    'D': 5.0,   # Tendance Directrice (King)
    'W': 3.0,   # Contexte Macro
    'M': 2.0    # Biais long terme
}
TOTAL_WEIGHT = sum(MTF_WEIGHTS.values())

# ==================== INDICATEURS TECHNIQUES ====================
def zlema(series, length):
    """Zero-Lag Exponential Moving Average"""
    lag = int((length - 1) / 2)
    src_adj = series + (series - series.shift(lag))
    return src_adj.ewm(span=length, adjust=False).mean()

def ema(series, length):
    """Standard Exponential Moving Average (Baseline)"""
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

# ==================== COEUR DU SYSTÈME (INSTITUTIONNEL) ====================
def calc_professional_trend(df, timeframe='D'):
    # Sécurité anti-crash si pas assez de données
    if df.empty or len(df) < LENGTH_ZLEMA + 5:
        return 'Range', 0, 'C', 0

    close = df['Close']
    high = df['High']
    low = df['Low']

    # 1. Calcul des Indicateurs
    zlema_val = zlema(close, LENGTH_ZLEMA)
    adx_val, plus_di, minus_di = adx(high, low, close, ADX_PERIOD)
    
    # Baseline 200 (Seulement si assez d'historique, sinon on ignore)
    has_baseline = len(df) >= LENGTH_BASELINE
    baseline = ema(close, LENGTH_BASELINE) if has_baseline else pd.Series(0, index=close.index)

    # Dernières valeurs
    curr_price = close.iloc[-1]
    curr_zlema = zlema_val.iloc[-1]
    curr_adx = adx_val.iloc[-1]
    curr_di_plus = plus_di.iloc[-1]
    curr_di_minus = minus_di.iloc[-1]
    
    # Valeur Baseline (si dispo)
    curr_baseline = baseline.iloc[-1] if has_baseline else 0

    # 2. Logique de Décision (Hedge Fund Logic)
    trend = "Range"
    strength_score = 0
    
    # --- SCÉNARIO AVEC BASELINE (Historique suffisant) ---
    if has_baseline:
        # BULLISH SCENARIOS
        if curr_price > curr_zlema:
            if curr_price > curr_baseline:
                trend = "Bullish" # Full Trend
                strength_score = 60
                if curr_zlema > curr_baseline: strength_score += 20 # Alignement parfait
            else:
                # Prix > ZLEMA mais sous la 200 = Contre-tendance ou Rebond technique
                trend = "Correction" if curr_di_plus > curr_di_minus else "Range"
                strength_score = 40
        
        # BEARISH SCENARIOS
        elif curr_price < curr_zlema:
            if curr_price < curr_baseline:
                trend = "Bearish" # Full Trend
                strength_score = 60
                if curr_zlema < curr_baseline: strength_score += 20 # Alignement parfait
            else:
                # Prix < ZLEMA mais au-dessus de la 200 = Correction dans tendance haussière
                trend = "Correction" if curr_di_minus > curr_di_plus else "Range"
                strength_score = 40
        
        # FILTER: Si coincé entre ZLEMA et Baseline avec ADX faible
        dist_ma = abs(curr_zlema - curr_baseline) / curr_baseline * 100
        if dist_ma < 0.2 and curr_adx < 20: 
            trend = "Range" # Squeeze extreme

    # --- SCÉNARIO SANS BASELINE (Ex: Weekly/Monthly avec peu de data) ---
    else:
        if curr_price > curr_zlema and curr_di_plus > curr_di_minus:
            trend = "Bullish"
            strength_score = 50
        elif curr_price < curr_zlema and curr_di_minus > curr_di_plus:
            trend = "Bearish"
            strength_score = 50
        else:
            trend = "Range"

    # 3. Bonus Momentum (ADX)
    if curr_adx > 25: strength_score += 10
    if curr_adx > 40: strength_score += 10 # Tendance très puissante

    # 4. Qualité
    strength_score = min(100, strength_score)
    if strength_score >= 80: quality = 'A+'
    elif strength_score >= 65: quality = 'A'
    elif strength_score >= 50: quality = 'B'
    else: quality = 'C'

    # Normalisation des termes pour l'affichage
    if trend == "Correction":
        # Pour simplifier le tableau MTF, on définit le sens de la correction
        # Si on corrige dans un marché haussier (Prix > 200), le biais reste Bullish mais faible
        if has_baseline and curr_price > curr_baseline: trend = "Range" # Zone d'achat potentielle
        elif has_baseline and curr_price < curr_baseline: trend = "Range" # Zone de vente potentielle
    
    return trend, strength_score, quality, round(curr_adx, 1)

# ==================== OANDA DATA (ROBUSTE) ====================
def get_oanda_data(instrument, granularity, count, account_id, access_token, max_retries=3):
    url = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {'granularity': granularity, 'count': count, 'price': 'M'}

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=15)
            if r.status_code != 200:
                time.sleep(1)
                continue
            
            data = r.json()
            if 'candles' not in data: return pd.DataFrame()
            
            candles = [c for c in data['candles'] if c.get('complete')]
            if not candles: return pd.DataFrame()
            
            df = pd.DataFrame([{
                'date': c['time'], 
                'Open': float(c['mid']['o']),
                'High': float(c['mid']['h']), 
                'Low': float(c['mid']['l']),
                'Close': float(c['mid']['c'])
            } for c in candles])
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        except:
            time.sleep(1)
    return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_oanda_data(instrument, granularity, count, account_id, access_token):
    return get_oanda_data(instrument, granularity, count, account_id, access_token)

# ==================== MOTEUR D'ANALYSE ====================
def analyze_forex_pairs(account_id, access_token):
    results = []
    
    # IMPORTANT: On demande 400 bougies pour assurer le calcul de la 200 EMA
    # Pour W et M, on prend le max possible via Daily pour simuler
    tf_params = {
        '15m': ('M15', 400), 
        '1H': ('H1', 400), 
        '4H': ('H4', 400),
        'D': ('D', 400), 
        'W': ('D', 1500), # On récupère bcp de Daily pour resampler
        'M': ('D', 2500)  # On récupère max Daily pour resampler
    }
    
    total = len(FOREX_PAIRS_EXTENDED)
    bar = st.progress(0)
    status = st.empty()

    for idx, pair in enumerate(FOREX_PAIRS_EXTENDED):
        instrument = f"{pair[:3]}_{pair[3:]}"
        status.text(f"Analyse Institutionnelle : {pair}...")
        data = {}
        ok = True

        for tf, (g, c) in tf_params.items():
            df = get_cached_oanda_data(instrument, g, c, account_id, access_token)
            
            # Resampling intelligent
            if tf == 'W' and not df.empty:
                df = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            if tf == 'M' and not df.empty:
                df = df.resample('ME').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            
            if df.empty:
                ok = False
                break
            data[tf] = df

        if not ok: continue

        trends = {tf: calc_professional_trend(data[tf], tf)[0] for tf in tf_params}
        qualities = {tf: calc_professional_trend(data[tf], tf)[2] for tf in tf_params}
        
        # Calcul du Score MTF Pondéré
        score_bull = sum(MTF_WEIGHTS[tf] for tf in trends if trends[tf] == 'Bullish')
        score_bear = sum(MTF_WEIGHTS[tf] for tf in trends if trends[tf] == 'Bearish')
        
        # Algorithme de décision finale
        # Si le Daily et H4 sont d'accord, c'est le signal dominant
        dominant_tf_trend = trends['D']
        
        if score_bull > score_bear and score_bull >= (TOTAL_WEIGHT * 0.4):
            mtf_trend = 'Bullish'
            power = (score_bull / TOTAL_WEIGHT) * 100
        elif score_bear > score_bull and score_bear >= (TOTAL_WEIGHT * 0.4):
            mtf_trend = 'Bearish'
            power = (score_bear / TOTAL_WEIGHT) * 100
        else:
            mtf_trend = 'Range'
            power = 0

        # Qualité Globale (basée sur le Daily)
        final_quality = qualities['D']

        results.append({
            'Paire': f"{pair[:3]}/{pair[3:]}",
            '15m': trends['15m'], '1H': trends['1H'], '4H': trends['4H'],
            'D': trends['D'], 'W': trends['W'], 'M': trends['M'],
            'MTF': f"{mtf_trend} ({power:.0f}%)",
            'Quality': final_quality
        })

        bar.progress((idx + 1) / total)

    bar.empty()
    status.empty()
    
    if not results: return pd.DataFrame()
    
    df = pd.DataFrame(results)
    # Tri intelligent : Bullish A+ en premier, puis Bullish A, etc.
    return df.sort_values(by=['MTF', 'Quality'], ascending=[False, True])

# ==================== EXPORTS & PDF ====================
def create_pdf_report(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Rapport Institutionnel - Bluestar MTF", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(8)
    col_w = pdf.w / (len(df.columns) + 1)
    pdf.set_font("Helvetica", "B", 8)
    for col in df.columns:
        pdf.cell(col_w, 8, col, border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for _, row in df.iterrows():
        for val in row:
            val_str = str(val)
            fill = False
            pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(0, 0, 0)
            
            if "Bullish" in val_str:
                pdf.set_fill_color(46, 204, 113); pdf.set_text_color(255,255,255); fill=True
            elif "Bearish" in val_str:
                pdf.set_fill_color(231, 76, 60); pdf.set_text_color(255,255,255); fill=True
            elif "Range" in val_str:
                pdf.set_fill_color(149, 165, 166); pdf.set_text_color(255,255,255); fill=True
                
            pdf.cell(col_w, 8, val_str, border=1, align="C", fill=fill, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln()
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer.getvalue()

def create_png_report(df):
    text = f"Bluestar Institutional Feed - {datetime.now():%d/%m/%Y %H:%M}\n" + "="*80 + "\n"
    text += df.to_string(index=False)
    font = ImageFont.load_default()
    lines = text.split('\n')
    h = len(lines) * 15 + 50
    img = Image.new('RGB', (1400, h), 'white')
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((20, 20 + i*15), line, fill='black', font=font)
    buf = BytesIO()
    img.save(buf, 'PNG')
    return buf.getvalue()

# ==================== INTERFACE UTILISATEUR ====================
def main():
    st.markdown("""
    <div style="text-align:center;padding:20px;background-color:#1e272e;color:white;border-radius:10px;margin-bottom:20px">
        <h1>🏛️ Bluestar Institutional Dashboard</h1>
        <p>Analyse Structurelle • Moyennes Mobiles 200 • Multi-Timeframe</p>
    </div>
    """, unsafe_allow_html=True)

    # Récupération Secrets
    try:
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
    except:
        st.warning("⚠️ Clés API non trouvées dans .streamlit/secrets.toml")
        st.stop()

    if st.button("🔄 SCANNER LE MARCHÉ (HEDGE FUND LOGIC)", type="primary", use_container_width=True):
        with st.spinner("Analyse des structures de marché en cours..."):
            df = analyze_forex_pairs(account_id, access_token)
        
        if df.empty:
            st.error("Erreur de connexion API ou pas de données.")
        else:
            st.session_state.df = df
            st.success("Analyse terminée.")

    if "df" in st.session_state:
        df = st.session_state.df

        def color_cell(val):
            if isinstance(val, str):
                if "Bullish" in val: return f"background-color: {TREND_COLORS_HEX['Bullish']}; color: white; font-weight: bold"
                if "Bearish" in val: return f"background-color: {TREND_COLORS_HEX['Bearish']}; color: white; font-weight: bold"
                if "Range" in val: return f"background-color: {TREND_COLORS_HEX['Range']}; color: white"
            return ""

        st.dataframe(
            df.style.map(color_cell, subset=['15m','1H','4H','D','W','M','MTF']),
            height=800,
            use_container_width=True
        )

        # Boutons Export
        c1, c2, c3 = st.columns(3)
        now = datetime.now().strftime("%Y%m%d_%H%M")
        with c1: st.download_button("📥 Télécharger PDF", create_pdf_report(df), f"Report_{now}.pdf", "application/pdf")
        with c2: st.download_button("🖼️ Télécharger PNG", create_png_report(df), f"Report_{now}.png", "image/png")
        with c3: st.download_button("📊 Télécharger CSV", df.to_csv(index=False).encode(), f"Report_{now}.csv", "text/csv")

if __name__ == "__main__":
    main()

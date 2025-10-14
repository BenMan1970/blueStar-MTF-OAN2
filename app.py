import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import requests
import time
from io import BytesIO

# Import pour l'image et le PDF
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF

# --- Constantes ---
OANDA_API_URL = "https://api-fxpractice.oanda.com"
FOREX_PAIRS_EXTENDED = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
    'EURGBP', 'EURJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD',
    'GBPJPY', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD',
    'AUDJPY', 'AUDCAD', 'AUDCHF', 'AUDNZD',
    'CADJPY', 'CADCHF',
    'CHFJPY',
    'NZDJPY', 'NZDCAD', 'NZDCHF'
]
TREND_COLORS_HEX = {'Bullish': '#008f7a', 'Bearish': '#d9534f', 'Neutral': '#808080'}
TREND_COLORS_RGB = {'Bullish': (0, 143, 122), 'Bearish': (217, 83, 79), 'Neutral': (128, 128, 128)}

# --- Paramètres Bluestar ---
LENGTH = 70
MULT = 1.2
USE_MOMENTUM_FILTER = True
USE_VOLUME_FILTER = True
TREND_CONFIRM_BARS = 3
VOLATILITY_PERIOD = 14

# --- Fonctions indicateurs techniques ---

def zlema(series, length):
    """Zero-Lag EMA"""
    if len(series) < length:
        return pd.Series([np.nan] * len(series), index=series.index)
    
    lag = int((length - 1) / 2)
    src_adjusted = series + (series - series.shift(lag))
    return src_adjusted.ewm(span=length, adjust=False).mean()

def atr(high, low, close, period):
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def rsi(series, period=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    """MACD"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calc_professional_trend(df, timeframe='D'):
    """
    Calcule la tendance professionnelle selon la logique Bluestar MTF Pro+
    """
    if df.empty or len(df) < 100:
        return 'Neutral', 0, 'D'
    
    # Ajustement des paramètres selon le timeframe
    is_monthly = timeframe == 'M'
    is_weekly = timeframe == 'W'
    
    adj_length = 12 if is_monthly else 26 if is_weekly else LENGTH
    adj_vol = 6 if is_monthly else 10 if is_weekly else VOLATILITY_PERIOD
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    open_price = df['Open']
    
    # Volume (si disponible, sinon on simule)
    if 'Volume' in df.columns:
        volume = df['Volume']
    else:
        volume = pd.Series([1] * len(df), index=df.index)
    
    # Calcul ZLEMA
    zlema_series = zlema(close, adj_length)
    
    # Calcul volatilité adaptative
    atr_adaptive = atr(high, low, close, adj_vol)
    atr_ma = atr_adaptive.rolling(window=adj_length).mean()
    volatility_ratio = atr_adaptive / atr_ma
    volatility = atr_adaptive.rolling(window=adj_length * 3).max() * MULT * volatility_ratio
    
    # Bandes
    upper_band = zlema_series + volatility
    lower_band = zlema_series - volatility
    
    # Structure de marché
    ema20 = close.ewm(span=20, adjust=False).mean()
    structure_bullish = (ema20.diff(3) > 0).fillna(False)
    structure_bearish = (ema20.diff(3) < 0).fillna(False)
    
    # Momentum
    rsi_series = rsi(close, 14)
    rsi_trend = np.where(rsi_series > 50, 1, np.where(rsi_series < 50, -1, 0))
    
    macd_line, macd_signal = macd(close)
    macd_trend = np.where(macd_line > macd_signal, 1, -1)
    
    momentum = close.diff(10)
    momentum_trend = np.where(momentum > 0, 1, -1)
    
    momentum_score = (rsi_trend + macd_trend + momentum_trend) / 3
    
    # Volume
    vol_ma = volume.rolling(window=20).mean()
    vol_ratio = volume / vol_ma
    strong_volume = vol_ratio > 1.5
    
    # Chandeliers
    bullish_candle = close > open_price
    bearish_candle = close < open_price
    candle_size = abs(close - open_price)
    avg_candle_size = candle_size.rolling(window=14).mean()
    strong_candle = candle_size > avg_candle_size * 1.2
    
    # Tendance brute (crossover/crossunder)
    raw_trend = pd.Series([0] * len(close), index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i] and close.iloc[i-1] <= upper_band.iloc[i-1]:
            raw_trend.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i] and close.iloc[i-1] >= lower_band.iloc[i-1]:
            raw_trend.iloc[i] = -1
        else:
            raw_trend.iloc[i] = raw_trend.iloc[i-1]
    
    # Système de scoring
    bullish_signals = pd.Series([0] * len(close), index=close.index, dtype=float)
    bearish_signals = pd.Series([0] * len(close), index=close.index, dtype=float)
    
    # Position par rapport à ZLEMA
    bullish_signals += np.where(close > zlema_series, 2, 0)
    bearish_signals += np.where(close <= zlema_series, 2, 0)
    
    # Structure
    bullish_signals += np.where(structure_bullish, 1, 0)
    bearish_signals += np.where(structure_bearish, 1, 0)
    
    # Momentum
    if USE_MOMENTUM_FILTER:
        bullish_signals += np.where(momentum_score > 0.3, 1, 0)
        bearish_signals += np.where(momentum_score < -0.3, 1, 0)
    
    # Volume
    if USE_VOLUME_FILTER:
        bullish_signals += np.where(strong_volume & bullish_candle & strong_candle, 1, 0)
        bearish_signals += np.where(strong_volume & bearish_candle & strong_candle, 1, 0)
    
    # Raw trend
    bullish_signals += np.where(raw_trend == 1, 2, 0)
    bearish_signals += np.where(raw_trend == -1, 2, 0)
    
    # Tendance potentielle
    potential_trend = np.where(
        bullish_signals > bearish_signals, 1,
        np.where(bearish_signals > bullish_signals, -1, 0)
    )
    
    # Confirmation sur plusieurs barres
    confirmed_trend = pd.Series([0] * len(close), index=close.index)
    confirmation_count = 0
    current_confirmed = 0
    
    for i in range(len(potential_trend)):
        if potential_trend[i] != current_confirmed:
            confirmation_count += 1
            if confirmation_count >= TREND_CONFIRM_BARS:
                current_confirmed = potential_trend[i]
                confirmation_count = 0
        else:
            confirmation_count = 0
        confirmed_trend.iloc[i] = current_confirmed
    
    # Force du signal
    price_distance = abs((close - zlema_series) / zlema_series * 100)
    signal_strength = abs(bullish_signals - bearish_signals)
    
    pd_norm = np.minimum(100, price_distance)
    mom_norm = abs(momentum_score) * 100
    sig_norm = signal_strength / 7.0 * 100
    
    strength = np.minimum(100, (pd_norm * 30 + mom_norm * 30 + sig_norm * 10) / 70.0)
    
    # Qualité du signal
    quality_score = pd.Series([0] * len(close), index=close.index)
    quality_score += np.where(abs(bullish_signals - bearish_signals) >= 4, 25, 0)
    quality_score += np.where(strong_volume, 25, 0)
    quality_score += np.where(volatility_ratio < 1.2, 25, 0)
    quality_score += np.where(
        (confirmed_trend == 1) & structure_bullish | (confirmed_trend == -1) & structure_bearish,
        25, 0
    )
    
    # Dernier signal
    last_trend = confirmed_trend.iloc[-1]
    last_strength = strength.iloc[-1]
    last_quality = quality_score.iloc[-1]
    
    quality_label = 'A+' if last_quality >= 75 else 'A' if last_quality >= 60 else 'B' if last_quality >= 45 else 'C' if last_quality >= 30 else 'D'
    
    trend_label = 'Bullish' if last_trend == 1 else 'Bearish' if last_trend == -1 else 'Neutral'
    
    return trend_label, last_strength, quality_label

def get_oanda_data(instrument, granularity, count, account_id, access_token):
    url = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {'granularity': granularity, 'count': count, 'price': 'M'}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        raw_data = response.json()
        candles_data = raw_data.get('candles', [])
        if not candles_data:
            return pd.DataFrame()
        records = [
            {
                'date': c['time'],
                'Open': float(c['mid']['o']),
                'High': float(c['mid']['h']),
                'Low': float(c['mid']['l']),
                'Close': float(c['mid']['c'])
            }
            for c in candles_data if c.get('complete', False)
        ]
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except requests.exceptions.RequestException as e:
        st.toast(f"Erreur réseau pour {instrument}: {e}", icon="🔥")
        return pd.DataFrame()
    except Exception as e:
        st.toast(f"Erreur de traitement des données OANDA pour {instrument}: {e}", icon="🔥")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_cached_oanda_data(instrument, granularity, count, account_id, access_token):
    """Version cachée de get_oanda_data pour éviter les appels API répétés"""
    return get_oanda_data(instrument, granularity, count, account_id, access_token)

def analyze_forex_pairs(account_id, access_token):
    results_internal = []
    
    # Mapping des timeframes
    timeframe_params_oanda = {
        '15m': {'granularity': 'M15', 'count': 300, 'tf_name': '15m'},
        '1H': {'granularity': 'H1', 'count': 300, 'tf_name': '1H'},
        '4H': {'granularity': 'H4', 'count': 300, 'tf_name': '4H'},
        'D': {'granularity': 'D', 'count': 300, 'tf_name': 'D'},
        'W': {'granularity': 'D', 'count': 900, 'tf_name': 'W'},
        'M': {'granularity': 'D', 'count': 2500, 'tf_name': 'M'}
    }
    
    total_pairs = len(FOREX_PAIRS_EXTENDED)
    progress_bar = st.progress(0, text=f"Analyse de 0 / {total_pairs} paires...")
    
    for i, pair_symbol in enumerate(FOREX_PAIRS_EXTENDED):
        oanda_instrument = f"{pair_symbol[:3]}_{pair_symbol[3:]}"
        
        try:
            data_sets = {}
            all_data_ok = True
            
            # Récupération des données pour chaque timeframe
            for tf_key, params in timeframe_params_oanda.items():
                df = get_cached_oanda_data(
                    oanda_instrument,
                    params['granularity'],
                    params['count'],
                    account_id,
                    access_token
                )
                
                # Resampling pour W et M
                if tf_key == 'W' and not df.empty:
                    df = df.resample('W-FRI').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last'
                    }).dropna()
                elif tf_key == 'M' and not df.empty:
                    df = df.resample('ME').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last'
                    }).dropna()
                
                if df.empty:
                    all_data_ok = False
                    break
                
                data_sets[tf_key] = df
            
            if not all_data_ok:
                continue
            
            # Calcul des tendances selon la logique Bluestar
            trend_15m, strength_15m, quality_15m = calc_professional_trend(data_sets['15m'], '15m')
            trend_1h, strength_1h, quality_1h = calc_professional_trend(data_sets['1H'], '1H')
            trend_4h, strength_4h, quality_4h = calc_professional_trend(data_sets['4H'], '4H')
            trend_d, strength_d, quality_d = calc_professional_trend(data_sets['D'], 'D')
            trend_w, strength_w, quality_w = calc_professional_trend(data_sets['W'], 'W')
            trend_m, strength_m, quality_m = calc_professional_trend(data_sets['M'], 'M')
            
            # Calcul du score d'alignement MTF (pondéré comme dans TradingView)
            trends = [trend_15m, trend_1h, trend_4h, trend_d, trend_w, trend_m]
            weights = [1, 1.5, 2, 2.5, 3, 3]  # Total = 13
            
            bullish_score = sum(w for t, w in zip(trends, weights) if t == 'Bullish')
            bearish_score = sum(w for t, w in zip(trends, weights) if t == 'Bearish')
            
            total_score = sum(weights)
            alignment_percent = max(bullish_score, bearish_score) / total_score * 100
            
            dominant_trend = 'Bullish' if bullish_score > bearish_score else 'Bearish' if bearish_score > bullish_score else 'Neutral'
            
            # Score pour le tri
            score = bullish_score - bearish_score
            
            results_internal.append({
                'Paire': f"{pair_symbol[:3]}/{pair_symbol[3:]}",
                '15m': trend_15m,
                '1H': trend_1h,
                '4H': trend_4h,
                'D': trend_d,
                'W': trend_w,
                'M': trend_m,
                'MTF': f"{dominant_trend} ({alignment_percent:.0f}%)",
                '_score_internal': score,
                '_alignment': alignment_percent
            })
            
        except Exception as e:
            st.error(f"Erreur inattendue lors de l'analyse de {oanda_instrument}: {e}")
        
        finally:
            progress_bar.progress(
                (i + 1) / total_pairs,
                text=f"Analyse de {i+1} / {total_pairs} paires..."
            )
            time.sleep(0.2)
    
    progress_bar.empty()
    
    if not results_internal:
        return pd.DataFrame()
    
    df_temp = pd.DataFrame(results_internal).sort_values(
        by='_score_internal',
        ascending=False
    )
    
    return df_temp[['Paire', '15m', '1H', '4H', 'D', 'W', 'M', 'MTF']]

# --- Fonctions pour le téléchargement ---

def create_image_report(df_report):
    report_title = "Classement des Paires Forex - Bluestar MTF Pro+"
    report_text = report_title + "\n" + ("-" * len(report_title)) + "\n"
    report_text += df_report.to_string(index=False) if not df_report.empty else "Aucune donnée."
    
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    text_bbox = temp_draw.multiline_textbbox((0, 0), report_text, font=font)
    padding = 20
    width = text_bbox[2] + 2 * padding
    height = text_bbox[3] + 2 * padding
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    draw.multiline_text((padding, padding), report_text, font=font, fill='black')
    output_buffer = BytesIO()
    img.save(output_buffer, format="PNG")
    return output_buffer.getvalue()

def create_pdf_report_simple(df_report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Classement Forex - Bluestar MTF Pro+', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 9)
    pdf.set_fill_color(220, 220, 220)
    col_width = pdf.w / (len(df_report.columns) + 0.5)
    for col_name in df_report.columns:
        pdf.cell(col_width, 8, col_name, 1, 0, 'C', 1)
    pdf.ln()
    pdf.set_font('Arial', '', 8)
    for _, row in df_report.iterrows():
        for col_name in df_report.columns:
            value = str(row[col_name])
            if value in TREND_COLORS_RGB:
                pdf.set_fill_color(*TREND_COLORS_RGB[value])
                pdf.set_text_color(255, 255, 255)
                fill = True
            else:
                pdf.set_fill_color(255, 255, 255)
                pdf.set_text_color(0, 0, 0)
                fill = False
            pdf.cell(col_width, 8, value, 1, 0, 'C', fill)
        pdf.ln()
    
    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()

# --- Fonction principale ---

def main():
    st.set_page_config(layout="wide")
    st.title("🌟 Classement Forex - Bluestar MTF Pro+ (via OANDA)")

    try:
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
    except (KeyError, FileNotFoundError):
        st.error("Erreur Critique: Les secrets OANDA ne sont pas configurés.")
        st.stop()

    if 'df_results' not in st.session_state:
        st.session_state.df_results = pd.DataFrame()
    if 'analysis_done_once' not in st.session_state:
        st.session_state.analysis_done_once = False

    if st.button("🚀 Analyser les Paires Forex"):
        with st.spinner("Analyse des paires avec la logique Bluestar MTF Pro+..."):
            st.session_state.df_results = analyze_forex_pairs(account_id, access_token)
            st.session_state.analysis_done_once = True

    if st.session_state.analysis_done_once:
        if not st.session_state.df_results.empty:
            # Filtres interactifs
            st.subheader("🎯 Filtres")
            col_filter1, col_filter2 = st.columns(2)
            
            with col_filter1:
                min_alignment = st.slider(
                    "Alignement MTF minimum (%)",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=5,
                    help="Afficher uniquement les paires avec un alignement supérieur à ce seuil"
                )
            
            with col_filter2:
                trend_filter = st.selectbox(
                    "Filtrer par tendance dominante",
                    options=["Tous", "Bullish uniquement", "Bearish uniquement"],
                    help="Afficher uniquement les paires avec une tendance spécifique"
                )
            
            # Application des filtres
            df_to_display = st.session_state.df_results.copy()
            df_to_display['_alignment_num'] = df_to_display['MTF'].str.extract(r'(\d+)%')[0].astype(float)
            df_to_display = df_to_display[df_to_display['_alignment_num'] >= min_alignment]
            
            if trend_filter == "Bullish uniquement":
                df_to_display = df_to_display[df_to_display['MTF'].str.contains('Bullish')]
            elif trend_filter == "Bearish uniquement":
                df_to_display = df_to_display[df_to_display['MTF'].str.contains('Bearish')]
            
            df_to_display = df_to_display.drop(columns=['_alignment_num'])
            
            st.subheader("📊 Classement des paires Forex")
            
            if df_to_display.empty:
                st.warning("Aucune paire ne correspond aux filtres sélectionnés.")
            else:
                st.info(f"**{len(df_to_display)}** paires affichées sur {len(st.session_state.df_results)}")
                
                def style_trends(val):
                    if val in TREND_COLORS_HEX:
                        return f'background-color: {TREND_COLORS_HEX[val]}; color: white; font-weight: bold;'
                    return ''
                
                styled_df = df_to_display.style.map(
                    style_trends,
                    subset=['15m', '1H', '4H', 'D', 'W', 'M']
                )
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    height=min((len(df_to_display) + 1) * 35 + 3, 600)
                )

            st.divider()
            st.subheader("📥 Télécharger le rapport")
            col1, col2, col3 = st.columns(3)
            now_str = datetime.now().strftime('%Y%m%d_%H%M')

            with col1:
                st.download_button(
                    label="📄 Télécharger en PDF",
                    data=create_pdf_report_simple(df_to_display),
                    file_name=f"classement_forex_bluestar_{now_str}.pdf",
                    mime='application/pdf',
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    label="🖼️ Télécharger en Image (PNG)",
                    data=create_image_report(df_to_display),
                    file_name=f"classement_forex_bluestar_{now_str}.png",
                    mime='image/png',
                    use_container_width=True
                )
            with col3:
                csv_data = df_to_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Télécharger en CSV",
                    data=csv_data,
                    file_name=f"classement_forex_bluestar_{now_str}.csv",
                    mime='text/csv',
                    use_container_width=True
                )

            st.subheader("ℹ️ À propos de la méthode Bluestar MTF Pro+")
            with st.expander("📖 Voir les détails de la logique de calcul"):
                st.markdown("""
                **Logique de calcul professionnelle :**
                - **ZLEMA (Zero-Lag EMA)** : Indicateur principal avec lag réduit
                - **Bandes de volatilité adaptatives** : Basées sur l'ATR avec ratio dynamique
                - **Système de scoring multicritères** :
                  - Position par rapport au ZLEMA (2 points)
                  - Structure de marché (1 point)
                  - Momentum RSI/MACD (1 point)
                  - Confirmation par volume et chandeliers (1 point)
                  - Cassure de bandes (2 points)
                - **Confirmation sur 3 barres** : Filtre les faux signaux
                - **Score d'alignement MTF pondéré** : 15m(1) → 1H(1.5) → 4H(2) → D(2.5) → W(3) → M(3)
                
                **Timeframes analysés :** 15m, 1H, 4H, Daily, Weekly, Monthly
                
                **Légende des couleurs :**
                - 🟢 **Bullish** (vert) : Tendance haussière confirmée
                - 🔴 **Bearish** (rouge adouci) : Tendance baissière confirmée
                - ⚪ **Neutral** (gris) : Pas de tendance claire
                """)
        else:
            st.warning("L'analyse n'a produit aucun résultat.")
    else:
        st.info("Cliquez sur 'Analyser' pour lancer l'analyse complète avec la méthode Bluestar MTF Pro+")

    st.markdown("---")
    st.caption("Données via OANDA v20 REST API | Logique Bluestar MTF Pro+ adaptée")

if __name__ == "__main__":
    main()

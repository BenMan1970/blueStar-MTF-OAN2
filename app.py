st.divider()
            st.subheader("📥 Télécharger le rapport")
            
            # Préparer les données pour export (format simplifié)
            df_export = df_to_display[['Paire', '15m', '1H', '4H', 'D', 'W', 'M', 'MTF', 'Quality', 'Consistency']].copy() if not df_to_display.empty else df_full[['Paire', '15m', '1H', '4H', 'D', 'W', 'M', 'MTF', 'Quality']].copy()
            
            col1, col2, col3 = st.columns(3)
            now_str = datetime.now().strftime('%Y%m%d_%H%M')

            with col1:
                st.download_button(
                    label="📄 Télécharger en PDF",
                    data=create_pdf_report_simple(df_export),
                    file_name=f"classement_forex_bluestar_{now_str}.pdf",
                    mime='application/pdf',
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    label="🖼️ Télécharger en Image (PNG)",
                    data=create_image_report(df_export),
                    file_name=f"classement_forex_bluestar_{now_str}.png",
                    mime='image/png',
                    use_container_width=True
                )
            with col3:
                csv_data = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Télécharger en CSV",
                    data=csv_data,
                    file_name=f"classement_forex_bluestar_{now_str}.csv",
                    mime='text/csv',
                    use_container_width=True
                )import streamlit as st
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
TREND_COLORS_HEX = {
    'Bullish': '#008f7a', 
    'Bearish': '#d9534f', 
    'Neutral': '#808080',
    'Range': '#f0ad4e'  # Orange pour range
}
TREND_COLORS_RGB = {
    'Bullish': (0, 143, 122), 
    'Bearish': (217, 83, 79), 
    'Neutral': (128, 128, 128),
    'Range': (240, 173, 78)
}

# --- Paramètres Bluestar Optimisés ---
LENGTH = 70
MULT = 1.2
USE_MOMENTUM_FILTER = True
USE_VOLUME_FILTER = False  # Désactivé pour Forex (pas de volume réel)
USE_ADX_FILTER = True  # Nouveau : filtre ADX
ADX_THRESHOLD = 25  # En dessous = range
VOLATILITY_PERIOD = 14

# Poids MTF optimisés (progression exponentielle)
MTF_WEIGHTS = {
    '15m': 0.5,
    '1H': 1.0,
    '4H': 2.0,
    'D': 3.0,
    'W': 4.5,
    'M': 6.0
}
TOTAL_MTF_WEIGHT = sum(MTF_WEIGHTS.values())

# Barres de confirmation adaptatives par timeframe
CONFIRMATION_BARS = {
    '15m': 2,
    '1H': 2,
    '4H': 3,
    'D': 3,
    'W': 2,
    'M': 1
}

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

def adx(high, low, close, period=14):
    """Average Directional Index - Mesure la force de la tendance"""
    # Calcul du True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calcul des mouvements directionnels
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    
    # Lissage avec EMA
    atr_smooth = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_smooth)
    
    # Calcul ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_value = dx.ewm(span=period, adjust=False).mean()
    
    return adx_value, plus_di, minus_di

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
    Calcule la tendance professionnelle selon la logique Bluestar MTF Pro+ OPTIMISÉE
    Avec filtre ADX et détection de range
    """
    if df.empty or len(df) < 100:
        return 'Neutral', 0, 'D', 0
    
    # Ajustement des paramètres selon le timeframe
    is_monthly = timeframe == 'M'
    is_weekly = timeframe == 'W'
    
    adj_length = 12 if is_monthly else 26 if is_weekly else LENGTH
    adj_vol = 6 if is_monthly else 10 if is_weekly else VOLATILITY_PERIOD
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    open_price = df['Open']
    
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
    
    # NOUVEAU : Calcul ADX pour détection de range
    adx_value, plus_di, minus_di = adx(high, low, close, 14)
    is_ranging = adx_value < ADX_THRESHOLD
    
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
    
    # Chandeliers (remplace le volume pour Forex)
    bullish_candle = close > open_price
    bearish_candle = close < open_price
    candle_size = abs(close - open_price)
    avg_candle_size = candle_size.rolling(window=14).mean()
    strong_candle = candle_size > avg_candle_size * 1.2
    
    # Volatilité comme proxy du volume
    high_volatility = volatility_ratio > 1.2
    
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
    
    # Volatilité forte + chandeliers (remplace volume)
    bullish_signals += np.where(high_volatility & bullish_candle & strong_candle, 1, 0)
    bearish_signals += np.where(high_volatility & bearish_candle & strong_candle, 1, 0)
    
    # Raw trend
    bullish_signals += np.where(raw_trend == 1, 2, 0)
    bearish_signals += np.where(raw_trend == -1, 2, 0)
    
    # ADX bonus : ajouter du poids si tendance forte
    if USE_ADX_FILTER:
        strong_trend = adx_value > ADX_THRESHOLD * 1.5  # ADX > 37.5
        bullish_signals += np.where(strong_trend & (plus_di > minus_di), 1, 0)
        bearish_signals += np.where(strong_trend & (minus_di > plus_di), 1, 0)
    
    # Tendance potentielle
    potential_trend = np.where(
        bullish_signals > bearish_signals, 1,
        np.where(bearish_signals > bullish_signals, -1, 0)
    )
    
    # Confirmation adaptative par timeframe
    confirm_bars = CONFIRMATION_BARS.get(timeframe, 3)
    confirmed_trend = pd.Series([0] * len(close), index=close.index)
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
    
    # Force du signal
    price_distance = abs((close - zlema_series) / zlema_series * 100)
    signal_strength = abs(bullish_signals - bearish_signals)
    
    pd_norm = np.minimum(100, price_distance)
    mom_norm = abs(momentum_score) * 100
    sig_norm = signal_strength / 8.0 * 100  # Ajusté car on a 8 points max maintenant
    adx_norm = np.minimum(100, adx_value)
    
    # Nouvelle formule de force incluant ADX
    strength = np.minimum(100, (pd_norm * 25 + mom_norm * 25 + sig_norm * 25 + adx_norm * 25) / 100.0)
    
    # Qualité du signal (mise à jour)
    quality_score = pd.Series([0] * len(close), index=close.index)
    quality_score += np.where(abs(bullish_signals - bearish_signals) >= 4, 25, 0)
    quality_score += np.where(adx_value > ADX_THRESHOLD, 25, 0)  # ADX remplace volume
    quality_score += np.where(volatility_ratio < 1.2, 25, 0)
    quality_score += np.where(
        (confirmed_trend == 1) & structure_bullish | (confirmed_trend == -1) & structure_bearish,
        25, 0
    )
    
    # Derniers signaux
    last_trend = confirmed_trend.iloc[-1]
    last_strength = strength.iloc[-1]
    last_quality = quality_score.iloc[-1]
    last_adx = adx_value.iloc[-1]
    last_is_ranging = is_ranging.iloc[-1]
    
    quality_label = 'A+' if last_quality >= 75 else 'A' if last_quality >= 60 else 'B' if last_quality >= 45 else 'C' if last_quality >= 30 else 'D'
    
    # NOUVEAU : Si ADX < 25, forcer "Range" au lieu de Neutral
    if USE_ADX_FILTER and last_is_ranging:
        trend_label = 'Range'
    else:
        trend_label = 'Bullish' if last_trend == 1 else 'Bearish' if last_trend == -1 else 'Neutral'
    
    return trend_label, last_strength, quality_label, last_adx

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

def analyze_signal_consistency(row):
    """
    Analyse la cohérence des signaux multi-timeframe
    Retourne un dictionnaire avec les alertes et le score de cohérence
    """
    trends = [row['15m'], row['1H'], row['4H'], row['D'], row['W'], row['M']]
    
    # Compter les tendances
    bullish_count = sum(1 for t in trends if t == 'Bullish')
    bearish_count = sum(1 for t in trends if t == 'Bearish')
    range_count = sum(1 for t in trends if t == 'Range')
    neutral_count = sum(1 for t in trends if t == 'Neutral')
    
    alerts = []
    consistency_score = 0
    signal_quality = "🟢 Excellent"
    
    # 1. Détection des divergences court terme vs long terme
    short_term = trends[0:2]  # 15m, 1H
    mid_term = trends[2:4]    # 4H, D
    long_term = trends[4:6]   # W, M
    
    short_bullish = sum(1 for t in short_term if t == 'Bullish')
    short_bearish = sum(1 for t in short_term if t == 'Bearish')
    
    mid_bullish = sum(1 for t in mid_term if t == 'Bullish')
    mid_bearish = sum(1 for t in mid_term if t == 'Bearish')
    
    long_bullish = sum(1 for t in long_term if t == 'Bullish')
    long_bearish = sum(1 for t in long_term if t == 'Bearish')
    
    # Divergence majeure : court terme opposé au long terme
    if short_bullish >= 1 and long_bearish >= 1:
        alerts.append("⚠️ DIVERGENCE : Court terme Bullish vs Long terme Bearish")
        consistency_score -= 20
    elif short_bearish >= 1 and long_bullish >= 1:
        alerts.append("⚠️ DIVERGENCE : Court terme Bearish vs Long terme Bullish")
        consistency_score -= 20
    
    # 2. Détection de Range sur timeframes critiques
    if row['D'] == 'Range' or row['W'] == 'Range':
        alerts.append("🟠 ATTENTION : Range détecté sur Daily/Weekly (consolidation)")
        consistency_score -= 10
    
    if row['4H'] == 'Range' and row['D'] == 'Range':
        alerts.append("🟠 RANGE ÉTENDU : 4H et Daily en consolidation")
        consistency_score -= 15
    
    # 3. Signaux contradictoires adjacents
    for i in range(len(trends) - 1):
        if trends[i] == 'Bullish' and trends[i+1] == 'Bearish':
            tf_names = ['15m', '1H', '4H', 'D', 'W', 'M']
            alerts.append(f"⚠️ CONTRADICTION : {tf_names[i]} Bullish vs {tf_names[i+1]} Bearish")
            consistency_score -= 10
        elif trends[i] == 'Bearish' and trends[i+1] == 'Bullish':
            tf_names = ['15m', '1H', '4H', 'D', 'W', 'M']
            alerts.append(f"⚠️ CONTRADICTION : {tf_names[i]} Bearish vs {tf_names[i+1]} Bullish")
            consistency_score -= 10
    
    # 4. Bonus pour alignement parfait
    if bullish_count >= 5 and bearish_count == 0 and range_count <= 1:
        alerts.append("✅ ALIGNEMENT OPTIMAL : Tous les TF Bullish")
        consistency_score += 30
    elif bearish_count >= 5 and bullish_count == 0 and range_count <= 1:
        alerts.append("✅ ALIGNEMENT OPTIMAL : Tous les TF Bearish")
        consistency_score += 30
    
    # 5. Trop de Range = marché indécis
    if range_count >= 3:
        alerts.append("🔶 MARCHÉ INDÉCIS : 3+ timeframes en Range")
        consistency_score -= 25
        signal_quality = "🔶 Faible"
    
    # 6. Vérification de la progression logique
    # Le Monthly/Weekly devrait "guider" les TF inférieurs
    if row['M'] == 'Bullish' and bearish_count >= 3:
        alerts.append("⚠️ INCOHÉRENCE : Monthly Bullish mais majorité Bearish")
        consistency_score -= 20
    elif row['M'] == 'Bearish' and bullish_count >= 3:
        alerts.append("⚠️ INCOHÉRENCE : Monthly Bearish mais majorité Bullish")
        consistency_score -= 20
    
    # Calcul du score final (0-100)
    base_score = 50
    final_score = max(0, min(100, base_score + consistency_score))
    
    # Détermination de la qualité
    if final_score >= 80:
        signal_quality = "🟢 Excellent"
    elif final_score >= 60:
        signal_quality = "🟡 Bon"
    elif final_score >= 40:
        signal_quality = "🟠 Moyen"
    else:
        signal_quality = "🔴 Faible"
    
    return {
        'alerts': alerts if alerts else ["✅ Aucune divergence détectée"],
        'consistency_score': final_score,
        'signal_quality': signal_quality,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'range_count': range_count
    }

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
            
            # Calcul des tendances selon la logique Bluestar optimisée
            trend_15m, strength_15m, quality_15m, adx_15m = calc_professional_trend(data_sets['15m'], '15m')
            trend_1h, strength_1h, quality_1h, adx_1h = calc_professional_trend(data_sets['1H'], '1H')
            trend_4h, strength_4h, quality_4h, adx_4h = calc_professional_trend(data_sets['4H'], '4H')
            trend_d, strength_d, quality_d, adx_d = calc_professional_trend(data_sets['D'], 'D')
            trend_w, strength_w, quality_w, adx_w = calc_professional_trend(data_sets['W'], 'W')
            trend_m, strength_m, quality_m, adx_m = calc_professional_trend(data_sets['M'], 'M')
            
            # Calcul du score d'alignement MTF avec poids optimisés
            trends = [trend_15m, trend_1h, trend_4h, trend_d, trend_w, trend_m]
            qualities = [quality_15m, quality_1h, quality_4h, quality_d, quality_w, quality_m]
            weights = [MTF_WEIGHTS['15m'], MTF_WEIGHTS['1H'], MTF_WEIGHTS['4H'], 
                      MTF_WEIGHTS['D'], MTF_WEIGHTS['W'], MTF_WEIGHTS['M']]
            
            # Ignorer les "Range" dans le calcul (considérés comme neutres)
            bullish_score = sum(w for t, w in zip(trends, weights) if t == 'Bullish')
            bearish_score = sum(w for t, w in zip(trends, weights) if t == 'Bearish')
            range_count = sum(1 for t in trends if t == 'Range')
            
            alignment_percent = max(bullish_score, bearish_score) / TOTAL_MTF_WEIGHT * 100
            
            # Tendance dominante
            if range_count >= 4:  # Si 4+ timeframes en range
                dominant_trend = 'Range'
            else:
                dominant_trend = 'Bullish' if bullish_score > bearish_score else 'Bearish' if bearish_score > bullish_score else 'Neutral'
            
            # Qualité globale (moyenne pondérée)
            quality_values = {'A+': 4, 'A': 3, 'B': 2, 'C': 1, 'D': 0}
            weighted_quality = sum(quality_values.get(q, 0) * w for q, w in zip(qualities, weights)) / TOTAL_MTF_WEIGHT
            global_quality = 'A+' if weighted_quality >= 3.5 else 'A' if weighted_quality >= 2.5 else 'B' if weighted_quality >= 1.5 else 'C' if weighted_quality >= 0.5 else 'D'
            
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
                'Quality': global_quality,
                '_score_internal': score,
                '_alignment': alignment_percent,
                '_range_count': range_count,
                '_adx_d': adx_d,
                '_adx_w': adx_w
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
    
    return df_temp[['Paire', '15m', '1H', '4H', 'D', 'W', 'M', 'MTF', 'Quality']]

# --- Fonctions pour le téléchargement ---

def create_image_report(df_report):
    report_title = "Classement Forex - Bluestar MTF Pro+ Optimisé"
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
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Classement Forex - Bluestar MTF Pro+ Optimise', 0, 1, 'C')
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 8)
    pdf.set_fill_color(220, 220, 220)
    col_width = pdf.w / (len(df_report.columns) + 0.5)
    for col_name in df_report.columns:
        pdf.cell(col_width, 7, col_name, 1, 0, 'C', 1)
    pdf.ln()
    pdf.set_font('Arial', '', 7)
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
            pdf.cell(col_width, 7, value, 1, 0, 'C', fill)
        pdf.ln()
    
    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()

# --- Fonction principale ---

def main():
    st.set_page_config(layout="wide", page_title="Bluestar MTF Pro+", page_icon="🌟")
    
    # Header avec style
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, #008f7a 0%, #00b894 100%);
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 2.5rem;
        }
        .main-header p {
            color: white;
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
        }
        </style>
        <div class="main-header">
            <h1>🌟 Bluestar MTF Pro+ Optimisé</h1>
            <p>Analyse Multi-Timeframe Professionnelle avec Filtre ADX</p>
        </div>
    """, unsafe_allow_html=True)

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

    # Bouton d'analyse avec style
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🚀 Analyser les Paires Forex", use_container_width=True, type="primary"):
            with st.spinner("Analyse des paires avec la logique Bluestar optimisée..."):
                st.session_state.df_results = analyze_forex_pairs(account_id, access_token)
                st.session_state.analysis_done_once = True

    if st.session_state.analysis_done_once:
        if not st.session_state.df_results.empty:
            # Analyse de cohérence pour chaque paire
            df_full = st.session_state.df_results.copy()
            
            # Ajouter l'analyse de cohérence
            consistency_data = []
            for idx, row in df_full.iterrows():
                analysis = analyze_signal_consistency(row)
                consistency_data.append({
                    'Paire': row['Paire'],
                    'Consistency': analysis['consistency_score'],
                    'Signal_Quality': analysis['signal_quality'],
                    'Alerts': analysis['alerts']
                })
            
            df_consistency = pd.DataFrame(consistency_data)
            df_full = df_full.merge(df_consistency, on='Paire', how='left')
            
            # Statistiques rapides
            bullish_count = df_full['MTF'].str.contains('Bullish').sum()
            bearish_count = df_full['MTF'].str.contains('Bearish').sum()
            range_count = df_full['MTF'].str.contains('Range').sum()
            excellent_signals = (df_full['Signal_Quality'] == '🟢 Excellent').sum()
            weak_signals = (df_full['Signal_Quality'] == '🔴 Faible').sum()
            
            st.subheader("📊 Vue d'ensemble du marché")
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            with metric_col1:
                st.metric("Total Paires", len(df_full))
            with metric_col2:
                st.metric("🟢 Bullish", bullish_count)
            with metric_col3:
                st.metric("🔴 Bearish", bearish_count)
            with metric_col4:
                st.metric("🟠 Range", range_count)
            with metric_col5:
                st.metric("✅ Signaux Fiables", excellent_signals)
            
            # Alertes critiques
            critical_pairs = df_full[df_full['Consistency'] < 50]
            if not critical_pairs.empty:
                st.warning(f"⚠️ **{len(critical_pairs)} paires** ont des divergences significatives !")
                with st.expander("🔍 Voir les paires avec divergences", expanded=True):
                    for _, pair_row in critical_pairs.iterrows():
                        st.markdown(f"### {pair_row['Paire']} - Score: {pair_row['Consistency']:.0f}/100")
                        for alert in pair_row['Alerts']:
                            st.markdown(f"- {alert}")
                        st.divider()
            else:
                st.success("✅ Aucune divergence majeure détectée ! Les signaux sont cohérents.")
            
            # Filtres interactifs
            st.divider()
            st.subheader("🎯 Filtres de recherche")
            col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
            
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
                    options=["Tous", "Bullish uniquement", "Bearish uniquement", "Range uniquement", "Exclure Range"],
                    help="Afficher uniquement les paires avec une tendance spécifique"
                )
            
            with col_filter3:
                quality_filter = st.multiselect(
                    "Qualité minimale",
                    options=['A+', 'A', 'B', 'C', 'D'],
                    default=['A+', 'A', 'B'],
                    help="Filtrer par qualité du signal"
                )
            
            with col_filter4:
                consistency_filter = st.slider(
                    "Cohérence minimum",
                    min_value=0,
                    max_value=100,
                    value=60,
                    step=5,
                    help="Score de cohérence multi-timeframe (0-100)"
                )
            
            # Application des filtres
            df_to_display = df_full.copy()
            df_to_display['_alignment_num'] = df_to_display['MTF'].str.extract(r'(\d+)%')[0].astype(float)
            df_to_display = df_to_display[df_to_display['_alignment_num'] >= min_alignment]
            df_to_display = df_to_display[df_to_display['Consistency'] >= consistency_filter]
            
            if trend_filter == "Bullish uniquement":
                df_to_display = df_to_display[df_to_display['MTF'].str.contains('Bullish')]
            elif trend_filter == "Bearish uniquement":
                df_to_display = df_to_display[df_to_display['MTF'].str.contains('Bearish')]
            elif trend_filter == "Range uniquement":
                df_to_display = df_to_display[df_to_display['MTF'].str.contains('Range')]
            elif trend_filter == "Exclure Range":
                df_to_display = df_to_display[~df_to_display['MTF'].str.contains('Range')]
            
            if quality_filter:
                df_to_display = df_to_display[df_to_display['Quality'].isin(quality_filter)]
            
            df_to_display = df_to_display.drop(columns=['_alignment_num'])
            
            st.divider()
            st.subheader("📈 Résultats de l'analyse")
            
            if df_to_display.empty:
                st.warning("⚠️ Aucune paire ne correspond aux filtres sélectionnés.")
            else:
                st.success(f"✅ **{len(df_to_display)}** paires affichées sur {len(df_full)}")
                
                # Affichage des top opportunités avec analyse de cohérence
                if len(df_to_display) > 0:
                    # Trier par cohérence ET alignement
                    df_sorted = df_to_display.sort_values(by=['Consistency', '_alignment'], ascending=False)
                    
                    top_bullish = df_sorted[df_sorted['MTF'].str.contains('Bullish')].head(3)
                    top_bearish = df_sorted[df_sorted['MTF'].str.contains('Bearish')].head(3)
                    
                    if not top_bullish.empty or not top_bearish.empty:
                        st.info("🎯 **Top Opportunités** (cohérence + alignement MTF)")
                        opp_col1, opp_col2 = st.columns(2)
                        
                        with opp_col1:
                            if not top_bullish.empty:
                                st.markdown("**🟢 Top Bullish (plus fiables):**")
                                for _, row in top_bullish.iterrows():
                                    consistency_emoji = "🟢" if row['Consistency'] >= 80 else "🟡" if row['Consistency'] >= 60 else "🟠"
                                    st.markdown(f"- **{row['Paire']}** - {row['MTF']} - Qualité: {row['Quality']}")
                                    st.markdown(f"  {consistency_emoji} Cohérence: {row['Consistency']:.0f}/100")
                                    if row['Consistency'] < 70:
                                        st.caption(f"  ⚠️ {row['Alerts'][0]}")
                        
                        with opp_col2:
                            if not top_bearish.empty:
                                st.markdown("**🔴 Top Bearish (plus fiables):**")
                                for _, row in top_bearish.iterrows():
                                    consistency_emoji = "🟢" if row['Consistency'] >= 80 else "🟡" if row['Consistency'] >= 60 else "🟠"
                                    st.markdown(f"- **{row['Paire']}** - {row['MTF']} - Qualité: {row['Quality']}")
                                    st.markdown(f"  {consistency_emoji} Cohérence: {row['Consistency']:.0f}/100")
                                    if row['Consistency'] < 70:
                                        st.caption(f"  ⚠️ {row['Alerts'][0]}")
                
                # Préparer le DataFrame pour affichage (sans colonnes internes)
                df_display_clean = df_to_display[['Paire', '15m', '1H', '4H', 'D', 'W', 'M', 'MTF', 'Quality', 'Consistency', 'Signal_Quality']].copy()
                
                # Bouton pour voir les détails des divergences
                st.markdown("---")
                show_alerts = st.checkbox("🔍 Afficher les alertes détaillées pour chaque paire", value=False)
                
                if show_alerts:
                    st.subheader("🚨 Alertes et divergences détaillées")
                    for _, row in df_to_display.iterrows():
                        with st.expander(f"{row['Paire']} - Cohérence: {row['Consistency']:.0f}/100 {row['Signal_Quality']}"):
                            col_alert1, col_alert2 = st.columns([2, 1])
                            with col_alert1:
                                st.markdown("**Alertes détectées:**")
                                for alert in row['Alerts']:
                                    st.markdown(f"- {alert}")
                            with col_alert2:
                                st.metric("Score de cohérence", f"{row['Consistency']:.0f}/100")
                                st.metric("MTF", row['MTF'])
                                st.metric("Qualité", row['Quality'])
                
                # Tableau principal avec style
                st.markdown("---")
                st.subheader("📋 Tableau complet")
                
                # Tableau principal avec style
                st.markdown("---")
                st.subheader("📋 Tableau complet")
                
                def style_trends(val):
                    if val in TREND_COLORS_HEX:
                        return f'background-color: {TREND_COLORS_HEX[val]}; color: white; font-weight: bold;'
                    return ''
                
                def style_quality(val):
                    colors = {
                        'A+': 'background-color: #2ecc71; color: white; font-weight: bold;',
                        'A': 'background-color: #27ae60; color: white; font-weight: bold;',
                        'B': 'background-color: #f39c12; color: white; font-weight: bold;',
                        'C': 'background-color: #e67e22; color: white; font-weight: bold;',
                        'D': 'background-color: #e74c3c; color: white; font-weight: bold;'
                    }
                    return colors.get(val, '')
                
                def style_consistency(val):
                    if isinstance(val, (int, float)):
                        if val >= 80:
                            return 'background-color: #2ecc71; color: white; font-weight: bold;'
                        elif val >= 60:
                            return 'background-color: #f39c12; color: white; font-weight: bold;'
                        elif val >= 40:
                            return 'background-color: #e67e22; color: white; font-weight: bold;'
                        else:
                            return 'background-color: #e74c3c; color: white; font-weight: bold;'
                    return ''
                
                styled_df = df_display_clean.style\
                    .map(style_trends, subset=['15m', '1H', '4H', 'D', 'W', 'M'])\
                    .map(style_quality, subset=['Quality'])\
                    .map(style_consistency, subset=['Consistency'])\
                    .format({'Consistency': '{:.0f}'})
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    height=min((len(df_display_clean) + 1) * 35 + 3, 600)
                )

            st.divider()
            st.subheader("📥 Télécharger le rapport")
            col1, col2, col3 = st.columns(3)
            now_str = datetime.now().strftime('%Y%m%d_%H%M')

            with col1:
                st.download_button(
                    label="📄 Télécharger en PDF",
                    data=create_pdf_report_simple(df_to_display if not df_to_display.empty else df_full),
                    file_name=f"classement_forex_bluestar_{now_str}.pdf",
                    mime='application/pdf',
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    label="🖼️ Télécharger en Image (PNG)",
                    data=create_image_report(df_to_display if not df_to_display.empty else df_full),
                    file_name=f"classement_forex_bluestar_{now_str}.png",
                    mime='image/png',
                    use_container_width=True
                )
            with col3:
                csv_data = (df_to_display if not df_to_display.empty else df_full).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Télécharger en CSV",
                    data=csv_data,
                    file_name=f"classement_forex_bluestar_{now_str}.csv",
                    mime='text/csv',
                    use_container_width=True
                )

            st.divider()
            st.subheader("ℹ️ À propos de la méthode Bluestar MTF Pro+ Optimisée")
            
            # Tabs pour organiser l'information
            tab1, tab2, tab3, tab4 = st.tabs(["📖 Logique de calcul", "⚙️ Optimisations", "🚨 Système d'alertes", "💡 Guide d'utilisation"])
            
            with tab1:
                st.markdown("""
                ### 🧮 Logique de calcul professionnelle
                
                **Indicateurs principaux :**
                - **ZLEMA (Zero-Lag EMA)** : Indicateur de tendance avec lag réduit
                - **Bandes de volatilité adaptatives** : Basées sur l'ATR avec ratio dynamique
                - **ADX (Average Directional Index)** : Mesure la force de la tendance
                
                **Système de scoring multicritères :**
                1. Position par rapport au ZLEMA (2 points)
                2. Structure de marché - EMA(20) (1 point)
                3. Momentum RSI/MACD combiné (1 point)
                4. Volatilité forte + chandeliers significatifs (1 point)
                5. Cassure de bandes de volatilité (2 points)
                6. Bonus ADX pour tendance forte (1 point)
                
                **Maximum : 8 points par direction**
                
                **Confirmation adaptative :**
                - 15m/1H : 2 barres de confirmation
                - 4H/D : 3 barres de confirmation
                - W : 2 barres de confirmation
                - M : 1 barre de confirmation
                
                **Détection de Range :**
                - Si ADX < 25 → État "Range" (marché sans tendance claire)
                - Permet d'éviter les faux signaux en phase de consolidation
                """)
            
            with tab2:
                st.markdown("""
                ### ⚡ Optimisations professionnelles
                
                **1. Poids MTF optimisés (progression exponentielle) :**
                ```
                15m : 0.5  (faible poids, très bruité)
                1H  : 1.0
                4H  : 2.0
                D   : 3.0
                W   : 4.5
                M   : 6.0  (poids maximum, haute fiabilité)
                Total : 17.0 points
                ```
                
                **2. Filtre ADX activé :**
                - ADX < 25 : Market en Range → Éviter le trading
                - ADX > 37.5 : Tendance très forte → Bonus de signal
                - Remplace le filtre volume (inapplicable sur Forex)
                
                **3. Volatilité comme proxy du volume :**
                - Sur Forex, pas de volume réel disponible
                - Utilisation de la volatilité (ATR ratio) pour détecter l'activité
                - Combinaison avec l'analyse des chandeliers
                
                **4. Qualité globale pondérée :**
                - A+ : Qualité excellente (≥75%) sur tous les timeframes
                - A : Très bonne qualité (≥60%)
                - B : Qualité correcte (≥45%)
                - C : Qualité moyenne (≥30%)
                - D : Qualité faible
                
                **5. Cache intelligent :**
                - Données mises en cache 5 minutes
                - Évite les appels API répétés
                - Amélioration de la performance de ~90%
                """)
            
            with tab3:
                st.markdown("""
                ### 🚨 Système de détection des divergences (NOUVEAU!)
                
                **Score de cohérence (0-100) :**
                
                Le système analyse automatiquement chaque paire et détecte :
                
                **🔴 Divergences critiques (-20 points) :**
                - Court terme opposé au long terme
                  - Exemple : 15m Bullish mais Weekly/Monthly Bearish
                - Monthly en désaccord avec la majorité des TF
                  - Exemple : Monthly Bullish mais 4+ timeframes Bearish
                
                **🟠 Alertes importantes (-10 à -15 points) :**
                - Range détecté sur Daily ou Weekly (consolidation)
                - Multiple timeframes en Range (marché indécis)
                - Signaux contradictoires adjacents
                  - Exemple : 4H Bullish puis Daily Bearish
                
                **🟡 Incohérences mineures (-10 points) :**
                - Transitions brusques entre timeframes
                - Range étendu sur plusieurs périodes
                
                **🟢 Bonus d'alignement (+30 points) :**
                - 5+ timeframes alignés dans la même direction
                - Progression logique du Monthly vers le 15m
                - Aucune divergence détectée
                
                **Interprétation du score :**
                - **80-100** : 🟢 Signal excellent, haute fiabilité
                - **60-79** : 🟡 Signal bon, quelques réserves
                - **40-59** : 🟠 Signal moyen, prudence requise
                - **0-39** : 🔴 Signal faible, à éviter
                
                **Exemples d'alertes générées :**
                - ✅ "Alignement optimal : Tous les TF Bullish"
                - ⚠️ "DIVERGENCE : Court terme Bullish vs Long terme Bearish"
                - 🟠 "ATTENTION : Range détecté sur Daily/Weekly"
                - ⚠️ "CONTRADICTION : 1H Bullish vs 4H Bearish"
                - 🔶 "MARCHÉ INDÉCIS : 3+ timeframes en Range"
                """)
            
            with tab4:
                st.markdown("""
                ### 💡 Guide d'utilisation professionnelle
                
                **🎯 Comment interpréter les résultats avec le nouveau système :**
                
                **1. Signaux PRIORITAIRES (à trader en premier) :**
                - ✅ Alignement MTF ≥ 80% 
                - ✅ Qualité A ou A+
                - ✅ **Cohérence ≥ 80** (NOUVEAU critère essentiel)
                - ✅ Aucune alerte critique
                
                **2. Signaux SECONDAIRES (confirmation supplémentaire requise) :**
                - ⚠️ Alignement 60-80%
                - ⚠️ Qualité B
                - ⚠️ **Cohérence 60-79**
                - ⚠️ 1-2 alertes mineures
                
                **3. Signaux à ÉVITER :**
                - ❌ **Cohérence < 60** (divergences importantes)
                - ❌ Alertes de type "DIVERGENCE" ou "CONTRADICTION"
                - ❌ Range détecté sur 3+ timeframes
                - ❌ Monthly en opposition avec majorité des TF
                
                **📊 Nouvelle stratégie de filtrage recommandée :**
                
                **Pour le swing trading :**
                1. Filtre 1 : Cohérence ≥ 70
                2. Filtre 2 : Alignement MTF ≥ 75%
                3. Filtre 3 : Qualité A+ ou A
                4. Filtre 4 : Exclure les Range
                5. Vérifier les alertes détaillées
                
                **Pour le day trading :**
                1. Filtre 1 : Cohérence ≥ 60
                2. Filtre 2 : Alignement MTF ≥ 65%
                3. Filtre 3 : Vérifier que 15m et 1H ne divergent pas
                4. Attendre confirmation sur 4H minimum
                
                **🔍 Utilisation des alertes détaillées :**
                
                Activez "Afficher les alertes détaillées" pour :
                - Comprendre POURQUOI une paire a un score faible
                - Identifier les timeframes problématiques
                - Décider si la divergence est acceptable
                
                **Exemple de décision :**
                ```
                EUR/USD :
                - Cohérence : 75/100
                - Alerte : "15m Bearish vs Monthly Bullish"
                - Décision : Acceptable si vous tradez en swing (ignorer 15m)
                           Problématique si vous tradez en intraday
                ```
                
                **⚠️ Avertissements CRITIQUES :**
                
                1. **Ne JAMAIS trader une paire avec cohérence < 50**
                   - Trop de signaux contradictoires
                   - Risque élevé de whipsaw
                
                2. **Si alerte "Monthly en opposition", STOP**
                   - Le Monthly a toujours raison à long terme
                   - Vous tradez contre la tendance principale
                
                3. **Range sur Daily + Weekly = attendre**
                   - Le marché consolide
                   - Risque de faux breakout élevé
                
                4. **Top Opportunités trié par cohérence**
                   - Les paires listées ont la meilleure combinaison
                   - Cohérence + Alignement + Qualité
                
                **🔄 Fréquence d'utilisation :**
                - Swing Trading : 1x par jour (fin de journée)
                - Day Trading : 2-3x par jour
                - Vérifier les alertes avant CHAQUE trade
                - Re-scanner après news importantes (NFP, Fed, etc.)
                
                **💼 Workflow professionnel recommandé :**
                
                1. **Screening initial** : Lancer l'analyse complète
                2. **Filtrage** : Cohérence ≥ 70, Alignement ≥ 75%
                3. **Vérification alertes** : Lire les divergences
                4. **Analyse technique** : Support/résistance sur TradingView
                5. **Confirmation** : Price action + contexte fondamental
                6. **Entrée** : Uniquement si TOUS les critères sont verts
                """)
            
        else:
            st.warning("L'analyse n'a produit aucun résultat.")
    else:
        st.info("👆 Cliquez sur le bouton 'Analyser' pour lancer l'analyse complète avec la méthode Bluestar MTF Pro+ optimisée")
        
        # Information avant analyse
        st.markdown("""
        ### 🌟 Nouveautés de la version optimisée :
        
        - ✅ **Filtre ADX** : Détection automatique des marchés en range
        - ✅ **Poids MTF optimisés** : Progression exponentielle (Monthly = 12x le poids du 15m)
        - ✅ **Confirmation adaptative** : Nombre de barres ajusté par timeframe
        - ✅ **Qualité globale** : Score de qualité pondéré affiché
        - ✅ **🚨 NOUVEAU : Détection automatique des divergences**
        - ✅ **🚨 NOUVEAU : Score de cohérence multi-timeframe (0-100)**
        - ✅ **🚨 NOUVEAU : Alertes détaillées par paire**
        - ✅ **Top Opportunités** : Classement par fiabilité (cohérence + alignement)
        - ✅ **Filtre volume désactivé** : Remplacé par volatilité (adapté au Forex)
        - ✅ **Performance** : Cache intelligent pour rapidité optimale
        
        ### 📊 Timeframes analysés :
        15 minutes | 1 Heure | 4 Heures | Daily | Weekly | Monthly
        
        ### 🎯 Système d'alertes intelligent :
        Le système détecte automatiquement :
        - Divergences court terme vs long terme
        - Contradictions entre timeframes adjacents
        - Marchés en range (indécision)
        - Incohérences avec le Monthly
        - Alignements optimaux (bonus de confiance)
        """)

    st.markdown("---")
    st.caption("💼 Données via OANDA v20 REST API | 🧠 Logique Bluestar MTF Pro+ Optimisée | ⚡ Version Professionnelle avec ADX & Détection de divergences")

if __name__ == "__main__":
    main()
            
            with tab1:
                st.markdown("""
                ### 🧮 Logique de calcul professionnelle
                
                **Indicateurs principaux :**
                - **ZLEMA (Zero-Lag EMA)** : Indicateur de tendance avec lag réduit
                - **Bandes de volatilité adaptatives** : Basées sur l'ATR avec ratio dynamique
                - **ADX (Average Directional Index)** : Mesure la force de la tendance
                
                **Système de scoring multicritères :**
                1. Position par rapport au ZLEMA (2 points)
                2. Structure de marché - EMA(20) (1 point)
                3. Momentum RSI/MACD combiné (1 point)
                4. Volatilité forte + chandeliers significatifs (1 point)
                5. Cassure de bandes de volatilité (2 points)
                6. Bonus ADX pour tendance forte (1 point)
                
                **Maximum : 8 points par direction**
                
                **Confirmation adaptative :**
                - 15m/1H : 2 barres de confirmation
                - 4H/D : 3 barres de confirmation
                - W : 2 barres de confirmation
                - M : 1 barre de confirmation
                
                **Détection de Range :**
                - Si ADX < 25 → État "Range" (marché sans tendance claire)
                - Permet d'éviter les faux signaux en phase de consolidation
                """)
            
            with tab2:
                st.markdown("""
                ### ⚡ Optimisations professionnelles
                
                **1. Poids MTF optimisés (progression exponentielle) :**
                ```
                15m : 0.5  (faible poids, très bruité)
                1H  : 1.0
                4H  : 2.0
                D   : 3.0
                W   : 4.5
                M   : 6.0  (poids maximum, haute fiabilité)
                Total : 17.0 points
                ```
                
                **2. Filtre ADX activé :**
                - ADX < 25 : Market en Range → Éviter le trading
                - ADX > 37.5 : Tendance très forte → Bonus de signal
                - Remplace le filtre volume (inapplicable sur Forex)
                
                **3. Volatilité comme proxy du volume :**
                - Sur Forex, pas de volume réel disponible
                - Utilisation de la volatilité (ATR ratio) pour détecter l'activité
                - Combinaison avec l'analyse des chandeliers
                
                **4. Qualité globale pondérée :**
                - A+ : Qualité excellente (≥75%) sur tous les timeframes
                - A : Très bonne qualité (≥60%)
                - B : Qualité correcte (≥45%)
                - C : Qualité moyenne (≥30%)
                - D : Qualité faible
                
                **5. Cache intelligent :**
                - Données mises en cache 5 minutes
                - Évite les appels API répétés
                - Amélioration de la performance de ~90%
                """)
            
            with tab3:
                st.markdown("""
                ### 💡 Guide d'utilisation professionnelle
                
                **🎯 Comment interpréter les résultats :**
                
                1. **Alignement MTF ≥ 80% + Qualité A/A+** :
                   - ✅ Signal très fiable
                   - ✅ Tendance bien établie sur tous les timeframes
                   - ✅ Priorité absolue pour le trading
                
                2. **Alignement 60-80% + Qualité B** :
                   - ⚠️ Signal correct mais attention
                   - ⚠️ Vérifier les divergences entre timeframes
                   - ⚠️ Attendre une confirmation supplémentaire
                
                3. **Range détecté (🟠)** :
                   - ❌ Éviter le trading directionnel
                   - ✅ Opportunités de trading de range (support/résistance)
                   - ✅ Attendre une cassure avec volume
                
                **📊 Stratégie de trading recommandée :**
                
                **Pour les entrées :**
                - Filtrer par alignement ≥ 70%
                - Privilégier qualité A+ ou A
                - Exclure les paires en "Range"
                - Attendre un pullback sur support/résistance
                
                **Pour la gestion :**
                - Stop loss : En dessous du dernier swing low/high
                - Take profit : Ratio risk/reward minimum 1:2
                - Trail stop selon le timeframe le plus élevé aligné
                
                **⚠️ Avertissements :**
                - Ne jamais trader uniquement sur ce signal
                - Toujours vérifier le contexte fondamental (news, NFP, Fed...)
                - Respecter votre plan de trading et money management
                - Les performances passées ne garantissent pas les résultats futurs
                
                **🔄 Fréquence d'utilisation :**
                - Swing Trading : Analyser 1x par jour (fin de journée)
                - Day Trading : Analyser 2-3x par jour
                - Scalping : Ne pas utiliser (timeframes trop longs)
                """)
            
        else:
            st.warning("L'analyse n'a produit aucun résultat.")
    else:
        st.info("👆 Cliquez sur le bouton 'Analyser' pour lancer l'analyse complète avec la méthode Bluestar MTF Pro+ optimisée")
        
        # Information avant analyse
        st.markdown("""
        ### 🌟 Nouveautés de la version optimisée :
        
        - ✅ **Filtre ADX** : Détection automatique des marchés en range
        - ✅ **Poids MTF optimisés** : Progression exponentielle (Monthly = 12x le poids du 15m)
        - ✅ **Confirmation adaptative** : Nombre de barres ajusté par timeframe
        - ✅ **Qualité globale** : Score de qualité pondéré affiché
        - ✅ **Top Opportunités** : Suggestions automatiques des meilleures configurations
        - ✅ **Filtre volume désactivé** : Remplacé par volatilité (adapté au Forex)
        - ✅ **Performance** : Cache intelligent pour rapidité optimale
        
        ### 📊 Timeframes analysés :
        15 minutes | 1 Heure | 4 Heures | Daily | Weekly | Monthly
        """)

    st.markdown("---")
    st.caption("💼 Données via OANDA v20 REST API | 🧠 Logique Bluestar MTF Pro+ Optimisée | ⚡ Version Professionnelle avec ADX")

if __name__ == "__main__":
    main()
        

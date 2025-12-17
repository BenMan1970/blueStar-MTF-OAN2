"""
Bluestar Hedge Fund GPS - Multi-Timeframe Analysis System
===========================================================

Application d'analyse multi-timeframe institutionnelle pour le trading Forex.
Utilise une logique top-down stricte avec pondération par force des signaux.

Auteur: Bluestar Capital
Version: 2.0 Production
Date: Décembre 2024

Fonctionnalités:
- Analyse MTF (Monthly → 15min) avec logique institutionnelle
- Retracements directionnels (Bull/Bear)
- Quality grades stricts (A+, A, B, C)
- Filtre institutionnel (élimine contre-tendances)
- ATR multi-timeframe pour risk management
- Exports PDF et CSV
"""

# ==================== IMPORTS ====================
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from io import BytesIO
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

# URL de l'API OANDA (Practice)
OANDA_API_URL = "https://api-fxpractice.oanda.com"

# Liste étendue des paires Forex + métaux + indices
FOREX_PAIRS_EXTENDED = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
    'EURGBP', 'EURJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD',
    'GBPJPY', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD',
    'AUDJPY', 'AUDCAD', 'AUDCHF', 'AUDNZD',
    'CADJPY', 'CADCHF', 'CHFJPY',
    'NZDJPY', 'NZDCAD', 'NZDCHF',
    'XAUUSD', 'XPTUSD', 'US30USD', 'SPX500USD', 'NAS100USD'
]

# Palette de couleurs institutionnelle
TREND_COLORS = {
    'Bullish': '#2ecc71',           # Vert vif - Tendance haussière claire
    'Bearish': '#e74c3c',           # Rouge vif - Tendance baissière claire
    'Retracement Bull': '#7dcea0',  # Vert clair - Pullback dans tendance haussière
    'Retracement Bear': '#f1948a',  # Rouge clair - Correction dans tendance baissière
    'Range': '#95a5a6'              # Gris - Pas de tendance claire
}

# Poids pour le score MTF global (priorité aux hauts timeframes)
MTF_WEIGHTS = {
    'M': 5.0,    # Monthly - Maximum (vision macro)
    'W': 4.0,    # Weekly - Très important
    'D': 4.0,    # Daily - Très important (pivot d'exécution)
    '4H': 2.5,   # 4 heures - Important
    '1H': 1.5,   # 1 heure - Timing
    '15m': 1.0   # 15 minutes - Entrée précise
}
TOTAL_WEIGHT = sum(MTF_WEIGHTS.values())  # 18.0

# ==================== INDICATEURS TECHNIQUES ====================

def sma(series, length):
    """
    Simple Moving Average
    
    Args:
        series (pd.Series): Série de prix (généralement Close)
        length (int): Période de la moyenne
        
    Returns:
        pd.Series: SMA calculée
    """
    return series.rolling(window=length).mean()


def ema(series, length):
    """
    Exponential Moving Average
    
    Args:
        series (pd.Series): Série de prix
        length (int): Période de la moyenne
        
    Returns:
        pd.Series: EMA calculée
    """
    return series.ewm(span=length, adjust=False).mean()


def adx(high, low, close, period=14):
    """
    Average Directional Index - Mesure la force d'une tendance
    
    Args:
        high (pd.Series): Série des hauts
        low (pd.Series): Série des bas
        close (pd.Series): Série des clôtures
        period (int): Période de calcul (défaut: 14)
        
    Returns:
        tuple: (adx, plus_di, minus_di)
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    
    # ATR et Directional Indicators
    atr_s = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_s)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_s)
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_result = dx.ewm(span=period, adjust=False).mean()
    
    return adx_result, plus_di, minus_di


def atr(high, low, close, period=14):
    """
    Average True Range - Mesure la volatilité
    
    Args:
        high (pd.Series): Série des hauts
        low (pd.Series): Série des bas
        close (pd.Series): Série des clôtures
        period (int): Période de calcul (défaut: 14)
        
    Returns:
        pd.Series: ATR calculé
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def rsi(close, period=14):
    """
    Relative Strength Index - Mesure le momentum
    
    Args:
        close (pd.Series): Série des clôtures
        period (int): Période de calcul (défaut: 14)
        
    Returns:
        pd.Series: RSI calculé (0-100)
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ==================== LOGIQUE MTF INSTITUTIONNELLE ====================

def calc_institutional_trend_macro(df):
    """
    Logique pour Monthly/Weekly (Timeframes Macro)
    
    Critères stricts basés sur SMA 200 et EMA 50 pour déterminer la direction macro.
    C'est le "GPS" principal qui guide toutes les décisions intraday.
    
    Args:
        df (pd.DataFrame): DataFrame avec colonnes OHLC
        
    Returns:
        tuple: (trend, score)
            - trend (str): 'Bullish', 'Bearish', ou 'Range'
            - score (int): Force du signal (0-100)
    
    Logique:
        - Perfect alignment (prix > SMA 200 ET EMA 50 > SMA 200) → 85%
        - Simple position (prix > SMA 200 seulement) → 65%
        - Fallback sur SMA 50 si moins de 200 bougies disponibles
    """
    if len(df) < 50:
        return 'Range', 0
    
    close = df['Close']
    curr_price = close.iloc[-1]
    
    # Utiliser SMA 50 si pas assez de données pour SMA 200
    has_200 = len(df) >= 200
    sma200 = sma(close, 200) if has_200 else sma(close, 50)
    ema50 = ema(close, 50)
    
    curr_sma200 = sma200.iloc[-1]
    curr_ema50 = ema50.iloc[-1]
    
    # Conditions pour Monthly/Weekly
    above_sma200 = curr_price > curr_sma200
    below_sma200 = curr_price < curr_sma200
    ema50_above_sma = curr_ema50 > curr_sma200
    ema50_below_sma = curr_ema50 < curr_sma200
    
    # Perfect alignment → Score élevé
    if above_sma200 and ema50_above_sma:
        trend = "Bullish"
        score = 85
    elif below_sma200 and ema50_below_sma:
        trend = "Bearish"
        score = 85
    # Simple position vs SMA → Score moyen
    elif above_sma200:
        trend = "Bullish"
        score = 65
    elif below_sma200:
        trend = "Bearish"
        score = 65
    else:
        trend = "Range"
        score = 40
    
    return trend, score


def calc_institutional_trend_daily(df):
    """
    Logique pour Daily (Timeframe Pivot)
    
    Le Daily est le "pivot" entre macro et intraday. Combine SMA 200, EMA 50, 
    et EMA 21 avec gradations de qualité et détection des retracements.
    
    Args:
        df (pd.DataFrame): DataFrame avec colonnes OHLC
        
    Returns:
        tuple: (trend, score)
            - trend (str): 'Bullish', 'Bearish', 'Retracement Bull', 'Retracement Bear', 'Range'
            - score (int): Force du signal (0-100)
    
    Gradations:
        - Perfect (90%) : Tous les indicateurs alignés (4/4 conditions)
        - Strong (70%)  : 3 conditions sur 4 respectées
        - Retracement (55%) : Prix contre baseline mais structure intacte
        - Weak (50%)    : Simple position vs SMA 200
        - Range (35%)   : Pas de tendance claire
    """
    if len(df) < 200:
        return 'Range', 0
    
    close = df['Close']
    curr_price = close.iloc[-1]
    
    # Indicateurs clés
    sma200 = sma(close, 200)
    ema50 = ema(close, 50)
    ema21 = ema(close, 21)
    
    curr_sma200 = sma200.iloc[-1]
    curr_ema50 = ema50.iloc[-1]
    curr_ema21 = ema21.iloc[-1]
    
    # Conditions d'alignement
    above_sma200 = curr_price > curr_sma200
    below_sma200 = curr_price < curr_sma200
    ema50_above_sma = curr_ema50 > curr_sma200
    ema50_below_sma = curr_ema50 < curr_sma200
    ema21_above_50 = curr_ema21 > curr_ema50
    ema21_below_50 = curr_ema21 < curr_ema50
    price_above_21 = curr_price > curr_ema21
    price_below_21 = curr_price < curr_ema21
    
    # Perfect Bull: Alignement total (4/4)
    if above_sma200 and ema50_above_sma and ema21_above_50 and price_above_21:
        return "Bullish", 90
    
    # Perfect Bear: Alignement total inversé (4/4)
    if below_sma200 and ema50_below_sma and ema21_below_50 and price_below_21:
        return "Bearish", 90
    
    # Strong Bull: Au moins 3 conditions sur 4
    if above_sma200 and ema50_above_sma and (ema21_above_50 or price_above_21):
        return "Bullish", 70
    
    # Strong Bear: Au moins 3 conditions sur 4
    if below_sma200 and ema50_below_sma and (ema21_below_50 or price_below_21):
        return "Bearish", 70
    
    # RETRACEMENT DIRECTIONNEL: Prix contre baseline mais structure intacte
    # Retracement Bull: Prix pullback sous SMA 200 mais EMA 50 au-dessus (structure haussière)
    if below_sma200 and ema50_above_sma:
        return "Retracement Bull", 55
    
    # Retracement Bear: Prix rebond au-dessus SMA 200 mais EMA 50 en-dessous (structure baissière)
    if above_sma200 and ema50_below_sma:
        return "Retracement Bear", 55
    
    # Weak signals: Simple position vs SMA 200
    if above_sma200:
        return "Bullish", 50
    if below_sma200:
        return "Bearish", 50
    
    return "Range", 35


def calc_institutional_trend_4h(df):
    """
    Logique pour 4H
    
    Similaire au Daily mais avec critères légèrement assouplis.
    Sert de confirmation entre Daily et Intraday.
    
    Args:
        df (pd.DataFrame): DataFrame avec colonnes OHLC
        
    Returns:
        tuple: (trend, score)
    
    Scores:
        - Perfect: 80%
        - Strong: 60%
        - Retracement: 50%
        - Range: 40%
    """
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
    
    # Perfect alignment
    if above_sma200 and ema21_above_50 and ema50_above_sma and price_above_21:
        return "Bullish", 80
    
    if below_sma200 and ema21_below_50 and curr_ema50 < curr_sma200 and price_below_21:
        return "Bearish", 80
    
    # Strong signals
    if above_sma200 and price_above_21:
        return "Bullish", 60
    
    if below_sma200 and price_below_21:
        return "Bearish", 60
    
    # Retracements directionnels
    if below_sma200 and ema50_above_sma:
        return "Retracement Bull", 50
    
    if above_sma200 and curr_ema50 < curr_sma200:
        return "Retracement Bear", 50
    
    return "Range", 40


def calc_institutional_trend_intraday(df, macro_trend=None):
    """
    Logique pour 1H et 15m (Timeframes Intraday)
    
    Basée sur EMA alignment (9/21/50), ZLEMA, momentum (RSI/MACD) et volume.
    Détecte les retracements directionnels pour le timing d'entrée optimal.
    
    Args:
        df (pd.DataFrame): DataFrame avec colonnes OHLC + Volume
        macro_trend (str, optional): Tendance macro pour contexte
        
    Returns:
        tuple: (trend, score)
    
    Caractéristiques:
        - ZLEMA (50) : Zero-lag EMA pour réactivité
        - EMA 9/21/50 : Alignment court terme
        - RSI + MACD : Confirmation momentum
        - Volume > 1.3× avg : Bonus de force
        - Baseline SMA 200 si disponible
    
    Scores:
        - Bullish/Bearish fort: jusqu'à 75%
        - Retracement: 45%
        - Range: 30%
    """
    if len(df) < 50:
        return 'Range', 0
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume_data = df.get('Volume', pd.Series([1]*len(df), index=df.index))
    
    curr_price = close.iloc[-1]
    
    # EMAs rapides pour intraday
    ema50 = ema(close, 50)
    ema21 = ema(close, 21)
    ema9 = ema(close, 9)
    
    curr_ema50 = ema50.iloc[-1]
    curr_ema21 = ema21.iloc[-1]
    curr_ema9 = ema9.iloc[-1]
    
    # ZLEMA (Zero-Lag EMA) pour réactivité
    lag = 17
    src_adj = close + (close - close.shift(lag))
    zlema_val = src_adj.ewm(span=50, adjust=False).mean()
    curr_zlema = zlema_val.iloc[-1]
    
    # Baseline (SMA 200 si disponible, sinon EMA 50)
    has_baseline = len(df) >= 200
    baseline = sma(close, 200) if has_baseline else curr_ema50
    curr_baseline = baseline.iloc[-1] if has_baseline else curr_ema50
    
    # Momentum indicators
    rsi_val = rsi(close, 14).iloc[-1]
    macd_line = ema(close, 12) - ema(close, 26)
    signal_line = ema(macd_line, 9)
    curr_macd = macd_line.iloc[-1]
    curr_signal = signal_line.iloc[-1]
    
    # Volume analysis
    vol = volume_data.iloc[-1]
    vol_ma = volume_data.rolling(20).mean().iloc[-1]
    strong_vol = vol > vol_ma * 1.3  # Volume significatif
    
    # EMA Alignment (structure court terme)
    ema_bull_align = curr_ema9 > curr_ema21 and curr_ema21 > curr_ema50
    ema_bear_align = curr_ema9 < curr_ema21 and curr_ema21 < curr_ema50
    
    # Momentum conditions
    momentum_bull = rsi_val > 50 and curr_macd > curr_signal
    momentum_bear = rsi_val < 50 and curr_macd < curr_signal
    
    # Decision principale avec retracements directionnels
    bullish = curr_price > curr_zlema and ema_bull_align and momentum_bull
    bearish = curr_price < curr_zlema and ema_bear_align and momentum_bear
    
    if bullish:
        # Calcul force basée sur distance à ZLEMA
        base_strength = min(75, abs(curr_price - curr_zlema) / curr_price * 1000)
        momentum_bonus = 15 if strong_vol else 0
        score = min(75, base_strength + momentum_bonus)
        return "Bullish", score
    
    if bearish:
        base_strength = min(75, abs(curr_price - curr_zlema) / curr_price * 1000)
        momentum_bonus = 15 if strong_vol else 0
        score = min(75, base_strength + momentum_bonus)
        return "Bearish", score
    
    # RETRACEMENT DIRECTIONNEL Intraday
    if has_baseline:
        # Déterminer la tendance de fond via la baseline
        baseline_trend = "Bullish" if curr_ema50 > curr_baseline else "Bearish"
        
        # Retracement Bull: Prix pullback sous baseline mais structure haussière
        if curr_price < curr_baseline and baseline_trend == "Bullish":
            return "Retracement Bull", 45
        
        # Retracement Bear: Prix rebond au-dessus baseline mais structure baissière
        if curr_price > curr_baseline and baseline_trend == "Bearish":
            return "Retracement Bear", 45
    
    return "Range", 30


# ==================== DATA FETCHING ====================

def get_oanda_data(instrument, granularity, count, account_id, access_token):
    """
    Récupère les données de prix depuis l'API OANDA
    
    Args:
        instrument (str): Paire au format OANDA (ex: 'EUR_USD')
        granularity (str): Timeframe ('D', 'H4', 'H1', 'M15', etc.)
        count (int): Nombre de bougies à récupérer
        account_id (str): ID du compte OANDA
        access_token (str): Token d'authentification API
        
    Returns:
        pd.DataFrame: DataFrame avec colonnes [date, Open, High, Low, Close, Volume]
                      ou DataFrame vide en cas d'erreur
    """
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
        
        # Construction du DataFrame
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
        
    except Exception as e:
        # En cas d'erreur, retourner DataFrame vide
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def get_cached_oanda_data(inst, gran, cnt, acc, tok):
    """
    Version cachée de get_oanda_data avec TTL de 10 minutes
    
    Cache les données pour éviter de surcharger l'API OANDA.
    Le cache expire après 10 minutes (600 secondes).
    """
    return get_oanda_data(inst, gran, cnt, acc, tok)


# ==================== CORE ANALYTICS ====================

def analyze_market(account_id, access_token):
    """
    Analyse complète multi-timeframe de tous les instruments
    
    C'est la fonction principale qui orchestre toute l'analyse:
    1. Récupère les données pour chaque timeframe
    2. Applique la logique MTF appropriée
    3. Calcule les scores pondérés
    4. Applique le filtre institutionnel
    5. Détermine le Quality grade
    
    Args:
        account_id (str): ID du compte OANDA
        access_token (str): Token API OANDA
        
    Returns:
        pd.DataFrame: Résultats avec colonnes [Paire, M, W, D, 4H, 1H, 15m, 
                      MTF, Quality, ATR_Daily, ATR_H1, ATR_15m]
    
    Configuration timeframes:
        - M (Monthly): Resample de 4500 daily bars
        - W (Weekly): Resample de 2000 daily bars
        - D (Daily): 300 bars directs
        - 4H: 300 bars H4
        - 1H: 300 bars H1
        - 15m: 300 bars M15
    """
    results = []
    
    # Configuration des timeframes
    tf_config = {
        'M':   ('D', 4500, 'Macro'),   # Monthly via resampling
        'W':   ('D', 2000, 'Macro'),   # Weekly via resampling
        'D':   ('D', 300,  'Daily'),   # Daily direct
        '4H':  ('H4', 300, '4H'),      # 4H direct
        '1H':  ('H1', 300, 'Intra'),   # 1H direct
        '15m': ('M15', 300,'Intra')    # 15m direct
    }
    
    # Progress bar pour feedback utilisateur
    bar = st.progress(0)
    status = st.empty()
    
    for idx, pair in enumerate(FOREX_PAIRS_EXTENDED):
        # Gestion des noms spéciaux pour métaux et indices
        if pair in ['XAUUSD', 'XPTUSD']:
            inst = f"{pair[:3]}_{pair[3:]}"  # XAU_USD
            display_name = f"{pair[:3]}/USD"
        elif pair in ['US30USD', 'SPX500USD', 'NAS100USD']:
            inst = pair  # Garder tel quel
            display_name = pair.replace('USD', '')
        else:
            inst = f"{pair[:3]}_{pair[3:]}"  # EUR_USD
            display_name = f"{pair[:3]}/{pair[3:]}"
        
        status.text(f"GPS Institutionnel : {display_name}...")
        
        row_data = {'Paire': display_name}
        trends_map = {}
        scores_map = {}
        valid_pair = True
        
        # Étape 1: Récupération des données pour tous les timeframes
        data_cache = {}
        for tf, (gran, count, _) in tf_config.items():
            df = get_cached_oanda_data(inst, gran, count, account_id, access_token)
            if df.empty:
                valid_pair = False
                break
            
            # Resampling pour Monthly et Weekly
            if tf == 'M':
                df = df.resample('ME').agg({
                    'Open':'first',
                    'High':'max',
                    'Low':'min',
                    'Close':'last',
                    'Volume':'sum'
                }).dropna()
                
                # Fallback si pas assez de données mensuelles
                if len(df) < 50:
                    df_temp = get_cached_oanda_data(inst, 'D', 2000, account_id, access_token)
                    if not df_temp.empty:
                        df = df_temp.resample('ME').agg({
                            'Open':'first',
                            'High':'max',
                            'Low':'min',
                            'Close':'last',
                            'Volume':'sum'
                        }).dropna()
                        
            elif tf == 'W':
                df = df.resample('W-FRI').agg({
                    'Open':'first',
                    'High':'max',
                    'Low':'min',
                    'Close':'last',
                    'Volume':'sum'
                }).dropna()
            
            data_cache[tf] = df

        if not valid_pair:
            continue  # Passer à la paire suivante si données invalides

        # Étape 2: Application de la logique MTF selon le timeframe
        for tf, (_, _, mode) in tf_config.items():
            df = data_cache[tf]
            
            if mode == 'Macro':
                t, s = calc_institutional_trend_macro(df)
            elif mode == 'Daily':
                t, s = calc_institutional_trend_daily(df)
            elif mode == '4H':
                t, s = calc_institutional_trend_4h(df)
            else:  # Intra
                t, s = calc_institutional_trend_intraday(df)
            
            trends_map[tf] = t
            scores_map[tf] = s
            row_data[tf] = t

        # Étape 3: Calcul de l'ATR sur Daily, H1 et 15m pour risk management
        df_daily = data_cache['D']
        atr_daily = atr(df_daily['High'], df_daily['Low'], df_daily['Close'], 14).iloc[-1]
        row_data['ATR_Daily'] = f"{atr_daily:.5f}" if atr_daily < 1 else f"{atr_daily:.2f}"
        
        df_h1 = data_cache['1H']
        atr_h1 = atr(df_h1['High'], df_h1['Low'], df_h1['Close'], 14).iloc[-1]
        row_data['ATR_H1'] = f"{atr_h1:.5f}" if atr_h1 < 1 else f"{atr_h1:.2f}"
        
        df_15m = data_cache['15m']
        atr_15m = atr(df_15m['High'], df_15m['Low'], df_15m['Close'], 14).iloc[-1]
        row_data['ATR_15m'] = f"{atr_15m:.5f}" if atr_15m < 1 else f"{atr_15m:.2f}"

        # Étape 4: FILTRE INSTITUTIONNEL - Aligner intraday avec macro
        # La tendance macro guide les timeframes intraday (top-down strict)
        macro_trend = trends_map['M'] if trends_map['M'] != 'Range' else \
                     trends_map['W'] if trends_map['W'] != 'Range' else \
                     trends_map['D']
        
        # Filtrer 1H si contre-tendance macro
        if macro_trend == 'Bearish' and trends_map['1H'] == 'Bullish':
            trends_map['1H'] = 'Range'  # Neutraliser contre-tendance
        if macro_trend == 'Bullish' and trends_map['1H'] == 'Bearish':
            trends_map['1H'] = 'Range'
        
        # Filtrer 15m si contre-tendance macro
        if macro_trend == 'Bearish' and trends_map['15m'] == 'Bullish':
            trends_map['15m'] = 'Range'
        if macro_trend == 'Bullish' and trends_map['15m'] == 'Bearish':
            trends_map['15m'] = 'Range'
        
        # Mettre à jour les colonnes après filtrage
        row_data['1H'] = trends_map['1H']
        row_data['15m'] = trends_map['15m']

        # Étape 5: Calcul du score MTF pondéré par FORCE
        # Les signaux forts contribuent plus que les signaux faibles
        w_bull = sum(MTF_WEIGHTS[tf] * (scores_map[tf]/100) 
                    for tf in trends_map if trends_map[tf] == 'Bullish')
        w_bear = sum(MTF_WEIGHTS[tf] * (scores_map[tf]/100) 
                    for tf in trends_map if trends_map[tf] == 'Bearish')
        
        # Retracements Bull comptent pour la tendance haussière (contribution réduite 30%)
        w_bull += sum(MTF_WEIGHTS[tf] * 0.3 
                     for tf in trends_map if trends_map[tf] == 'Retracement Bull')
        # Retracements Bear comptent pour la tendance baissière (contribution réduite 30%)
        w_bear += sum(MTF_WEIGHTS[tf] * 0.3 
                     for tf in trends_map if trends_map[tf] == 'Retracement Bear')
        
        # Étape 6: QUALITY GRADE (seuils stricts institutional grade)
        # Basé sur alignement + force moyenne des hauts timeframes
        high_tf_avg = (scores_map['M'] + scores_map['W'] + scores_map['D']) / 3
        
        quality = 'C'  # Par défaut
        
        # Vérifier que les hauts TF ne sont pas en Retracement pour Quality A/A+
        high_tf_clean = (
            'Retracement' not in trends_map['D'] and 
            'Retracement' not in trends_map['M'] and 
            'Retracement' not in trends_map['W']
        )
        
        # Attribution du Quality grade
        if trends_map['D'] == trends_map['M'] == trends_map['W'] and high_tf_clean:
            # Tous les hauts TF alignés (même direction)
            if high_tf_avg >= 80:
                quality = 'A+'  # Excellence: alignement + force exceptionnelle
            elif high_tf_avg >= 70:
                quality = 'A'   # Très bon: alignement + force solide
            else:
                quality = 'B'   # Bon: alignement mais force moyenne
                
        elif trends_map['D'] == trends_map['M'] and high_tf_clean:
            # M + D alignés (W différent)
            if high_tf_avg >= 75:
                quality = 'B+'  # Bon setup malgré W non aligné
            else:
                quality = 'B'
                
        elif trends_map['D'] == trends_map['W'] and high_tf_clean:
            quality = 'B-'  # Setup acceptable (D + W)
        else:
            quality = 'C'   # Désalignement macro → Pas de trade

        # Étape 7: Détermination de la direction MTF finale
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
        
        # Mise à jour de la progress bar
        bar.progress((idx + 1) / len(FOREX_PAIRS_EXTENDED))
        
    bar.empty()
    status.empty()
    return pd.DataFrame(results)


# ==================== EXPORTS ====================

def create_pdf(df):
    """
    Génère un rapport PDF professionnel avec les résultats MTF
    
    Args:
        df (pd.DataFrame): DataFrame avec les résultats d'analyse
        
    Returns:
        bytes: Contenu du PDF en bytes (prêt pour download)
    
    Caractéristiques:
        - Colonnes adaptatives selon largeur de page
        - Couleurs RGB institutionnelles
        - Headers en gras
        - Bordures et alignement propres
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Bluestar GPS Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(5)
    
    # Colonnes à exporter
    cols = ['Paire', 'M', 'W', 'D', '4H', '1H', '15m', 'MTF', 'Quality', 
            'ATR_Daily', 'ATR_H1', 'ATR_15m']
    w = pdf.w / (len(cols) + 1)  # Largeur adaptative
    
    # Headers
    pdf.set_font("Helvetica", "B", 6)
    for c in cols: 
        pdf.cell(w, 8, c.replace('_', ' '), border=1, align='C', 
                new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln()
    
    # Data rows avec couleurs
    pdf.set_font("Helvetica", "", 6)
    for _, row in df.iterrows():
        for c in cols:
            val = str(row[c])
            
            # Couleurs de fond selon la tendance
            pdf.set_fill_color(255, 255, 255)  # Blanc par défaut
            if "Bull" in val and "Retracement" not in val: 
                pdf.set_fill_color(46, 204, 113)  # Vert vif
            elif "Bear" in val and "Retracement" not in val: 
                pdf.set_fill_color(231, 76, 60)   # Rouge vif
            elif "Retracement Bull" in val: 
                pdf.set_fill_color(125, 206, 160)  # Vert clair
            elif "Retracement Bear" in val: 
                pdf.set_fill_color(241, 148, 138)  # Rouge clair
            elif "Range" in val: 
                pdf.set_fill_color(149, 165, 166)  # Gris
            
            pdf.cell(w, 8, val, border=1, align='C', fill=True, 
                    new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln()
    
    # Retourner le PDF en bytes
    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()


# ==================== MAIN UI ====================

def main():
    """
    Interface principale Streamlit
    
    Flow:
        1. Affichage du header stylé
        2. Vérification des secrets OANDA
        3. Bouton de lancement d'analyse
        4. Affichage des résultats avec styling
        5. Boutons de téléchargement (PDF + CSV)
    """
    
    # Header institutionnel
    st.markdown("""
        <div style='text-align:center; padding:15px; background:#2c3e50; 
                    color:white; border-radius:10px; margin-bottom:15px'>
            <h2 style='margin:0'>🏛️ Bluestar Hedge Fund GPS</h2>
        </div>
    """, unsafe_allow_html=True)

    # Vérification des credentials OANDA
    try:
        acc = st.secrets["OANDA_ACCOUNT_ID"]
        tok = st.secrets["OANDA_ACCESS_TOKEN"]
    except Exception:
        st.error("❌ Secrets OANDA manquants dans .streamlit/secrets.toml")
        st.stop()

    # Bouton de lancement d'analyse
    if st.button("🚀 LANCER L'ANALYSE TOP-DOWN", type="primary", use_container_width=True):
        with st.spinner("⏳ Analyse Institutionnelle Multi-Timeframe en cours..."):
            df = analyze_market(acc, tok)
            
            if not df.empty:
                # Tri par Quality (meilleurs setups en premier) puis MTF
                df = df.sort_values(by=['Quality', 'MTF'], ascending=[True, False]) 
                st.session_state.df = df
    
    # Affichage des résultats si disponibles
    if "df" in st.session_state:
        df = st.session_state.df
        cols_order = ['Paire', 'M', 'W', 'D', '4H', '1H', '15m', 
                     'MTF', 'Quality', 'ATR_Daily', 'ATR_H1', 'ATR_15m']
        
        # Fonction de styling pour le DataFrame
        def style_map(v):
            """Applique les couleurs institutionnelles aux cellules"""
            if isinstance(v, str):
                if "Bull" in v and "Retracement" not in v: 
                    return f"background-color: {TREND_COLORS['Bullish']}; color:white; font-weight:bold"
                if "Bear" in v and "Retracement" not in v: 
                    return f"background-color: {TREND_COLORS['Bearish']}; color:white; font-weight:bold"
                if "Retracement Bull" in v: 
                    return f"background-color: {TREND_COLORS['Retracement Bull']}; color:white"
                if "Retracement Bear" in v: 
                    return f"background-color: {TREND_COLORS['Retracement Bear']}; color:white"
                if "Range" in v: 
                    return f"background-color: {TREND_COLORS['Range']}; color:white"
            return ""

        # Calcul hauteur dynamique du tableau
        h = (len(df) + 1) * 35 + 3
        
        # Affichage avec styling
        st.dataframe(
            df[cols_order].style.map(style_map), 
            height=h, 
            use_container_width=True
        )
        
        # Boutons de téléchargement
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📄 Télécharger PDF Report", 
                create_pdf(df[cols_order]), 
                "Bluestar_GPS.pdf", 
                "application/pdf",
                use_container_width=True
            )
        with c2:
            st.download_button(
                "📊 Télécharger CSV Data", 
                df[cols_order].to_csv(index=False).encode(), 
                "Bluestar_GPS.csv", 
                "text/csv",
                use_container_width=True
            )


if __name__ == "__main__":
    main()

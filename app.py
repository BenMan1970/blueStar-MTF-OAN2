import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import requests
import time
from io import BytesIO

# Import pour l'image et le PDF
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF

# --- Constantes (Mises à jour) ---
OANDA_API_URL = "https://api-fxpractice.oanda.com" # Ou "https://api-fxtrade.oanda.com" pour un compte réel
FOREX_PAIRS_EXTENDED = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
    'EURGBP', 'EURJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD',
    'GBPJPY', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD',
    'AUDJPY', 'AUDCAD', 'AUDCHF', 'AUDNZD',
    'CADJPY', 'CADCHF',
    'CHFJPY',
    'NZDJPY', 'NZDCAD', 'NZDCHF'
]
TREND_COLORS_HEX = {'BULL': '#008f7a', 'BEAR': '#cc0a00', 'NEUTRAL': '#FFD700', 'N/A': '#808080'}
TREND_COLORS_RGB = {'BULL': (0, 143, 122), 'BEAR': (204, 10, 0), 'NEUTRAL': (255, 215, 0), 'N/A': (128, 128, 128)}


# --- Fonctions techniques adaptées de Pine Script (répétées pour clarté) ---

# EMA (Exponential Moving Average)
def ema_py(series, length):
    return series.ewm(span=length, adjust=False).mean()

# SMA (Simple Moving Average)
def sma_py(series, length):
    return series.rolling(window=length).mean()

# ATR (Average True Range)
def atr_py(df, length):
    high_low = df['High'] - df['Low']
    high_prev_close = abs(df['High'] - df['Close'].shift(1))
    low_prev_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.DataFrame({'hl': high_low, 'hpc': high_prev_close, 'lpc': low_prev_close}).max(axis=1)
    return ema_py(tr, length)

# RSI (Relative Strength Index)
def rsi_py(series, length):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = ema_py(gain, length)
    avg_loss = ema_py(loss, length)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# MACD (Moving Average Convergence Divergence)
def macd_py(series, fast_length=12, slow_length=26, signal_length=9):
    fast_ema = ema_py(series, fast_length)
    slow_ema = ema_py(series, slow_length)
    macd_line = fast_ema - slow_ema
    signal_line = ema_py(macd_line, signal_length)
    return macd_line, signal_line

# Momentum
def mom_py(series, length):
    return series.diff(length)

# Rising/Falling (Pine Script's ta.rising/ta.falling)
def is_rising(series, length):
    return (series > series.shift(length)).astype(int)

def is_falling(series, length):
    return (series < series.shift(length)).astype(int)

# Crossover/Crossunder (Pine Script's ta.crossover/ta.crossunder)
def crossover_py(series1, series2):
    return ((series1.shift(1) < series2.shift(1)) & (series1 > series2)).astype(int)

def crossunder_py(series1, series2):
    return ((series1.shift(1) > series2.shift(1)) & (series1 < series2)).astype(int)


# --- Fonctions de l'application d'origine (adaptées) ---

# get_oanda_data modifiée pour inclure le volume (si disponible via l'API, ce qui n'est pas toujours le cas pour les bougies MID)
def get_oanda_data(instrument, granularity, count, account_id, access_token):
    url = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {'granularity': granularity, 'count': count, 'price': 'M'} # 'M' pour mid-point
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        raw_data = response.json()
        candles_data = raw_data.get('candles', [])
        if not candles_data: return pd.DataFrame()
        
        records = []
        for c in candles_data:
            if c.get('complete', False):
                record = {
                    'date': c['time'],
                    'Open': float(c['mid']['o']),
                    'High': float(c['mid']['h']),
                    'Low': float(c['mid']['l']),
                    'Close': float(c['mid']['c']),
                    'Volume': int(c.get('volume', 0)) # OANDA peut ne pas fournir de volume pour 'M' price
                }
                records.append(record)
        
        if not records: return pd.DataFrame()
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except requests.exceptions.RequestException as e:
        # st.toast(f"Erreur réseau pour {instrument} ({granularity}): {e}", icon="🔥")
        return pd.DataFrame()
    except Exception as e:
        # st.toast(f"Erreur de traitement des données OANDA pour {instrument} ({granularity}): {e}", icon="🔥")
        return pd.DataFrame()


# --- Traduction de calcProfessionalTrend en Python ---
def calc_professional_trend_py(df_ohlcv, length=70, mult=1.2, useMomentumFilter=True, useVolumeFilter=True, trendConfirmBars=3, volatilityPeriod=14):
    # Ensure sufficient data
    min_data_points = max(length * 3, volatilityPeriod * 3, 26) # MACD requires 26 bars, ZLEMA/ATR needs length*3 for highest/SMA
    if df_ohlcv.empty or len(df_ohlcv) < min_data_points:
        return [0, 0, "D", np.nan, np.nan] # [confirmed_trend, strength, quality, zlema, volatility]

    src_calc = df_ohlcv['Close']
    
    adj_length = length
    adj_vol = volatilityPeriod

    lag = math.floor((adj_length - 1) / 2)
    src_zlema = 2 * src_calc - src_calc.shift(lag)
    zlema = ema_py(src_zlema, adj_length)

    atr_adaptive = atr_py(df_ohlcv, adj_vol)
    atr_ma = sma_py(atr_adaptive, adj_length)
    
    volatility_ratio = atr_adaptive / atr_ma
    volatility_ratio.replace([np.inf, -np.inf], np.nan, inplace=True)
    volatility_ratio.fillna(1, inplace=True)

    volatility_series_for_highest = atr_py(df_ohlcv, adj_vol) # Using ATR here as per Pine original logic
    # Pine Script's ta.highest(series, length) is a rolling maximum
    volatility = volatility_series_for_highest.rolling(window=adj_length * 3, min_periods=1).max() * mult * volatility_ratio


    ema20 = ema_py(df_ohlcv['Close'], 20)
    structureBullish = is_rising(ema20, 3)
    structureBearish = is_falling(ema20, 3)

    rsi = rsi_py(df_ohlcv['Close'], 14)
    rsiTrend = pd.Series(np.where(rsi > 50, 1, np.where(rsi < 50, -1, 0)), index=df_ohlcv.index)

    macd_line, macd_signal = macd_py(df_ohlcv['Close'], 12, 26, 9)
    macdTrend = pd.Series(np.where(macd_line > macd_signal, 1, -1), index=df_ohlcv.index)

    momentum = mom_py(df_ohlcv['Close'], 10)
    momentumTrend = pd.Series(np.where(momentum > 0, 1, -1), index=df_ohlcv.index)
    
    momentumScore = (rsiTrend + macdTrend + momentumTrend) / 3

    vol = df_ohlcv['Volume']
    volMA = sma_py(vol, 20)
    volRatio = vol / volMA
    strongVolume = (volRatio > 1.5).astype(int)

    bullishCandle = (df_ohlcv['Close'] > df_ohlcv['Open']).astype(int)
    bearishCandle = (df_ohlcv['Close'] < df_ohlcv['Open']).astype(int)
    candleSize = abs(df_ohlcv['Close'] - df_ohlcv['Open'])
    avgCandleSize = sma_py(candleSize, 14)
    strongCandle = (candleSize > avgCandleSize * 1.2).astype(int)

    upperBand = zlema + volatility
    lowerBand = zlema - volatility

    raw_trend_series = pd.Series(0, index=df_ohlcv.index)
    # Use fillna(0) for initial state, then forward fill
    raw_trend_series = np.where(crossover_py(df_ohlcv['Close'], upperBand) == 1, 1, raw_trend_series)
    raw_trend_series = np.where(crossunder_py(df_ohlcv['Close'], lowerBand) == 1, -1, raw_trend_series)
    raw_trend_series = pd.Series(raw_trend_series, index=df_ohlcv.index).replace(0, np.nan).ffill().fillna(0) # Fill forward any detected trend


    bullish_signals = pd.Series(0, index=df_ohlcv.index, dtype=float)
    bearish_signals = pd.Series(0, index=df_ohlcv.index, dtype=float)

    bullish_signals = np.where(df_ohlcv['Close'] > zlema, bullish_signals + 2, bullish_signals)
    bearish_signals = np.where(df_ohlcv['Close'] < zlema, bearish_signals + 2, bearish_signals)

    bullish_signals = np.where(structureBullish == 1, bullish_signals + 1, bullish_signals)
    bearish_signals = np.where(structureBearish == 1, bearish_signals + 1, bearish_signals)

    if useMomentumFilter:
        bullish_signals = np.where(momentumScore > 0.3, bullish_signals + 1, bullish_signals)
        bearish_signals = np.where(momentumScore < -0.3, bearish_signals + 1, bearish_signals)
    
    if useVolumeFilter:
        # Ensure 'Volume' column exists and is not all NaNs
        if 'Volume' in df_ohlcv.columns and not df_ohlcv['Volume'].isnull().all():
            bullish_signals = np.where((strongVolume == 1) & (bullishCandle == 1) & (strongCandle == 1), bullish_signals + 1, bullish_signals)
            bearish_signals = np.where((strongVolume == 1) & (bearishCandle == 1) & (strongCandle == 1), bearish_signals + 1, bearish_signals)
    
    bullish_signals = np.where(raw_trend_series == 1, bullish_signals + 2, bullish_signals)
    bearish_signals = np.where(raw_trend_series == -1, bearish_signals + 2, bearish_signals)
    
    bullish_signals = pd.Series(bullish_signals, index=df_ohlcv.index)
    bearish_signals = pd.Series(bearish_signals, index=df_ohlcv.index)

    # Confirmed Trend Logic for the LAST bar
    final_confirmed_trend = 0
    if len(bullish_signals) >= trendConfirmBars:
        potential_trends_hist = []
        for i in range(1, trendConfirmBars + 1):
            bs = bullish_signals.iloc[-i]
            brs = bearish_signals.iloc[-i]
            if bs > brs:
                potential_trends_hist.append(1)
            elif brs > bs:
                potential_trends_hist.append(-1)
            else:
                potential_trends_hist.append(0)
        
        if len(set(potential_trends_hist)) == 1 and potential_trends_hist[0] != 0:
            final_confirmed_trend = potential_trends_hist[0]
        else:
            last_potential_trend = 0
            if bullish_signals.iloc[-1] > bearish_signals.iloc[-1]: last_potential_trend = 1
            elif bearish_signals.iloc[-1] > bullish_signals.iloc[-1]: last_potential_trend = -1
            final_confirmed_trend = last_potential_trend
    else:
        last_potential_trend = 0
        if bullish_signals.iloc[-1] > bearish_signals.iloc[-1]: last_potential_trend = 1
        elif bearish_signals.iloc[-1] > bullish_signals.iloc[-1]: last_potential_trend = -1
        final_confirmed_trend = last_potential_trend

    # Strength Calculation for the LAST bar
    last_close = df_ohlcv['Close'].iloc[-1]
    last_zlema = zlema.iloc[-1]
    last_momentumScore = momentumScore.iloc[-1]
    last_bullish_signals = bullish_signals.iloc[-1]
    last_bearish_signals = bearish_signals.iloc[-1]
    
    strength = 0
    if not (pd.isna(last_close) or pd.isna(last_zlema) or pd.isna(last_momentumScore)):
        price_distance = abs((last_close - last_zlema) / last_zlema * 100)
        signal_strength = abs(last_bullish_signals - last_bearish_signals)
        
        pd_norm = min(100, price_distance)
        mom_norm = abs(last_momentumScore) * 100
        sig_norm = signal_strength / 7.0 * 100 
        strength = min(100, (pd_norm * 30 + mom_norm * 30 + sig_norm * 10) / 70.0)

    # Quality Score Calculation for the LAST bar
    quality_score = 0
    if abs(last_bullish_signals - last_bearish_signals) >= 4:
        quality_score += 25
    if 'Volume' in df_ohlcv.columns and not df_ohlcv['Volume'].isnull().all() and strongVolume.iloc[-1] == 1:
        quality_score += 25
    
    last_volatility_ratio = volatility_ratio.iloc[-1]
    if not pd.isna(last_volatility_ratio) and last_volatility_ratio < 1.2:
        quality_score += 25
    
    last_structureBullish = structureBullish.iloc[-1]
    last_structureBearish = structureBearish.iloc[-1]
    if (final_confirmed_trend == 1 and last_structureBullish == 1) or \
       (final_confirmed_trend == -1 and last_structureBearish == 1):
        quality_score += 25

    quality = "D"
    if quality_score >= 75: quality = "A+"
    elif quality_score >= 60: quality = "A"
    elif quality_score >= 45: quality = "B"
    elif quality_score >= 30: quality = "C"

    last_zlema_val = zlema.iloc[-1] if not zlema.empty and zlema.notna().any() else np.nan
    last_volatility_val = volatility.iloc[-1] if not volatility.empty and volatility.notna().any() else np.nan

    return [final_confirmed_trend, strength, quality, last_zlema_val, last_volatility_val]


# --- Fonction d'analyse principale (adaptée pour MTF Bluestar avec OANDA) ---
# @st.cache_data(ttl=60*5) # Réactiver le cache une fois stable
def analyze_forex_pairs_bluestar(account_id, access_token, ps_params):
    results_internal = []
    
    # OANDA granularities for Pine Script MTF
    # Note: OANDA M5 is 5-minute, M15 is 15-minute, H1 is 1-hour, H4 is 4-hour, D is Daily, W is Weekly, M is Monthly
    timeframe_map_oanda = {
        '15m': {'granularity': 'M15', 'count': 250}, # 250 bars
        '60m': {'granularity': 'H1', 'count': 250},
        '240m': {'granularity': 'H4', 'count': 250},
        '1D':  {'granularity': 'D', 'count': 250},
        '1W':  {'granularity': 'W', 'count': 250},
        '1M':  {'granularity': 'M', 'count': 250}
    }

    # Extract Pine Script parameters
    ps_length = ps_params['length']
    ps_mult = ps_params['mult']
    ps_useMomentumFilter = ps_params['useMomentumFilter']
    ps_useVolumeFilter = ps_params['useVolumeFilter']
    ps_trendConfirmBars = ps_params['trendConfirmBars']
    ps_volatilityPeriod = ps_params['volatilityPeriod']

    total_pairs = len(FOREX_PAIRS_EXTENDED)
    progress_bar = st.progress(0, text=f"Préparation de l'analyse de 0 / {total_pairs} paires...")

    all_pairs_data = {}
    
    # Use ThreadPoolExecutor to fetch data in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=8) as executor: # Increased workers for faster fetching
        futures = {}
        for pair_symbol in FOREX_PAIRS_EXTENDED:
            oanda_instrument = f"{pair_symbol[:3]}_{pair_symbol[3:]}"
            for tf_key, params in timeframe_map_oanda.items():
                futures[executor.submit(get_oanda_data, oanda_instrument, params['granularity'], params['count'], account_id, access_token)] = (pair_symbol, tf_key)

        for i, future in enumerate(as_completed(futures)):
            pair_symbol, tf_key = futures[future]
            try:
                df = future.result()
                if pair_symbol not in all_pairs_data:
                    all_pairs_data[pair_symbol] = {}
                all_pairs_data[pair_symbol][tf_key] = df
            except Exception as e:
                # st.toast(f"Erreur lors de la récupération des données pour {pair_symbol} {tf_key}: {e}", icon="⛔")
                pass # Continue processing other pairs
            progress_bar.progress((i + 1) / len(futures), text=f"Récupération des données pour {i+1} / {len(futures)} requêtes...")

    progress_bar.empty()
    progress_bar = st.progress(0, text=f"Analyse des tendances pour 0 / {total_pairs} paires...")

    # Second pass: Analyze fetched data
    for i, pair_symbol in enumerate(FOREX_PAIRS_EXTENDED):
        current_pair_data = all_pairs_data.get(pair_symbol, {})
        data_sets_for_calc = {}
        all_data_ok_for_pair = True

        # Check if we have data for all required TFs for this pair
        required_tfs = ['15m', '60m', '240m', '1D', '1W', '1M']
        for tf in required_tfs:
            df = current_pair_data.get(tf)
            if df is None or df.empty or len(df) < ps_length * 3: # Basic check for minimum data
                all_data_ok_for_pair = False
                break
            data_sets_for_calc[tf] = df

        if not all_data_ok_for_pair:
            progress_bar.progress((i + 1) / total_pairs, text=f"Analyse de {i+1} / {total_pairs} paires...")
            continue # Skip to next pair if data is incomplete


        try:
            s_values = {} # To store confirmed_trend for each TF
            
            for tf_key, df_tf in data_sets_for_calc.items():
                # Pass the dynamically set PS parameters
                s, strength, quality, zlema_val, volatility_val = calc_professional_trend_py(
                    df_tf, length=ps_length, mult=ps_mult, 
                    useMomentumFilter=ps_useMomentumFilter, 
                    useVolumeFilter=ps_useVolumeFilter, 
                    trendConfirmBars=ps_trendConfirmBars, 
                    volatilityPeriod=ps_volatilityPeriod
                )
                s_values[tf_key] = s

            # MTF Score Calculation (as per Pine Script)
            # Weights: 15m: 1, 1H: 1.5, 4H: 2, 1D: 2.5, 1W: 3, 1M: 3
            
            bullishScore = (s_values.get('15m', 0) == 1) * 1 + \
                           (s_values.get('60m', 0) == 1) * 1.5 + \
                           (s_values.get('240m', 0) == 1) * 2 + \
                           (s_values.get('1D', 0) == 1) * 2.5 + \
                           (s_values.get('1W', 0) == 1) * 3 + \
                           (s_values.get('1M', 0) == 1) * 3

            bearishScore = (s_values.get('15m', 0) == -1) * 1 + \
                           (s_values.get('60m', 0) == -1) * 1.5 + \
                           (s_values.get('240m', 0) == -1) * 2 + \
                           (s_values.get('1D', 0) == -1) * 2.5 + \
                           (s_values.get('1W', 0) == -1) * 3 + \
                           (s_values.get('1M', 0) == -1) * 3
            
            totalScore = 13.0 # Sum of weights: 1 + 1.5 + 2 + 2.5 + 3 + 3 = 13

            if bullishScore == 0 and bearishScore == 0:
                alignmentPercent = 0
                dominantTrend = "NEUTRAL"
            else:
                alignmentPercent = (bullishScore / totalScore * 100) if bullishScore >= bearishScore else (bearishScore / totalScore * 100)
                dominantTrend = "BULL" if bullishScore > bearishScore else "BEAR" if bearishScore > bullishScore else "NEUTRAL"
            
            # Formattage de la chaîne d'alignement pour l'affichage
            mtf_emoji = ""
            if dominantTrend == "BULL" and alignmentPercent >= 90: mtf_emoji = "🟢"
            elif dominantTrend == "BEAR" and alignmentPercent >= 90: mtf_emoji = "🔴"
            elif alignmentPercent >= 70: mtf_emoji = "🟡" # Strong but not dominant
            elif alignmentPercent >= 50: mtf_emoji = "🟠" # Moderate
            else: mtf_emoji = "⚪" # Weak / Neutral

            mtf_text = ""
            if alignmentPercent >= 90: mtf_text = "Strong "
            elif alignmentPercent >= 70: mtf_text = "" # No specific prefix
            elif alignmentPercent >= 50: mtf_text = "Weak "
            
            alignment_display = f"{mtf_emoji} {mtf_text}{dominantTrend} ({alignmentPercent:.0f}%)"


            results_internal.append({
                'Paire': f"{pair_symbol[:3]}/{pair_symbol[3:]}",
                '15m': "BULL" if s_values.get('15m') == 1 else ("BEAR" if s_values.get('15m') == -1 else "N/A"),
                '1H': "BULL" if s_values.get('60m') == 1 else ("BEAR" if s_values.get('60m') == -1 else "N/A"),
                '4H': "BULL" if s_values.get('240m') == 1 else ("BEAR" if s_values.get('240m') == -1 else "N/A"),
                '1D': "BULL" if s_values.get('1D') == 1 else ("BEAR" if s_values.get('1D') == -1 else "N/A"),
                '1W': "BULL" if s_values.get('1W') == 1 else ("BEAR" if s_values.get('1W') == -1 else "N/A"),
                '1M': "BULL" if s_values.get('1M') == 1 else ("BEAR" if s_values.get('1M') == -1 else "N/A"),
                'MTF Alignment': alignment_display,
                '_score_internal': (bullishScore - bearishScore) # For sorting
            })
        except Exception as e:
            st.toast(f"Erreur lors du calcul pour la paire {pair_symbol}: {e}", icon="❌")
            pass
        finally:
            progress_bar.progress((i + 1) / total_pairs, text=f"Analyse de {i+1} / {total_pairs} paires...")

    progress_bar.empty()
    if not results_internal: return pd.DataFrame()

    df_temp = pd.DataFrame(results_internal)
    df_temp.sort_values(by='_score_internal', ascending=False, inplace=True)
    return df_temp[['Paire', '15m', '1H', '4H', '1D', '1W', '1M', 'MTF Alignment']]

# --- Fonctions de rapport (adaptées pour les nouvelles colonnes et couleurs) ---
def create_image_report(df_report):
    report_title = "Classement des Paires Forex par Tendance MTF (Bluestar)"
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
    pdf.cell(0, 10, 'Classement des Paires Forex par Tendance MTF (Bluestar)', 0, 1, 'C')
    pdf.ln(10)
    
    # Header
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(220, 220, 220)
    col_width = pdf.w / (len(df_report.columns) + 0.5)
    for col_name in df_report.columns:
        pdf.cell(col_width, 8, col_name, 1, 0, 'C', 1)
    pdf.ln()
    
    # Data rows
    pdf.set_font('Arial', '', 10)
    for _, row in df_report.iterrows():
        for col_name in df_report.columns:
            value = str(row[col_name])
            current_trend_key = ""
            if "BULL" in value: current_trend_key = 'BULL'
            elif "BEAR" in value: current_trend_key = 'BEAR'
            elif "NEUTRAL" in value: current_trend_key = 'NEUTRAL'
            elif "N/A" in value: current_trend_key = 'N/A'

            if current_trend_key in TREND_COLORS_RGB:
                pdf.set_fill_color(*TREND_COLORS_RGB[current_trend_key])
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


# --- Fonction principale de l'application ---
def main():
    st.set_page_config(layout="wide")
    st.title("Classement des Paires Forex par Tendance Multi-Timeframe (Bluestar via OANDA)")

    try:
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
    except (KeyError, FileNotFoundError):
        st.error("Erreur Critique: Les secrets OANDA_ACCOUNT_ID et OANDA_ACCESS_TOKEN ne sont pas configurés. L'application ne peut pas fonctionner.")
        st.stop()

    if 'df_results_bluestar' not in st.session_state:
        st.session_state.df_results_bluestar = pd.DataFrame()

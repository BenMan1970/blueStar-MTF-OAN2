import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import requests
import time

# --- Fonctions et Constantes (Définies en premier) ---

# URL de l'API OANDA (utilisez la bonne pour votre compte : practice ou live)
OANDA_API_URL = "https://api-fxpractice.oanda.com" # Ou "https://api-fxtrade.oanda.com" pour un compte réel

# Constante de la liste des paires
FOREX_PAIRS_EXTENDED = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
    'EURGBP', 'EURJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD',
    'GBPJPY', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD',
    'AUDJPY', 'AUDCAD', 'AUDCHF', 'AUDNZD',
    'CADJPY', 'CADCHF',
    'CHFJPY',
    'NZDJPY', 'NZDCAD', 'NZDCHF'
]

# Fonction HMA (inchangée)
def hma(series, length):
    length = int(length)
    if len(series) < (length + int(math.sqrt(length)) - 1): return pd.Series([np.nan] * len(series), index=series.index, name='HMA')
    wma1_period = int(length / 2)
    sqrt_length_period = int(math.sqrt(length))
    if wma1_period < 1 or sqrt_length_period < 1: return pd.Series([np.nan] * len(series), index=series.index, name='HMA')
    wma1 = series.rolling(window=wma1_period).mean() * 2
    wma2 = series.rolling(window=length).mean()
    raw_hma = wma1 - wma2
    hma_series = raw_hma.rolling(window=sqrt_length_period).mean()
    return hma_series

# Fonction de tendance (inchangée)
def get_trend(fast, slow):
    if fast is None or slow is None or fast.empty or slow.empty: return 'Neutral'
    fast_last_scalar = fast.dropna().iloc[-1] if not fast.dropna().empty else np.nan
    slow_last_scalar = slow.dropna().iloc[-1] if not slow.dropna().empty else np.nan
    if pd.isna(fast_last_scalar) or pd.isna(slow_last_scalar): return 'Neutral'
    if fast_last_scalar > slow_last_scalar: return 'Bullish'
    elif fast_last_scalar < slow_last_scalar: return 'Bearish'
    else: return 'Neutral'

# --- NOUVELLE FONCTION DE RÉCUPÉRATION DES DONNÉES OANDA ---
def get_oanda_data(instrument, granularity, count, account_id, access_token):
    """
    Récupère les données de chandelier depuis l'API OANDA v20.
    """
    url = f"{OANDA_API_URL}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        'granularity': granularity,
        'count': count,
        'price': 'M'  # 'M' pour Midpoint, 'B' pour Bid, 'A' pour Ask
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP 4xx/5xx
        raw_data = response.json()

        candles_data = raw_data.get('candles', [])
        if not candles_data:
            st.toast(f"OANDA: Aucune donnée de chandelier reçue pour {instrument}", icon="⚠️")
            return pd.DataFrame()

        records = []
        for candle in candles_data:
            # On ne garde que les bougies complètes
            if candle.get('complete', False):
                records.append({
                    'date': candle['time'],
                    'Open': float(candle['mid']['o']),
                    'High': float(candle['mid']['h']),
                    'Low': float(candle['mid']['l']),
                    'Close': float(candle['mid']['c']),
                })

        if not records: return pd.DataFrame()

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


# --- Fonction d'analyse principale (adaptée pour OANDA) ---
# @st.cache_data(ttl=60*5) # Le cache est désactivé pour le débogage. Vous pourrez le réactiver.
def analyze_forex_pairs(account_id, access_token):
    results_internal = []
    
    # Paramètres de récupération pour OANDA
    timeframe_params_oanda = {
        'H1': {'granularity': 'H1', 'count': 200}, # 200 bougies H1 suffisent pour les calculs
        'H4': {'granularity': 'H4', 'count': 200}, # 200 bougies H4 suffisent
        'D':  {'granularity': 'D',  'count': 250}, # ~1 an de données journalières
        'W':  {'granularity': 'D',  'count': 750}  # ~3 ans de données journalières pour les ré-échantillonner en Weekly
    }
    
    total_pairs = len(FOREX_PAIRS_EXTENDED)
    progress_bar = st.progress(0, text=f"Analyse de 0 / {total_pairs} paires...")
    
    for i, pair_symbol in enumerate(FOREX_PAIRS_EXTENDED):
        oanda_instrument = f"{pair_symbol[:3]}_{pair_symbol[3:]}" # Format OANDA (ex: EUR_USD)
        try:
            data_sets = {}
            all_data_ok = True
            for tf_key, params in timeframe_params_oanda.items():
                df = get_oanda_data(oanda_instrument, params['granularity'], params['count'], account_id, access_token)
                
                # Le ré-échantillonnage pour le Weekly se fait ici, après avoir récupéré les données Daily
                if tf_key == 'W' and not df.empty:
                    df = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()

                if df.empty:
                    all_data_ok = False
                    break
                data_sets[tf_key] = df
            
            if not all_data_ok:
                st.toast(f"Données manquantes pour {oanda_instrument}, la paire est ignorée.", icon="ℹ️")
                continue

            # Les calculs de tendance restent les mêmes
            data_h1, data_h4, data_d, data_w = data_sets['H1'], data_sets['H4'], data_sets['D'], data_sets['W']
            trend_h1 = get_trend(hma(data_h1['Close'], 12), data_h1['Close'].ewm(span=20, adjust=False).mean())
            trend_h4 = get_trend(hma(data_h4['Close'], 12), data_h4['Close'].ewm(span=20, adjust=False).mean())
            trend_d  = get_trend(data_d['Close'].ewm(span=20, adjust=False).mean(), data_d['Close'].ewm(span=50, adjust=False).mean())
            trend_w  = get_trend(data_w['Close'].ewm(span=20, adjust=False).mean(), data_w['Close'].ewm(span=50, adjust=False).mean())
            
            score = sum([1 if t == 'Bullish' else -1 if t == 'Bearish' else 0 for t in [trend_h1, trend_h4, trend_d, trend_w]])
            results_internal.append({
                'Paire': f"{pair_symbol[:3]}/{pair_symbol[3:]}", 
                'H1': trend_h1, 'H4': trend_h4, 'D': trend_d, 'W': trend_w, 
                '_score_internal': score
            })
        except Exception as e:
            st.error(f"Erreur inattendue lors de l'analyse de {oanda_instrument}: {e}")
            continue
        finally:
            progress_bar.progress((i + 1) / total_pairs, text=f"Analyse de {i+1} / {total_pairs} paires...")
            time.sleep(0.2) # Petite pause pour respecter les limites de l'API OANDA

    progress_bar.empty()
    if not results_internal: return pd.DataFrame()
    
    df_temp = pd.DataFrame(results_internal)
    df_temp.sort_values(by='_score_internal', ascending=False, inplace=True)
    return df_temp[['Paire', 'H1', 'H4', 'D', 'W']]


# --- Fonction principale de l'application (contient l'interface) ---
def main():
    st.set_page_config(layout="wide")
    st.title("Classement des Paires Forex par Tendance MTF (via OANDA)")

    # Vérification des clés API OANDA
    try:
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
    except (KeyError, FileNotFoundError):
        st.error("Erreur Critique: Les secrets `OANDA_ACCOUNT_ID` et `OANDA_ACCESS_TOKEN` ne sont pas configurés. L'application ne peut pas fonctionner.")
        st.info("Veuillez configurer les secrets requis dans les paramètres de votre application Streamlit.")
        st.stop()

    # Initialisation de l'état de session
    if 'df_results' not in st.session_state:
        st.session_state.df_results = pd.DataFrame()
    if 'analysis_done_once' not in st.session_state:
        st.session_state.analysis_done_once = False

    # Logique du bouton
    if st.button("🚀 Analyser les Paires Forex"):
        with st.spinner("Analyse des paires via OANDA en cours..."):
            st.session_state.df_results = analyze_forex_pairs(account_id, access_token)
            st.session_state.analysis_done_once = True

    # Logique d'affichage (inchangée)
    if st.session_state.analysis_done_once:
        if not st.session_state.df_results.empty:
            st.subheader("Classement des paires Forex")
            df_to_display = st.session_state.df_results.copy()
            df_to_display['Paire'] = df_to_display['Paire'].fillna('Erreur de Paire')

            def style_trends(val):
                colors = {'Bullish': '#2E7D32', 'Bearish': '#C62828', 'Neutral': '#FFD700'}
                color = colors.get(val, '')
                text_color = 'white' if val in ['Bullish', 'Bearish'] else 'black'
                return f'background-color: {color}; color: {text_color};'
            
            styled_df = df_to_display.style.map(style_trends, subset=['H1', 'H4', 'D', 'W'])
            height_dynamic = (len(df_to_display) + 1) * 35 + 3
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=height_dynamic)

            st.subheader("Résumé des Indicateurs")
            st.markdown("- **H1, H4**: Tendance basée sur HMA(12) vs EMA(20).\n- **D, W**: Tendance basée sur EMA(20) vs EMA(50).")
        else:
            st.warning("L'analyse n'a produit aucun résultat. Cela peut être dû à un problème de connexion avec l'API OANDA ou des données manquantes pour toutes les paires. Vérifiez les notifications (toasts) pour plus de détails.")
    else:
        st.info("Cliquez sur 'Analyser' pour charger les données et voir le classement.")

    st.markdown("---")
    st.caption("Données via OANDA v20 REST API.")

# --- Point d'entrée de l'application ---
if __name__ == "__main__":
    main()

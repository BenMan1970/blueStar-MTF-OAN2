import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import requests
import time
from io import BytesIO

### AJOUT : IMPORTS POUR L'IMAGE ET LE PDF ###
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF # <-- NOUVEL IMPORT POUR LE PDF

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
# Dictionnaire des couleurs pour la cohérence
TREND_COLORS = {'Bullish': '#2E7D32', 'Bearish': '#C62828', 'Neutral': '#FFD700'}


# --- Fonctions de base (inchangées) ---
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

def get_trend(fast, slow):
    if fast is None or slow is None or fast.empty or slow.empty: return 'Neutral'
    fast_last_scalar = fast.dropna().iloc[-1] if not fast.dropna().empty else np.nan
    slow_last_scalar = slow.dropna().iloc[-1] if not slow.dropna().empty else np.nan
    if pd.isna(fast_last_scalar) or pd.isna(slow_last_scalar): return 'Neutral'
    if fast_last_scalar > slow_last_scalar: return 'Bullish'
    elif fast_last_scalar < slow_last_scalar: return 'Bearish'
    else: return 'Neutral'

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
            st.toast(f"OANDA: Aucune donnée de chandelier reçue pour {instrument}", icon="⚠️")
            return pd.DataFrame()
        records = [{'date': c['time'], 'Open': float(c['mid']['o']), 'High': float(c['mid']['h']), 'Low': float(c['mid']['l']), 'Close': float(c['mid']['c'])} for c in candles_data if c.get('complete', False)]
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


# --- Fonctions de génération de rapports (Image et PDF) ---

def create_simple_image_report(df_report):
    report_title = "Classement des Paires Forex par Tendance MTF"
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

### --- NOUVELLE FONCTION DE CRÉATION DE PDF --- ###
def create_pdf_report(df_report):
    """Crée un rapport PDF à partir du DataFrame des résultats."""
    pdf = FPDF()
    pdf.add_page()
    
    # IMPORTANT : Ajouter une police supportant l'UTF-8 comme DejaVu
    # L'utilisateur doit placer 'DejaVuSans.ttf' dans le même dossier.
    try:
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        pdf.set_font('DejaVu', '', 14)
    except RuntimeError:
        # Plan B si la police n'est pas trouvée
        pdf.set_font('Arial', '', 14)
        st.toast("Police DejaVu non trouvée. Le PDF est généré avec Arial.", icon="⚠️")

    # Titre
    pdf.cell(0, 10, 'Classement des Paires Forex par Tendance MTF', 0, 1, 'C')
    pdf.ln(5)

    # Entêtes du tableau
    pdf.set_font_size(10)
    pdf.set_fill_color(220, 220, 220) # Gris clair pour l'entête
    pdf.set_text_color(0, 0, 0)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(0.3)
    
    col_width = pdf.w / (len(df_report.columns) + 1)  # Largeur de colonne approximative
    
    for col_name in df_report.columns:
        pdf.cell(col_width, 8, col_name, 1, 0, 'C', 1)
    pdf.ln()

    # Contenu du tableau
    pdf.set_font_size(9)
    for index, row in df_report.iterrows():
        for col_name in df_report.columns:
            value = str(row[col_name])
            
            # Appliquer les couleurs
            if value in TREND_COLORS:
                hex_color = TREND_COLORS[value].lstrip('#')
                rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                pdf.set_fill_color(rgb_color[0], rgb_color[1], rgb_color[2])
                pdf.set_text_color(255, 255, 255) # Texte blanc pour les fonds colorés
                fill = True
            else:
                pdf.set_fill_color(255, 255, 255) # Fond blanc
                pdf.set_text_color(0, 0, 0) # Texte noir
                fill = False

            pdf.cell(col_width, 8, value, 1, 0, 'C', fill)
        pdf.ln()

    # Retourner les données binaires du PDF
    return pdf.output(dest='S').encode('latin-1')


# --- Fonction d'analyse principale (inchangée) ---
def analyze_forex_pairs(account_id, access_token):
    results_internal = []
    timeframe_params_oanda = {'H1': {'granularity': 'H1', 'count': 200},'H4': {'granularity': 'H4', 'count': 200},'D':  {'granularity': 'D',  'count': 250},'W':  {'granularity': 'D',  'count': 750}}
    total_pairs = len(FOREX_PAIRS_EXTENDED)
    progress_bar = st.progress(0, text=f"Analyse de 0 / {total_pairs} paires...")
    for i, pair_symbol in enumerate(FOREX_PAIRS_EXTENDED):
        oanda_instrument = f"{pair_symbol[:3]}_{pair_symbol[3:]}"
        try:
            data_sets = {}
            all_data_ok = True
            for tf_key, params in timeframe_params_oanda.items():
                df = get_oanda_data(oanda_instrument, params['granularity'], params['count'], account_id, access_token)
                if tf_key == 'W' and not df.empty:
                    df = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
                if df.empty:
                    all_data_ok = False
                    break
                data_sets[tf_key] = df
            if not all_data_ok:
                st.toast(f"Données manquantes pour {oanda_instrument}, la paire est ignorée.", icon="ℹ️")
                continue
            data_h1, data_h4, data_d, data_w = data_sets['H1'], data_sets['H4'], data_sets['D'], data_sets['W']
            trend_h1 = get_trend(hma(data_h1['Close'], 12), data_h1['Close'].ewm(span=20, adjust=False).mean())
            trend_h4 = get_trend(hma(data_h4['Close'], 12), data_h4['Close'].ewm(span=20, adjust=False).mean())
            trend_d  = get_trend(data_d['Close'].ewm(span=20, adjust=False).mean(), data_d['Close'].ewm(span=50, adjust=False).mean())
            trend_w  = get_trend(data_w['Close'].ewm(span=20, adjust=False).mean(), data_w['Close'].ewm(span=50, adjust=False).mean())
            score = sum([1 if t == 'Bullish' else -1 if t == 'Bearish' else 0 for t in [trend_h1, trend_h4, trend_d, trend_w]])
            results_internal.append({'Paire': f"{pair_symbol[:3]}/{pair_symbol[3:]}", 'H1': trend_h1, 'H4': trend_h4, 'D': trend_d, 'W': trend_w, '_score_internal': score})
        except Exception as e:
            st.error(f"Erreur inattendue lors de l'analyse de {oanda_instrument}: {e}")
            continue
        finally:
            progress_bar.progress((i + 1) / total_pairs, text=f"Analyse de {i+1} / {total_pairs} paires...")
            time.sleep(0.2)
    progress_bar.empty()
    if not results_internal: return pd.DataFrame()
    df_temp = pd.DataFrame(results_internal)
    df_temp.sort_values(by='_score_internal', ascending=False, inplace=True)
    return df_temp[['Paire', 'H1', 'H4', 'D', 'W']]


# --- Fonction principale de l'application (contient l'interface) ---
def main():
    st.set_page_config(layout="wide")
    st.title("Classement des Paires Forex par Tendance MTF (via OANDA)")

    try:
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
    except (KeyError, FileNotFoundError):
        st.error("Erreur Critique: Les secrets `OANDA_ACCOUNT_ID` et `OANDA_ACCESS_TOKEN` ne sont pas configurés.")
        st.stop()

    if 'df_results' not in st.session_state:
        st.session_state.df_results = pd.DataFrame()
    if 'analysis_done_once' not in st.session_state:
        st.session_state.analysis_done_once = False

    if st.button("🚀 Analyser les Paires Forex"):
        with st.spinner("Analyse des paires via OANDA en cours..."):
            st.session_state.df_results = analyze_forex_pairs(account_id, access_token)
            st.session_state.analysis_done_once = True

    if st.session_state.analysis_done_once:
        if not st.session_state.df_results.empty:
            st.subheader("Classement des paires Forex")
            df_to_display = st.session_state.df_results.copy()
            
            def style_trends(val):
                color = TREND_COLORS.get(val, '')
                text_color = 'white' if val in ['Bullish', 'Bearish'] else 'black'
                return f'background-color: {color}; color: {text_color};'
            
            styled_df = df_to_display.style.map(style_trends, subset=['H1', 'H4', 'D', 'W'])
            height_dynamic = (len(df_to_display) + 1) * 35 + 3
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=height_dynamic)

            ### --- SECTION DE TÉLÉCHARGEMENT MODIFIÉE --- ###
            st.divider()
            st.subheader("Options de téléchargement")
            
            col1, col2 = st.columns(2)

            with col1:
                # Générer l'image en mémoire
                image_bytes = create_simple_image_report(st.session_state.df_results)
                st.download_button(
                    label="🖼️ Télécharger en PNG",
                    data=image_bytes,
                    file_name=f"classement_forex_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                    mime='image/png',
                    use_container_width=True
                )
            
            with col2:
                # Générer le PDF en mémoire
                pdf_bytes = create_pdf_report(st.session_state.df_results)
                st.download_button(
                    label="📄 Télécharger en PDF",
                    data=pdf_bytes,
                    file_name=f"classement_forex_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime='application/pdf',
                    use_container_width=True
                )
            ### --- FIN DE LA SECTION MODIFIÉE --- ###

            st.subheader("Résumé des Indicateurs")
            st.markdown("- **H1, H4**: Tendance basée sur HMA(12) vs EMA(20).\n- **D, W**: Tendance basée sur EMA(20) vs EMA(50).")
        else:
            st.warning("L'analyse n'a produit aucun résultat.")
    else:
        st.info("Cliquez sur 'Analyser' pour charger les données et voir le classement.")

    st.markdown("---")
    st.caption("Données via OANDA v20 REST API.")

if __name__ == "__main__":
    main()

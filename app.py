import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import requests
import time
from io import BytesIO

# Imports pour la génération de l'image et du PDF
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
# Dictionnaire centralisé des couleurs pour la cohérence
TREND_COLORS = {'Bullish': '#2E7D32', 'Bearish': '#C62828', 'Neutral': '#FFD700'}


# --- Fonctions de calcul et d'API ---
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
        if not candles_data: return pd.DataFrame()
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


# --- Fonctions de génération de rapports ---

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

def create_pdf_report(df_report):
    pdf = FPDF()
    pdf.add_page()
    
    # GESTION ROBUSTE DE LA POLICE :
    # Tente d'utiliser la police DejaVu. Si le fichier n'est pas dans le dépôt,
    # il bascule sur Arial sans faire planter l'application.
    try:
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        pdf.set_font('DejaVu', '', 14)
    except FileNotFoundError:
        pdf.set_font('Arial', '', 14)
        print("Avertissement: Le fichier de police 'DejaVuSans.ttf' n'a pas été trouvé. Le PDF est généré avec la police Arial par défaut.")

    # Titre
    pdf.cell(0, 10, 'Classement des Paires Forex par Tendance MTF', 0, 1, 'C')
    pdf.ln(5)

    # Entêtes du tableau
    pdf.set_font_size(10)
    pdf.set_fill_color(220, 220, 220)
    pdf.set_text_color(0, 0, 0)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(0.3)
    available_width = pdf.w - 2 * pdf.l_margin
    col_width = available_width / len(df_report.columns)
    
    for col_name in df_report.columns:
        pdf.cell(col_width, 8, col_name, 1, 0, 'C', 1)
    pdf.ln()

    # Contenu du tableau
    pdf.set_font_size(9)
    for _, row in df_report.iterrows():
        for col_name in df_report.columns:
            value = str(row[col_name])
            if value in TREND_COLORS:
                hex_color = TREND_COLORS[value].lstrip('#')
                rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                pdf.set_fill_color(*rgb_color)
                pdf.set_text_color(255, 255, 255)
                fill = True
            else:
                pdf.set_fill_color(255, 255, 255)
                pdf.set_text_color(0, 0, 0)
                fill = False
            pdf.cell(col_width, 8, value, 1, 0, 'C', fill)
        pdf.ln()

    return pdf.output(dest='S').encode('latin-1')


# --- Fonction d'analyse principale ---
def analyze_forex_pairs(account_id, access_token):
    results_internal = []
    timeframe_params = {'H1': {'granularity': 'H1', 'count': 200}, 'H4': {'granularity': 'H4', 'count': 200}, 'D': {'granularity': 'D', 'count': 250}, 'W': {'granularity': 'D', 'count': 750}}
    progress_bar = st.progress(0, text=f"Analyse de 0 / {len(FOREX_PAIRS_EXTENDED)} paires...")
    for i, pair_symbol in enumerate(FOREX_PAIRS_EXTENDED):
        oanda_instrument = f"{pair_symbol[:3]}_{pair_symbol[3:]}"
        try:
            data_sets = {}
            all_data_ok = True
            for tf_key, params in timeframe_params.items():
                df = get_oanda_data(oanda_instrument, params['granularity'], params['count'], account_id, access_token)
                if tf_key == 'W' and not df.empty:
                    df = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
                if df.empty:
                    all_data_ok = False; break
                data_sets[tf_key] = df
            if not all_data_ok: continue
            
            trend_h1 = get_trend(hma(data_sets['H1']['Close'], 12), data_sets['H1']['Close'].ewm(span=20, adjust=False).mean())
            trend_h4 = get_trend(hma(data_sets['H4']['Close'], 12), data_sets['H4']['Close'].ewm(span=20, adjust=False).mean())
            trend_d  = get_trend(data_sets['D']['Close'].ewm(span=20, adjust=False).mean(), data_sets['D']['Close'].ewm(span=50, adjust=False).mean())
            trend_w  = get_trend(data_sets['W']['Close'].ewm(span=20, adjust=False).mean(), data_sets['W']['Close'].ewm(span=50, adjust=False).mean())
            score = sum([1 if t == 'Bullish' else -1 if t == 'Bearish' else 0 for t in [trend_h1, trend_h4, trend_d, trend_w]])
            results_internal.append({'Paire': f"{pair_symbol[:3]}/{pair_symbol[3:]}", 'H1': trend_h1, 'H4': trend_h4, 'D': trend_d, 'W': trend_w, '_score_internal': score})
        except Exception as e:
            st.error(f"Erreur inattendue sur {oanda_instrument}: {e}")
        finally:
            progress_bar.progress((i + 1) / len(FOREX_PAIRS_EXTENDED), text=f"Analyse de {i+1} / {len(FOREX_PAIRS_EXTENDED)} paires...")
            time.sleep(0.2)
    progress_bar.empty()
    if not results_internal: return pd.DataFrame()
    df_temp = pd.DataFrame(results_internal).sort_values(by='_score_internal', ascending=False)
    return df_temp[['Paire', 'H1', 'H4', 'D', 'W']]


# --- Interface principale de l'application Streamlit ---
def main():
    st.set_page_config(layout="wide")
    st.title("Classement des Paires Forex par Tendance MTF (via OANDA)")

    try:
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
    except (KeyError, FileNotFoundError):
        st.error("Erreur Critique: Les secrets `OANDA_ACCOUNT_ID` et `OANDA_ACCESS_TOKEN` ne sont pas configurés.")
        st.stop()

    if 'df_results' not in st.session_state: st.session_state.df_results = pd.DataFrame()
    if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False

    if st.button("🚀 Analyser les Paires Forex"):
        with st.spinner("Analyse des paires via OANDA en cours..."):
            st.session_state.df_results = analyze_forex_pairs(account_id, access_token)
            st.session_state.analysis_done = True

    if st.session_state.analysis_done:
        if not st.session_state.df_results.empty:
            st.subheader("Classement des paires Forex")
            
            def style_trends(val):
                return f'background-color: {TREND_COLORS.get(val, "")}; color: {"white" if val in ["Bullish", "Bearish"] else "black"};'
            
            styled_df = st.session_state.df_results.style.map(style_trends, subset=['H1', 'H4', 'D', 'W'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=(len(st.session_state.df_results) + 1) * 35 + 3)

            st.divider()
            st.subheader("Options de téléchargement")
            col1, col2 = st.columns(2)
            now_str = datetime.now().strftime('%Y%m%d_%H%M')

            with col1:
                st.download_button(
                    label="🖼️ Télécharger en PNG",
                    data=create_simple_image_report(st.session_state.df_results),
                    file_name=f"classement_forex_{now_str}.png",
                    mime='image/png',
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    label="📄 Télécharger en PDF",
                    data=create_pdf_report(st.session_state.df_results),
                    file_name=f"classement_forex_{now_str}.pdf",
                    mime='application/pdf',
                    use_container_width=True
                )

            st.subheader("Résumé des Indicateurs")
            st.markdown("- **H1, H4**: Tendance basée sur HMA(12) vs EMA(20).\n- **D, W**: Tendance basée sur EMA(20) vs EMA(50).")
        else:
            st.warning("L'analyse n'a produit aucun résultat.")
    else:
        st.info("Cliquez sur 'Analyser' pour lancer l'analyse des paires.")

    st.markdown("---")
    st.caption("Données via OANDA v20 REST API.")

if __name__ == "__main__":
    main()

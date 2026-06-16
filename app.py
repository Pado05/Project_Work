import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# ============================================
# IMPORTA CLASSI CUSTOM
# ============================================
from model_utils import text_cleaner, LemmatizzatoreTransformer

# ============================================
# CONFIGURAZIONE STREAMLIT
# ============================================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📰 Guardian Environment News - Sentiment Analysis")

# ============================================
# CARICAMENTO MODELLO
# ============================================
@st.cache_resource
def load_model():
    return joblib.load("model/pipe.pkl")

# ============================================
# FUNZIONI UTILI
# ============================================
def load_dataset(filepath):
    """Carica il CSV e prepara i dati"""
    df = pd.read_csv(filepath)
    df['Date Published'] = pd.to_datetime(df['Date Published'])
    return df

def get_sentiment_by_year(df):
    """Estrae conteggio sentimenti per anno"""
    df['Year'] = df['Date Published'].dt.year
    sentiment_by_year = df.groupby(['Year', 'sentiment']).size().unstack(fill_value=0)
    return sentiment_by_year

def plot_sentiment_timeline(sentiment_by_year):
    """Crea grafico temporale interattivo"""
    fig = go.Figure()
    
    if 'positive' in sentiment_by_year.columns:
        fig.add_trace(go.Bar(
            x=sentiment_by_year.index,
            y=sentiment_by_year['positive'],
            name='Positive ✓',
            marker_color='#2ecc71',
            hovertemplate='<b>Year: %{x}</b><br>Positive: %{y}<extra></extra>'
        ))
    
    if 'negative' in sentiment_by_year.columns:
        fig.add_trace(go.Bar(
            x=sentiment_by_year.index,
            y=sentiment_by_year['negative'],
            name='Negative ✗',
            marker_color='#e74c3c',
            hovertemplate='<b>Year: %{x}</b><br>Negative: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Sentiment Distribution by Year',
        xaxis_title='Year',
        yaxis_title='Number of Articles',
        barmode='group',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

# ============================================
# SIDEBAR - NAVIGAZIONE
# ============================================
page = st.sidebar.radio(
    "📍 Navigazione",
    ["📊 Dashboard", "📤 Carica e Predici"]
)

# ============================================
# PAGINA 1: DASHBOARD
# ============================================
if page == "📊 Dashboard":
    st.header("📊 Analisi Temporale Sentimenti")
    
    try:
        # Carica dataset base
        dataset_path = "dataset/guardian_environment_news_cleaned.csv"
        
        if os.path.exists(dataset_path):
            st.info("📂 Caricando dati dal dataset base...")
            df_base = load_dataset(dataset_path)
            
            # Statistiche generali
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Articles", len(df_base))
            with col2:
                positive_count = (df_base['sentiment'] == 'positive').sum()
                st.metric("Positive", positive_count)
            with col3:
                negative_count = (df_base['sentiment'] == 'negative').sum()
                st.metric("Negative", negative_count)
            with col4:
                positive_pct = (positive_count / len(df_base) * 100)
                st.metric("Positive %", f"{positive_pct:.1f}%")
            
            st.divider()
            
            # Grafico temporale
            sentiment_by_year = get_sentiment_by_year(df_base)
            fig = plot_sentiment_timeline(sentiment_by_year)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabella dettagli per anno
            with st.expander("📋 Dettagli per Anno"):
                st.dataframe(
                    sentiment_by_year.reset_index().rename(
                        columns={'Year': 'Anno', 'positive': 'Positivi', 'negative': 'Negativi'}
                    ),
                    use_container_width=True
                )
        else:
            st.error("❌ Dataset non trovato in 'dataset/guardian_environment_news_cleaned.csv'")
    
    except Exception as e:
        st.error(f"❌ Errore nel caricamento: {str(e)}")

# ============================================
# PAGINA 2: CARICA E PREDICI
# ============================================
elif page == "📤 Carica e Predici":
    st.header("📤 Carica CSV e Applica Modello")
    
    try:
        # Carica il modello
        pipe = load_model()
        st.success("✅ Modello predittivo caricato!")
        
        # Upload file
        uploaded_file = st.file_uploader(
            "Scegli un file CSV (con colonna 'Article Text')",
            type=['csv'],
            help="Il file deve contenere una colonna 'Article Text' o 'text'"
        )
        
        if uploaded_file is not None:
            st.info(f"📄 File caricato: {uploaded_file.name}")
            
            try:
                # Leggi il file caricato
                df_new = pd.read_csv(uploaded_file)
                
                # Verifica colonne disponibili
                text_col = None
                if 'Article Text' in df_new.columns:
                    text_col = 'Article Text'
                elif 'text' in df_new.columns:
                    text_col = 'text'
                else:
                    st.error(f"❌ Nessuna colonna di testo trovata. Colonne disponibili: {list(df_new.columns)}")
                    st.stop()
                
                st.write(f"**Righe nel file:** {len(df_new)}")
                
                # Preview del file
                with st.expander("👁️ Anteprima dati"):
                    st.dataframe(df_new.head(), use_container_width=True)
                
                # Applica predizioni
                if st.button("🚀 Applica Modello Predittivo", use_container_width=True):
                    with st.spinner("⏳ Elaborazione in corso..."):
                        df_new['predicted_sentiment'] = pipe.predict(df_new[text_col])
                    
                    st.success("✅ Predizioni completate!")
                    
                    # Statistiche predizioni
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Articoli Positivi", (df_new['predicted_sentiment'] == 'positive').sum())
                    with col2:
                        st.metric("Articoli Negativi", (df_new['predicted_sentiment'] == 'negative').sum())
                    with col3:
                        pos_pct = ((df_new['predicted_sentiment'] == 'positive').sum() / len(df_new) * 100)
                        st.metric("% Positivi", f"{pos_pct:.1f}%")
                    
                    st.divider()
                    
                    # Grafico pie chart
                    sentiment_counts = df_new['predicted_sentiment'].value_counts()
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=sentiment_counts.index,
                        values=sentiment_counts.values,
                        marker=dict(colors=['#2ecc71', '#e74c3c']),
                        textposition='inside',
                        textinfo='label+percent'
                    )])
                    fig_pie.update_layout(title='Distribuzione Sentimenti', height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Aggiorna il grafico temporale se hanno Date Published
                    if 'Date Published' in df_new.columns:
                        st.subheader("📊 Timeline Aggiornata (Dati Nuovi)")
                        df_new['Date Published'] = pd.to_datetime(df_new['Date Published'])
                        sentiment_new_by_year = get_sentiment_by_year(df_new)
                        fig_timeline = plot_sentiment_timeline(sentiment_new_by_year)
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Download risultati
                    csv_results = df_new.to_csv(index=False)
                    st.download_button(
                        label="💾 Scarica Risultati (CSV)",
                        data=csv_results,
                        file_name=f"sentiment_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Anteprima risultati
                    with st.expander("👁️ Anteprima Risultati"):
                        st.dataframe(
                            df_new[[text_col, 'predicted_sentiment']].head(10),
                            use_container_width=True
                        )
            
            except Exception as e:
                st.error(f"❌ Errore nell'elaborazione del file: {str(e)}")
        
        else:
            st.info("👆 Carica un file CSV per iniziare")
    
    except Exception as e:
        st.error(f"❌ Errore nel caricamento del modello: {str(e)}")


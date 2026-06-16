# 📰 Guardian Environment News - Sentiment Analysis

Una web app Streamlit per analizzare il sentimento degli articoli di notizie ambientali da The Guardian.

## 🎯 Funzionalità

- **📊 Dashboard**: Visualizza grafici temporali dei sentimenti positivi/negativi anno per anno
- **📤 Upload & Predici**: Carica un CSV personalizzato e applica il modello di sentiment analysis
- **📉 Grafici Interattivi**: Timeline e pie chart dinamici con Plotly
- **💾 Export**: Scarica i risultati delle predizioni in CSV

## 🚀 Come Eseguire

### Prerequisiti
- Python 3.8+
- pip (gestore pacchetti Python)

### Setup

1. **Installa le dipendenze:**
```bash
pip install -r requirements.txt
```

2. **Assicurati che i file siano nella posizione corretta:**
```
project_folder/
├── app.py
├── requirements.txt
├── dataset/
│   └── guardian_environment_news_cleaned.csv
├── model/
│   └── pipe.pkl
└── README.md
```

3. **Esegui l'app Streamlit:**
```bash
streamlit run app.py
```

L'app si aprirà nel browser su `http://localhost:8501`

## 📋 Struttura del Progetto

- **app.py**: Applicazione Streamlit principale
- **dataset/**: Contiene i dati CSV per la dashboard
- **model/**: Contiene il modello predittivo (pipe.pkl)
- **requirements.txt**: Dipendenze Python

## 🎨 Navigazione

### 📊 Dashboard
- Visualizza statistiche generali (totale articoli, positivi, negativi)
- Grafico a barre con andamento storico anno per anno
- Tabella dettagli

### 📤 Carica e Predici
- Carica un file CSV con una colonna di testo ("Article Text" o "text")
- Il modello predittivo analizza il sentimento
- Visualizza risultati con grafici
- Scarica risultati in CSV

## 📝 Formato CSV Richiesto

Il CSV deve contenere almeno una colonna di testo con uno di questi nomi:
- `Article Text`
- `text`

Opzionalmente può contenere:
- `Date Published`: Per generare timeline aggiornate

## 🔧 Modello Utilizzato

Il modello è una **Pipeline Scikit-learn** che include:
1. Text Cleaner (rimozione URL, numeri, punteggiatura)
2. Lemmatizer (normalizzazione parole)
3. CountVectorizer (bag-of-words)
4. TF-IDF Transformer
5. LinearSVC Classifier

## 📊 Risultati Attesi

- Classificazione binaria: **Positive** ✓ / **Negative** ✗
- Accuratezza: ~91%
- F1 Score: ~0.68

## 📦 Dipendenze Principali

- **streamlit**: Framework web app
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning pipeline
- **plotly**: Visualizzazione interattiva

## 🐛 Troubleshooting

**Errore: "Modello non trovato"**
- Verifica che il file `model/pipe.pkl` esista

**Errore: "Dataset non trovato"**
- Verifica che il file `dataset/guardian_environment_news_cleaned.csv` esista

**Errore: "Colonna non trovata"**
- Assicurati che il CSV caricato contenga "Article Text" o "text"
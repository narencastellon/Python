import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from xgboost import XGBClassifier
import shap
import os  # Para chequeo de archivo

# Configuración de la página para estilo profesional (inspirado en Power BI: azules, grises claros, sombras suaves)
st.set_page_config(page_title="Detección de Fraude en Transacciones Bancarias", layout="wide", initial_sidebar_state="expanded")

# CSS personalizado para estilo Power BI-like: fondo gris claro, elementos con bordes y sombras, colores azul corporativo


@st.cache_data
def load_data(path='./bank_fraud_transactions.csv'):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return pd.DataFrame()
    else:
        st.error(f"Archivo '{path}' no encontrado. Asegúrate de generarlo primero.")
        return pd.DataFrame()
    return df

fraud_df_raw = load_data()
fraud_df_raw2 = fraud_df_raw.drop(columns= ['device_new', 'velocity_24h'])



# Preprocesamiento: Codificar categóricas para todo el df (para consistencia en EDA y SHAP)
@st.cache_data
def preprocess_data(df):
    cat_cols = ['transaction_type', 'merchant_category', 'device_type']
    existing = [col for col in cat_cols if col in df.columns]
    missing = [col for col in cat_cols if col not in df.columns]

    if missing:
        st.warning(f"Columnas faltantes: {missing}")

    # Codificar categóricas
    df_encoded = pd.get_dummies(df, columns=existing, prefix=existing)

    # Asegurar que todas las columnas sean numéricas (float64)
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').astype(float)

    return df_encoded



fraud_df = preprocess_data(fraud_df_raw2)

# Entrenar modelo si no existe
@st.cache_resource
def train_model():
    X = fraud_df.drop(columns=['transaction_id', 'is_fraud', 'customer_id',])
    y = fraud_df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Métricas de evaluación
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    return model, X.columns, report, roc_auc  # Retorna modelo, features, reporte y ROC-AUC

model, features, eval_report, eval_roc_auc = train_model()

# Función para predicción
def predict_fraud(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[features]  # Alinear columnas
    prob = model.predict_proba(input_df)[0][1]
    return prob

# Fondo para SHAP (muestra pequeña del dataset)
X_background = fraud_df.drop(columns=['transaction_id', 'is_fraud']).sample(1000)

# Función get_shap_explainer corregida (con _model para no hashear)
@st.cache_resource
def get_shap_explainer(_model, X_background):
    explainer = shap.TreeExplainer(_model, X_background)
    return explainer

explainer = get_shap_explainer(model, X_background)


# Sidebar para navegación
st.sidebar.title("Navegación")
st.sidebar.markdown("Selecciona una sección para explorar el análisis de fraude.")
section = st.sidebar.radio("Selecciona una Sección", [
    "1. Datos Históricos",
    "2. Análisis Exploratorio de Datos (EDA)",
    "3. Predicciones y Probabilidades",
    "4. Análisis de SHAP"
])

# Sección 1: Datos Históricos
if section == "1. Datos Históricos":
    st.title("Datos Históricos de Transacciones")
    st.markdown("Visualización del dataset completo de transacciones bancarias.")
    
    #st.dataframe(fraud_df_raw.style.background_gradient(cmap='viridis', subset=['amount', 'account_balance']))
    st.dataframe(fraud_df)
    
    st.subheader("Resumen Estadístico")
    st.dataframe(fraud_df_raw.describe())

    st.dataframe(features)

# Sección 2: Análisis Exploratorio de Datos (EDA)
elif section == "2. Análisis Exploratorio de Datos (EDA)":
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.markdown("Exploración visual y estadística del dataset para entender patrones de fraude.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Fraude")
        fig_dist = plt.figure(figsize=(6, 4))
        sns.countplot(x='is_fraud', data=fraud_df_raw, palette='viridis')
        st.pyplot(fig_dist)
    
    with col2:
        st.subheader("Monto por Clase de Fraude")
        fig_box = plt.figure(figsize=(6, 4))
        sns.boxplot(x='is_fraud', y='amount', data=fraud_df_raw, palette='viridis')
        plt.yscale('log')
        st.pyplot(fig_box)
    
    st.subheader("Matriz de Correlación (Variables Numéricas)")
    numeric_cols = fraud_df_raw.select_dtypes(include=[np.number]).columns
    corr_matrix = fraud_df_raw[numeric_cols].corr()
    fig_corr = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(fig_corr)

# Sección 3: Predicciones y Probabilidades
elif section == "3. Predicciones y Probabilidades":
    st.title("Predicciones y Probabilidades de Fraude")
    st.markdown("Ingrese los parámetros de una transacción para predecir si es fraudulenta, con probabilidad y clasificación.")
    
    # Formulario interactivo para inputs (basado en features del dataset)
    with st.form(key='fraud_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Monto (USD)", min_value=0.0, value=100.0)
            hour_of_day = st.slider("Hora del Día (0-23)", 0, 23, 12)
            #day_of_week = st.slider("Día de la Semana (0-6)", 0, 6, 3)
            customer_age = st.slider("Edad del Cliente", 18, 80, 35)
            account_balance = st.number_input("Saldo de Cuenta (USD)", min_value=0.0, value=5000.0)

        
        with col2:
            distance_from_home = st.number_input("Distancia desde Hogar (km)", min_value=0.0, value=5.0)
            is_foreign = st.selectbox("Transacción Extranjera", [0, 1])
            #num_recent_transactions = st.number_input("Transacciones Recientes (24h)", min_value=0, value=3)
            transaction_type = st.selectbox("Tipo de Transacción", ['online', 'in_store', 'atm'])
            merchant_category = st.selectbox("Categoría del Comercio", ['grocery', 'online_retail', 'travel', 'entertainment', 'fuel'])
            device_type = st.selectbox("Tipo de Dispositivo", ['mobile', 'desktop', 'atm'])
        
        submit = st.form_submit_button("Predecir Fraude")

    if submit:
        # Crear input_data con todas las features (rellenar ceros para one-hot no seleccionadas)
        input_data = {
            'amount': amount,
            'hour_of_day': hour_of_day,
            #'day_of_week': day_of_week,
            'customer_age': customer_age,
            'account_balance': account_balance,

            'distance_from_home': distance_from_home,
            'is_foreign': is_foreign,
            #'device_new' : device_new,
            #'velocity_24h': velocity_24h,
            #'num_recent_transactions': num_recent_transactions,
            # One-hot dummies (simulado; en real, alinea todas)
            'transaction_type_atm': 1 if transaction_type == 'atm' else 0,
            'transaction_type_in_store': 1 if transaction_type == 'in_store' else 0,
            'transaction_type_online': 1 if transaction_type == 'online' else 0,
            
            'merchant_category_entertainment': 1 if merchant_category == 'entertainment' else 0,
            'merchant_category_fuel': 1 if merchant_category == 'fuel' else 0,
            'merchant_category_grocery': 1 if merchant_category == 'grocery' else 0,
            'merchant_category_online_retail': 1 if merchant_category == 'online_retail' else 0,
            'merchant_category_travel': 1 if merchant_category == 'travel' else 0,
            
            'device_type_atm': 1 if device_type == 'atm' else 0,
            'device_type_desktop': 1 if device_type == 'desktop' else 0,
            'device_type_mobile': 1 if device_type == 'mobile' else 0,
            # Asegúrate de que todas las dummies del modelo estén aquí con 0 si no seleccionadas
        }
        input_df = pd.DataFrame([input_data])
        
        # Predicción
        prob_fraud = model.predict_proba(input_df)[0][1]
        is_fraud = 1 if prob_fraud > 0.5 else 0
        
        st.subheader("Resultados de Predicción")
        st.markdown(f"**Probabilidad de Fraude:** {prob_fraud:.2%}")
        st.markdown(f"**Clasificación:** {'Fraude' if is_fraud == 1 else 'Legítima'}")

# Sección 4: Análisis de SHAP
else:
    st.title("Análisis de SHAP para Explicabilidad")
    st.markdown("SHAP explica la contribución de cada feature a las predicciones del modelo. Selecciona una muestra para visualizar.")
    
    sample_size = st.slider("Número de Muestras para SHAP", 100, 1000, 500)
    
    X_sample = fraud_df.drop(columns=['transaction_id', 'is_fraud']).sample(sample_size)
    shap_values = explainer.shap_values(X_sample)
    
    st.subheader("Resumen SHAP (Importancia Global)")
    fig_summary = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig_summary)
    
    st.subheader("Force Plot para una Muestra Específica")
    sample_idx = st.slider("Selecciona Muestra", 0, sample_size - 1, 0)
    fig_force = plt.figure(figsize=(10, 4))
    shap.initjs()  # Para visualización interactiva (si Streamlit lo soporta)
    st.write(shap.force_plot(explainer.expected_value, shap_values[sample_idx], X_sample.iloc[sample_idx]))
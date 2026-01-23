import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import lightgbm as lgb
import shap
import os
import time

# Configuración de la página para estilo profesional (inspirado en Power BI: azules, blancos, sombras)
st.set_page_config(page_title="Detección de Fraude en Tarjetas de Crédito", layout="wide", initial_sidebar_state="expanded")

# CSS personalizado para look Power BI: fondo gris claro, botones azules, sombras en elementos, tipografía limpia


# Cargar o generar datos (cacheado)
@st.cache_data
def load_or_generate_data():
    if os.path.exists('credit_fraud_data.csv'):
        df = pd.read_csv('credit_fraud_data.csv')
    else:
        # Generar si no existe (función del código anterior)
        df = generate_credit_fraud_data(num_cards=500, num_days=200)
        df.to_csv('credit_fraud_data.csv', index=False)
    df['ds'] = pd.to_datetime(df['ds'])
    return df

fraud_df = load_or_generate_data()

# Preprocesamiento y entrenamiento del modelo (cacheado)
@st.cache_resource
def preprocess_and_train():
    # Codificar categóricas
    cat_cols = ['merchant_category', 'transaction_type', 'device_type']
    le_dict = {col: LabelEncoder().fit(fraud_df[col]) for col in cat_cols}
    for col in cat_cols:
        fraud_df[col] = le_dict[col].transform(fraud_df[col])
    
    # Escalar numéricas
    num_cols = ['amount', 'distance_from_home', 'account_balance', 'num_recent_transactions']
    scaler = StandardScaler()
    fraud_df[num_cols] = scaler.fit_transform(fraud_df[num_cols])
    
    # Dividir train/test
    X = fraud_df.drop(columns=['transaction_id', 'card_id', 'ds', 'is_fraud'])
    y = fraud_df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Entrenar LightGBM
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': True,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'random_state': 42
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data], callbacks=[lgb.early_stopping(stopping_rounds=50)])
    
    # Métricas
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_prob > 0.5).astype(int)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    fraud_rate = y.mean() * 100
    
    return model, X.columns, scaler, le_dict, roc_auc, pr_auc, fraud_rate, confusion_matrix(y_test, y_pred), classification_report(y_test, y_pred, output_dict=True)

model, features, scaler, le_dict, roc_auc, pr_auc, fraud_rate, cm, report = preprocess_and_train()

# Función para predecir fraude en una o varias transacciones
def detect_fraud(input_df):
    # Preprocesar input (codificar y escalar)
    for col in le_dict:
        if col in input_df.columns:
            input_df[col] = input_df[col].map(lambda x: le_dict[col].transform([x])[0] if x in le_dict[col].classes_ else -1)
    
    num_cols = ['amount', 'distance_from_home', 'account_balance', 'num_recent_transactions']
    if all(col in input_df.columns for col in num_cols):
        input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Alinear features
    input_df = input_df.reindex(columns=features, fill_value=0)
    
    # Predicción
    probs = model.predict(input_df, num_iteration=model.best_iteration)
    preds = (probs > 0.5).astype(int)
    
    return preds, probs

# SHAP explainer (cacheado)
@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_shap_explainer(model)

# Sidebar para navegación (menú tipo Power BI: secciones claras)
st.sidebar.title("Navegación")
section = st.sidebar.radio("Selecciona una Sección", ["Dashboard de Indicadores", "Datos Históricos", "Análisis Exploratorio", "Detección de Fraude Individual", "Detección de Fraude Batch", "Explicabilidad SHAP"])

# Dashboard de Indicadores (principal, con KPIs)
if section == "Dashboard de Indicadores":
    st.title("Dashboard de Detección de Fraude")
    st.markdown("Indicadores clave del modelo y dataset para monitoreo de fraude en tarjetas de crédito.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="kpi-box"><div class="kpi-title">Tasa de Fraude Global</div><div class="kpi-value">{:.2f}%</div></div>'.format(fraud_rate), unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="kpi-box"><div class="kpi-title">ROC-AUC del Modelo</div><div class="kpi-value">{:.4f}</div></div>'.format(roc_auc), unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="kpi-box"><div class="kpi-title">PR-AUC del Modelo</div><div class="kpi-value">{:.4f}</div></div>'.format(pr_auc), unsafe_allow_html=True)
    
    st.subheader("Matriz de Confusión del Modelo")
    fig_cm = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig_cm)
    
    st.subheader("Reporte de Clasificación")
    st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='viridis'))

# Sección Datos Históricos
elif section == "Datos Históricos":
    st.title("Datos Históricos de Transacciones")
    st.dataframe(fraud_df.style.background_gradient(cmap='viridis', subset=['amount']))

# Sección Análisis Exploratorio
elif section == "Análisis Exploratorio":
    st.title("Análisis Exploratorio de Datos (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        fig_dist = plt.figure(figsize=(6, 4))
        sns.countplot(x='is_fraud', data=fraud_df)
        st.pyplot(fig_dist)
    with col2:
        fig_box = plt.figure(figsize=(6, 4))
        sns.boxplot(x='is_fraud', y='amount', data=fraud_df)
        plt.yscale('log')
        st.pyplot(fig_box)

# Sección Detección Individual
elif section == "Detección de Fraude Individual":
    st.title("Detección de Fraude para Transacción Individual")
    with st.form("individual_form"):
        amount = st.number_input("Monto (USD)", 0.0)
        merchant_category = st.selectbox("Categoría de Comercio", ['grocery', 'online', 'travel', 'electronics', 'fuel'])
        transaction_type = st.selectbox("Tipo de Transacción", ['online', 'pos', 'atm'])
        distance_from_home = st.number_input("Distancia desde Hogar (km)", 0.0)
        is_foreign = st.selectbox("Transacción Extranjera", [0, 1])
        hour_of_day = st.slider("Hora del Día", 0, 23)
        day_of_week = st.slider("Día de la Semana", 0, 6)
        device_type = st.selectbox("Tipo de Dispositivo", ['mobile', 'desktop', 'pos'])
        user_age = st.slider("Edad del Usuario", 18, 80)
        account_balance = st.number_input("Saldo de Cuenta (USD)", 0.0)
        num_recent_transactions = st.number_input("Transacciones Recientes", 0)
        submit = st.form_submit_button("Detectar Fraude")
    
    if submit:
        input_data = pd.DataFrame([{
            'amount': amount,
            'merchant_category': merchant_category,
            'transaction_type': transaction_type,
            'distance_from_home': distance_from_home,
            'is_foreign': is_foreign,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'device_type': device_type,
            'user_age': user_age,
            'account_balance': account_balance,
            'num_recent_transactions': num_recent_transactions
        }])
        preds, probs = detect_fraud(input_data)
        st.markdown(f"**Predicción:** {'Fraude' if preds[0] == 1 else 'No Fraude'}")
        st.markdown(f"**Probabilidad de Fraude:** {probs[0]:.2%}")

# Sección Detección Batch
elif section == "Detección de Fraude Batch":
    st.title("Detección de Fraude para Múltiples Transacciones")
    uploaded_file = st.file_uploader("Sube un CSV con transacciones (columnas como en dataset)", type="csv")
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        preds, probs = detect_fraud(batch_df)
        batch_df['fraud_prediction'] = preds
        batch_df['fraud_probability'] = probs
        st.dataframe(batch_df)
        fraud_users = batch_df[batch_df['fraud_prediction'] == 1]['card_id'].unique()
        st.markdown(f"**Usuarios con Fraude Detectado:** {', '.join(fraud_users) if fraud_users.size > 0 else 'Ninguno'}")

# Sección Explicabilidad SHAP
else:
    st.title("Explicabilidad con SHAP")
    sample_size = st.slider("Tamaño de Muestra para SHAP", 100, 1000, 500)
    X_sample = X_test.sample(sample_size)
    shap_values = explainer.shap_values(X_sample)
    fig_summary = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample)
    st.pyplot(fig_summary)

# Pie de página
st.markdown("---")
st.caption("App para detección de fraude en tarjetas de crédito. Datos sintéticos para demostración.")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import shap
import os

# Configuración de la página para estilo profesional (similar a Power BI: azules, blancos, sombras)
st.set_page_config(page_title="Predicción de Aprobación de Crédito Hipotecario", layout="wide", initial_sidebar_state="expanded")


# Cargar o generar datos (cacheado)
@st.cache_data
def load_or_generate_data():
    if os.path.exists('mortgage_approval_data.csv'):
        df = pd.read_csv('mortgage_approval_data.csv')
    else:
        # Generar si no existe (función del código anterior)
        df = generate_mortgage_approval_data(num_applicants=500, num_days=200)
        df.to_csv('mortgage_approval_data.csv', index=False)
    df['ds'] = pd.to_datetime(df['ds'])
    return df

mortgage_df = load_or_generate_data()

# Preprocesamiento y entrenamiento del modelo Bagging (cacheado)
@st.cache_resource
def preprocess_and_train():
    # Codificar categóricas
    cat_cols = ['employment_status', 'marital_status', 'residence_type']
    le_dict = {col: LabelEncoder().fit(mortgage_df[col]) for col in cat_cols}
    for col in cat_cols:
        mortgage_df[col] = le_dict[col].transform(mortgage_df[col])
    
    # Escalar numéricas
    num_cols = ['applicant_age', 'annual_income', 'credit_score', 'debt_to_income_ratio', 'loan_amount_requested', 'property_value', 'down_payment', 'interest_rate', 'num_dependents', 'years_at_current_job', 'bank_account_balance']
    scaler = StandardScaler()
    mortgage_df[num_cols] = scaler.fit_transform(mortgage_df[num_cols])
    
    # Dividir train/test
    X = mortgage_df.drop(columns=['application_id', 'applicant_id', 'ds', 'is_approved'])
    y = mortgage_df['is_approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Bagging model
    bagging_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=50,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    bagging_model.fit(X_train, y_train)
    
    # Métricas
    y_pred = bagging_model.predict(X_test)
    y_prob = bagging_model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    cm_matrix = confusion_matrix(y_test, y_pred)
    
    return bagging_model, X.columns, scaler, le_dict, report, roc_auc, pr_auc, cm_matrix, X_train, X_test, y_test, y_prob

model, features, scaler, le_dict, report, roc_auc, pr_auc, cm, X_train, X_test, y_test, y_prob = preprocess_and_train()

# Función para predecir aprobación y probabilidad
def predict_approval(input_data):
    input_df = pd.DataFrame([input_data])
    
    # Codificar categóricas usando encoders
    for col in le_dict:
        if col in input_df.columns:
            input_df[col] = le_dict[col].transform([input_df[col][0]])
    
    # Escalar numéricas
    num_cols = ['applicant_age', 'annual_income', 'credit_score', 'debt_to_income_ratio', 'loan_amount_requested', 'property_value', 'down_payment', 'interest_rate', 'num_dependents', 'years_at_current_job', 'bank_account_balance']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Alinear features
    input_df = input_df.reindex(columns=features, fill_value=0)
    
    # Predicción
    prob = model.predict_proba(input_df)[0][1]
    approved = 1 if prob > 0.5 else 0
    
    return approved, prob



# SHAP explainer (cacheado)
@st.cache_resource
def get_shap_explainer(_model, _X_background):
    return shap.Explainer(_model.predict, _X_background)  # Usar model.predict como callable para Explainer (corrige error)

explainer = get_shap_explainer(model, X_train.sample(100))  # Muestra pequeña de X_train para background (eficiencia)


# Sidebar para navegación (menú tipo Power BI: secciones claras)

with st.sidebar:

    # Mostrar imagen/banner en la parte superior
    st.image("./imagen/credi_hipotecario.png")

    st.sidebar.title("Menú")
    section = st.sidebar.radio("Selecciona una Sección", ["Datos Históricos", "Análisis Exploratorio (EDA)", "Predicciones", "Evaluación del Modelo", "Análisis con SHAP", "Medir Riesgo de Clientes"])


# Sección 1: Datos Históricos
if section == "Datos Históricos":
    st.title("Datos Históricos de Solicitudes de Hipoteca")
    st.markdown("Visualización del dataset completo de solicitudes de crédito hipotecario.")
    
    #st.dataframe(mortgage_df.style.background_gradient(cmap='viridis', subset=['annual_income', 'credit_score']))
    st.dataframe(mortgage_df)

# Sección 2: Análisis Exploratorio (EDA)
elif section == "Análisis Exploratorio (EDA)":
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.markdown("Exploración visual del dataset para entender patrones en aprobaciones de hipoteca.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Aprobación")
        fig_dist = plt.figure(figsize=(6, 4))
        sns.countplot(x='is_approved', data=mortgage_df, palette='viridis')
        st.pyplot(fig_dist)
    
    with col2:
        st.subheader("Monto Solicitado por Aprobación")
        fig_box = plt.figure(figsize=(6, 4))
        sns.boxplot(x='is_approved', y='loan_amount_requested', data=mortgage_df, palette='viridis')
        plt.yscale('log')
        st.pyplot(fig_box)
    
    st.subheader("Matriz de Correlación (Variables Numéricas)")
    numeric_cols = mortgage_df.select_dtypes(include=[np.number]).columns
    corr_matrix = mortgage_df[numeric_cols].corr()
    fig_corr = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(fig_corr)

# Sección 3: Predicciones
elif section == "Predicciones":
    st.title("Predicción de Aprobación de Crédito Hipotecario")
    st.markdown("Ingrese los datos del cliente para predecir si la hipoteca será aprobada y ver la probabilidad.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            applicant_age = st.slider("Edad del Solicitante", 25, 65, 40)
            annual_income = st.number_input("Ingreso Anual (USD)", min_value=0.0, value=80000.0)
            employment_status = st.selectbox("Estado Laboral", ['employed', 'self-employed', 'retired'])
            credit_score = st.slider("Score Crediticio", 500, 850, 700)
            debt_to_income_ratio = st.number_input("Ratio Deuda/Ingreso", min_value=0.0, max_value=0.5, value=0.2)
            loan_amount_requested = st.number_input("Monto Solicitado (USD)", min_value=50000.0, max_value=500000.0, value=200000.0)
            property_value = st.number_input("Valor de la Propiedad (USD)", min_value=50000.0, value=250000.0)
        
        with col2:
            down_payment = st.number_input("Pago Inicial (USD)", min_value=0.0, value=50000.0)
            loan_term_years = st.selectbox("Término del Préstamo (años)", [15, 20, 25, 30])
            interest_rate = st.number_input("Tasa de Interés (%)", min_value=0.0, value=4.5)
            marital_status = st.selectbox("Estado Civil", ['single', 'married', 'divorced'])
            num_dependents = st.slider("Número de Dependientes", 0, 4, 0)
            residence_type = st.selectbox("Tipo de Residencia Actual", ['rent', 'own_other', 'family'])
            years_at_current_job = st.slider("Años en Empleo Actual", 0, 30, 10)
            has_previous_mortgage = st.selectbox("Tiene Hipoteca Previa", [0, 1])
            previous_default = st.selectbox("Default Previo", [0, 1])
            bank_account_balance = st.number_input("Saldo Bancario (USD)", min_value=0.0, value=10000.0)
        
        submit = st.form_submit_button("Predecir Aprobación")

    if submit:
        input_data = {
            'applicant_age': applicant_age,
            'annual_income': annual_income,
            'employment_status': employment_status,
            'credit_score': credit_score,
            'debt_to_income_ratio': debt_to_income_ratio,
            'loan_amount_requested': loan_amount_requested,
            'property_value': property_value,
            'down_payment': down_payment,
            'loan_term_years': loan_term_years,
            'interest_rate': interest_rate,
            'marital_status': marital_status,
            'num_dependents': num_dependents,
            'residence_type': residence_type,
            'years_at_current_job': years_at_current_job,
            'has_previous_mortgage': has_previous_mortgage,
            'previous_default': previous_default,
            'bank_account_balance': bank_account_balance
        }
        
        approved, prob = predict_approval(input_data)
        st.subheader("Resultados de Predicción")
        st.markdown(f"**Aprobación Predicha:** {'Aprobado' if approved == 1 else 'No Aprobado'}")
        st.markdown(f"**Probabilidad de Aprobación:** {prob:.2%}")

# Sección 4: Evaluación del Modelo
elif section == "Evaluación del Modelo":
    st.title("Evaluación del Modelo")
    st.markdown("Métricas y gráficos de rendimiento del modelo de bagging para predicción de aprobación de hipoteca.")
    
    st.subheader("Reporte de Clasificación")
    st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='viridis'))
    
    st.subheader("ROC-AUC y PR-AUC")
    st.markdown(f"**ROC-AUC:** {roc_auc:.4f}")
    st.markdown(f"**PR-AUC:** {pr_auc:.4f}")
    
    st.subheader("Matriz de Confusión")
    fig_cm = plt.figure(figsize=(16, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig_cm)

# Sección 5: Análisis con SHAP
elif section == "Análisis con SHAP":
    st.title("Análisis con SHAP para Explicabilidad")
    st.markdown("SHAP explica la contribución de cada variable a las predicciones del modelo.")
    
    sample_size = st.slider("Tamaño de Muestra para SHAP", 100, 1000, 500)
    
    X_sample = X_test.sample(sample_size)
    shap_values = explainer(X_sample)
    
    st.subheader("Resumen SHAP (Importancia Global)")
    fig_summary = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar")
    st.pyplot(fig_summary)
    
    st.subheader("Resumen Detallado SHAP")
    fig_detail = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample)
    st.pyplot(fig_detail)
    
    st.subheader("Waterfall Plot para una Muestra Específica")
    sample_idx = st.slider("Selecciona Muestra", 0, sample_size - 1, 0)
    fig_waterfall = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[sample_idx])
    st.pyplot(fig_waterfall)

# Sección 6: Medir Riesgo de Clientes
elif section == "Medir Riesgo de Clientes":
    st.title("Medir Riesgo de Clientes")
    st.markdown("Categoriza el riesgo de aprobación basado en la probabilidad predicha: Bajo (>70%), Medio (30-70%), Alto (<30%). Evalúa una muestra de clientes del test set.")
    
    num_samples = st.slider("Número de Clientes a Evaluar", 10, 100, 50)
    X_risk_sample = X_test.sample(num_samples)
    y_risk_prob = model.predict_proba(X_risk_sample)[:, 1]
    
    risk_levels = []
    for prob in y_risk_prob:
        if prob > 0.7:
            risk_levels.append("Riesgo Bajo")
        elif prob > 0.3:
            risk_levels.append("Riesgo Medio")
        else:
            risk_levels.append("Riesgo Alto")
    
    risk_df = pd.DataFrame({
        'Cliente ID (Muestra)': range(1, num_samples + 1),
        'Probabilidad de Aprobación': y_risk_prob,
        'Nivel de Riesgo': risk_levels
    })
    
    st.dataframe(risk_df.style.background_gradient(cmap='viridis', subset=['Probabilidad de Aprobación']))
    
    st.subheader("Distribución de Riesgo")
    fig_risk = plt.figure(figsize=(18, 6))
    sns.countplot(y='Nivel de Riesgo', data=risk_df, palette='viridis')
    st.pyplot(fig_risk)

# Pie de página
st.markdown("---")
st.caption("App para predicción de aprobación de crédito hipotecario. Datos sintéticos para demostración.")
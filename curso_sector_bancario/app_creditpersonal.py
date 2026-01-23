import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import os

# Configuración de la página para estilo profesional (similar a Power BI: azules, blancos, sombras)
st.set_page_config(page_title="Predicción de Aprobación de Crédito Personal", layout="wide", initial_sidebar_state="expanded")

# CSS personalizado para look Power BI: fondo gris claro, botones azules, sombras en bloques, tipografía limpia


# Cargar o generar datos (cacheado)
@st.cache_data
def load_or_generate_data():
    if os.path.exists('credit_approval_data.csv'):
        df = pd.read_csv('credit_approval_data.csv')
    else:
        # Generar si no existe (función del código anterior)
        df = generate_credit_approval_data(num_applicants=500, num_days=200)
        df.to_csv('credit_approval_data.csv', index=False)
    df['ds'] = pd.to_datetime(df['ds'])
    return df

credit_df = load_or_generate_data()

# Preprocesamiento y entrenamiento del modelo Stacking (cacheado)
@st.cache_resource
def preprocess_and_train():
    # Codificar categóricas
    cat_cols = ['employment_status', 'loan_purpose', 'marital_status', 'residence_type']
    le_dict = {col: LabelEncoder().fit(credit_df[col]) for col in cat_cols}
    for col in cat_cols:
        credit_df[col] = le_dict[col].transform(credit_df[col])
    
    # Escalar numéricas
    num_cols = ['applicant_age', 'annual_income', 'credit_score', 'debt_to_income_ratio', 'loan_amount_requested', 'num_dependents', 'years_at_current_address', 'years_employed', 'bank_account_balance']
    scaler = StandardScaler()
    credit_df[num_cols] = scaler.fit_transform(credit_df[num_cols])
    
    # Dividir train/test
    X = credit_df.drop(columns=['application_id', 'applicant_id', 'ds', 'is_approved'])
    y = credit_df['is_approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Modelos base y stacking
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ]
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)
    stacking_model.fit(X_train, y_train)
    
    # Métricas
    y_pred = stacking_model.predict(X_test)
    y_prob = stacking_model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    cm_matrix = confusion_matrix(y_test, y_pred)
    
    return stacking_model, X.columns, scaler, le_dict, report, roc_auc, pr_auc, cm_matrix

model, features, scaler, le_dict, report, roc_auc, pr_auc, cm = preprocess_and_train()

# Función para predecir aprobación y probabilidad
def predict_approval(input_data):
    input_df = pd.DataFrame([input_data])
    
    # Codificar categóricas usando encoders
    for col in le_dict:
        input_df[col] = le_dict[col].transform([input_df[col][0]])
    
    # Escalar numéricas
    num_cols = ['applicant_age', 'annual_income', 'credit_score', 'debt_to_income_ratio', 'loan_amount_requested', 'num_dependents', 'years_at_current_address', 'years_employed', 'bank_account_balance']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Alinear features
    input_df = input_df.reindex(columns=features, fill_value=0)
    
    # Predicción
    prob = model.predict_proba(input_df)[0][1]
    approved = 1 if prob > 0.5 else 0
    
    return approved, prob

# Sidebar para navegación (menú tipo Power BI)
with st.sidebar:

    # Mostrar imagen/banner en la parte superior
    st.image("./imagen/credi_personal.png", width = 1000)
    st.sidebar.title("Menú")
    section = st.sidebar.radio("Selecciona una Sección", ["Datos Históricos", "Análisis Exploratorio (EDA)", "Predicciones", "Evolución del Modelo"])
    st.caption("By. Naren Castellon")
    
    

# Sección 1: Datos Históricos
if section == "Datos Históricos":
    st.title("Datos Históricos de Solicitudes de Crédito")
    st.markdown("Visualización del dataset completo de solicitudes de crédito personal.")
    
    #st.dataframe(credit_df.style.background_gradient(cmap='viridis', subset=['annual_income', 'credit_score']))
    st.dataframe(credit_df)

# Sección 2: Análisis Exploratorio (EDA)
elif section == "Análisis Exploratorio (EDA)":
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.markdown("Exploración visual del dataset para entender patrones en aprobaciones de crédito.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Aprobación")
        fig_dist = plt.figure(figsize=(6, 4))
        sns.countplot(x='is_approved', data=credit_df, palette='viridis')
        st.pyplot(fig_dist)
    
    with col2:
        st.subheader("Monto Solicitado por Aprobación")
        fig_box = plt.figure(figsize=(6, 4))
        sns.boxplot(x='is_approved', y='loan_amount_requested', data=credit_df, palette='viridis')
        plt.yscale('log')
        st.pyplot(fig_box)
    
    st.subheader("Matriz de Correlación (Variables Numéricas)")
    numeric_cols = credit_df.select_dtypes(include=[np.number]).columns
    corr_matrix = credit_df[numeric_cols].corr()
    fig_corr = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(fig_corr)

# Sección 3: Predicciones
elif section == "Predicciones":
    st.title("Predicción de Aprobación de Crédito Personal")
    st.markdown("Ingrese los datos del cliente para predecir si el crédito será aprobado y ver la probabilidad.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            applicant_age = st.slider("Edad del Solicitante", 18, 70, 35)
            annual_income = st.number_input("Ingreso Anual (USD)", min_value=0.0, value=50000.0)
            employment_status = st.selectbox("Estado Laboral", ['employed', 'unemployed', 'self-employed'])
            credit_score = st.slider("Score Crediticio", 300, 850, 650)
            debt_to_income_ratio = st.number_input("Ratio Deuda/Ingreso", min_value=0.0, max_value=1.0, value=0.3)
            loan_amount_requested = st.number_input("Monto Solicitado (USD)", min_value=0.0, value=10000.0)
        
        with col2:
            loan_purpose = st.selectbox("Propósito del Préstamo", ['home', 'car', 'education', 'personal'])
            marital_status = st.selectbox("Estado Civil", ['single', 'married', 'divorced'])
            num_dependents = st.slider("Número de Dependientes", 0, 5, 0)
            residence_type = st.selectbox("Tipo de Residencia", ['own', 'rent', 'family'])
            years_at_current_address = st.slider("Años en Dirección Actual", 0, 20, 5)
            years_employed = st.slider("Años Empleado", 0, 40, 10)
            has_previous_loan = st.selectbox("Tiene Préstamo Previo", [0, 1])
            previous_loan_default = st.selectbox("Default en Préstamo Previo", [0, 1])
            bank_account_balance = st.number_input("Saldo Bancario (USD)", min_value=0.0, value=2000.0)
        
        submit = st.form_submit_button("Predecir Aprobación")

    if submit:
        input_data = {
            'applicant_age': applicant_age,
            'annual_income': annual_income,
            'employment_status': employment_status,
            'credit_score': credit_score,
            'debt_to_income_ratio': debt_to_income_ratio,
            'loan_amount_requested': loan_amount_requested,
            'loan_purpose': loan_purpose,
            'marital_status': marital_status,
            'num_dependents': num_dependents,
            'residence_type': residence_type,
            'years_at_current_address': years_at_current_address,
            'years_employed': years_employed,
            'has_previous_loan': has_previous_loan,
            'previous_loan_default': previous_loan_default,
            'bank_account_balance': bank_account_balance
        }
        
        approved, prob = predict_approval(input_data)
        st.subheader("Resultados de Predicción")
        st.markdown(f"**Aprobación Predicha:** {'Aprobado' if approved == 1 else 'No Aprobado'}")
        st.markdown(f"**Probabilidad de Aprobación:** {prob:.2%}")

# Sección 4: Evolución del Modelo
else:
    st.title("Evolución del Modelo")
    st.markdown("Métricas y gráficos de rendimiento del modelo de stacking para predicción de aprobación de crédito.")
    
    st.subheader("Reporte de Clasificación")
    st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='viridis'))
    
    st.subheader("ROC-AUC y PR-AUC")
    st.markdown(f"**ROC-AUC:** {roc_auc:.4f}")
    st.markdown(f"**PR-AUC:** {pr_auc:.4f}")
    
    st.subheader("Matriz de Confusión")
    fig_cm = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig_cm)

# Pie de página
st.markdown("---")
st.caption("App para predicción de aprobación de crédito personal. Datos sintéticos para demostración.")
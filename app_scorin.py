import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Configuración inicial de la página para estilo profesional (similar a Power BI)
st.set_page_config(page_title="Análisis de Crédito - Banco XYZ", layout="wide", initial_sidebar_state="expanded")

# Estilo CSS personalizado para un look moderno y profesional (colores actualizados)
st.markdown("""<style>
    .main { background-color: #f0f4f8; }  /* Fondo principal cambiado a gris claro suave para mejor visibilidad de textos */
    .stButton>button { background-color: #0056b3; color: white; border-radius: 5px; transition: background-color 0.3s; }
    .stButton>button:hover { background-color: #003f88; }
    .stSlider .stSliderLabel { color: #333333; font-family: Arial, sans-serif; }
    .stSelectbox { background-color: #ffffff; border: 1px solid #ced4da; border-radius: 5px; }
    h1, h2, h3 { color: #1a1a1a; font-family: Arial, sans-serif; }
    .sidebar .sidebar-content { background-color: #e9ecef; box-shadow: 2px 0 5px rgba(0,0,0,0.1); }  /* Fondo sidebar gris medio para contraste */
    .block-container { padding: 20px; background-color: "#6fab6c"; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    body { color: #333333; }  /* Asegurar texto oscuro para legibilidad */
</style>
""", unsafe_allow_html=True)



# Definir columnas numéricas a escalar (global para evitar NameError)
num_cols_to_scale = ['age', 'monthly_income', 'job_tenure', 'loan_amount', 'loan_term', 'credit_history', 
                     'open_accounts', 'current_debt', 'past_delinquencies']

credit_df = pd.read_csv('bank_credit_dataset.csv')

# Generar y preprocesar datos
@st.cache_resource
def load_and_train_model():
    #credit_df = generate_credit_data()

    credit_df = pd.read_csv('bank_credit_dataset.csv')
    
    # Codificar categóricas
    cat_cols = ['gender', 'marital_status', 'education_level', 'home_ownership']
    credit_df = pd.get_dummies(credit_df, columns=cat_cols, drop_first=True)
    
    # Escalar numéricas
    scaler = StandardScaler()
    credit_df[num_cols_to_scale] = scaler.fit_transform(credit_df[num_cols_to_scale])
    
    # Dividir en X/y
    X = credit_df.drop(columns=['customer_id', 'credit_approved'])
    y = credit_df['credit_approved']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    
    return rf_clf, scaler, X.columns, y  # Retornar también y para dashboard

model, scaler, features, y_global = load_and_train_model()

# Función para análisis de sensibilidad
def sensitivity_analysis(input_data, model, feature, variations=np.linspace(-0.2, 0.2, 9)):
    results = []
    original_prob = model.predict_proba(input_data)[0][1]  # Probabilidad de aprobación
    
    for var in variations:
        modified_data = input_data.copy()
        original_value = modified_data[feature].values[0]
        new_value = original_value * (1 + var)
        modified_data[feature] = new_value
        
        new_prob = model.predict_proba(modified_data)[0][1]
        results.append({
            'Variation (%)': var * 100,
            'New Value': new_value,
            'Probability': new_prob
        })
    
    return pd.DataFrame(results)

# Clasificación de riesgo
def classify_risk(prob):
    if prob > 0.7:
        return "Bajo Riesgo", "green"
    elif prob > 0.3:
        return "Riesgo Medio", "orange"
    else:
        return "Alto Riesgo", "red"

# Interfaz de la App
st.sidebar.title("Navegación")
page = st.sidebar.radio("Secciones", ["Dashboard General", "Análisis de Cliente Específico"])

if page == "Dashboard General":
    st.title("Dashboard de Análisis de Crédito - Banco XYZ")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Aprobaciones")
        fig_dist = plt.figure(figsize=(6, 4))
        sns.countplot(x=y_global, palette='viridis')
        st.pyplot(fig_dist)
    
    with col2:
        st.subheader("Importancia de Features")
        feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)[:10]
        fig_imp = plt.figure(figsize=(6, 4))
        sns.barplot(x=feature_importance.values, y=feature_importance.index, palette='viridis')
        st.pyplot(fig_imp)
    
    st.subheader("Matriz de Correlación")
    num_cols = num_cols_to_scale  # Usar lista definida
    corr_matrix = pd.DataFrame(np.random.rand(len(num_cols), len(num_cols)), columns=num_cols, index=num_cols)  # Placeholder
    fig_corr = plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot(fig_corr)

else:
    st.title("Análisis de Crédito para Cliente Específico")
    
    st.markdown("Ingrese los datos del cliente para evaluar su riesgo de crédito.")
    
    # Formulario interactivo para inputs del usuario
    with st.form(key='client_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Edad", 18, 70, 35)
            gender = st.selectbox("Género", ['Male', 'Female'])
            monthly_income = st.number_input("Ingreso Mensual (USD)", 2000, 15000, 5000)
            job_tenure = st.slider("Antigüedad Laboral (años)", 0, 40, 5)
            loan_amount = st.number_input("Monto Solicitado (USD)", 1000, 50000, 10000)
        
        with col2:
            loan_term = st.slider("Plazo del Préstamo (meses)", 12, 60, 24)
            credit_history = st.slider("Historial Crediticio (300-850)", 300, 850, 700)
            open_accounts = st.slider("Cuentas Abiertas", 1, 10, 3)
            current_debt = st.number_input("Deuda Actual (USD)", 0, 30000, 5000)
            past_delinquencies = st.slider("Pagos Atrasados (últimos 2 años)", 0, 10, 1)
            marital_status = st.selectbox("Estado Civil", ['Single', 'Married', 'Divorced'])
            education_level = st.selectbox("Nivel Educativo", ['High School', 'Bachelor', 'Master', 'PhD'])
            home_ownership = st.selectbox("Propiedad Vivienda", ['Rent', 'Own'])
        
        submit_button = st.form_submit_button(label="Evaluar Riesgo de Crédito")

    if submit_button:
        # Crear DataFrame con inputs del usuario
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'monthly_income': [monthly_income],
            'job_tenure': [job_tenure],
            'loan_amount': [loan_amount],
            'loan_term': [loan_term],
            'credit_history': [credit_history],
            'open_accounts': [open_accounts],
            'current_debt': [current_debt],
            'past_delinquencies': [past_delinquencies],
            'marital_status': [marital_status],
            'education_level': [education_level],
            'home_ownership': [home_ownership]
        })
        
        # Preprocesar inputs (codificar categóricas y escalar numéricas)
        input_data = pd.get_dummies(input_data, columns=['gender', 'marital_status', 'education_level', 'home_ownership'], drop_first=True)
        
        # Alinear columnas con las del modelo (agregar missing si faltan)
        for col in features:
            if col not in input_data.columns:
                input_data[col] = 0
        
        input_data = input_data[features]  # Ordenar columnas igual que entrenamiento
        
        # Escalar numéricas
        input_data[num_cols_to_scale] = scaler.transform(input_data[num_cols_to_scale])
        
        # Predicción de probabilidad de aprobación (1: aprobado, 0: rechazado)
        prob_approve = model.predict_proba(input_data)[0][1]
        risk_prob = 1 - prob_approve  # Probabilidad de riesgo (no aprobación)
        
        # Clasificación de riesgo
        if risk_prob < 0.3:
            risk_class = "Bajo Riesgo"
            color = "green"
        elif risk_prob < 0.7:
            risk_class = "Riesgo Medio"
            color = "orange"
        else:
            risk_class = "Alto Riesgo"
            color = "red"
        
        # Mostrar resultados
        st.subheader("Resultados del Análisis de Crédito")
        st.markdown(f"**Probabilidad de Riesgo:** {risk_prob:.2%} ({risk_class})", unsafe_allow_html=True)
        
        # Análisis de Sensibilidad para features clave
        st.subheader("Análisis de Sensibilidad")
        sens_features = ['monthly_income', 'credit_history', 'current_debt']
        
        for feature in sens_features:
            sens_df = sensitivity_analysis(input_data, model, feature)
            st.markdown(f"**Sensibilidad para {feature.replace('_', ' ').title()}**")
            fig_sens = plt.figure(figsize=(8, 4))
            sns.lineplot(x='Variation (%)', y='Probability', data=sens_df, marker='o', color=color)
            plt.title(f'Impacto en Probabilidad de Aprobación al Variar {feature.replace("_", " ").title()}')
            plt.xlabel('Variación (%)')
            plt.ylabel('Probabilidad de Aprobación')
            plt.grid(True)
            st.pyplot(fig_sens)
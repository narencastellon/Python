import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import shap
import os

# Configuración de la página para estilo profesional (similar a Power BI: azules, blancos, sombras)
st.set_page_config(page_title="Predicción de Riesgo en Seguros de Vida", layout="wide", initial_sidebar_state="expanded")

# CSS personalizado para look Power BI: fondo gris claro, botones azules, sombras en bloques, tipografía limpia

# Cargar o generar datos (cacheado)
@st.cache_data
def load_or_generate_data():
    if os.path.exists('life_insurance_risk_data.csv'):
        df = pd.read_csv('life_insurance_risk_data.csv')
    else:
        # Generar si no existe (función del código anterior)
        df = generate_life_insurance_data(num_clients=500, num_days=200)
        df.to_csv('life_insurance_risk_data.csv', index=False)
    df['ds'] = pd.to_datetime(df['ds'])
    return df

insurance_df = load_or_generate_data()

# Preprocesamiento y entrenamiento del modelo CatBoost (cacheado)
@st.cache_resource
def preprocess_and_train():
    # Identificar columnas categóricas
    cat_cols = ['gender', 'smoking_status', 'occupation_risk_level', 'location_risk', 'policy_type']
    cat_features_indices = [insurance_df.columns.get_loc(col) for col in cat_cols]
    
    # Dividir train/test
    X = insurance_df.drop(columns=['client_id', 'ds', 'claim_probability'])
    y = insurance_df['claim_probability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pools para CatBoost
    train_pool = Pool(X_train, y_train, cat_features = cat_cols)
    test_pool = Pool(X_test, y_test, cat_features = cat_cols)
    
    # Modelo CatBoost
    cat_model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function='RMSE',
        random_seed=42,
        verbose=False
    )
    cat_model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)
    
    # Métricas
    y_pred = cat_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return cat_model, X.columns, cat_features_indices, mae, rmse, r2, X_train, X_test, y_test, y_pred

model, features, cat_features, mae, rmse, r2, X_train, X_test, y_test, y_pred = preprocess_and_train()

# Función para predecir probabilidad
def predict_claim(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[features]  # Alinear columnas
    prob = model.predict(input_df)[0]
    return prob

# Función para medir riesgo y calcular prima
def measure_risk_and_premium(prob, coverage):
    if prob > 0.3:
        risk = "Riesgo Alto"
    elif prob > 0.1:
        risk = "Riesgo Medio"
    else:
        risk = "Riesgo Bajo"
    
    base_premium = 500
    factor = 1.5
    premium = base_premium + prob * coverage * factor
    return risk, premium

# SHAP explainer (cacheado)
@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_shap_explainer(model)

# Sidebar para navegación (menú tipo Power BI)
with st.sidebar:
    # Mostrar imagen/banner en la parte superior
    st.image("./imagen/seguro_vida.png")
    st.sidebar.title("Menú")
    section = st.sidebar.radio("Selecciona una Sección", ["Datos Históricos", "Análisis Exploratorio (EDA)", "Predicciones", "Evaluación del Modelo", "Análisis con SHAP", "Medir Riesgo de Clientes", "Análisis de Sensibilidad"])


# Sección 1: Datos Históricos
if section == "Datos Históricos":
    st.title("Datos Históricos de Clientes y Seguros")
    st.markdown("Visualización del dataset completo de clientes con seguros de vida vinculados a cuentas de ahorro.")
    
    #st.dataframe(insurance_df.style.background_gradient(cmap='viridis', subset=['savings_balance', 'health_score']))
    st.dataframe(insurance_df)

# Sección 2: Análisis Exploratorio (EDA)
elif section == "Análisis Exploratorio (EDA)":
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.markdown("Exploración visual del dataset para entender patrones en riesgo de fallecimiento/invalidez.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Probabilidad de Claim")
        fig_dist = plt.figure(figsize=(6, 4))
        sns.histplot(insurance_df['claim_probability'], bins=50, kde=True, color='#0078d4')
        st.pyplot(fig_dist)
    
    with col2:
        st.subheader("Probabilidad vs Edad")
        fig_scatter = plt.figure(figsize=(6, 4))
        sns.scatterplot(x='age', y='claim_probability', data=insurance_df, alpha=0.5, color='#0078d4')
        st.pyplot(fig_scatter)
    
    st.subheader("Matriz de Correlación (Variables Numéricas)")
    numeric_cols = insurance_df.select_dtypes(include=[np.number]).columns
    corr_matrix = insurance_df[numeric_cols].corr()
    fig_corr = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(fig_corr)

# Sección 3: Predicciones
elif section == "Predicciones":
    st.title("Predicción de Riesgo en Seguros de Vida")
    st.markdown("Ingrese los datos del cliente para predecir la probabilidad de fallecimiento/invalidez y calcular prima.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Edad", 25, 80, 45)
            gender = st.selectbox("Género", ['male', 'female'])
            annual_income = st.number_input("Ingreso Anual (USD)", min_value=0.0, value=80000.0)
            savings_balance = st.number_input("Saldo en Cuenta de Ahorro (USD)", min_value=0.0, value=20000.0)
            bmi = st.number_input("BMI", min_value=18.0, max_value=40.0, value=28.0)
            smoking_status = st.selectbox("Fumador", ['yes', 'no'])
            has_chronic_disease = st.selectbox("Enfermedad Crónica", [0, 1])
            family_history_risk = st.selectbox("Historia Familiar de Riesgo", [0, 1])
        
        with col2:
            occupation_risk_level = st.selectbox("Nivel de Riesgo Ocupacional", ['low', 'medium', 'high'])
            location_risk = st.selectbox("Riesgo por Ubicación", ['urban', 'rural'])
            policy_type = st.selectbox("Tipo de Seguro Solicitado", ['life', 'disability', 'combined'])
            coverage_amount = st.number_input("Monto de Cobertura (USD)", min_value=50000.0, value=300000.0)
            policy_term_years = st.selectbox("Plazo del Seguro (años)", [10, 15, 20, 25, 30])
            health_score = st.slider("Score de Salud", 0, 100, 75)
        
        submit = st.form_submit_button("Predecir Riesgo")

    if submit:
        input_data = {
            'age': age,
            'gender': gender,
            'annual_income': annual_income,
            'savings_balance': savings_balance,
            'bmi': bmi,
            'smoking_status': smoking_status,
            'has_chronic_disease': has_chronic_disease,
            'family_history_risk': family_history_risk,
            'occupation_risk_level': occupation_risk_level,
            'location_risk': location_risk,
            'policy_type': policy_type,
            'coverage_amount': coverage_amount,
            'policy_term_years': policy_term_years,
            'health_score': health_score
        }
        
        prob = predict_claim(input_data)
        risk, premium = measure_risk_and_premium(prob, coverage_amount)
        
        st.subheader("Resultados de Predicción")
        st.markdown(f"**Probabilidad de Fallecimiento/Invalidez:** {prob:.2%}")
        st.markdown(f"**Nivel de Riesgo:** {risk}")
        st.markdown(f"**Prima Sugerida Anual:** ${premium:.2f}")
        
        st.subheader("Perfil del Cliente y Seguro Solicitado")
        st.markdown(f"**Tipo de Seguro:** {policy_type}")
        st.markdown(f"**Plazo:** {policy_term_years} años")
        st.markdown(f"**Cobertura:** ${coverage_amount:.2f}")

# Sección 4: Evaluación del Modelo
elif section == "Evaluación del Modelo":
    st.title("Evaluación del Modelo")
    st.markdown("Métricas y gráficos de rendimiento del modelo CatBoost para predicción de riesgo.")
    
    st.subheader("Métricas de Regresión")
    st.markdown(f"**MAE:** {mae:.4f}")
    st.markdown(f"**RMSE:** {rmse:.4f}")
    st.markdown(f"**R²:** {r2:.4f}")
    
    st.subheader("Predicción vs Real")
    fig_pred = plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='#0078d4')
    plt.plot([0, 0.5], [0, 0.5], 'r--')
    plt.xlabel('Real')
    plt.ylabel('Predicho')
    st.pyplot(fig_pred)

# Sección 5: Análisis con SHAP
elif section == "Análisis con SHAP":
    st.title("Análisis con SHAP para Explicabilidad")
    st.markdown("SHAP explica la contribución de cada variable a las predicciones del modelo.")
    
    sample_size = st.slider("Tamaño de Muestra para SHAP", 100, 1000, 500)
    
    X_sample = X_test.sample(sample_size)
    shap_values = explainer.shap_values(X_sample)
    
    st.subheader("Resumen SHAP (Importancia Global)")
    fig_summary = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar")
    st.pyplot(fig_summary)
    
    st.subheader("Resumen Detallado SHAP")
    fig_detail = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample)
    st.pyplot(fig_detail)
    
    st.subheader("Force Plot para una Muestra Específica")
    sample_idx = st.slider("Selecciona Muestra", 0, sample_size - 1, 0)
    fig_force = plt.figure(figsize=(10, 4))
    shap.initjs()
    st.write(shap.force_plot(explainer.expected_value, shap_values[sample_idx], X_sample.iloc[sample_idx]))

# Sección 6: Medir Riesgo de Clientes
elif section == "Medir Riesgo de Clientes":
    st.title("Medir Riesgo de Clientes")
    st.markdown("Categoriza el riesgo basado en la probabilidad predicha: Bajo (<10%), Medio (10-30%), Alto (>30%). Evalúa una muestra de clientes.")
    
    num_samples = st.slider("Número de Clientes a Evaluar", 10, 100, 50)
    X_risk_sample = X_test.sample(num_samples)
    y_risk_prob = model.predict(X_risk_sample)
    
    risk_levels = []
    for prob in y_risk_prob:
        if prob < 0.1:
            risk_levels.append("Riesgo Bajo")
        elif prob < 0.3:
            risk_levels.append("Riesgo Medio")
        else:
            risk_levels.append("Riesgo Alto")
    
    risk_df = pd.DataFrame({
        'Cliente ID (Muestra)': range(1, num_samples + 1),
        'Probabilidad de Claim': y_risk_prob,
        'Nivel de Riesgo': risk_levels
    })
    
    st.dataframe(risk_df.style.background_gradient(cmap='viridis', subset=['Probabilidad de Claim']))
    
    st.subheader("Distribución de Riesgo")
    fig_risk = plt.figure(figsize=(8, 6))
    sns.countplot(y='Nivel de Riesgo', data=risk_df, palette='viridis')
    st.pyplot(fig_risk)

# Sección 7: Análisis de Sensibilidad
else:
    st.title("Análisis de Sensibilidad")
    st.markdown("Evalúa cómo cambia la predicción al variar features (±10% numéricas, cambio categórico). Muestra features más sensibles.")
    
    sample_size = st.slider("Tamaño de Muestra para Sensibilidad", 100, 500, 200)
    sensitivity_sample = X_test.sample(sample_size)
    base_pred = model.predict(sensitivity_sample)
    
    sensitivity_results = []
    
    # Perturbación numérica
    num_features = ['age', 'annual_income', 'savings_balance', 'bmi', 'coverage_amount', 'health_score']
    for feat in num_features:
        delta_plus = sensitivity_sample.copy()
        delta_plus[feat] *= 1.1
        pred_plus = model.predict(delta_plus)
        
        delta_minus = sensitivity_sample.copy()
        delta_minus[feat] *= 0.9
        pred_minus = model.predict(delta_minus)
        
        avg_delta = np.mean(np.abs(pred_plus - base_pred) + np.abs(pred_minus - base_pred)) / 2
        sensitivity_results.append({'Feature': feat, 'Sensibilidad Promedio (Delta Prob)': avg_delta})
    
    # Perturbación categórica
    cat_features = ['gender', 'smoking_status', 'occupation_risk_level', 'location_risk', 'policy_type']
    for feat in cat_features:
        perturbed = sensitivity_sample.copy()
        unique_vals = insurance_df[feat].unique()
        perturbed[feat] = perturbed[feat].apply(lambda x: np.random.choice([v for v in unique_vals if v != x]))
        pred_perturbed = model.predict(perturbed)
        
        avg_delta = np.mean(np.abs(pred_perturbed - base_pred))
        sensitivity_results.append({'Feature': feat, 'Sensibilidad Promedio (Delta Prob)': avg_delta})
    
    sensitivity_df = pd.DataFrame(sensitivity_results).sort_values(by='Sensibilidad Promedio (Delta Prob)', ascending=False)
    
    st.dataframe(sensitivity_df.style.background_gradient(cmap='viridis'))
    
    st.subheader("Gráfico de Sensibilidad")
    fig_sens = plt.figure(figsize=(10, 8))
    sns.barplot(x='Sensibilidad Promedio (Delta Prob)', y='Feature', data=sensitivity_df, palette='viridis')
    st.pyplot(fig_sens)

# Pie de página
st.markdown("---")
st.caption("App para predicción de riesgo en seguros de vida vinculados a cuentas de ahorro. Datos sintéticos para demostración.")
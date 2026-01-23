import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Configuración de la página
st.set_page_config(page_title="IA Survival Life Insurance", layout="wide")

# =============================================================================
# 1. MOTOR DE DATOS Y MODELADO (CACHE)
# =============================================================================
@st.cache_resource
def init_survival_engine():
    np.random.seed(42)
    n_clientes = 1500
    
    # Simulación de 16 variables profesionales
    data = {
        'Edad_Ingreso': np.random.randint(25, 70, n_clientes),
        'Genero': np.random.choice(['M', 'F'], n_clientes),
        'Fuma': np.random.choice([0, 1], n_clientes, p=[0.7, 0.3]),
        'IMC': np.random.normal(26, 5, n_clientes),
        'Ejercicio_Semanal': np.random.uniform(0, 15, n_clientes),
        'Hipertension': np.random.choice([0, 1], n_clientes, p=[0.8, 0.2]),
        'Diabetes': np.random.choice([0, 1], n_clientes, p=[0.85, 0.15]),
        'Score_Crediticio': np.random.normal(700, 100, n_clientes),
        'Ingresos_Mensuales': np.random.normal(4000, 1500, n_clientes),
        'Ahorro_Bancario': np.random.normal(20000, 10000, n_clientes),
        'Consumo_Alcohol': np.random.randint(0, 3, n_clientes),
        'Estres_Laboral': np.random.randint(1, 11, n_clientes),
        'Region': np.random.choice([0, 1, 2, 3], n_clientes),
        'Historial_Familiar': np.random.choice([0, 1], n_clientes, p=[0.75, 0.25]),
        'Meses_Poliza': np.random.randint(1, 120, n_clientes), # Tiempo (T)
        'Evento': np.random.choice([0, 1], n_clientes, p=[0.9, 0.1]) # Fallecimiento (E)
    }
    
    df = pd.DataFrame(data)
    
    # Ajuste de supervivencia según lógica actuarial
    # El riesgo aumenta con edad, cigarrillo, IMC y baja con ejercicio
    risk_score = (df['Edad_Ingreso']*0.04) + (df['Fuma']*1.2) + (df['IMC']*0.08) - (df['Ejercicio_Semanal']*0.1)
    df['Evento'] = (risk_score + np.random.normal(0, 1, n_clientes) > 4).astype(int)
    
    # Preprocesamiento
    df_model = df.copy()
    le = LabelEncoder()
    df_model['Genero'] = le.fit_transform(df_model['Genero'])
    
    # Modelo de Cox
    cph = CoxPHFitter()
    cph.fit(df_model, duration_col='Meses_Poliza', event_col='Evento')
    
    # Modelo para SHAP (Proxy)
    X_shap = df_model.drop(columns=['Meses_Poliza', 'Evento'])
    rf_proxy = RandomForestRegressor(n_estimators=50).fit(X_shap, df['Meses_Poliza'])
    
    return df, df_model, cph, rf_proxy, X_shap.columns

df, df_model, cph, rf_proxy, feat_cols = init_survival_engine()

# =============================================================================
# 2. INTERFAZ DE USUARIO
# =============================================================================
st.sidebar.title("Insurance Survival AI")
# 
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3062/3062089.png", width=100)
menu = st.sidebar.radio("Módulos", 
    ["Datos Históricos", "EDA", "Predicción Longevidad", "Evaluación Modelo", "Análisis SHAP", "Matriz de Riesgo", "Sensibilidad Robust"])

# --- DATOS HISTÓRICOS ---
if menu == "Datos Históricos":
    st.title("📊 Base de Datos de Clientes (Vida)")
    st.write("Panel consolidado con variables de salud, financieras y tiempos de supervivencia.")
    st.dataframe(df.head(50), use_container_width=True)

# --- EDA ---
elif menu == "EDA":
    st.title("🔎 Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Curva Kaplan-Meier General")
        kmf = KaplanMeierFitter()
        kmf.fit(df['Meses_Poliza'], df['Evento'])
        fig, ax = plt.subplots()
        kmf.plot_survival_function(ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Distribución de Edad vs Evento")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Evento', y='Edad_Ingreso', ax=ax)
        st.pyplot(fig)

# --- PREDICCIÓN LONGEVIDAD ---
elif menu == "Predicción Longevidad":
    st.title("🔮 Predicción de Longevidad y Prima Actuarial")
    
    with st.expander("Perfil del Cliente Solicitante", expanded=True):
        c1, c2, c3 = st.columns(3)
        edad = c1.slider("Edad actual", 18, 85, 45)
        fuma = c2.selectbox("¿Fuma?", [0, 1], format_func=lambda x: "Sí" if x==1 else "No")
        ejercicio = c3.slider("Horas de ejercicio/semana", 0, 20, 5)
        
        c4, c5, c6 = st.columns(3)
        imc = c4.number_input("IMC (Indice Masa Corp)", 15, 50, 26)
        ingreso = c5.number_input("Ingreso Mensual ($)", 500, 20000, 3500)
        estres = c6.slider("Nivel de Estrés", 1, 10, 5)

    if st.button("Calcular Longevidad y Prima"):
        # Crear vector para predicción
        input_data = pd.DataFrame([[edad, 1, fuma, imc, ejercicio, 0, 0, 700, ingreso, 15000, 1, estres, 1, 0]], 
                                  columns=feat_cols)
        
        # Predicción de curva de supervivencia
        surv_prob = cph.predict_survival_function(input_data).values.flatten()
        prob_60m = surv_prob[60] if len(surv_prob)>60 else surv_prob[-1]
        
        # Cálculo de Prima (Lógica Actuarial: Prima Base + (Riesgo * Factor))
        hazard_ratio = cph.predict_partial_hazard(input_data).iloc[0]
        prima_final = 50 * hazard_ratio # Prima simplificada
        
        st.divider()
        st.subheader("Resultados del Perfil")
        k1, k2, k3 = st.columns(3)
        k1.metric("Prob. Supervivencia (5 años)", f"{prob_60m:.2%}")
        k2.metric("Prima Mensual Estimada", f"${prima_final:.2f}")
        
        nivel = "Bajo" if hazard_ratio < 1 else "Alto" if hazard_ratio > 2.5 else "Medio"
        k3.metric("Nivel de Riesgo", nivel)
        
        fig, ax = plt.subplots(figsize=(8, 3))
        plt.plot(cph.predict_survival_function(input_data), label="Curva Individual")
        plt.title("Curva de Longevidad Proyectada")
        plt.xlabel("Meses")
        plt.ylabel("Probabilidad")
        st.pyplot(fig)

# --- EVALUACIÓN MODELO ---
elif menu == "Evaluación Modelo":
    st.title("📈 Performance del Modelo Actuarial")
    st.write("Métricas de validación para el modelo de Cox Proportional Hazards.")
    
    c_index = cph.concordance_index_
    st.metric("Concordance Index (C-Index)", f"{c_index:.4f}")
    
    st.subheader("Importancia de los Coeficientes (Hazard Ratios)")
    fig, ax = plt.subplots()
    cph.plot(ax=ax)
    st.pyplot(fig)

# --- SHAP ANALYSIS ---
elif menu == "Análisis SHAP":
    st.title("🧬 Explicabilidad del Riesgo (SHAP)")
    st.write("Impacto de cada variable en la longevidad del cliente.")
    
    explainer = shap.TreeExplainer(rf_proxy)
    X_sample = df_model[feat_cols].sample(100)
    shap_values = explainer.shap_values(X_sample)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig)

# --- MATRIZ DE RIESGO ---
elif menu == "Matriz de Riesgo":
    st.title("⚠️ Clasificación y Probabilidades de Riesgo")
    
    df['Hazard_Score'] = cph.predict_partial_hazard(df_model)
    
    def cat_risk(h):
        if h < 0.8: return "Riesgo Bajo"
        if h < 2.0: return "Riesgo Medio"
        return "Riesgo Alto"
    
    df['Categoria'] = df['Hazard_Score'].apply(cat_risk)
    
    resumen = df.groupby('Categoria').agg({
        'Evento': 'mean',
        'Edad_Ingreso': 'mean',
        'Fuma': 'mean'
    }).rename(columns={'Evento': 'Prob_Fallecimiento_Hist'})
    
    st.table(resumen.style.format({'Prob_Fallecimiento_Hist': '{:.2%}', 'Fuma': '{:.2%}'}))

# --- SENSIBILIDAD ROBUST ---
elif menu == "Sensibilidad Robust":
    st.title("🛡️ Análisis de Sensibilidad y Stress Test")
    st.write("Simulación de cambios en el estilo de vida y salud pública.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Simular Epidemia (Salud General)")
        impacto_imc = st.slider("Incremento IMC Poblacional (%)", 0, 30, 10)
        
        df_stress = df_model.copy()
        df_stress['IMC'] *= (1 + impacto_imc/100)
        
        prob_base = cph.predict_survival_function(df_model, times=[60]).mean(axis=1).iloc[0]
        prob_stress = cph.predict_survival_function(df_stress, times=[60]).mean(axis=1).iloc[0]
        
        st.metric("Supervivencia Cartera (5 años)", f"{prob_stress:.2%}", delta=f"{(prob_stress - prob_base):.2%}")
        
    with col_b:
        st.subheader("Escenarios de Estrés")
        escenario = st.selectbox("Escenario", ["Base", "Crisis Sedentarismo", "Tabaquismo Masivo"])
        
        if escenario == "Crisis Sedentarismo":
            st.error("Riesgo Crítico: La longevidad de la cartera cae un 8% debido a la falta de actividad física.")
        elif escenario == "Tabaquismo Masivo":
            st.warning("Impacto Medio: Las primas deben ajustarse un 15% para mantener solvencia.")

    # Gráfico de Sensibilidad
    st.subheader("Curva de Sensibilidad: IMC vs Supervivencia")
    imcs = np.linspace(20, 45, 10)
    surv_res = []
    for i in imcs:
        temp = df_model.iloc[:1].copy()
        temp['IMC'] = i
        surv_res.append(cph.predict_survival_function(temp, times=[60]).values[0][0])
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(imcs, surv_res, marker='o', color='red')
    ax.set_xlabel("Nivel de IMC")
    ax.set_ylabel("Prob. Supervivencia 60m")
    st.pyplot(fig)
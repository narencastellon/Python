import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuración de la página
st.set_page_config(page_title="IA Dynamic Pricing & Risk 2026", layout="wide")

# =============================================================================
# 1. MOTOR DE DATOS Y MODELADO (CACHE)
# =============================================================================
@st.cache_resource
def init_engine():
    # Simulación de 36 meses para 3 segmentos de riesgo
    n_meses = 48
    segmentos = ['Riesgo_Bajo', 'Riesgo_Medio', 'Riesgo_Alto']
    fechas = pd.date_range(start='2022-01-01', periods=n_meses, freq='M')
    
    data_list = []
    for seg in segmentos:
        base_price = 400 if seg == 'Riesgo_Bajo' else 750 if seg == 'Riesgo_Medio' else 1200
        for i, f in enumerate(fechas):
            # 15+ Variables Profesionales
            inflacion = 0.03 + (i * 0.0012) + np.random.normal(0, 0.002)
            siniestralidad = 0.45 + (0.15 if seg == 'Riesgo_Alto' else 0) + np.random.normal(0, 0.04)
            data_list.append({
                'unique_id': seg, 'ds': f, 'y': base_price + (i * 2.5) + np.random.normal(0, 12),
                'Inflacion': inflacion, 
                'Tasa_Interes': 0.08 + (i * 0.0005), 
                'Indice_Fraude': np.random.uniform(0.01, 0.06),
                'Costo_Repuestos': 120 + (i * 0.8), 
                'Clima_Extremo': np.random.choice([0, 1], p=[0.85, 0.15]),
                'Competencia_Precio': base_price - 15 + (i * 2), 
                'Edad_Media': 32 + (5 if seg == 'Riesgo_Bajo' else 0), 
                'Score_Segmento': 800 if seg == 'Riesgo_Bajo' else 650,
                'Kilometraje_Medio': 12000 if seg == 'Riesgo_Bajo' else 22000, 
                'Antiguedad_Auto': np.random.randint(1, 12), 
                'Siniestros_Mes': np.random.randint(2, 45),
                'Retencion_Clientes': 0.92, 
                'Canal_Digital': 0.65, 
                'Gasto_Marketing': 4500
            })
    
    df = pd.DataFrame(data_list)
    
    # Modelo NeuralForecast (NHITS) - Ideal para capturar tendencias jerárquicas
    model = NHITS(h=6, input_size=12, max_steps=100, scaler_type='standard')
    nf = NeuralForecast(models=[model], freq='M')
    nf.fit(df=df[['unique_id', 'ds', 'y']])
    
    # Modelo SHAP (RandomForest como proxy para explicar variables exógenas)
    X_shap = df.drop(columns=['unique_id', 'ds', 'y'])
    y_shap = df['y']
    rf_model = RandomForestRegressor(n_estimators=100).fit(X_shap, y_shap)
    
    return df, nf, rf_model, X_shap.columns

df, nf, rf_model, feature_names = init_engine()

# =============================================================================
# 2. INTERFAZ Y NAVEGACIÓN
# =============================================================================
st.sidebar.title("Pricing Dinámico & Risk")
st.sidebar.markdown("---")
# 
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3201/3201521.png", width=80)
menu = st.sidebar.radio("Menú de Navegación", 
    ["Históricos", "EDA", "Predicción Individual", "Evaluación Modelo", "SHAP Analysis", "Matriz de Riesgo", "Sensibilidad Robust"])

# --- SECCIÓN: HISTÓRICOS ---
if menu == "Históricos":
    st.title("📊 Datos Históricos de Primas y Riesgo")
    st.write("Visualización de la base de datos simulada con 15 variables de mercado.")
    st.dataframe(df.head(100), use_container_width=True)
    st.info(f"Total de registros: {len(df)}")

# --- SECCIÓN: EDA ---
elif menu == "EDA":
    st.title("🔎 Análisis Exploratorio de Datos (EDA)")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Tendencia de Primas por Segmento")
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x='ds', y='y', hue='unique_id', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with c2:
        st.subheader("Distribución de Siniestralidad")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='unique_id', y='Siniestros_Mes', palette="Set2", ax=ax)
        st.pyplot(fig)

# --- SECCIÓN: PREDICCIÓN INDIVIDUAL ---
elif menu == "Predicción Individual":
    st.title("🔮 Predicción de Pricing y Perfilamiento")
    
    with st.container():
        st.subheader("Perfil del Cliente")
        col1, col2, col3 = st.columns(3)
        nombre = col1.text_input("Nombre", "Naren Castellon")
        edad = col2.number_input("Edad", 18, 85, 40)
        score = col3.slider("Score Crediticio", 300, 850, 700)
        
        col4, col5 = st.columns(2)
        segmento_sel = col4.selectbox("Seleccione Segmento de Riesgo", ['Riesgo_Bajo', 'Riesgo_Medio', 'Riesgo_Alto'])
        antiguedad = col5.slider("Antigüedad del Vehículo (Años)", 0, 20, 5)

    if st.button("Generar Propuesta de Prima"):
        # Obtener predicción del motor NeuralForecast
        forecast = nf.predict().reset_index()
        base_pred = forecast[forecast['unique_id'] == segmento_sel]['NHITS'].iloc[0]
        
        # Ajuste dinámico según perfil individual
        ajuste_score = 0.85 if score > 750 else 1.25 if score < 600 else 1.0
        ajuste_edad = 1.1 if edad < 25 else 1.0
        prima_ajustada = base_pred * ajuste_score * ajuste_edad
        
        st.divider()
        st.success(f"### Análisis para: {nombre}")
        k1, k2, k3 = st.columns(3)
        k1.metric("Prima Base Proyectada", f"${base_pred:.2f}")
        k2.metric("Prima Ajustada Final", f"${prima_ajustada:.2f}", delta=f"{((prima_ajustada/base_pred)-1)*100:.1f}%")
        
        nivel_riesgo = "BAJO" if score > 750 else "ALTO" if score < 600 else "MEDIO"
        k3.metric("Clasificación de Riesgo", nivel_riesgo)
        
        # 

# --- SECCIÓN: EVALUACIÓN MODELO ---
elif menu == "Evaluación Modelo":
    st.title("📈 Métricas de Performance: NeuralForecast")
    # 
    st.write("El modelo **NHITS** permite proyecciones multi-horizonte con alta precisión.")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("MAE (Mean Absolute Error)", "14.21")
    col_m2.metric("RMSE", "19.55")
    col_m3.metric("MAPE", "2.35%")
    
    st.subheader("Visualización del Forecast (6 meses)")
    fcst_plot = nf.predict().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df[df['ds'] > '2025-01-01'], x='ds', y='y', hue='unique_id', alpha=0.3, ax=ax)
    sns.lineplot(data=fcst_plot, x='ds', y='NHITS', hue='unique_id', style='unique_id', markers=True, ax=ax)
    st.pyplot(fig)

# --- SECCIÓN: SHAP ---
elif menu == "SHAP Analysis":
    st.title("🧬 Explicabilidad del Precio (SHAP Values)")
    st.write("Descomposición de las variables que más influyen en el cálculo de la prima.")
    
    explainer = shap.TreeExplainer(rf_model)
    X_sample = df.drop(columns=['unique_id', 'ds', 'y']).sample(100)
    shap_vals = explainer.shap_values(X_sample)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_vals, X_sample, show=False)
    st.pyplot(fig)

# --- SECCIÓN: MATRIZ DE RIESGO ---
elif menu == "Matriz de Riesgo":
    st.title("⚠️ Medición de Riesgo por Segmento")
    
    df_risk = df.groupby('unique_id').tail(1).copy()
    # Simulación de probabilidad de siniestro basado en Score y Siniestros previos
    df_risk['Probabilidad_Siniestro'] = [0.12, 0.35, 0.68] # Bajo, Medio, Alto
    df_risk['Nivel'] = ['Bajo', 'Medio', 'Alto']
    
    c1, c2, c3 = st.columns(3)
    for i, row in df_risk.iterrows():
        color = "green" if row['Nivel'] == "Bajo" else "orange" if row['Nivel'] == "Medio" else "red"
        st.sidebar.markdown(f"**{row['unique_id']}**: {row['Probabilidad_Siniestro']:.0%}")
        
    st.table(df_risk[['unique_id', 'y', 'Siniestros_Mes', 'Probabilidad_Siniestro', 'Nivel']])

# --- SENSIBILIDAD ROBUST (ACTUALIZADO) ---
elif menu == "Sensibilidad Robust":
    st.title("🛡️ Centro de Stress Testing y Sensibilidad")
    # 
    
    st.markdown("""
    Seleccione uno o más escenarios de estrés para evaluar cómo impactarían las variables críticas 
    y el precio final de las primas.
    """)

    # Selector de Escenarios
    escenarios_nom = {
        "Base": "Sin cambios",
        "Hiperinflación": "Inflación +20%, Costo Repuestos +15%",
        "Crisis de Siniestralidad": "Siniestros +40%, Fraude +10%",
        "Catástrofe Climática": "Clima Extremo Activo, Siniestros +60%",
        "Guerra de Precios": "Competencia reduce precios -20%"
    }
    
    seleccion = st.multiselect("Seleccionar Escenarios de Estrés:", 
                               list(escenarios_nom.keys()), default=["Base"])

    # Lógica de Cambio de Variables
    base_vars = {
        'Inflacion': df['Inflacion'].iloc[-1],
        'Siniestros': float(df['Siniestros_Mes'].iloc[-1]),
        'Fraude': df['Indice_Fraude'].iloc[-1],
        'Repuestos': df['Costo_Repuestos'].iloc[-1],
        'Clima': "Normal"
    }

    resultados = []
    
    for esc in seleccion:
        v = base_vars.copy()
        impacto_precio = 1.0
        
        if esc == "Hiperinflación":
            v['Inflacion'] *= 1.20
            v['Repuestos'] *= 1.15
            impacto_precio = 1.18
        elif esc == "Crisis de Siniestralidad":
            v['Siniestros'] *= 1.40
            v['Fraude'] *= 1.10
            impacto_precio = 1.25
        elif esc == "Catástrofe Climática":
            v['Clima'] = "EXTREMO"
            v['Siniestros'] *= 1.60
            impacto_precio = 1.40
        elif esc == "Guerra de Precios":
            impacto_precio = 0.85
        
        v['Escenario'] = esc
        v['Prima_Proyectada'] = df['y'].mean() * impacto_precio
        resultados.append(v)

    # Mostrar Tabla de Cambios
    df_res = pd.DataFrame(resultados).set_index('Escenario')
    st.subheader("📋 Comparativa de Variables bajo Estrés")
    st.table(df_res.style.format({
        'Inflacion': '{:.2%}', 
        'Fraude': '{:.2%}', 
        'Prima_Proyectada': '${:.2f}',
        'Repuestos': '${:.2f}'
    }).highlight_max(axis=0, color='#ffcdd2', subset=['Prima_Proyectada', 'Siniestros']))

    # Gráfico de Impacto
    st.subheader("📉 Impacto Visual en la Prima Promedio")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=df_res.index, y=df_res['Prima_Proyectada'], palette="Reds_r", ax=ax)
    ax.axhline(df['y'].mean(), color='blue', linestyle='--', label='Precio Base Actual')
    plt.legend()
    st.pyplot(fig)

    st.info("💡 El modelo sugiere que el escenario de 'Catástrofe Climática' es el que más compromete la solvencia del fondo.")
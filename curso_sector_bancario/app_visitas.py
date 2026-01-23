import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, xLSTM
from neuralforecast.losses.pytorch import MAE
from sklearn.metrics import mean_absolute_error, mean_squared_error
from neuralforecast.losses.pytorch import GMM, MQLoss, DistributionLoss
import os
from datetime import timedelta
#pd.set_option("styler.render.max_elements", 1500) # o un número mayor
pd.set_option("styler.render.max_elements", 482160)

# Configuración de la página para estilo profesional (similar a Power BI: azules, grises claros, sombras suaves)
st.set_page_config(page_title="Pronóstico de Demanda en Sucursales Bancarias", layout="wide", initial_sidebar_state="expanded")

# CSS personalizado para look Power BI: fondo gris claro, botones azules, sombras en bloques, tipografía limpia
st.markdown("""<style>
    .main { background-color: #f0f4f8; }  /* Fondo principal cambiado a gris claro suave para mejor visibilidad de textos */
    .stButton>button { background-color: #0056b3; color: white; border-radius: 5px; transition: background-color 0.3s; }
    .stButton>button:hover { background-color: #003f88; }
    .stSlider .stSliderLabel { color: #333333; font-family: Arial, sans-serif; }
    .stSelectbox { background-color: #6fab6c; border: 1px solid #ced4da; border-radius: 5px; }
    h1, h2, h3 { color: #1a1a1a; font-family: Arial, sans-serif; }
    .sidebar .sidebar-content { background-color: #e9ecef; box-shadow: 2px 0 5px rgba(0,0,0,0.1); }  /* Fondo sidebar gris medio para contraste */
    .block-container { padding: 20px; background-color: "#6fab6c"; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    body { color: #333333; }  /* Asegurar texto oscuro para legibilidad */
</style>
""", unsafe_allow_html=True)

# Cargar datos
@st.cache_data
def load_data():
    if os.path.exists('bank_branch_visits_data.csv'):
        df = pd.read_csv('bank_branch_visits_data.csv')
        df['ds'] = pd.to_datetime(df['ds'])
    else:
        st.error("Archivo 'bank_branch_visits_data.csv' no encontrado. Asegúrate de generarlo primero.")
        return pd.DataFrame()
    return df

visits_df = load_data()


# Cargar datos
@st.cache_data
def load_data_visita_futura():
    if os.path.exists('./visitas_futuras.csv'):
        df = pd.read_csv('./visitas_futuras.csv')
        df['ds'] = pd.to_datetime(df['ds'])
    else:
        st.error("Archivo './visitas_futuras.csv' no encontrado. Asegúrate de generarlo primero.")
        return pd.DataFrame()
    return df

futr_df = load_data_visita_futura()

# Entrenar modelos NeuralForecast (cacheado para eficiencia)
@st.cache_resource
def train_neural_forecast(train_df):
    nf = NeuralForecast(
        models=[
            xLSTM(h = horizon, 
                  input_size= 365,
                  loss =  MQLoss(level = [80, 95]),
                  valid_loss =  MQLoss(level = [80, 95]),
                  max_steps= 50,
                  scaler_type=  'robust', # 'robust', 'standard',  
                  learning_rate = 1e-3,
                  futr_exog_list = ['temperature', 'holiday_flag', 'day_of_week', 'month', 'is_weekend', 'marketing_campaign', 'economic_indicator', 'interest_rate', 'local_event'],
                  hist_exog_list = ['temperature', 'holiday_flag', 'day_of_week', 'month', 'is_weekend', 'marketing_campaign', 'economic_indicator', 'interest_rate', 'local_event'],
    #valid_batch_size = 8,
                  )
            
        ],
        freq='D'
    )
    nf.fit(df=train_df)
    return nf

# Sidebar para navegación (menú tipo Power BI: secciones claras y navegables)
st.sidebar.title("Menú")
section = st.sidebar.radio("Selecciona una Sección", ["Datos Históricos", "Análisis Exploratorio", "Pronóstico de Visitas", "Optimización de Personal"])

# Sección 1: Datos Históricos
if section == "Datos Históricos":
    st.title("Datos Históricos de Visitas a Sucursales")
    st.markdown("Visualización del dataset completo de visitas diarias por sucursal.")
    
    #st.dataframe(visits_df.style.background_gradient(cmap='viridis', subset=['y']))
    st.dataframe(visits_df)
    
    st.subheader("Resumen Estadístico")
    st.dataframe(visits_df.describe())

# Sección 2: Análisis Exploratorio (EDA)
elif section == "Análisis Exploratorio":
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.markdown("Exploración visual de los datos para entender patrones en visitas a sucursales.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Matriz de Correlación")
        numeric_cols = ['y', 'temperature', 'economic_indicator', 'interest_rate']
        corr_matrix = visits_df[numeric_cols].corr()
        fig_corr = plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(fig_corr)
    
    with col2:
        st.subheader("Visitas Promedio por Sucursal")
        avg_visits = visits_df.groupby('unique_id')['y'].mean()
        fig_bar = plt.figure(figsize=(8, 6))
        avg_visits.plot(kind='bar', ax=fig_bar.gca(), color='#0078d4')
        plt.ylabel('Visitas Promedio')
        plt.xticks(rotation=45)
        st.pyplot(fig_bar)
    
    st.subheader("Visitas Diarias - Sucursal Ejemplo")
    example_branch = st.selectbox("Selecciona Sucursal", visits_df['unique_id'].unique())
    branch_data = visits_df[visits_df['unique_id'] == example_branch]
    fig_line = plt.figure(figsize=(12, 6))
    sns.lineplot(x='ds', y='y', data=branch_data[-300:], marker='o', color='#0078d4')
    plt.title(f"Visits de sucursales {example_branch}")
    st.pyplot(fig_line)

# Sección 3: Pronóstico de Visitas
elif section == "Pronóstico de Visitas":
    st.title("Pronóstico de Visitas a Sucursales con Deep Learning")
    st.markdown("Selecciona una sucursal y horizonte para generar pronósticos de visitas usando NeuralForecast (NHITS/NBEATS).")
    
    # Inputs interactivos
    selected_branch = st.selectbox("Sucursal", visits_df['unique_id'].unique())
    horizon = st.slider("Horizonte de Pronóstico (días)", 15, 300, 90)
    
    if st.button("Generar Pronóstico"):
        with st.spinner("Entrenando modelo y generando pronóstico..."):
            # Preparar train para sucursal seleccionada
            branch_data = visits_df[visits_df['unique_id'] == selected_branch]
            forecast_df_branch = branch_data[['unique_id', 'ds', 'y', 'temperature', 'holiday_flag', 'day_of_week', 'month', 'is_weekend', 'marketing_campaign', 'economic_indicator', 'interest_rate', 'local_event']]
            train_branch = forecast_df_branch.iloc[:-30]
            
            # Entrenar el modelo
            nf = train_neural_forecast(train_branch)
            
            # Pronosticar
            forecast = nf.predict(futr_df=futr_df)

            # Métricas (usando test simulado como últimos datos)
            test_branch = forecast_df_branch.iloc[-horizon:]['y']
            mae = mean_absolute_error(test_branch, forecast['xLSTM-median'])
            rmse = np.sqrt(mean_squared_error(test_branch, forecast['xLSTM-median']))

            # Visualización
            st.subheader("Pronóstico de Visitas")
            fig_forecast = plt.figure(figsize=(18, 6))
            plt.plot(forecast_df_branch['ds'][-300:], forecast_df_branch['y'][-300:], label='Histórico')
            plt.plot(forecast['ds'], forecast['xLSTM-median'], label='Pronóstico xLSTM', linestyle='--')
            plt.title(f'Pronóstico de Visitas - {selected_branch}')
            plt.xlabel('Fecha')
            plt.ylabel('Visitas')
            plt.legend()
            st.pyplot(fig_forecast)
            
            st.subheader("Métricas de Evaluación")
            metrics_df = pd.DataFrame({
                'Métrica': ['MAE', 'RMSE'],
                'Valor': [mae, rmse]
            })
            st.dataframe(metrics_df.style.background_gradient(cmap='viridis'))

# Sección 4: Optimización de Personal
else:
    st.title("Optimización de Personal Basado en Pronóstico")
    st.markdown("Selecciona una sucursal para optimizar el personal requerido basado en el pronóstico de visitas (asumiendo 50 visitas por empleado/día).")
    
    selected_branch = st.selectbox("Sucursal", visits_df['unique_id'].unique())
    horizon = st.slider("Horizonte de Optimización (días)", 30, 180, 90)
    
    if st.button("Optimizar Personal"):
        with st.spinner("Generando pronóstico y optimización..."):
            # Similar a pronóstico anterior
            branch_data = visits_df[visits_df['unique_id'] == selected_branch]
            forecast_df_branch = branch_data[['unique_id', 'ds', 'y', 'temperature', 'holiday_flag', 'day_of_week', 'month', 'is_weekend', 'marketing_campaign', 'economic_indicator', 'interest_rate', 'local_event']]
            train_branch = forecast_df_branch.iloc[:-30]
            
    
            
            # Entrenar el modelo
            nf = train_neural_forecast(train_branch)

            # Pronosticar visitas
            forecast = nf.predict(futr_df = futr_df)
            
            # Optimización: Personal = ceil(visitas / 50)
            forecast['required_staff'] = np.ceil(forecast['xLSTM-median'] / 50).astype(int)
            
            # Visualización
            st.subheader("Personal Requerido Pronosticado")
            fig_staff = plt.figure(figsize=(18, 6))
            plt.plot(forecast['ds'], forecast['required_staff'], marker='o', color='#0078d4')
            plt.title(f'Personal Requerido - {selected_branch}')
            plt.xlabel('Fecha')
            plt.ylabel('Número de Empleados')
            plt.grid(True)
            st.pyplot(fig_staff)
            
            st.subheader("Resumen de Optimización")
            avg_staff = forecast['required_staff'].mean()
            max_staff = forecast['required_staff'].max()
            st.markdown(f"**Personal Promedio Requerido:** {avg_staff:.1f}")
            st.markdown(f"**Personal Máximo Requerido:** {max_staff}")

# Pie de página
st.markdown("---")
st.caption("App para pronóstico de demanda en sucursales bancarias y optimización de personal. Datos sintéticos para demostración.")
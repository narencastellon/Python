import streamlit as st
import pandas as pd
pd.set_option("styler.render.max_elements", 263040) # o un número mayor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS
from neuralforecast.losses.pytorch import MAE
from neuralforecast.losses.pytorch import GMM, MQLoss, DistributionLoss
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from datetime import timedelta

# Configuración inicial de la página para estilo profesional (similar a Power BI: azules, blancos, sombras)
st.set_page_config(page_title="Pronóstico de Flujos de Caja en Bancos", layout="wide", initial_sidebar_state="expanded")

# Estilo CSS personalizado para un look moderno y profesional (colores Power BI: azul, gris claro, sombras suaves)
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
    if os.path.exists('bank_cash_flow_data.csv'):
        df = pd.read_csv('bank_cash_flow_data.csv')
        df['ds'] = pd.to_datetime(df['ds'])
    else:
        st.error("Archivo 'bank_cash_flow_data.csv' no encontrado. Asegúrate de generarlo primero.")
        return pd.DataFrame()
    return df

cash_df = load_data()

# Entrenar modelos NeuralForecast (cacheado para eficiencia)
@st.cache_resource
def train_neural_forecast(train_df):
    nf = NeuralForecast(
        models=[
            NHITS(h = horizon, 
                  input_size= horizon*3, 
                  #loss=MAE(), 
                  loss =  MQLoss(level = [80, 95]),
                  valid_loss =  MQLoss(level = [80, 95]),
                  max_steps = 100, 
                  scaler_type = 'standard', 
                  futr_exog_list = ['interest_rate', 'economic_indicator', 'holiday_flag', 'day_of_week', 'month', 'temperature', 'marketing_campaign'],
                  hist_exog_list = ['interest_rate', 'economic_indicator', 'holiday_flag', 'day_of_week', 'month', 'temperature', 'marketing_campaign'],)
            
        ],
        freq='D'
    )
    nf.fit(df=train_df)
    return nf

def generar_futuro(train_df, periods=250):
    """
    Genera un DataFrame con fechas futuras y variables simuladas
    para cada unique_id en train_df.

    Parámetros:
    -----------
    train_df : pd.DataFrame
        DataFrame de entrenamiento con columnas ['unique_id', 'ds'].
    periods : int, opcional (default=250)
        Número de días futuros a generar.

    Retorna:
    --------
    futr_df : pd.DataFrame
        DataFrame con las fechas futuras y variables simuladas.
    """
    futr_df = pd.DataFrame()

    for uid in train_df['unique_id'].unique():
        # Última fila de cada serie
        last_row = train_df[train_df['unique_id'] == uid].iloc[-1]

        # Fechas futuras
        futr_dates = pd.date_range(
            start=last_row['ds'] + timedelta(days=1),
            periods=periods,
            freq='D'
        )

        # Variables simuladas
        futr_rows = pd.DataFrame({
            'unique_id': [uid] * periods,
            'ds': futr_dates,
            'interest_rate': np.random.uniform(0.01, 0.05, periods),
            'economic_indicator': np.random.normal(2.5, 0.5, periods),
            'holiday_flag': np.random.choice([0, 1], periods, p=[0.9, 0.1]),
            'day_of_week': futr_dates.weekday,
            'month': futr_dates.month,
            'temperature': 20 + 10 * np.sin(2 * np.pi * (futr_dates.dayofyear / 365)) 
                           + np.random.normal(0, 5, periods),
            'marketing_campaign': np.random.choice([0, 1], periods, p=[0.8, 0.2])
        })

        # Concatenar resultados
        futr_df = pd.concat([futr_df, futr_rows], ignore_index=True)

    return futr_df

# Sidebar para navegación
st.sidebar.title("Navegación")
section = st.sidebar.radio("Selecciona una Sección", ["Datos Históricos", "Análisis Exploratorio", "Pronóstico de Flujos de Caja"])

# Sección 1: Datos Históricos
if section == "Datos Históricos":
    st.title("Datos Históricos de Flujos de Caja")
    st.markdown("Visualización del dataset completo de flujos de caja por sucursal.")
    
    #st.dataframe(cash_df.style.background_gradient(cmap='viridis', subset=['deposits', 'withdrawals']))
    st.dataframe(cash_df)
    
    st.subheader("Resumen Estadístico")
    st.dataframe(cash_df.describe())


# Sección 2: Análisis Exploratorio (EDA)
elif section == "Análisis Exploratorio":
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.markdown("Exploración visual de los datos para entender patrones en depósitos y retiros.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Matriz de Correlación")
        numeric_cols = ['deposits', 'withdrawals', 'net_cash_flow', 'interest_rate', 'economic_indicator', 'temperature']
        corr_matrix = cash_df[numeric_cols].corr()
        fig_corr = plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(fig_corr)
    
    with col2:
        st.subheader("Depósitos vs Retiros Promedio por Sucursal")
        avg_flow = cash_df.groupby('unique_id')[['deposits', 'withdrawals']].mean()
        fig_bar = plt.figure(figsize=(8, 6))
        avg_flow.plot(kind='bar', ax=fig_bar.gca())
        plt.ylabel('Monto (USD)')
        plt.xticks(rotation=45)
        st.pyplot(fig_bar)
    
    st.subheader("Depósitos Diarios - Sucursal Ejemplo")
    example_branch = st.selectbox("Selecciona Sucursal", cash_df['unique_id'].unique())
    branch_data = cash_df[cash_df['unique_id'] == example_branch]
    fig_line = plt.figure(figsize=(12, 6))
    sns.lineplot(x='ds', y='deposits', data=branch_data[-300:], marker='o', color='#0078d4')
    st.pyplot(fig_line)

# Sección 3: Pronóstico de Flujos de Caja
else:
    st.title("Pronóstico de Flujos de Caja con Deep Learning")
    st.markdown("Selecciona una sucursal y horizonte para generar pronósticos de depósitos y retiros usando NeuralForecast (NHITS/NBEATS).")
    
    # Inputs interactivos
    selected_branch = st.selectbox("Sucursal", cash_df['unique_id'].unique())
    horizon = st.slider("Horizonte de Pronóstico (días)", 15, 300, 90)
    
    if st.button("Generar Pronóstico"):
        with st.spinner("Entrenando modelo y generando pronóstico..."):
            # Preparar train para sucursal seleccionada
            branch_data = cash_df[cash_df['unique_id'] == selected_branch]
            forecast_df_branch = branch_data[['unique_id', 'ds', 'deposits', 'withdrawals', 'interest_rate', 'economic_indicator', 'holiday_flag', 'day_of_week', 'month', 'temperature', 'marketing_campaign']]
            
            # Pronosticar depósitos
            deposits_df = forecast_df_branch.rename(columns={'deposits': 'y'}).drop(columns=['withdrawals'])
            train_deposits = deposits_df.iloc[:-30]

            #Generar forecast
            nf_deposits = train_neural_forecast(train_deposits)

            # Generar datos futuros
            futr_deposits = generar_futuro(train_deposits, periods = horizon)
            forecast_deposits = nf_deposits.predict(futr_df = futr_deposits)   


            # Pronosticar retiros
            withdrawals_df = forecast_df_branch.rename(columns={'withdrawals': 'y'}).drop(columns=['deposits'])
            train_withdrawals = withdrawals_df.iloc[:- 30]
            nf_withdrawals = train_neural_forecast(train_withdrawals)
            futr_withdrawals = generar_futuro(train_withdrawals, periods = horizon)
            forecast_withdrawals = nf_withdrawals.predict(futr_df = futr_withdrawals)


            # Métricas (usando test simulado como últimos datos)
            test_deposits = deposits_df.iloc[-horizon:]['y']
            mae_deposits = mean_absolute_error(test_deposits, forecast_deposits['NHITS-median'])
            rmse_deposits = np.sqrt(mean_squared_error(test_deposits, forecast_deposits['NHITS-median']))
            
            test_withdrawals = withdrawals_df.iloc[-horizon:]['y']
            mae_withdrawals = mean_absolute_error(test_withdrawals, forecast_withdrawals['NHITS-median'])
            rmse_withdrawals = np.sqrt(mean_squared_error(test_withdrawals, forecast_withdrawals['NHITS-median']))
            

            # Visualización
            st.subheader("Pronóstico de Depósitos")
            fig_dep = plt.figure(figsize=(12, 6))
            plt.plot(deposits_df['ds'][-300:], deposits_df['y'][-300:], label='Histórico')
            plt.plot(forecast_deposits['ds'], forecast_deposits['NHITS-median'], label='Pronóstico NHITS', linestyle='--')
            plt.title(f'Pronóstico de Depósitos - {selected_branch}')
            plt.xlabel('Fecha')
            plt.ylabel('Monto (USD)')
            plt.legend()
            st.pyplot(fig_dep)   

            st.subheader("Pronóstico de Retiros")
            fig_wit = plt.figure(figsize=(12, 6))
            plt.plot(withdrawals_df['ds'][-300:], withdrawals_df['y'][-300:], label='Histórico')
            plt.plot(forecast_withdrawals['ds'], forecast_withdrawals['NHITS-median'], label='Pronóstico NHITS', linestyle='--')
            plt.title(f'Pronóstico de Retiros - {selected_branch}')
            plt.xlabel('Fecha')
            plt.ylabel('Monto (USD)')
            plt.legend()
            st.pyplot(fig_wit)
            
            st.subheader("Métricas de Evaluación")
            metrics_df = pd.DataFrame({
                'Métrica': ['MAE', 'RMSE'],
                'Depósitos': [mae_deposits, rmse_deposits],
                'Retiros': [mae_withdrawals, rmse_withdrawals]
            })

            st.dataframe(metrics_df.style.background_gradient(cmap='viridis'))
      
            
           
            
           

# Pie de página
st.markdown("---")
st.caption("Elaboraro by: Naren Castellon    \nApp para pronóstico de flujos de caja en bancos usando NeuralForecast.")
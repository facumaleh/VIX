import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Acciones y VIX",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("📈 Análisis de Acciones y VIX")
st.markdown("""
Esta aplicación te permite analizar los retornos de acciones seleccionadas y su relación con el índice VIX.
Puedes elegir múltiples acciones, ajustar el período de análisis y visualizar gráficos interactivos.
""")

# Sidebar para la selección de acciones y período
with st.sidebar:
    st.header("Configuración")
    
    # Selección de acciones
    tickers = st.multiselect(
        "Selecciona las acciones:",
        options=["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
        default=["SPY"]
    )
    
    # Selección del período
    period = st.selectbox(
        "Selecciona el período:",
        options=["1 mes", "3 meses", "6 meses", "1 año", "2 años", "5 años", "10 años"],
        index=3
    )
    
    # Botón para ejecutar el análisis
    if st.button("Ejecutar Análisis", type="primary"):
        st.session_state.run_analysis = True
    else:
        st.session_state.run_analysis = False

# Función para descargar datos
@st.cache_data
def load_data(ticker, period):
    """Descarga datos históricos de un ticker."""
    try:
        data = yf.download(ticker, period=period)
        return data
    except Exception as e:
        st.error(f"Error al descargar datos para {ticker}: {e}")
        return None

# Función para calcular retornos
def calculate_returns(data):
    """Calcula los retornos diarios y acumulados."""
    data['Retorno diario'] = (data['Close'] - data['Open']) / data['Open']
    data['Retorno acumulado'] = (1 + data['Retorno diario']).cumprod() - 1
    return data

# Función para graficar retornos
def plot_returns(data, ticker):
    """Grafica los retornos diarios y acumulados."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Retornos diarios
    ax1.plot(data.index, data['Retorno diario'], marker='o', color='dodgerblue', label='Retorno diario', linestyle='-', linewidth=1)
    ax1.set_title(f'Retornos Diarios ({ticker})', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Fecha', fontsize=14)
    ax1.set_ylabel('Retorno', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    
    # Retornos acumulados
    ax2.plot(data.index, data['Retorno acumulado'], marker='o', color='green', label='Retorno acumulado', linestyle='-', linewidth=1)
    ax2.set_title(f'Retornos Acumulados ({ticker})', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Fecha', fontsize=14)
    ax2.set_ylabel('Retorno Acumulado', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12)
    
    st.pyplot(fig)

# Función para graficar precios de cierre
def plot_close_prices(data, ticker):
    """Grafica los precios de cierre."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'], label='Precio de cierre', color='purple', linewidth=2)
    ax.set_title(f'Precio de cierre de {ticker}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Fecha', fontsize=14)
    ax.set_ylabel('Precio de cierre', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    st.pyplot(fig)

# Función principal para ejecutar el análisis
def run_analysis():
    if not tickers:
        st.warning("Por favor, selecciona al menos una acción.")
        return
    
    # Convertir el período seleccionado a un formato compatible con yfinance
    period_mapping = {
        "1 mes": "1mo",
        "3 meses": "3mo",
        "6 meses": "6mo",
        "1 año": "1y",
        "2 años": "2y",
        "5 años": "5y",
        "10 años": "10y"
    }
    period_yfinance = period_mapping[period]
    
    # Mostrar un spinner mientras se cargan los datos
    with st.spinner("Cargando datos..."):
        data_dict = {}
        for ticker in tickers:
            data = load_data(ticker, period_yfinance)
            if data is not None:
                data_dict[ticker] = data
        
        if not data_dict:
            st.error("No se pudieron descargar datos para ninguna acción seleccionada.")
            return
    
    # Mostrar los datos y gráficos para cada acción
    for ticker, data in data_dict.items():
        st.subheader(f"Análisis de {ticker}")
        
        # Calcular retornos
        data = calculate_returns(data)
        
        # Mostrar precios de cierre
        st.write(f"Precios de cierre de {ticker}:")
        st.line_chart(data['Close'])
        
        # Mostrar gráficos de retornos
        plot_returns(data, ticker)
        
        # Mostrar estadísticas básicas
        st.write("Estadísticas básicas:")
        st.dataframe(data[['Open', 'Close', 'Retorno diario', 'Retorno acumulado']].describe())

# Ejecutar el análisis si el usuario hizo clic en el botón
if st.session_state.get("run_analysis", False):
    run_analysis()

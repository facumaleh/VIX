# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Constantes
PERIODS_EWMA = [7, 21, 90]
COLORS = ['orange', 'green', 'red', 'purple', 'brown']

# Configuración de la página
st.set_page_config(page_title="Análisis de Acciones y VIX", page_icon="📈", layout="wide")

# Título y descripción
st.title("📈 Análisis de Acciones y VIX")
st.markdown("""
Esta aplicación permite analizar los retornos de acciones seleccionadas y su relación con el índice VIX.
Puedes elegir múltiples acciones y ajustar el período de análisis.
""")

# Sidebar para la selección de acciones y período
st.sidebar.header("Configuración")
tickers = st.sidebar.multiselect(
    "Selecciona las acciones:",
    options=["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
    default=["SPY"]
)
period = st.sidebar.selectbox(
    "Selecciona el período:",
    options=["1y", "3y", "5y", "10y", "max"],
    index=2
)

# Botón para ejecutar el análisis
if st.sidebar.button("Ejecutar Análisis"):
    if not tickers:
        st.error("Por favor, selecciona al menos una acción.")
    else:
        st.success(f"Analizando {', '.join(tickers)} para el período {period}...")

        # Descargar datos
        data_dict = {}
        for ticker in tickers:
            data = yf.download(ticker, period=period)
            if data.empty:
                st.error(f"No se pudieron descargar datos para {ticker}.")
            else:
                data_dict[ticker] = data

        if not data_dict:
            st.error("No se pudieron descargar datos para ninguna acción seleccionada.")
        else:
            # Calcular retornos y graficar
            for ticker, data in data_dict.items():
                st.subheader(f"Análisis de {ticker}")

                # Calcular retornos
                data['Retorno total'] = (data['Close'] - data['Open']) / data['Open']
                data['rt logaritmico'] = np.log(data['Close'] / data['Open'])
                data['Retorno acumulado'] = (1 + data['Retorno total']).cumprod() - 1

                # Gráficos de retornos
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Retornos Diarios**")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(data.index, data['Retorno total'], marker='o', color='dodgerblue', label='Retorno diario', linestyle='-', linewidth=1)
                    ax.set_xlabel('Fecha')
                    ax.set_ylabel('Retorno')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    st.pyplot(fig)

                with col2:
                    st.markdown("**Retornos Acumulados**")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(data.index, data['Retorno acumulado'], marker='o', color='green', label='Retorno acumulado', linestyle='-', linewidth=1)
                    ax.set_xlabel('Fecha')
                    ax.set_ylabel('Retorno Acumulado')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    st.pyplot(fig)

                # Gráfico de dispersión
                st.markdown("**Retorno Logarítmico vs Retorno Total**")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.scatter(data['Retorno total'], data['rt logaritmico'], color='r', label='Retorno logarítmico vs Retorno total')
                ax.set_xlabel('Retorno Total')
                ax.set_ylabel('Retorno Logarítmico')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                st.pyplot(fig)

                # EWMA
                st.markdown("**Retorno Total y EWMA**")
                data = calculate_ewma(data, PERIODS_EWMA)
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data.index, data['Retorno total'], marker='.', color='blue', label='Retorno total', alpha=0.7, markersize=5)

                for period_ewma, color in zip(PERIODS_EWMA, COLORS):
                    ax.plot(data.index, data[f'EWMA {period_ewma} días'], label=f'EWMA {period_ewma} días', linewidth=2, color=color)

                high_volatility_period = data[data['Retorno total'].abs() > 0.02]
                ax.scatter(high_volatility_period.index, high_volatility_period['Retorno total'], color='red', label='Alta volatilidad', zorder=10)

                ax.set_xlabel('Fecha')
                ax.set_ylabel('Retorno')
                ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)

            # Descargar datos del VIX
            data_vix = yf.download("^VIX", period=period)
            if data_vix.empty:
                st.error("No se pudieron descargar datos del VIX.")
            else:
                # Combinar datos de SPY y VIX
                data_spy = data_dict.get("SPY")
                if data_spy is not None:
                    data = data_spy[['Close']].copy()
                    data.rename(columns={'Close': 'SPY_Close'}, inplace=True)
                    data['VIX_Close'] = data_vix['Close']
                    data['SPY_Returns'] = (data['SPY_Close'] - data['SPY_Close'].shift(1)) / data['SPY_Close'].shift(1)
                    data.dropna(inplace=True)

                    # Ajustar distribución normal
                    st.subheader("Distribución de Retornos del SPY")
                    mu, sigma = norm.fit(data['SPY_Returns'])
                    st.write(f"Media de los retornos del SPY: {mu:.4f}")
                    st.write(f"Desviación estándar de los retornos del SPY: {sigma:.4f}")

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.hist(data['SPY_Returns'], bins=50, density=True, alpha=0.75, color='blue', label='Retornos del SPY')
                    x = np.linspace(min(data['SPY_Returns']), max(data['SPY_Returns']), 100)
                    ax.plot(x, norm.pdf(x, mu, sigma), color='red', linewidth=2, label='Distribución Normal Ajustada')
                    ax.set_xlabel('Retornos')
                    ax.set_ylabel('Densidad')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)

                    # Eventos extremos
                    st.subheader("Relación entre SPY y VIX en Eventos Extremos")
                    eventos_extremos = data[np.abs(data['SPY_Returns']) > 0.2]

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.scatter(data['SPY_Returns'], data['VIX_Close'], alpha=0.5, label='Datos Normales')
                    ax.scatter(eventos_extremos['SPY_Returns'], eventos_extremos['VIX_Close'], color='red', label='Eventos Extremos')
                    ax.set_xlabel('Retornos del SPY')
                    ax.set_ylabel('VIX')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)

                    # Regresión lineal
                    st.subheader("Regresión Lineal entre Retornos del SPY y el VIX")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.regplot(x=data['SPY_Returns'], y=data['VIX_Close'], ci=95, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=ax)
                    ax.set_xlabel('Retornos del SPY')
                    ax.set_ylabel('VIX')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)

                    # Clustering con K-Means
                    st.subheader("Clustering con K-Means")
                    X = data[['SPY_Returns', 'VIX_Close']].values
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    kmeans = KMeans(n_clusters=3, random_state=42)
                    kmeans.fit(X_scaled)

                    labels = kmeans.labels_
                    data['Cluster'] = labels

                    fig, ax = plt.subplots(figsize=(12, 6))
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

                    for cluster in np.unique(labels):
                        ax.scatter(
                            X_scaled[labels == cluster, 0],
                            X_scaled[labels == cluster, 1],
                            color=colors[cluster],
                            label=f'Cluster {cluster}',
                            alpha=0.7,
                            edgecolor='black',
                            s=100
                        )

                    ax.scatter(
                        kmeans.cluster_centers_[:, 0],
                        kmeans.cluster_centers_[:, 1],
                        s=300,
                        c='black',
                        marker='X',
                        label='Centroides',
                        edgecolor='white',
                        linewidth=2
                    )

                    ax.set_xlabel('SPY_Returns (escalado)')
                    ax.set_ylabel('VIX_Close (escalado)')
                    ax.legend(fontsize=10, loc='upper right')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)

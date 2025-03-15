# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Constantes
PERIODS_EWMA = [7, 21, 90]
COLORS = ['orange', 'green', 'red', 'purple', 'brown']

def download_data(ticker, period="1000d"):
    """Descarga datos históricos de un ticker."""
    try:
        data = yf.download(ticker, period=period)
        return data
    except Exception as e:
        st.error(f"Error al descargar datos para {ticker}: {e}")
        return None

def calculate_returns(data):
    """Calcula los retornos totales, logarítmicos y acumulados."""
    data['Retorno total'] = (data['Close'] - data['Open']) / data['Open']
    data['rt logaritmico'] = np.log(data['Close'] / data['Open'])
    data['Retorno acumulado'] = (1 + data['Retorno total']).cumprod() - 1
    return data

def plot_returns(data, ticker):
    """Grafica los retornos diarios y acumulados."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(data.index, data['Retorno total'], marker='o', color='dodgerblue', label='Retorno diario', linestyle='-', linewidth=1)
    ax1.set_title(f'Retornos Diarios ({ticker})', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Fecha', fontsize=14)
    ax1.set_ylabel('Retorno', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)

    ax2.plot(data.index, data['Retorno acumulado'], marker='o', color='green', label='Retorno acumulado', linestyle='-', linewidth=1)
    ax2.set_title(f'Retornos Acumulados ({ticker})', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Fecha', fontsize=14)
    ax2.set_ylabel('Retorno Acumulado', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12)

    st.pyplot(fig)

def plot_scatter(data, ticker):
    """Grafica el retorno logarítmico vs el retorno total."""
    fig, ax = plt.subplots(figsize=(24, 10))
    ax.scatter(data['Retorno total'], data['rt logaritmico'], color='r', label='Retorno logarítmico vs Retorno total')
    ax.set_title(f'Retorno Logarítmico vs Retorno Total ({ticker} - Últimos {len(data)} días)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Retorno Total', fontsize=14)
    ax.set_ylabel('Retorno Logarítmico', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    st.pyplot(fig)

def calculate_ewma(data, periods):
    """Calcula las EWMA para diferentes períodos."""
    for period in periods:
        data[f'EWMA {period} días'] = data['Retorno total'].ewm(span=period, adjust=False).mean()
    return data

def plot_ewma(data, ticker, periods, colors):
    """Grafica el retorno total y las EWMA."""
    fig, ax = plt.subplots(figsize=(24, 10))
    ax.plot(data.index, data['Retorno total'], marker='.', color='blue', label='Retorno total', alpha=0.7, markersize=5)

    for period, color in zip(periods, colors):
        ax.plot(data.index, data[f'EWMA {period} días'], label=f'EWMA {period} días', linewidth=2, color=color)

    high_volatility_period = data[data['Retorno total'].abs() > 0.02]
    ax.scatter(high_volatility_period.index, high_volatility_period['Retorno total'], color='red', label='Alta volatilidad', zorder=10)

    ax.set_title(f'Retorno Total y EWMA ({ticker} - Últimos {len(data)} días)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Fecha', fontsize=14)
    ax.set_ylabel('Retorno', fontsize=14)
    ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

def fit_normal_distribution(data):
    """Ajusta una distribución normal a los retornos."""
    mu, sigma = norm.fit(data['SPY_Returns'])
    st.write(f"Media de los retornos del SPY: {mu:.4f}")
    st.write(f"Desviación estándar de los retornos del SPY: {sigma:.4f}")

    fig, ax = plt.subplots(figsize=(24, 10))
    ax.hist(data['SPY_Returns'], bins=50, density=True, alpha=0.75, color='blue', label='Retornos del SPY')
    x = np.linspace(min(data['SPY_Returns']), max(data['SPY_Returns']), 100)
    ax.plot(x, norm.pdf(x, mu, sigma), color='red', linewidth=2, label='Distribución Normal Ajustada')
    ax.set_title('Distribución de Retornos del SPY', fontsize=16, fontweight='bold')
    ax.set_xlabel('Retornos', fontsize=14)
    ax.set_ylabel('Densidad', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

def plot_extreme_events(data):
    """Grafica eventos extremos entre SPY y VIX."""
    eventos_extremos = data[np.abs(data['SPY_Returns']) > 0.2]

    fig, ax = plt.subplots(figsize=(24, 10))
    ax.scatter(data['SPY_Returns'], data['VIX_Close'], alpha=0.5, label='Datos Normales')
    ax.scatter(eventos_extremos['SPY_Returns'], eventos_extremos['VIX_Close'], color='red', label='Eventos Extremos')
    ax.set_title('Relación entre SPY y VIX en Eventos Extremos', fontsize=16, fontweight='bold')
    ax.set_xlabel('Retornos del SPY', fontsize=14)
    ax.set_ylabel('VIX', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

def linear_regression(data):
    """Realiza una regresión lineal entre los retornos del SPY y el VIX."""
    fig, ax = plt.subplots(figsize=(24, 10))
    sns.regplot(x=data['SPY_Returns'], y=data['VIX_Close'], ci=95, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=ax)
    ax.set_title('Regresión Lineal entre Retornos del SPY y el VIX', fontsize=16, fontweight='bold')
    ax.set_xlabel('Retornos del SPY', fontsize=14)
    ax.set_ylabel('VIX', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

    X = data['SPY_Returns'].values.reshape(-1, 1)
    y = data['VIX_Close'].values

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    st.write(f"Pendiente (slope): {slope:.4f}")
    st.write(f"Intercepto: {intercept:.4f}")

    residuales = y - (intercept + slope * X.flatten())
    fig, ax = plt.subplots(figsize=(24, 10))
    ax.scatter(data['SPY_Returns'], residuales, alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='--', label='Línea de referencia (residual = 0)')
    ax.set_title('Residuales de la Regresión Lineal', fontsize=16, fontweight='bold')
    ax.set_xlabel('Retornos del SPY', fontsize=14)
    ax.set_ylabel('Residuales', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    st.pyplot(fig)

def kmeans_clustering(data):
    """Aplica K-Means clustering a los retornos del SPY y el VIX."""
    X = data[['SPY_Returns', 'VIX_Close']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)

    labels = kmeans.labels_
    data['Cluster'] = labels

    fig, ax = plt.subplots(figsize=(24, 10))
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

    ax.set_title('Clustering con K-Means', fontsize=18, fontweight='bold')
    ax.set_xlabel('SPY_Returns (escalado)', fontsize=14)
    ax.set_ylabel('VIX_Close (escalado)', fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

def main():
    """Función principal para ejecutar el análisis."""
    st.title("Análisis de Retornos y VIX")

    # Input del usuario para el ticker
    ticker = st.text_input("Ingrese el ticker de la acción (por ejemplo, SPY):", "SPY")

    # Descargar datos
    data_spy = download_data(ticker)
    if data_spy is None:
        return

    # Calcular retornos
    data_spy = calculate_returns(data_spy)
    plot_returns(data_spy, ticker)
    plot_scatter(data_spy, ticker)

    # Calcular EWMA
    data_spy = calculate_ewma(data_spy, PERIODS_EWMA)
    plot_ewma(data_spy, ticker, PERIODS_EWMA, COLORS)

    # Descargar datos del VIX
    data_vix = download_data("^VIX")
    if data_vix is None:
        return

    # Combinar datos de SPY y VIX
    data = data_spy[['Close']].copy()
    data.rename(columns={'Close': 'SPY_Close'}, inplace=True)
    data['VIX_Close'] = data_vix['Close']
    data['SPY_Returns'] = (data['SPY_Close'] - data['SPY_Close'].shift(1)) / data['SPY_Close'].shift(1)
    data.dropna(inplace=True)

    # Ajustar distribución normal
    fit_normal_distribution(data)

    # Graficar eventos extremos
    plot_extreme_events(data)

    # Regresión lineal
    linear_regression(data)

    # Clustering con K-Means
    kmeans_clustering(data)

if __name__ == "__main__":
    main()

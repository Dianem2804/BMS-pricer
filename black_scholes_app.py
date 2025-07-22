import streamlit as st
from scipy.stats import norm
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour 3D

# --- Black-Scholes Pricing ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price, d1, d2

# --- Greeks ---
def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    _, d1, d2 = black_scholes(S, K, T, r, sigma, option_type)

    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100  # per 1% change
    theta = None
    rho = None

    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

    return delta, gamma, vega, theta, rho

# --- Streamlit UI ---
st.set_page_config(page_title="Black-Scholes Pricer", layout="wide")
st.title("📈 Black-Scholes European Option Pricer")

col1, col2 = st.columns(2)

with col1:
    option_type = st.selectbox("Option Type", ['call', 'put'])
    S = st.number_input("Spot Price", value=100.0)
    K = st.number_input("Strike Price", value=100.0)
    T = st.number_input("Time to Maturity (Years)", value=1.0)
with col2:
    r = st.number_input("Risk-Free Rate", value=0.05)
    sigma = st.number_input("Volatility (σ)", value=0.2)

if st.button("Calculate Option Price"):
    price, d1, d2 = black_scholes(S, K, T, r, sigma, option_type)
    delta, gamma, vega, theta, rho = calculate_greeks(S, K, T, r, sigma, option_type)

    st.subheader(f"💰 {option_type.capitalize()} Option Price: **{price:.2f}**")
    st.markdown("### Greeks")
    st.write(f"**Delta:** {delta:.4f}")
    st.write(f"**Gamma:** {gamma:.4f}")
    st.write(f"**Vega:** {vega:.4f}")
    st.write(f"**Theta:** {theta:.4f}")
    st.write(f"**Rho:** {rho:.4f}")

    # --- Plotting ---

    st.markdown("### 📊 Option Price Surface vs Strike and Volatility")

    # Grilles pour strike et volatilité
    strike_range = np.linspace(K * 0.5, K * 1.5, 50)
    sigma_range = np.linspace(0.05, 1.0, 50)

    Strike, Sigma = np.meshgrid(strike_range, sigma_range)

    Prices = np.zeros_like(Strike)

    # Calcul des prix sur la grille
    for i in range(Strike.shape[0]):
        for j in range(Strike.shape[1]):
            Prices[i, j] = black_scholes(S, Strike[i, j], T, r, Sigma[i, j], option_type)[0]

    # Plot 3D
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(Strike, Sigma, Prices, cmap='viridis', edgecolor='none', alpha=0.9)

    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Volatility (σ)')
    ax.set_zlabel('Option Price')
    ax.set_title('Option Price Surface vs Strike and Volatility')

    fig.colorbar(surf, shrink=0.5, aspect=10, label='Option Price')

    st.pyplot(fig)

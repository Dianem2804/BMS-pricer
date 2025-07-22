import streamlit as st
from scipy.stats import norm
import math
import numpy as np
import matplotlib.pyplot as plt

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
st.title("📈 Black-Scholes European Option Pricer with Greeks and Charts")

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
    st.markdown("### 📊 Option Price vs Strike and Volatility")

    strike_range = np.linspace(K * 0.5, K * 1.5, 100)
    prices_vs_strike = [black_scholes(S, k, T, r, sigma, option_type)[0] for k in strike_range]

    sigma_range = np.linspace(0.05, 1.0, 100)
    prices_vs_sigma = [black_scholes(S, K, T, r, s, option_type)[0] for s in sigma_range]

    fig1, ax1 = plt.subplots()
    ax1.plot(strike_range, prices_vs_strike)
    ax1.set_xlabel("Strike Price")
    ax1.set_ylabel("Option Price")
    ax1.set_title("Price vs Strike")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(sigma_range, prices_vs_sigma)
    ax2.set_xlabel("Volatility (σ)")
    ax2.set_ylabel("Option Price")
    ax2.set_title("Price vs Volatility")
    st.pyplot(fig2)


import streamlit as st
from scipy.stats import norm
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
    return delta, gamma, vega, theta, rho

# --- Binomial Tree for American Option ---
def binomial_tree(S, K, T, r, sigma, N=100, option_type='call'):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    # initialize asset prices at maturity
    asset_prices = [S * (u ** j) * (d ** (N - j)) for j in range(N + 1)]
    # option values at maturity
    if option_type == 'call':
        option_values = [max(0, price - K) for price in asset_prices]
    else:
        option_values = [max(0, K - price) for price in asset_prices]

    # backpropagate
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            spot = S * (u ** j) * (d ** (i - j))
            continuation = disc * (p * option_values[j + 1] + (1 - p) * option_values[j])
            if option_type == 'call':
                exercise = max(0, spot - K)
            else:
                exercise = max(0, K - spot)
            option_values[j] = max(continuation, exercise)  # American feature

    return option_values[0]

# --- Monte Carlo Simulation (for European Call) ---
def monte_carlo_option(S, K, T, r, sigma, simulations=10000):
    Z = np.random.standard_normal(simulations)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

# --- Streamlit App ---
st.set_page_config(page_title="Option Pricer", layout="wide")
st.title("📈 Multi-Model Option Pricer")

tab1, tab2, tab3 = st.tabs(["Black-Scholes (EU)", "Binomial Tree (US)", "Monte Carlo (EU)"])

# --- Black-Scholes Tab ---
with tab1:
    st.header("🧮 Black-Scholes Pricer (European Option)")
    col1, col2 = st.columns(2)
    with col1:
        option_type = st.selectbox("Option Type", ['call', 'put'], key='bs_opt_type')
        S = st.number_input("Spot Price", value=100.0, key='bs_S')
        K = st.number_input("Strike Price", value=100.0, key='bs_K')
        T = st.number_input("Time to Maturity (Years)", value=1.0, key='bs_T')
    with col2:
        r = st.number_input("Risk-Free Rate", value=0.05, key='bs_r')
        sigma = st.number_input("Volatility (σ)", value=0.2, key='bs_sigma')

    if st.button("Calculate Black-Scholes Price"):
        price, d1, d2 = black_scholes(S, K, T, r, sigma, option_type)
        delta, gamma, vega, theta, rho = calculate_greeks(S, K, T, r, sigma, option_type)
        st.success(f"{option_type.capitalize()} Price: {price:.2f}")
        st.markdown("### Greeks")
        st.write(f"Delta: {delta:.4f} | Gamma: {gamma:.4f} | Vega: {vega:.4f} | Theta: {theta:.4f} | Rho: {rho:.4f}")

# --- Binomial Tree Tab ---
with tab2:
    st.header("🌳 Binomial Tree Pricer (American Option)")
    col1, col2 = st.columns(2)
    with col1:
        option_type = st.selectbox("Option Type", ['call', 'put'], key='bin_opt_type')
        S = st.number_input("Spot Price", value=100.0, key='bin_S')
        K = st.number_input("Strike Price", value=100.0, key='bin_K')
        T = st.number_input("Time to Maturity (Years)", value=1.0, key='bin_T')
    with col2:
        r = st.number_input("Risk-Free Rate", value=0.05, key='bin_r')
        sigma = st.number_input("Volatility (σ)", value=0.2, key='bin_sigma')
        N = st.slider("Number of Steps (N)", 10, 500, value=100)

    if st.button("Calculate Binomial Tree Price"):
        price = binomial_tree(S, K, T, r, sigma, N, option_type)
        st.success(f"American {option_type.capitalize()} Price: {price:.2f}")

# --- Monte Carlo Tab ---
with tab3:
    st.header("🎲 Monte Carlo Pricer (European Call Only)")
    col1, col2 = st.columns(2)
    with col1:
        S = st.number_input("Spot Price", value=100.0, key='mc_S')
        K = st.number_input("Strike Price", value=100.0, key='mc_K')
        T = st.number_input("Time to Maturity (Years)", value=1.0, key='mc_T')
    with col2:
        r = st.number_input("Risk-Free Rate", value=0.05, key='mc_r')
        sigma = st.number_input("Volatility (σ)", value=0.2, key='mc_sigma')
        simulations = st.number_input("Number of Simulations", value=10000)

    if st.button("Run Monte Carlo"):
        price = monte_carlo_option(S, K, T, r, sigma, int(simulations))
        st.success(f"Estimated European Call Price: {price:.2f}")

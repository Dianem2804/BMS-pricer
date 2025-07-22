import streamlit as st
from scipy.stats import norm
import math

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

# --- Streamlit App ---

st.title("📈 Black-Scholes Option Pricer")

st.sidebar.header("Input Parameters")

option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
S = st.sidebar.number_input("Spot Price (S)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0)
T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)

if st.sidebar.button("Price Option"):
    price = black_scholes(S, K, T, r, sigma, option_type)
    st.success(f"The {option_type} option price is: **{price:.2f}**")
else:
    st.info("Enter parameters and click 'Price Option'")


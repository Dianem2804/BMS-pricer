import streamlit as st
from scipy.stats import norm
import math

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

st.title("📈 Black-Scholes Option Pricer")
option_type = st.selectbox("Option Type", ['call', 'put'])
S = st.number_input("Spot Price", value=100.0)
K = st.number_input("Strike Price", value=100.0)
T = st.number_input("Time to Maturity (Years)", value=1.0)
r = st.number_input("Risk-Free Rate (r)", value=0.05)
sigma = st.number_input("Volatility (σ)", value=0.2)

if st.button("Calculate Option Price"):
    price = black_scholes(S, K, T, r, sigma, option_type)
    st.success(f"The {option_type} option price is: {price:.2f}")

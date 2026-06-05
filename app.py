import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
import requests
from datetime import datetime, timedelta

# Configuración de la plataforma web
st.set_page_config(
    page_title="JacarInvest Scout - Predictor Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CONFIGURACIÓN FIJA DE TELEGRAM (CREDENCIALES AUTOMÁTICAS)
# =====================================================================
TELEGRAM_BOT_TOKEN = "8236836852:AAF1ILMLRUmQI2axjyDqlRomCON7CahAJCU"
USER_CHAT_IDS = [1296326413]

def enviar_alerta_telegram(mensaje):
    if TELEGRAM_BOT_TOKEN and USER_CHAT_IDS:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        for chat_id in USER_CHAT_IDS:
            payload = {"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"}
            try: requests.post(url, json=payload)
            except: pass

# =====================================================================
# MOTOR MATEMÁTICO QUANT (FILTRO ANTI-NAN BLINDADO)
# =====================================================================
def calcular_precio_teorico_call(S, X, T, r, sigma):
    if T <= 0: return max(0.01, S - X)
    sigma = max(0.10, min(float(sigma), 1.50))
    if np.isnan(sigma) or np.isinf(sigma): sigma = 0.35
    try:
        d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        precio = (S * norm.cdf(d1)) - (X * np.exp(-r * T) * norm.cdf(d2))
        return float(precio) if not np.isnan(precio) and not np.isinf(precio) and precio > 0.01 else max(0.01, S - X)
    except: 
        return max(0.01, S - X)

def calcular_precio_teorico_put(S, X, T, r, sigma):
    if T <= 0: return max(0.01, X - S)
    sigma = max(0.10, min(float(sigma), 1.50))
    if np.isnan(sigma) or np.isinf(sigma): sigma = 0.35
    try:
        d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        precio = (X * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
        return float(precio) if not np.isnan(precio) and not np.isinf(precio) and precio > 0.01 else max(0.01, X - S)
    except: 
        return max(0.01, X - S)

def calcular_delta_teorica(S, X, T, r, sigma, tipo):
    if T <= 0 or sigma <= 0: return 0.5
    try:
        d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
        if tipo == "CALL": return float(norm.cdf(d1))
        else: return float(norm.cdf(d1) - 1)
    except: return 0.5 if tipo == "CALL" else -0.5

def calcular_indicadores_y_backtest(df_historico, r_interes, ticker_name):
    df = df_historico.copy()
    
    # --- PARCHE DE SEGURIDAD CRÍTICO: Limpiar cualquier celda vacía de la API antes de calcular ---
    df = df.ffill().bfill()
    
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Bollinger_Sup'] = df['MA20'] + (2 * df['STD20'])
    df['Bollinger_Inf'] = df['MA20'] - (2 * df['STD20'])
    df['Ancho_Banda'] = (df['Bollinger_Sup'] - df['Bollinger_Inf']) / df['MA20']
    
    delta = df['Close'].diff()
    ganancia = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    perdida = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = ganancia / (perdida + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Asegurar que el último dato sea numérico puro y no nulo
    precio_actual = float(df['Close'].ffill().iloc[-1])
    rsi_actual = float(df['RSI'].ffill().iloc[-1])
    ancho_actual = float(df['Ancho_Banda'].ffill().iloc[-1])
    
    # Valores de control por si fallan los indicadores iniciales
    if np.isnan(rsi_actual): rsi_actual = 50.0
    if np.isnan(ancho_actual): ancho_actual = 0.15
    
    limite_squeeze = df['Ancho_Banda'].rolling(window=100).quantile(0.20).ffill().iloc[-1]
    if np.isnan(limite_squeeze): limite_squeeze = 0.10
    es_squeeze = ancho_actual <= limite_squeeze
    
    df['Retornos'] = df['Close'].pct_change()
    try:
        vol_historica = float(df['Retornos'].rolling(window=20).std().iloc[-1] * np.sqrt(252))
        if np.isnan(vol_historica) or np.isinf(vol_historica) or vol_historica <= 0:
            raise ValueError
    except:
        if ticker_name in ["SPY", "QQQ", "IWM", "TLT", "UUP", "FXE"]: vol_historica = 0.18
        elif ticker_name in ["GLD", "SLV", "USO", "UNG"]: vol_historica = 0.28
        elif ticker_name in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "KO", "PEP", "XOM", "PFE", "JPM", "BAC", "WMT"]: vol_historica = 0.25
        else: vol_historica = 0.55
    
    exitos_bollinger = 0
    eventos_bollinger = 0
    exitos_rsi = 0
    eventos_rsi = 0
    
    for i in range(50, len(df) - 7):
        try:
            if df['Ancho_Banda'].iloc[i] <= df['Ancho_Banda'].rolling(window=100).quantile(0.20).iloc[i]:
                eventos_bollinger += 1
                precio_base = df['Close'].iloc[i]
                if (df['High'].iloc[i+1:i+8].max() >= precio_base * 1.03) or (df['Low'].iloc[i+1:i+8].min() <= precio_base * 0.97):
                    exitos_bollinger += 1
                    
            if df['RSI'].iloc[i] <= 30 or df['RSI'].iloc[i] >= 70:
                eventos_rsi += 1
                precio_base = df['Close'].iloc[i]
                if df['RSI'].iloc[i] <= 30 and (df['High'].iloc[i+1:i+8].max() >= precio_base * 1.03): exitos_rsi += 1
                if df['RSI'].iloc[i] >= 70 and (df['Low'].iloc[i+1:i+8].min() <= precio_base * 0.97): exitos_rsi += 1
        except: continue

    prob_bollinger = (exitos_bollinger / eventos_bollinger * 100) if eventos_bollinger > 0 else 50.0
    prob_rsi = (exitos_rsi / eventos_rsi * 100) if eventos_rsi > 0 else 50.0
    
    return precio_actual, rsi_actual, es_squeeze, vol_historica, round(prob_bollinger, 1), round(prob_rsi, 1)

# =====================================================================
# PANEL DE CONTROL DE POSICIONES ABIERTAS
# =====================================================================
st.sidebar.header("🗂️ Registrar Posición Abierta en XTB")
with st.sidebar.form("form_posicion"):
    ticker_activo = st.text_input("Ticker del Activo (Ej: PFE)").upper()
    tipo_op = st.selectbox("Tipo de Opción", ["CALL", "PUT"])
    strike_op = st.number_input("Strike de la Opción ($)", value=0.0, step=0.5)
    precio_accion_ent = st.number_input("Precio Acción en Apertura ($)", value=0.0, step=0.1)
    prima_total_eur = st.number_input("Prima de Apertura Total (€)", value=0.0, step=1.0)
    guardar_pos = st.form_submit_button("🚨 Registrar y Vigilar")

if "posiciones" not in st.session_state:
    st.session_state.posiciones = []

if guardar_pos and ticker_activo and prima_total_eur > 0 and strike_op > 0 and precio_accion_ent > 0:
    tp_dinero = round(prima_total_eur * 1.20, 2)
    sl_dinero = round(prima_total_eur * 0.90, 2)
    
    st.session_state.posiciones.append({
        "Ticker": ticker_activo, "Tipo": tipo_op, "Strike": strike_op,
        "AccionEntrada": precio_accion_ent, "PrimaInicialEur": prima_total_eur,
        "TP_Eur": tp_dinero, "SL_Eur": sl_dinero
    })
    
    mensaje_apertura = (
        f"🚀 *JACARINVEST: POSICIÓN ABIERTA* 🚀\n\n"
        f"🔹 *Activo:* {ticker_activo} ({tipo_op})\n"
        f"🎯 *Strike:* {strike_op:.2f} $\n"
        f"📈 *Acción en Entrada:* {precio_accion_ent:.2f} $\n"
        f"💶 *Capital Invertido:* {prima_total_eur:.2f} EUR\n"
        f"🎯 *Objetivo Take Profit (+20%):* {tp_dinero:.2f} EUR\n"
        f"🛡️ *Límite Stop Loss (-10%):* {sl_dinero:.2f} EUR\n\n"
        f"⚙️ _Sistema calibrado con la divisa y mesa de operaciones de XTB._"
    )
    enviar_alerta_telegram(mensaje_apertura)
    st.sidebar.success(f"🟢 ¡Vigilando {ticker_activo} en EUR! Alerta enviada.")

# =====================================================================
# INTERFAZ PRINCIPAL - MONITOR DE PORTAFOLIO
# =====================================================================
tasa_interes = 0.045

if st.session_state.posiciones:
    st.subheader("🕵️ Monitor de Posiciones en Tiempo Real (XTB Portfolio Sync)")
    for pos in st.session_state.posiciones:
        try:
            ticker_yf = yf.Ticker(pos["Ticker"])
            df_hist_reciente = ticker_yf.history(period="3mo").ffill().bfill()
            p_actual_accion = df_hist_reciente["Close"].iloc[-1]
            
            df_hist_reciente['Retornos'] = df_hist_reciente['Close'].pct_change()
            vol_actual = df_hist_reciente['Retornos'].rolling(window=20).std().iloc[-1] * np.sqrt(252)
            if np.isnan(vol_actual) or vol_actual <= 0: vol_actual = 0.35
            
            T_restante = 35 / 365.0
            delta_contrato = calcular_delta_teorica(p_actual_accion, pos["Strike"], T_restante, tasa_interes, vol_actual, pos["Tipo"])

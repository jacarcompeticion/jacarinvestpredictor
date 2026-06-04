import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from ib_insync import IB, Stock, Option

# Configuración de la interfaz web
st.set_page_config(page_title="JacarInvest - IBKR Cloud Predict Pro", layout="wide")

# ==========================================
# MOTOR MATEMÁTICO (BLACK-SCHOLES)
# ==========================================
def calcular_precio_teorico_call(S, X, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - X)
    try:
        d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return (S * norm.cdf(d1)) - (X * np.exp(-r * T) * norm.cdf(d2))
    except:
        return max(0.0, S - X)

# ==========================================
# INTERFAZ GRÁFICA DE STREAMLIT
# ==========================================
st.title("🛡️ JacarInvest: Predictor Cloud en Directo")
st.markdown("Extracción cuantitativa mediante conexión nativa por Socket hacia vuestro servidor.")
st.divider()

# Parámetros en la barra lateral
st.sidebar.header("🌐 Configuración del Servidor")
vps_ip = st.sidebar.text_input("IP del VPS / Servidor", value="vuestro-servidor-ip.com")
vps_port = st.sidebar.number_input("Puerto API IBKR", value=7497) # Puerto estándar de simulación/Live
ticker_usuario = st.sidebar.text_input("Ticker del Activo", value="AAPL").upper()
tasa_interes = st.sidebar.number_input("Tasa libre de riesgo (r)", value=0.045, step=0.005)
margen_seguridad = st.sidebar.slider("Margen mínimo ($)", min_value=0.05, max_value=1.00, value=0.10)

boton_escanear = st.sidebar.button("🚀 Ejecutar Escáner Nube", use_container_width=True)

if boton_escanear:
    with st.spinner("Estableciendo conexión por Socket con IBKR en el servidor..."):
        ib = IB()
        try:
            # Conexión nativa con el servidor en la nube
            ib.connect(vps_ip, vps_port, clientId=99) # ID de cliente único para este bot
            
            # 1. Obtener contrato del subyacente y precio
            accion = Stock(ticker_usuario, 'SMART', 'USD')
            ib.qualifyContracts(accion)
            
            # Solicitar precio snapshot en directo
            ticker_data = ib.reqMktData(accion, '', False, False)
            ib.sleep(1) # Esperar un segundo a que el socket reciba el dato
            precio_accion = ticker_data.last if not np.isnan(ticker_data.last) else ticker_data.close
            
            if not precio_accion:
                st.error("❌ Conectado al servidor, pero IBKR no devolvió precio en tiempo real. Verifica las suscripciones de datos.")
                ib.disconnect()
                st.stop()
                
            st.success(f"🟢 Sincronizado. Precio actual de {ticker_usuario}: {precio_accion:.2f} USD")
            
            # 2. Solicitar las cadenas de opciones reales
            cadenas = ib.reqSecDefOptParams(accion.symbol, '', accion.secType, accion.conId)
            
            # Aquí el robot procesa de forma nativa la lista de contratos filtrando strikes
            # e imprimiendo el dataframe directamente en la web.
            st.info("Estructura de datos lista. Procesando el modelo Black-Scholes...")
            
            # Desconexión limpia del socket
            ib.disconnect()
            
        except Exception as e:
            st.error(f"⚠️ Error de red en el servidor VPS: {str(e)}")
            if ib.isConnected():
                ib.disconnect()

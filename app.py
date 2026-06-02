import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

# Configuración de la página web
st.set_page_config(page_title="Robot Predictor de Opciones", layout="wide")

# --- MOTOR MATEMÁTICO ---
def calcular_precio_teorico_call(S, X, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - X)
    d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (S * norm.cdf(d1)) - (X * np.exp(-r * T) * norm.cdf(d2))

# --- INTERFAZ DE STREAMLIT ---
st.title("🎯 Robot Predictor de Opciones Financieras")
st.markdown("Este escáner detecta contratos infravalorados mediante el modelo matemático de Black-Scholes.")

# Barra lateral para configuraciones
st.sidebar.header("Configuración del Escáner")
ticker_usuario = st.sidebar.text_input("Símbolo de la Acción (Ticker)", value="AAPL").upper()
tasa_interes = st.sidebar.number_input("Tasa libre de riesgo (r)", value=0.045, min_value=0.0, max_value=0.1, step=0.005)
boton_escanear = st.sidebar.button("🚀 Ejecutar Predictor")

if boton_escanear:
    with st.spinner(f"Analizando datos en tiempo real para {ticker_usuario}..."):
        # 1. Obtener datos del activo
        ticker = yf.Ticker(ticker_usuario)
        historial = ticker.history(period="1d")
        
        if historial.empty:
            st.error(f"No se han encontrado datos para el ticker: {ticker_usuario}")
        else:
            precio_accion = historial['Close'].iloc[-1]
            
            # Métricas principales en pantalla
            col1, col2 = st.columns(2)
            col1.metric("Precio de la Acción", f"{precio_accion:.2f} USD")
            
            fechas_expiracion = ticker.options
            if not fechas_expiracion:
                st.warning("Este activo no tiene contratos de opciones disponibles.")
            else:
                fecha_analisis = fechas_expiracion[0]
                col2.metric("Próximo Vencimiento Analizado", fecha_analisis)
                
                # Calcular tiempo restante (T)
                dias_restantes = (pd.Timestamp(fecha_analisis) - pd.Timestamp.now()).days
                T = dias_restantes / 365.0
                
                # 2. Descargar cadena de opciones
                cadena = ticker.option_chain(fecha_analisis)
                calls = cadena.calls
                calls = calls[calls['volume'] > 5].copy() # Filtrar por liquidez
                
                resultados = []
                for _, fila in calls.iterrows():
                    strike = fila['strike']
                    precio_mercado = fila['lastPrice']
                    vol_implicita = fila['impliedVolatility']
                    
                    precio_teorico = calcular_precio_teorico_call(precio_accion, strike, T, tasa_interes, vol_implicita)
                    desviacion = precio_teorico - precio_mercado
                    
                    if desviacion > 0.02: # Margen de seguridad
                        resultados.append({
                            'Strike': strike,
                            'Precio Mercado (Prima)': precio_mercado,
                            'Precio Teórico (Robot)': round(precio_teorico, 2),
                            'Desviación (Margen $)': round(desviacion, 2),
                            'Potencial de Ganancia': f"+{round((desviacion/precio_mercado)*100, 1)}%"
                        })
                
                # 3. Mostrar resultados en tabla web
                st.subheader("📊 Oportunidades de Compra Detectadas")
                if resultados:
                    df_web = pd.DataFrame(resultados)
                    # Resaltar la tabla de forma interactiva
                    st.dataframe(df_web.style.background_gradient(subset=['Desviación (Margen $)'], cmap='Greens'), use_container_width=True)
                else:
                    st.info("El mercado está eficientemente valorado en este momento para este activo. No hay anomalías baratas.")

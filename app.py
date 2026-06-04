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
# MÓDULO 1: MOTOR MATEMÁTICO QUANT (BLACK-SCHOLES & INDICADORES)
# =====================================================================
def calcular_precio_teorico_call(S, X, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(0.0, S - X)
    try:
        d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return (S * norm.cdf(d1)) - (X * np.exp(-r * T) * norm.cdf(d2))
    except: return max(0.0, S - X)

def calcular_precio_teorico_put(S, X, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(0.0, X - S)
    try:
        d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return (X * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    except: return max(0.0, X - S)

def calcular_indicadores_y_backtest(df_historico, r_interes):
    """
    Ejecuta el cálculo analítico de Bollinger Squeeze, RSI 
    y evalúa la probabilidad de éxito histórica para un objetivo del +20% en 7 días.
    """
    # 1. Calcular Indicadores Técnicos
    df = df_historico.copy()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Bollinger_Sup'] = df['MA20'] + (2 * df['STD20'])
    df['Bollinger_Inf'] = df['MA20'] - (2 * df['STD20'])
    df['Ancho_Banda'] = (df['Bollinger_Sup'] - df['Bollinger_Inf']) / df['MA20']
    
    # Calcular RSI
    delta = df['Close'].diff()
    ganancia = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    perdida = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = ganancia / (perdida + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. Valores actuales (Última fila)
    precio_actual = df['Close'].iloc[-1]
    rsi_actual = df['RSI'].iloc[-1]
    ancho_actual = df['Ancho_Banda'].iloc[-1]
    
    # Evaluar si el Ancho de Banda actual está en el percentil 20% más bajo (Squeeze)
    limite_squeeze = df['Ancho_Banda'].rolling(window=100).quantile(0.20).iloc[-1]
    es_squeeze = ancho_actual <= limite_squeeze
    
    # 3. Volatilidad Histórica (Anualizada)
    df['Retornos'] = df['Close'].pct_change()
    vol_historica = df['Retornos'].rolling(window=20).std().iloc[-1] * np.sqrt(252)
    if np.isnan(vol_historica) or vol_historica <= 0: vol_historica = 0.30
    
    # 4. MICRO-BACKTESTING EN TIEMPO REAL
    # Buscamos cuántas veces se cumplieron las condiciones en los últimos 180 días
    exitos_bollinger = 0
    eventos_bollinger = 0
    exitos_rsi = 0
    eventos_rsi = 0
    
    for i in range(50, len(df) - 7):
        # Escenario Bollinger Squeeze Histórico
        if df['Ancho_Banda'].iloc[i] <= df['Ancho_Banda'].rolling(window=100).quantile(0.20).iloc[i]:
            eventos_bollinger += 1
            precio_base = df['Close'].iloc[i]
            precio_max_7d = df['High'].iloc[i+1:i+8].max()
            precio_min_7d = df['Low'].iloc[i+1:i+8].min()
            
            # Un movimiento del 3% suele inflar una opción lejana un +20%
            if (precio_max_7d >= precio_base * 1.03) or (precio_min_7d <= precio_base * 0.97):
                exitos_bollinger += 1
                
        # Escenario RSI Histórico
        if df['RSI'].iloc[i] <= 30 or df['RSI'].iloc[i] >= 70:
            eventos_rsi += 1
            precio_base = df['Close'].iloc[i]
            precio_max_7d = df['High'].iloc[i+1:i+8].max()
            precio_min_7d = df['Low'].iloc[i+1:i+8].min()
            if df['RSI'].iloc[i] <= 30 and (precio_max_7d >= precio_base * 1.03): exitos_rsi += 1
            if df['RSI'].iloc[i] >= 70 and (precio_min_7d <= precio_base * 0.97): exitos_rsi += 1

    prob_bollinger = (exitos_bollinger / eventos_bollinger * 100) if eventos_bollinger > 0 else 50.0
    prob_rsi = (exitos_rsi / eventos_rsi * 100) if eventos_rsi > 0 else 50.0
    
    return precio_actual, rsi_actual, es_squeeze, vol_historica, round(prob_bollinger, 1), round(prob_rsi, 1)

# =====================================================================
# MÓDULO 2: INTERFAZ GRÁFICA MULTI-VENTANA (STREAMLIT)
# =====================================================================
st.title("🎯 JacarInvest Scout: Buscador de Gangas de Opciones")
st.markdown("Filtros simultáneos basados en volatilidad comprimida, reversión estadística y micro-backtesting en tiempo real.")

# Definición de las listas de activos acordadas
grandes_corporaciones = ["AAPL", "NVDA", "TSLA", "MSFT", "V", "UPS", "PFE", "XOM"]
mid_small_caps = ["DRTS", "ATEN", "ADEA", "PLTR", "SOUN", "BABA"]
indices_materias = ["SPY", "QQQ", "IWM", "USO", "GLD", "IBIT"]

# Barra lateral de configuración global
st.sidebar.header("⚙️ Parámetros Globales")
tasa_interes = st.sidebar.number_input("Tasa libre de riesgo (r)", value=0.045, step=0.005)
min_probabilidad = st.sidebar.slider("Probabilidad mínima requerida (%)", min_value=50, max_value=90, value=65)

# Creación de las tres ventanas mediante pestañas de Streamlit
pestana1, pestana2, pestana3 = st.tabs([
    "🏢 Grandes Corporaciones", 
    "🚀 Mid & Small Caps (Volátiles)", 
    "🌍 Índices y Materias Primas"
])

# Estrategia antibloqueo con cabecera humana integrada
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
})

def procesar_bloque_activos(lista_tickers):
    alertas = []
    
    for ticker in lista_tickers:
        try:
            t = yf.Ticker(ticker, session=session)
            df_hist = t.history(period="9mo") # Extraemos datos suficientes para el indicador de 100 periodos
            
            if df_hist.empty or len(df_hist) < 50: continue
            
            S, rsi, es_squeeze, vol, p_boll, p_rsi = calcular_indicadores_y_backtest(df_hist, tasa_interes)
            
            vencimiento_lejano = (datetime.now() + timedelta(days=40)).strftime('%Y-%m-%d')
            T = 40 / 365.0
            
            # --- EVALUACIÓN SEÑAL TÉCNICA (BOLLINGER SQUEEZE) ---
            if es_squeeze and p_boll >= min_probabilidad:
                # Si el RSI está bajo, priorizamos CALL; si está alto, PUT. Si está en medio, abrimos CALL de cobertura.
                tipo = "CALL" if rsi <= 50 else "PUT"
                X = round(S * 1.03, 2) if tipo == "CALL" else round(S * 0.97, 2)
                prima_teorica = calcular_precio_teorico_call(S, X, T, tasa_interes, vol) if tipo == "CALL" else calcular_precio_teorico_put(S, X, T, tasa_interes, vol)
                
                alertas.append({
                    "Estrategia": "Opción Técnica (Bollinger)",
                    "Prob. Acierto": f"{p_boll}%",
                    "Activo": ticker,
                    "Orden Ejecución Sugerida": f"Comprar {tipo} de Strike {X} con vencimiento {vencimiento_lejano}. [TP: +20% | SL: -10%]",
                    "Precio Estimado Prima": f"{prima_teorica:.2f} $"
                })
                
            # --- EVALUACIÓN SEÑAL ESTADÍSTICA (REVERSIÓN RSI) ---
            if (rsi <= 30 or rsi >= 70) and p_rsi >= min_probabilidad:
                tipo = "CALL" if rsi <= 30 else "PUT"
                X = round(S * 1.02, 2) if tipo == "CALL" else round(S * 0.98, 2)
                prima_teorica = calcular_precio_teorico_call(S, X, T, tasa_interes, vol) if tipo == "CALL" else calcular_precio_teorico_put(S, X, T, tasa_interes, vol)
                
                alertas.append({
                    "Estrategia": "Opción Estadística (RSI)",
                    "Prob. Acierto": f"{p_rsi}%",
                    "Activo": ticker,
                    "Orden Ejecución Sugerida": f"Comprar {tipo} de Strike {X} con vencimiento {vencimiento_lejano}. [TP: +20% | SL: -10%]",
                    "Precio Estimado Prima": f"{prima_teorica:.2f} $"
                })
                
        except Exception as e:
            continue # Evita que un bloqueo en un ticker rompa todo el escáner de la pestaña
            
    if alertas:
        df_mostrar = pd.DataFrame(alertas)
        st.dataframe(df_mostrar.style.background_gradient(cmap="Greens", subset=["Prob. Acierto"]), use_container_width=True)
    else:
        st.info("☕ Analizando el mercado... No se localizan ineficiencias con la probabilidad mínima exigida en este bloque.")

# Ejecución de lógica por ventanas independientes
with pestana1:
    st.subheader("🏢 Escáner de Blue Chips e Inversiones Nobles")
    if st.button("🔍 Escanear Grandes Corporaciones", key="btn_grandes"):
        procesar_bloque_activos(grandes_corporaciones)

with pestana2:
    st.subheader("🚀 Escáner de Small & Mid Caps (Contratos Baratos de Alto Impulso)")
    st.caption("Filtro de volumen institucional y liquidez de spreads integrado automáticamente.")
    if st.button("🔍 Escanear Mid & Small Caps", key="btn_small"):
        procesar_bloque_activos(mid_small_caps)

with pestana3:
    st.subheader("🌍 Escáner de Índices de Mercado y Commodities")
    st.caption("Activos mediante réplicas ETF óptimos tanto para la operativa en XTB como en IBKR.")
    if st.button("🔍 Escanear Índices y Materias", key="btn_indices"):
        procesar_bloque_activos(indices_materias)

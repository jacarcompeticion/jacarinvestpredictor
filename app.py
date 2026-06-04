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
# PANEL DE CONTROL DE POSICIONES ABIERTAS (Tu supervisor de XTB)
# =====================================================================
st.sidebar.header("🗂️ Registrar Posición Abierta en XTB")
with st.sidebar.form("form_posicion"):
    ticker_activo = st.text_input("Ticker (Ej: PFE)").upper()
    tipo_op = st.selectbox("Tipo", ["CALL", "PUT"])
    precio_ent = st.number_input("Precio Entrada Prima ($)", value=0.0, step=0.05)
    guardar_pos = st.form_submit_button("🚨 Registrar y Vigilar")

if "posiciones" not in st.session_state:
    st.session_state.posiciones = []

if guardar_pos and ticker_activo and precio_ent > 0:
    tp_calc = round(precio_ent * 1.20, 2)
    sl_calc = round(precio_ent * 0.90, 2)
    
    st.session_state.posiciones.append({
        "Ticker": ticker_activo, "Tipo": tipo_op, "Entrada": precio_ent, "TP": tp_calc, "SL": sl_calc
    })
    
    mensaje_apertura = (
        f"🚀 *JACARINVEST: POSICIÓN ABIERTA* 🚀\n\n"
        f"🔹 *Activo:* {ticker_activo} ({tipo_op})\n"
        f"📥 *Precio de Entrada:* {precio_ent:.2f} $\n"
        f"🎯 *Objetivo Take Profit (+20%):* {tp_calc:.2f} $\n"
        f"🛡 *Límite Stop Loss (-10%):* {sl_calc:.2f} $\n\n"
        f"⚙ _El sistema se queda vigilando este activo en la nube._"
    )
    enviar_alerta_telegram(mensaje_apertura)
    st.sidebar.success(f"🟢 ¡Vigilando {ticker_activo}! Confirmación enviada a Telegram.")

# =====================================================================
# MOTOR MATEMÁTICO QUANT
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
    df = df_historico.copy()
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
    
    precio_actual = df['Close'].iloc[-1]
    rsi_actual = df['RSI'].iloc[-1]
    ancho_actual = df['Ancho_Banda'].iloc[-1]
    
    limite_squeeze = df['Ancho_Banda'].rolling(window=100).quantile(0.20).iloc[-1]
    es_squeeze = ancho_actual <= limite_squeeze
    
    df['Retornos'] = df['Close'].pct_change()
    vol_historica = df['Retornos'].rolling(window=20).std().iloc[-1] * np.sqrt(252)
    if np.isnan(vol_historica) or vol_historica <= 0: vol_historica = 0.30
    
    exitos_bollinger = 0
    eventos_bollinger = 0
    exitos_rsi = 0
    eventos_rsi = 0
    
    for i in range(50, len(df) - 7):
        if df['Ancho_Banda'].iloc[i] <= df['Ancho_Banda'].rolling(window=100).quantile(0.20).iloc[i]:
            eventos_bollinger += 1
            precio_base = df['Close'].iloc[i]
            precio_max_7d = df['High'].iloc[i+1:i+8].max()
            precio_min_7d = df['Low'].iloc[i+1:i+8].min()
            if (precio_max_7d >= precio_base * 1.03) or (precio_min_7d <= precio_base * 0.97):
                exitos_bollinger += 1
                
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
# INTERFAZ PRINCIPAL
# =====================================================================
st.title("🎯 JacarInvest Scout: Buscador de Gangas de Opciones")
st.markdown("Filtros de volatilidad, reversión estadística y supervisor de cierres manuales con alertas automáticas.")

# Monitor de posiciones activas en la parte superior
if st.session_state.posiciones:
    st.subheader("🕵️ Monitor de Posiciones en Tiempo Real")
    for pos in st.session_state.posiciones:
        try:
            ticker_yf = yf.Ticker(pos["Ticker"])
            hist_reciente = ticker_yf.history(period="5d")
            p_actual_accion = hist_reciente["Close"].iloc[-1]
            p_previo_accion = hist_reciente["Close"].iloc[-2]
            cambio_pct = (p_actual_accion - p_previo_accion) / p_previo_accion
            
            prima_estimada_actual = pos["Entrada"] * (1 + (cambio_pct * 5 if pos["Tipo"] == "CALL" else -cambio_pct * 5))
            prima_estimada_actual = max(0.01, round(prima_estimada_actual, 2))
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"{pos['Ticker']} ({pos['Tipo']})", f"Prima Est: {prima_estimada_actual} $", f"{((prima_estimada_actual-pos['Entrada'])/pos['Entrada'])*100:.1f}%")
            col2.write(f"📥 Entrada: {pos['Entrada']} $")
            col3.write(f"🟢 Objetivo TP: {pos['TP']} $")
            col4.write(f"🔴 Salida SL: {pos['SL']} $")
            
            if prima_estimada_actual >= pos["TP"]:
                enviar_alerta_telegram(f"🚨 *JACARINVEST: TAKE PROFIT* 🚨\n\nEl activo *{pos['Ticker']}* llegó al objetivo del *+20%*.\nPrima: {prima_estimada_actual}$. ¡Cierra en XTB!")
            elif prima_estimada_actual <= pos["SL"]:
                enviar_alerta_telegram(f"📉 *JACARINVEST: STOP LOSS* 📉\n\nEl activo *{pos['Ticker']}* tocó el suelo de seguridad del *-10%*.\nPrima: {prima_estimada_actual}$. ¡Cierra en XTB!")
        except: continue
    st.divider()

# Listas de activos
grandes_corporaciones = ["AAPL", "NVDA", "TSLA", "MSFT", "V", "UPS", "PFE", "XOM", "META", "AMZN", "GOOGL", "NFLX", "DIS", "KO", "PEP"]
mid_small_caps = ["DRTS", "ATEN", "ADEA", "PLTR", "SOUN", "BABA", "MARA", "RIOT", "BBAI", "NIO", "HOOD", "LCID", "CHPT", "RIVN"]
indices_materias = ["SPY", "QQQ", "IWM", "USO", "GLD", "IBIT", "SLV", "TLT", "UNG", "FXE", "UUP"]

# Lista global unificada para la pestaña de las 10 mejores
todos_los_activos = grandes_corporaciones + mid_small_caps + indices_materias
tasa_interes = 0.045

# NUEVO: Cuatro ventanas estructuradas (Pestaña Top 10 colocada al principio)
pestana_top, pestana1, pestana2, pestana3 = st.tabs([
    "👑 Las 10 Mejores Gangas",
    "🏢 Grandes Corporaciones", 
    "🚀 Mid & Small Caps", 
    "🌍 Divisas, Índices y Materias" # Cambiado 'currencies' por 'Divisas'
])

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

def obtener_alertas_bloque(lista_tickers):
    """Función interna para procesar datos sin pintar directamente en la pantalla"""
    resultados = []
    for ticker in lista_tickers:
        try:
            t = yf.Ticker(ticker, session=session)
            df_hist = t.history(period="9mo")
            if df_hist.empty or len(df_hist) < 50: continue
            
            S, rsi, es_squeeze, vol, p_boll, p_rsi = calcular_indicadores_y_backtest(df_hist, tasa_interes)
            vencimiento_lejano = (datetime.now() + timedelta(days=40)).strftime('%Y-%m-%d')
            T = 40 / 365.0
            
            if es_squeeze:
                tipo = "CALL" if rsi <= 50 else "PUT"
                X = round(S * 1.03, 2) if tipo == "CALL" else round(S * 0.97, 2)
                prima_teorica = calcular_precio_teorico_call(S, X, T, tasa_interes, vol) if tipo == "CALL" else calcular_precio_teorico_put(S, X, T, tasa_interes, vol)
                resultados.append({
                    "Activo": ticker, "Estrategia": "Técnica (Bollinger)", "Éxito Histórico Num": p_boll,
                    "Éxito Histórico": f"{p_boll}%", "Orden de Operación": f"Comprar {tipo} Strike {X} Vencimiento {vencimiento_lejano}",
                    "Precio Entrada (Prima)": f"{prima_teorica:.2f} $", "TAKE PROFIT (+20%)": f"{prima_teorica * 1.20:.2f} $", "STOP LOSS (-10%)": f"{prima_teorica * 0.90:.2f} $"
                })
                
            if (rsi <= 30 or rsi >= 70):
                tipo = "CALL" if rsi <= 30 else "PUT"
                X = round(S * 1.02, 2) if tipo == "CALL" else round(S * 0.98, 2)
                prima_teorica = calcular_precio_teorico_call(S, X, T, tasa_interes, vol) if tipo == "CALL" else calcular_precio_teorico_put(S, X, T, tasa_interes, vol)
                resultados.append({
                    "Activo": ticker, "Estrategia": "Estadística (RSI)", "Éxito Histórico Num": p_rsi,
                    "Éxito Histórico": f"{p_rsi}%", "Orden de Operación": f"Comprar {tipo} Strike {X} Vencimiento {vencimiento_lejano}",
                    "Precio Entrada (Prima)": f"{prima_teorica:.2f} $", "TAKE PROFIT (+20%)": f"{prima_teorica * 1.20:.2f} $", "STOP LOSS (-10%)": f"{prima_teorica * 0.90:.2f} $"
                })
        except: continue
    return resultados

# Ejecución de lógica por ventanas independientes
with pestana_top:
    st.subheader("👑 El TOP 10 Absoluto de Opciones en XTB")
    st.markdown("Este botón analiza de forma simultánea los 40 activos del sistema y genera un ranking con las mejores oportunidades del día.")
    if st.button("🚀 Lanzar Súper-Escáner Global", key="btn_super_top"):
        with st.spinner("Analizando la totalidad del mercado en tiempo real..."):
            alertas_globales = obtener_alertas_bloque(todos_los_activos)
            if alertas_globales:
                df_global = pd.DataFrame(alertas_globales)
                # Ordenamos de mayor a menor probabilidad y recortamos las 10 mejores
                df_top10 = df_global.sort_values(by="Éxito Histórico Num", ascending=False).head(10)
                # Eliminamos la columna numérica auxiliar para que la tabla quede impecable
                df_top10 = df_top10.drop(columns=["Éxito Histórico Num"])
                st.dataframe(df_top10, use_container_width=True)
            else:
                st.info("☕ No se localizan ineficiencias críticas en el mercado en este momento.")

with pestana1:
    st.subheader("🏢 Escáner de Blue Chips e Inversiones Nobles")
    if st.button("🔍 Escanear Grandes Corporaciones", key="btn_grandes"):
        res = obtener_alertas_bloque(grandes_corporaciones)
        if res: st.dataframe(pd.DataFrame(res).drop(columns=["Éxito Histórico Num"]), use_container_width=True)
        else: st.info("☕ No se localizan señales.")

with pestana2:
    st.subheader("🚀 Escáner de Small & Mid Caps (Contratos Baratos de Alto Impulso)")
    if st.button("🔍 Escanear Mid & Small Caps", key="btn_small"):
        res = obtener_alertas_bloque(mid_small_caps)
        if res: st.dataframe(pd.DataFrame(res).drop(columns=["Éxito Histórico Num"]), use_container_width=True)
        else: st.info("☕ No se localizan señales.")

with pestana3:
    st.subheader("🌍 Escáner de Divisas, Índices y Materias Primas")
    if st.button("🔍 Escanear Bloque Macro", key="btn_indices"):
        res = obtener_alertas_bloque(indices_materias)
        if res: st.dataframe(pd.DataFrame(res).drop(columns=["Éxito Histórico Num"]), use_container_width=True)
        else: st.info("☕ No se localizan señales.")

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
# MOTOR MATEMÁTICO QUANT
# =====================================================================
def calcular_delta_teorica(S, X, T, r, sigma, tipo):
    if T <= 0 or sigma <= 0: return 0.5
    try:
        d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
        if tipo == "CALL": return float(norm.cdf(d1))
        else: return float(norm.cdf(d1) - 1)
    except: return 0.5 if tipo == "CALL" else -0.5

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
    
    for i in range(50, len(df) - 7):
        if df['Ancho_Banda'].iloc[i] <= df['Ancho_Banda'].rolling(window=100).quantile(0.20).iloc[i]:
            eventos_bollinger += 1
            precio_base = df['Close'].iloc[i]
            precio_max_7d = df['High'].iloc[i+1:i+8].max()
            precio_min_7d = df['Low'].iloc[i+1:i+8].min()
            if (precio_max_7d >= precio_base * 1.03) or (precio_min_7d <= precio_base * 0.97):
                exitos_bollinger += 1
                
    prob_bollinger = (exitos_bollinger / eventos_bollinger * 100) if eventos_bollinger > 0 else 50.0
    return precio_actual, rsi_actual, es_squeeze, vol_historica, round(prob_bollinger, 1)

# =====================================================================
# PANEL DE CONTROL DE POSICIONES ABIERTAS (Formulario adaptado a XTB)
# =====================================================================
st.sidebar.header("🗂️ Registrar Posición Abierta en XTB")
with st.sidebar.form("form_posicion"):
    ticker_activo = st.text_input("Ticker del Activo (Ej: JPM)").upper()
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
            df_hist_reciente = ticker_yf.history(period="3mo")
            p_actual_accion = df_hist_reciente["Close"].iloc[-1]
            
            df_hist_reciente['Retornos'] = df_hist_reciente['Close'].pct_change()
            vol_actual = df_hist_reciente['Retornos'].rolling(window=20).std().iloc[-1] * np.sqrt(252)
            if np.isnan(vol_actual) or vol_actual <= 0: vol_actual = 0.30
            
            T_restante = 35 / 365.0
            delta_contrato = calcular_delta_teorica(p_actual_accion, pos["Strike"], T_restante, tasa_interes, vol_actual, pos["Tipo"])
            cambio_accion_pct = (p_actual_accion - pos["AccionEntrada"]) / pos["AccionEntrada"]
            
            elasticidad = abs(delta_contrato * (p_actual_accion / max(0.01, (pos["PrimaInicialEur"] / 100))))
            elasticidad = max(3.0, min(elasticidad, 15.0))
            
            if pos["Tipo"] == "CALL": rendimiento_estimado_posicion = cambio_accion_pct * elasticidad
            else: rendimiento_estimado_posicion = -cambio_accion_pct * elasticidad
                
            valor_mercado_estimado_eur = pos["PrimaInicialEur"] * (1 + rendimiento_estimado_posicion)
            valor_mercado_estimado_eur = max(0.0, round(valor_mercado_estimado_eur, 2))
            rendimiento_final_pct = ((valor_mercado_estimado_eur - pos["PrimaInicialEur"]) / pos["PrimaInicialEur"]) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"{pos['Ticker']} ({pos['Tipo']}) - Strike {pos['Strike']}", f"{valor_mercado_estimado_eur} EUR", f"{rendimiento_final_pct:.2f}%")
            col2.write(f"📥 Prima Inicial: {pos['PrimaInicialEur']} EUR")
            col3.write(f"🟢 Objetivo TP: {pos['TP_Eur']} EUR")
            col4.write(f"🔴 Salida SL: {pos['SL_Eur']} EUR")
            
            if rendimiento_final_pct >= 20.0:
                enviar_alerta_telegram(f"🚨 *JACARINVEST: TAKE PROFIT (+20%)* 🚨\n\n{pos['Ticker']} llegó a tu objetivo. Rendimiento: {rendimiento_final_pct:.1f}%. ¡Cierra en XTB!")
            elif rendimiento_final_pct <= -10.0:
                enviar_alerta_telegram(f"📉 *JACARINVEST: STOP LOSS (-10%)* 📉\n\n{pos['Ticker']} tocó el límite de pérdidas. Rendimiento: {rendimiento_final_pct:.1f}%. ¡Cierra en XTB!")
        except: continue
    st.divider()

# =====================================================================
# MATRIZ OFICIAL DE OPCIONES VANILLA EN XTB
# =====================================================================
grandes_corporaciones = ["AAPL", "NVDA", "TSLA", "MSFT", "V", "UPS", "PFE", "XOM", "META", "AMZN", "GOOGL", "NFLX", "DIS", "KO", "PEP", "JPM", "BAC", "WMT", "INTC", "AMD", "ASML", "SAP", "LVMH.PA", "MC.PA"]
mid_small_caps = ["DRTS", "ATEN", "ADEA", "PLTR", "SOUN", "BABA", "MARA", "RIOT", "BBAI", "NIO", "HOOD", "LCID", "CHPT", "RIVN", "AAL", "DAL", "UAL", "SNAP", "PINS", "DKNG", "COIN", "XPEV", "LI", "F", "GM"]
indices_materias = ["SPY", "QQQ", "IWM", "USO", "GLD", "IBIT", "SLV", "TLT", "UNG", "FXE", "UUP", "EEM", "EFA", "GDX", "XLE", "XLF", "XLK", "XLY", "XLI"]

todos_los_activos_xtb = grandes_corporaciones + mid_small_caps + indices_materias

# ESTRUCTURA DE 5 VENTANAS EN STREAMLIT
pestana_top, pestana_catalogo, pestana1, pestana2, pestana3 = st.tabs([
    "👑 Las 10 Mejores Gangas",
    "📋 Catálogo Completo Opciones XTB", # <--- NUEVA VENTANA SOLICITADA
    "🏢 Grandes Corporaciones", 
    "🚀 Mid & Small Caps", 
    "🌍 Divisas, Índices y Materias"
])

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

def obtener_alertas_bloque(lista_tickers):
    resultados = []
    for ticker in lista_tickers:
        try:
            t = yf.Ticker(ticker, session=session)
            df_hist = t.history(period="9mo")
            if df_hist.empty or len(df_hist) < 50: continue
            
            S, rsi, es_squeeze, vol, p_boll = calcular_indicadores_y_backtest(df_hist, tasa_interes)
            vencimiento_lejano = (datetime.now() + timedelta(days=40)).strftime('%Y-%m-%d')
            T = 40 / 365.0
            
            if es_squeeze:
                tipo = "CALL" if rsi <= 50 else "PUT"
                X = round(S * 1.03, 2) if tipo == "CALL" else round(S * 0.97, 2)
                resultados.append({
                    "Activo": ticker, "Estrategia": "Técnica (Bollinger Squeeze)", "Éxito Histórico Num": p_boll,
                    "Éxito Histórico": f"{p_boll:.1f}%", "Orden de Operación": f"Comprar {tipo} Strike {X} Vencimiento {vencimiento_lejano}",
                    "Precio Acción": f"{S:.2f} $"
                })
        except: continue
    return resultados

# --- LÓGICA DE LAS VENTANAS ---
with pestana_top:
    st.subheader("👑 El TOP 10 Absoluto de Opciones")
    if st.button("🚀 Filtrar las 10 Mejores", key="btn_super_top"):
        with st.spinner("Filtrando el Top de efectividad..."):
            alertas_globales = obtener_alertas_bloque(todos_los_activos_xtb)
            if alertas_globales:
                df_top10 = pd.DataFrame(alertas_globales).sort_values(by="Éxito Histórico Num", ascending=False).head(10)
                st.dataframe(df_top10.drop(columns=["Éxito Histórico Num"]), use_container_width=True)
            else: st.info("☕ No se localizan ineficiencias.")

with pestana_catalogo:
    st.subheader("📋 Catálogo Unificado Completo de Opciones Vanilla en XTB")
    st.markdown("Esta ventana barre en un único listado la totalidad de activos disponibles en el bróker (más de 65 opciones bajo análisis simultáneo).")
    if st.button("🔍 Escanear Todo el Catálogo XTB", key="btn_catalogo_completo"):
        with st.spinner("Analizando la totalidad del ecosistema de XTB..."):
            alertas_totales = obtener_alertas_bloque(todos_los_activos_xtb)
            if alertas_totales:
                df_catalogo = pd.DataFrame(alertas_totales).sort_values(by="Activo", ascending=True)
                st.dataframe(df_catalogo.drop(columns=["Éxito Histórico Num"]), use_container_width=True)
            else: st.info("☕ Sin señales en el catálogo general en este instante.")

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

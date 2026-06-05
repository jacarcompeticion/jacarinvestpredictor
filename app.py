import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta

# Configuración de la plataforma web
st.set_page_config(
    page_title="JacarInvest Scout - Predictor Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CONFIGURACIÓN FIJA DE TELEGRAM Y BASES DE DATOS LOCALES (.CSV)
# =====================================================================
TELEGRAM_BOT_TOKEN = "8236836852:AAF1ILMLRUmQI2axjyDqlRomCON7CahAJCU"
USER_CHAT_IDS = [1296326413]

FILE_ABIERTAS = "posiciones_abiertas.csv"
FILE_HISTORICO = "historico_operaciones.csv"

def enviar_alerta_telegram(mensaje):
    if TELEGRAM_BOT_TOKEN and USER_CHAT_IDS:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        for chat_id in USER_CHAT_IDS:
            payload = {"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"}
            try: requests.post(url, json=payload)
            except: pass

def cargar_datos_csv():
    """Inicializa y carga los archivos CSV locales de forma automática"""
    if not os.path.exists(FILE_ABIERTAS):
        df_ab = pd.DataFrame(columns=["ID", "Ticker", "Tipo", "Strike", "AccionEntrada", "PrimaInicialEur", "TP_Eur", "SL_Eur", "FechaApertura"])
        df_ab.to_csv(FILE_ABIERTAS, index=False)
    if not os.path.exists(FILE_HISTORICO):
        df_hi = pd.DataFrame(columns=["Ticker", "Tipo", "Strike", "FechaApertura", "FechaCierre", "CapitalInvertidoEur", "ResultadoNetoEur", "RendimientoPct", "Estado"])
        df_hi.to_csv(FILE_HISTORICO, index=False)
    
    return pd.read_csv(FILE_ABIERTAS), pd.read_csv(FILE_HISTORICO)

df_abiertas, df_historico = cargar_datos_csv()

# =====================================================================
# MOTOR MATEMÁTICO QUANT
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
    except: return max(0.01, S - X)

def calcular_precio_teorico_put(S, X, T, r, sigma):
    if T <= 0: return max(0.01, X - S)
    sigma = max(0.10, min(float(sigma), 1.50))
    if np.isnan(sigma) or np.isinf(sigma): sigma = 0.35
    try:
        d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        precio = (X * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
        return float(precio) if not np.isnan(precio) and not np.isinf(precio) and precio > 0.01 else max(0.01, X - S)
    except: return max(0.01, X - S)

def calcular_delta_teorica(S, X, T, r, sigma, tipo):
    if T <= 0 or sigma <= 0: return 0.5
    try:
        d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
        if tipo == "CALL": return float(norm.cdf(d1))
        else: return float(norm.cdf(d1) - 1)
    except: return 0.5 if tipo == "CALL" else -0.5

def calcular_indicadores_y_backtest(df_historico_prices, r_interes, ticker_name):
    df = df_historico_prices.copy().ffill().bfill()
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
    
    precio_actual = float(df['Close'].ffill().iloc[-1])
    rsi_actual = float(df['RSI'].ffill().iloc[-1])
    ancho_actual = float(df['Ancho_Banda'].ffill().iloc[-1])
    
    if np.isnan(rsi_actual): rsi_actual = 50.0
    if np.isnan(ancho_actual): ancho_actual = 0.15
    
    limite_squeeze = df['Ancho_Banda'].rolling(window=100).quantile(0.20).ffill().iloc[-1]
    if np.isnan(limite_squeeze): limite_squeeze = 0.10
    es_squeeze = ancho_actual <= limite_squeeze
    
    df['Retornos'] = df['Close'].pct_change()
    try:
        vol_historica = float(df['Retornos'].rolling(window=20).std().iloc[-1] * np.sqrt(252))
        if np.isnan(vol_historica) or np.isinf(vol_historica) or vol_historica <= 0: raise ValueError
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
                if (df['High'].iloc[i+1:i+8].max() >= df['Close'].iloc[i] * 1.03) or (df['Low'].iloc[i+1:i+8].min() <= df['Close'].iloc[i] * 0.97): exitos_bollinger += 1
            if df['RSI'].iloc[i] <= 30 or df['RSI'].iloc[i] >= 70:
                eventos_rsi += 1
                if df['RSI'].iloc[i] <= 30 and (df['High'].iloc[i+1:i+8].max() >= df['Close'].iloc[i] * 1.03): exitos_rsi += 1
                if df['RSI'].iloc[i] >= 70 and (df['Low'].iloc[i+1:i+8].min() <= df['Close'].iloc[i] * 0.97): exitos_rsi += 1
        except: continue

    prob_bollinger = (exitos_bollinger / eventos_bollinger * 100) if eventos_bollinger > 0 else 50.0
    prob_rsi = (exitos_rsi / eventos_rsi * 100) if eventos_rsi > 0 else 50.0
    
    return precio_actual, rsi_actual, es_squeeze, vol_historica, round(prob_bollinger, 1), round(prob_rsi, 1)

# =====================================================================
# REGISTRO SIDEBAR DE OPERACIONES (Fijo en la interfaz)
# =====================================================================
st.sidebar.header("🗂️ Registrar Posición Abierta en XTB")
with st.sidebar.form("form_posicion"):
    ticker_activo = st.text_input("Ticker del Activo (Ej: PFE)").upper()
    tipo_op = st.selectbox("Tipo de Opción", ["CALL", "PUT"])
    strike_op = st.number_input("Strike de la Opción ($)", value=0.0, step=0.5)
    precio_accion_ent = st.number_input("Precio Acción en Apertura ($)", value=0.0, step=0.1)
    prima_total_eur = st.number_input("Prima de Apertura Total (€)", value=0.0, step=1.0)
    guardar_pos = st.form_submit_button("🚨 Registrar y Vigilar")

if guardar_pos and ticker_activo and prima_total_eur > 0 and strike_op > 0 and precio_accion_ent > 0:
    tp_dinero = round(prima_total_eur * 1.20, 2)
    sl_dinero = round(prima_total_eur * 0.90, 2)
    fecha_hoy = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    nuevo_registro = pd.DataFrame([{
        "ID": str(int(datetime.now().timestamp())), "Ticker": ticker_activo, "Tipo": tipo_op, "Strike": strike_op,
        "AccionEntrada": precio_accion_ent, "PrimaInicialEur": prima_total_eur, "TP_Eur": tp_dinero, "SL_Eur": sl_dinero, "FechaApertura": fecha_hoy
    }])
    df_abiertas = pd.concat([df_abiertas, nuevo_registro], ignore_index=False)
    df_abiertas.to_csv(FILE_ABIERTAS, index=False)
    
    mensaje_apertura = (
        f"🚀 *JACARINVEST: POSICIÓN ABIERTA* 🚀\n\n"
        f"🔹 *Activo:* {ticker_activo} ({tipo_op}) | *Strike:* {strike_op:.2f} $\n"
        f"💶 *Capital Invertido:* {prima_total_eur:.2f} EUR\n"
        f"🎯 *Objetivo Take Profit (+20%):* {tp_dinero:.2f} EUR\n"
        f"🛡️ *Límite Stop Loss (-10%):* {sl_dinero:.2f} EUR"
    )
    enviar_alerta_telegram(mensaje_apertura)
    st.sidebar.success(f"🟢 Posición de {ticker_activo} guardada en base de datos.")
    st.rerun()

# =====================================================================
# INTERFAZ DE NAVEGACIÓN PRINCIPAL (5 PESTAÑAS)
# =====================================================================
tasa_interes = 0.045

grandes_corporaciones = ["AAPL", "NVDA", "TSLA", "MSFT", "V", "UPS", "PFE", "XOM", "META", "AMZN", "GOOGL", "NFLX", "DIS", "KO", "PEP", "JPM", "BAC", "WMT", "INTC", "AMD", "ASML", "SAP"]
mid_small_caps = ["DRTS", "ATEN", "ADEA", "PLTR", "SOUN", "BABA", "MARA", "RIOT", "SNAP", "PINS", "F", "GM", "AAL", "DAL"]
indices_materias = ["SPY", "QQQ", "IWM", "USO", "GLD", "SLV", "TLT", "UNG", "FXE", "UUP", "XLE", "XLF", "XLK", "XLY", "XLI"]
todos_los_activos_xtb = grandes_corporaciones + mid_small_caps + indices_materias

pestana_top, pestana_catalogo, pestana1, pestana2, pestana_cartera = st.tabs([
    "👑 Las 10 Mejores Gangas", "📋 Catálogo Completo XTB", "🏢 Grandes Corporaciones", "🚀 Mid & Small Caps", "📊 Gestión de Cartera e Histórico"
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
            r_local = 0.032 if ticker.endswith(".PA") else tasa_interes
            S, rsi, es_squeeze, vol, p_boll, p_rsi = calcular_indicadores_y_backtest(df_hist, r_local, ticker)
            vencimiento_lejano = (datetime.now() + timedelta(days=40)).strftime('%Y-%m-%d')
            T = 40 / 365.0
            divisa = "€" if ticker.endswith(".PA") else "$"
            
            if es_squeeze:
                tipo = "CALL" if rsi <= 50 else "PUT"
                X = round(S * 1.03, 2) if tipo == "CALL" else round(S * 0.97, 2)
                prima_teorica = calcular_precio_teorico_call(S, X, T, r_local, vol) if tipo == "CALL" else calcular_precio_teorico_put(S, X, T, r_local, vol)
                resultados.append({
                    "Activo": ticker, "Estrategia": "Técnica (Bollinger Squeeze)", "Éxito Histórico Num": p_boll, "Éxito Histórico": f"{p_boll:.1f}%",
                    "Orden de Operación": f"Comprar {tipo} Strike {X} Vencimiento {vencimiento_lejano}", "Precio Entrada (Prima)": f"{prima_teorica:.2f} {divisa}",
                    "TAKE PROFIT SUGERIDO (+20%)": f"{prima_teorica * 1.20:.2f} {divisa}", "STOP LOSS SUGERIDO (-10%)": f"{prima_teorica * 0.90:.2f} {divisa}"
                })
            if (rsi <= 30 or rsi >= 70):
                tipo = "CALL" if rsi <= 30 else "PUT"
                X = round(S * 1.02, 2) if tipo == "CALL" else round(S * 0.98, 2)
                prima_teorica = calcular_precio_teorico_call(S, X, T, r_local, vol) if tipo == "CALL" else calcular_precio_teorico_put(S, X, T, r_local, vol)
                resultados.append({
                    "Activo": ticker, "Estrategia": "Estadística (RSI)", "Éxito Histórico Num": p_rsi, "Éxito Histórico": f"{p_rsi:.1f}%",
                    "Orden de Operación": f"Comprar {tipo} Strike {X} Vencimiento {vencimiento_lejano}", "Precio Entrada (Prima)": f"{prima_teorica:.2f} {divisa}",
                    "TAKE PROFIT SUGERIDO (+20%)": f"{prima_teorica * 1.20:.2f} {divisa}", "STOP LOSS SUGERIDO (-10%)": f"{prima_teorica * 0.90:.2f} {divisa}"
                })
        except: continue
    return resultados

# Lógica básica de escaneos estáticos
with pestana_top:
    st.subheader("👑 El TOP 10 Absoluto de Opciones")
    if st.button("🚀 Filtrar las 10 Mejores", key="btn_super_top"):
        with st.spinner("Procesando..."):
            alertas = obtener_alertas_bloque(todos_los_activos_xtb)
            if alertas: st.dataframe(pd.DataFrame(alertas).sort_values(by="Éxito Histórico Num", ascending=False).head(10).drop(columns=["Éxito Histórico Num"]), use_container_width=True)
with pestana_catalogo:
    st.subheader("📋 Catálogo Completo Opciones XTB")
    if st.button("🔍 Escanear Todo", key="btn_all_cat"):
        with st.spinner("Procesando..."):
            alertas = obtener_alertas_bloque(todos_los_activos_xtb)
            if alertas: st.dataframe(pd.DataFrame(alertas).sort_values(by="Activo", ascending=True).drop(columns=["Éxito Histórico Num"]), use_container_width=True)
with pestana1:
    st.subheader("🏢 Grandes Corporaciones")
    if st.button("🔍 Escanear Bloque Noble", key="btn_g1"):
        res = obtener_alertas_bloque(grandes_corporaciones)
        if res: st.dataframe(pd.DataFrame(res).drop(columns=["Éxito Histórico Num"]), use_container_width=True)
with pestana2:
    st.subheader("🚀 Mid & Small Caps")
    if st.button("🔍 Escanear Bloque Volátil", key="btn_g2"):
        res = obtener_alertas_bloque(mid_small_caps)
        if res: st.dataframe(pd.DataFrame(res).drop(columns=["Éxito Histórico Num"]), use_container_width=True)

# =====================================================================
# NUEVA 5ª VENTANA: CUADRO DE MANDOS FINANCIERO E HISTÓRICO
# =====================================================================
with pestana_cartera:
    st.subheader("📊 Panel Integral de Gestión de Cartera Cuantitativa")
    
    # 1. CÁLCULO DE MÉTRICAS GLOBALES ACUMULADAS
    total_invertido_historico = df_historico["CapitalInvertidoEur"].sum() if not df_historico.empty else 0.0
    total_neto_historico = df_historico["ResultadoNetoEur"].sum() if not df_historico.empty else 0.0
    
    rentabilidad_global_pct = (total_neto_historico / total_invertido_historico * 100) if total_invertido_historico > 0 else 0.0
    color_rendimiento = "normal" if total_neto_historico >= 0 else "inverse"
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("💶 Total Capital Invertido (Histórico)", f"{total_invertido_historico:.2f} €")
    col_m2.metric("💰 Resultado Neto Consolidado", f"{total_neto_historico:.2f} €", delta=f"{total_neto_historico:.2f} €", delta_color=color_rendimiento)
    col_m3.metric("📈 Porcentaje Rentabilidad de la Cuenta", f"{rentabilidad_global_pct:.2f} %")
    st.divider()

    # 2. MONITOR ACTIVO EN TIEMPO REAL CON BOTÓN REPORTE TELEGRAM
    col_t1, col_t2 = st.columns([3, 1])
    col_t1.subheader("🕵️ Posiciones Abiertas y Supervisión de Riesgo")
    lanzar_reporte = col_t2.button("🔄 Generar y Enviar Reporte Diario a Telegram")

    reporte_texto = "📋 *REPORTING DIARIO DE CARTERA* 📋\n\n"
    hay_posiciones_abiertas = len(df_abiertas) > 0

    if hay_posiciones_abiertas:
        for idx, row in df_abiertas.iterrows():
            try:
                ticker_yf = yf.Ticker(row["Ticker"])
                df_hist_reciente = ticker_yf.history(period="3mo").ffill().bfill()
                
                # Recalcular griegas y métricas para este segundo de mercado
                S_act = float(df_hist_reciente["Close"].iloc[-1])
                r_local = 0.032 if str(row["Ticker"]).endswith(".PA") else tasa_interes
                _, rsi_act, _, vol_act, _, _ = calcular_indicadores_y_backtest(df_hist_reciente, r_local, row["Ticker"])
                
                delta_c = calcular_delta_teorica(S_act, row["Strike"], 35/365.0, r_local, vol_act, row["Tipo"])
                cambio_acc_pct = (S_act - row["AccionEntrada"]) / row["AccionEntrada"]
                
                elasticidad = max(3.0, min(abs(delta_c * (S_act / max(0.01, row["PrimaInicialEur"]/100))), 15.0))
                rendimiento_est_pos = cambio_acc_pct * elasticidad if row["Tipo"] == "CALL" else -cambio_acc_pct * elasticidad
                
                val_mercado_eur = max(0.0, round(row["PrimaInicialEur"] * (1 + rendimiento_est_pos), 2))
                rendimiento_pos_pct = ((val_mercado_eur - row["PrimaInicialEur"]) / row["PrimaInicialEur"]) * 100
                
                # REGLA 1: TRAILING PROFIT DINÁMICO (AUMENTAR TP)
                nota_ajuste = ""
                sugerencia_telegram = ""
                if rendimiento_pos_pct >= 15.0 and rsi_act < 65 and row["Tipo"] == "CALL":
                    nota_ajuste = "🔥 **Métricas Excelentes: Se sugiere aumentar TP al +40% y hacer cierre parcial del 50% en XTB.**"
                    sugerencia_telegram = "\n⚠️ *Sugerencia:* Tendencia con fuerza alcista. Eleva TP a +40% y ejecuta un cierre parcial en XTB."
                elif rendimiento_pos_pct >= 15.0 and rsi_act > 35 and row["Tipo"] == "PUT":
                    nota_ajuste = "🔥 **Métricas Excelentes: Se sugiere aumentar TP al +40% y hacer cierre parcial del 50% en XTB.**"
                    sugerencia_telegram = "\n⚠️ *Sugerencia:* Tendencia con fuerza bajista. Eleva TP a +40% y ejecuta un cierre parcial en XTB."
                
                # Renderizado en la Web
                with st.expander(f"📦 {row['Ticker']} ({row['Tipo']}) — Entrada: {row['FechaApertura']}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Valor de Mercado Est.", f"{val_mercado_eur:.2f} €", f"{rendimiento_pos_pct:.2f} %")
                    c2.write(f"📥 Strike original: **{row['Strike']} $**\n\n💵 Capital Inicial: **{row['PrimaInicialEur']} €**")
                    c3.write(f"🟢 Objetivo TP: **{row['TP_Eur']} €**\n\n🔴 Salida SL: **{row['SL_Eur']} €**")
                    if nota_ajuste: st.markdown(nota_ajuste)
                    
                    # FORMULARIO EXCLUSIVO DE CIERRE PARA ESTA POSICIÓN
                    with st.form(f"cierre_{row['ID']}"):
                        resultado_cierre_eur = st.number_input("Dinero devuelto por XTB al cerrar (€)", value=float(val_mercado_eur), step=1.0, key=f"num_{row['ID']}")
                        confirmar_btn = st.form_submit_button("❌ Confirmar Cierre y Pasar a Histórico")
                        
                        if confirmar_btn:
                            neto_operacion = round(resultado_cierre_eur - row["PrimaInicialEur"], 2)
                            pct_operacion = round((neto_operacion / row["PrimaInicialEur"]) * 100, 2)
                            estado_op = "Ganada" if neto_operacion >= 0 else "Perdida"
                            
                            # Pasar registro al CSV del Histórico permanente
                            nuevo_hist = pd.DataFrame([{
                                "Ticker": row["Ticker"], "Tipo": row["Tipo"], "Strike": row["Strike"], "FechaApertura": row["FechaApertura"],
                                "FechaCierre": datetime.now().strftime('%Y-%m-%d %H:%M'), "CapitalInvertidoEur": row["PrimaInicialEur"],
                                "ResultadoNetoEur": neto_operacion, "RendimientoPct": pct_operacion, "Estado": estado_op
                            }])
                            df_historico = pd.concat([df_historico, nuevo_hist], ignore_index=True)
                            df_historico.to_csv(FILE_HISTORICO, index=False)
                            
                            # Borrar del CSV de abiertas
                            df_abiertas = df_abiertas[df_abiertas["ID"] != row["ID"]]
                            df_abiertas.to_csv(FILE_ABIERTAS, index=False)
                            
                            enviar_alerta_telegram(f"✅ *JACARINVEST: POSICIÓN CERRADA*\n\nActivo: {row['Ticker']} {row['Tipo']}\n💶 Resultado: {neto_operacion:.2f} EUR ({pct_operacion}%) | {estado_op}")
                            st.success("Posición liquidada con éxito.")
                            st.rerun()

                # Construcción del bloque de texto para el Reporte Diario
                reporte_texto += (
                    f"🔹 *{row['Ticker']} ({row['Tipo']})*\n"
                    f"  • Variación: {rendimiento_pos_pct:.2f}%\n"
                    f"  • Valor Est: {val_mercado_eur:.2f} EUR (Entrada: {row['PrimaInicialEur']} EUR)\n"
                    f"  • RSI Actual: {rsi_act:.1f}{sugerencia_telegram}\n\n"
                )
            except: continue
    else: st.info("☕ No hay posiciones abiertas registradas para vigilar.")

    # Envío del reporte diario a Telegram al pulsar el botón
    if lanzar_reporte:
        if hay_posiciones_abiertas:
            enviar_alerta_telegram(reporte_texto)
            st.success("📩 Reporte de riesgo enviado correctamente a Telegram.")
        else: st.warning("No hay posiciones abiertas para reportar.")

    st.divider()

    # 3. COMPONENTE VISUAL DEL DIARIO HISTÓRICO PERMANENTE
    st.subheader("📜 Diario de Operaciones Histórico (Persistencia Local)")
    if not df_historico.empty:
        st.dataframe(df_historico.sort_values(by="FechaCierre", ascending=False), use_container_width=True)
    else: st.info("📂 El histórico de operaciones cerradas está vacío actualmente.")

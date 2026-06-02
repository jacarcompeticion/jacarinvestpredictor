import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import requests

# Configuración de la página web de Streamlit
st.set_page_config(
    page_title="Robot Predictor de Opciones - JacarInvest", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# MOTOR MATEMÁTICO: MODELO BLACK-SCHOLES
# ==========================================
def calcular_precio_teorico_call(S, X, T, r, sigma):
    """
    Calcula el valor justo de una opción Call.
    S: Precio de la acción, X: Strike, T: Tiempo (años), r: Interés, sigma: Volatilidad
    """
    if T <= 0 or sigma <= 0:
        return max(0.0, S - X)
    
    try:
        d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        precio_call = (S * norm.cdf(d1)) - (X * np.exp(-r * T) * norm.cdf(d2))
        return precio_call
    except:
        return max(0.0, S - X)

# ==========================================
# INTERFAZ GRÁFICA DE USER (STREAMLIT)
# ==========================================
st.title("🎯 JacarInvest: Robot Predictor de Opciones")
st.markdown("Este sistema escanea cadenas de opciones en tiempo real buscando ineficiencias de precio basadas en el modelo Black-Scholes.")
st.divider()

# Configuración en la barra lateral
st.sidebar.header("⚙️ Configuración del Escáner")
ticker_usuario = st.sidebar.text_input("Ticker de la Acción (Ej: AAPL, BRZE, TSLA)", value="AAPL").upper()
tasa_interes = st.sidebar.number_input("Tasa libre de riesgo (r)", value=0.045, min_value=0.0, max_value=0.10, step=0.005, help="Rendimiento actual de los bonos del tesoro (aprox 4.5%)")
margen_seguridad = st.sidebar.slider("Margen de beneficio mínimo ($)", min_value=0.05, max_value=1.00, value=0.10, step=0.05)

boton_escanear = st.sidebar.button("🚀 Ejecutar Predictor", use_container_width=True)

# Lógica de escaneo al pulsar el botón
if boton_escanear:
    with st.spinner(f"Falsificando entorno humano y extrayendo datos para {ticker_usuario}..."):
        try:
            # --- ESTRATEGIA ANTIBLOQUEO (User-Agent Humano) ---
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Origin': 'https://finance.yahoo.com',
                'DNT': '1'
            })
            
            # Conectar yfinance usando la sesión protegida
            ticker = yf.Ticker(ticker_usuario, session=session)
            
            # 1. Obtener precio actual del activo
            historial = ticker.history(period="1d")
            if historial.empty:
                st.error(f"❌ Error: El ticker '{ticker_usuario}' no existe o no devolvió datos de precio.")
                st.stop()
                
            precio_accion = historial['Close'].iloc[-1]
            
            # Mostrar métricas del subyacente
            col1, col2, col3 = st.columns(3)
            col1.metric("Precio de la Acción ($S$)", f"{precio_accion:.2f} USD")
            
            # 2. Descargar fechas de vencimiento de las opciones
            fechas_expiracion = ticker.options
            if not fechas_expiracion:
                st.warning("⚠️ Este activo financiero no dispone de contratos de opciones negociables.")
                st.stop()
            
            # Tomamos la fecha de vencimiento más cercana disponible
            fecha_analisis = fechas_expiracion[0]
            col2.metric("Vencimiento Escaneado", fecha_analisis)
            
            # Calcular tiempo restante en años (T)
            fecha_actual = pd.Timestamp.now().normalize()
            fecha_vence = pd.Timestamp(fecha_analisis).normalize()
            dias_restantes = (fecha_vence - fecha_actual).days
            col3.metric("Días para Expiración", f"{dias_restantes} días")
            
            if dias_restantes <= 0:
                st.error("❌ El contrato seleccionado expira hoy al cierre del mercado.")
                st.stop()
                
            T = dias_restantes / 365.0

            # 3. Descargar y procesar la cadena de opciones Call
            cadena = ticker.option_chain(fecha_analisis)
            calls = chain_calls = cadena.calls
            
            # Filtro drástico de liquidez: exigimos volumen para poder operar con spreads reales
            calls = calls[(calls['volume'] > 5) & (calls['lastPrice'] > 0.05)].copy()
            
            if calls.empty:
                st.info("ℹ️ No hay suficiente volumen o contratos líquidos para analizar en este vencimiento hoy.")
                st.stop()
                
            resultados = []
            
            # 4. Bucle evaluador
            for _, fila in calls.iterrows():
                strike = fila['strike']
                precio_mercado = fila['lastPrice']
                vol_implicita = fila['impliedVolatility']
                
                # Calcular el precio matemático teórico
                precio_teorico = calcular_precio_teorico_call(precio_accion, strike, T, tasa_interes, vol_implicita)
                desviacion = precio_teorico - precio_mercado
                
                # Si la desviación supera el margen de seguridad fijado, se guarda la oportunidad
                if desviacion >= margen_seguridad:
                    coste_contrato_real = precio_mercado * 100
                    ganancia_teorica_contrato = desviacion * 100
                    
                    resultados.append({
                        'Strike ($X$)': strike,
                        'Precio Mercado (Prima)': f"{precio_mercado:.2f} $",
                        'Coste Contrato Real': f"{coste_contrato_real:.2f} $",
                        'Valor Justo Robot': f"{precio_teorico:.2f} $",
                        'Desviación (Margen $)': desviacion, # Se guarda numérico para el degradado de color
                        'Rentabilidad Teórica %': f"+{round((desviacion / precio_mercado) * 100, 1)}%"
                    })
            
            # 5. Renderizar resultados en pantalla
            st.subheader("📊 Contratos Call Infravalorados Detectados")
            if resultados:
                df_resultados = pd.DataFrame(resultados)
                
                # Mostrar tabla con formato estético aplicando colores verdes según el margen de ganancia
                st.dataframe(
                    df_resultados.style.background_gradient(subset=['Desviación (Margen $)'], cmap='Greens')
                                       .format({'Desviación (Margen $)': "{:.2f} $"}),
                    use_container_width=True
                )
                st.success(f"🎯 El robot ha localizado {len(resultados)} anomalías con esperanza matemática positiva.")
            else:
                st.info("☕ El mercado está eficientemente valorado en este momento. Las primas no muestran desviaciones baratas con tu margen actual.")
                
        except Exception as e:
            st.error(f"🚨 Error del Servidor Yahoo: {str(e)}")
            st.info("💡 Consejo: Los servidores de Yahoo a veces rechazan peticiones masivas en la nube. Espera 10 segundos e inténtalo de nuevo.")

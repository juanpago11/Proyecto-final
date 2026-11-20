import os
import pandas as pd
import streamlit as st
from influxdb_client import InfluxDBClient

st.set_page_config(page_title="Sistema de riego inteligente", page_icon="🌱", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.18);
    padding: 14px; border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# Configuración de InfluxDB. El token se lee desde secrets o una variable de entorno.
DEFAULT_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
DEFAULT_ORG = "Clase de computacion fisica"
DEFAULT_BUCKET = "Sprout"
MEASUREMENT = "tanque_planta"

# Umbrales de humedad definidos en el firmware del ESP32.
HUMEDAD_SECA = 20.0
HUMEDAD_HUMEDA = 55.0

# Umbrales provisionales del nivel del tanque. Ajustar tras calibrar el sensor.
NIVEL_BAJO = 20.0
NIVEL_ALTO = 60.0


def obtener_configuracion():
    try:
        url = st.secrets.get("INFLUXDB_URL", DEFAULT_URL)
        org = st.secrets.get("INFLUXDB_ORG", DEFAULT_ORG)
        bucket = st.secrets.get("INFLUXDB_BUCKET", DEFAULT_BUCKET)
        token = st.secrets.get("INFLUXDB_TOKEN", os.getenv("INFLUXDB_TOKEN", ""))
    except Exception:
        url = os.getenv("INFLUXDB_URL", DEFAULT_URL)
        org = os.getenv("INFLUXDB_ORG", DEFAULT_ORG)
        bucket = os.getenv("INFLUXDB_BUCKET", DEFAULT_BUCKET)
        token = os.getenv("INFLUXDB_TOKEN", "")
    return url, org, bucket, token


@st.cache_resource
def crear_cliente(url, token, org):
    return InfluxDBClient(url=url, token=token, org=org, timeout=15000)


def consultar_influx(url, token, org, bucket, periodo):
    cliente = crear_cliente(url, token, org)
    flux = f"""
from(bucket: "{bucket}")
    |> range(start: -{periodo})
    |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT}")
    |> filter(fn: (r) =>
        r["_field"] == "nivel_agua" or
        r["_field"] == "humedad_suelo" or
        r["_field"] == "nivel_raw" or
        r["_field"] == "humedad_raw" or
        r["_field"] == "estado_suelo" or
        r["_field"] == "estado_nivel" or
        r["_field"] == "alerta_nivel" or
        r["_field"] == "alerta_humedad" or
        r["_field"] == "valvula"
    )
    |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> sort(columns: ["_time"])
"""
    resultado = cliente.query_api().query_data_frame(query=flux, org=org)
    if isinstance(resultado, list):
        frames = [x for x in resultado if isinstance(x, pd.DataFrame) and not x.empty]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
    else:
        df = resultado
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "_time" in df.columns:
        df["_time"] = pd.to_datetime(df["_time"], errors="coerce", utc=True)
        df = df.dropna(subset=["_time"]).sort_values("_time")
    for col in ["nivel_agua", "humedad_suelo", "nivel_raw", "humedad_raw"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def interpretar_humedad(valor):
    if pd.isna(valor):
        return "SIN DATOS", "No se recibió una medición válida de humedad."
    if valor < HUMEDAD_SECA:
        return "SECO", "El suelo tiene poca humedad. El sistema identifica una necesidad de riego."
    if valor < HUMEDAD_HUMEDA:
        return "INTERMEDIO", "El suelo conserva humedad, pero no alcanza el umbral configurado como húmedo."
    return "HÚMEDO", "El suelo tiene humedad suficiente según el umbral configurado."


def interpretar_nivel(valor):
    if pd.isna(valor):
        return "SIN DATOS", "No se recibió una medición válida del nivel de agua."
    if valor < NIVEL_BAJO:
        return "BAJO", "El tanque tiene poca agua. Conviene rellenarlo."
    if valor < NIVEL_ALTO:
        return "MEDIO", "El tanque tiene una reserva intermedia. Vigila su evolución."
    return "ALTO", "El tanque tiene una reserva de agua alta."


def mostrar_estado(titulo, estado, explicacion, icono, valor):
    st.markdown(f"### {icono} {titulo}")
    st.markdown(f"**Estado: {estado}**")
    st.write(explicacion)
    if not pd.isna(valor):
        st.progress(max(0.0, min(float(valor) / 100.0, 1.0)))


# ----------------------- Interfaz -----------------------
st.title("🌱 Sistema inteligente de monitoreo de riego")
st.write(
    "Panel de seguimiento e interpretación de la humedad del suelo y del nivel "
    "de agua del tanque. Los datos se consultan directamente desde InfluxDB."
)

url, org, bucket, token = obtener_configuracion()

with st.sidebar:
    st.header("⚙️ Configuración")
    periodo_label = st.selectbox(
        "Periodo de consulta",
        ["Última hora", "Últimas 6 horas", "Últimas 24 horas", "Últimos 7 días"],
        index=2,
    )
    periodos = {
        "Última hora": "1h",
        "Últimas 6 horas": "6h",
        "Últimas 24 horas": "24h",
        "Últimos 7 días": "7d",
    }
    st.caption(f"Measurement: `{MEASUREMENT}`")
    st.caption(f"Bucket: `{bucket}`")
    st.divider()
    st.subheader("Umbrales")
    st.write(f"Humedad seca: menos de {HUMEDAD_SECA:.0f} %")
    st.write(f"Humedad adecuada: desde {HUMEDAD_HUMEDA:.0f} %")
    st.caption("Los umbrales del nivel del tanque son provisionales.")
    st.write(f"Nivel bajo: menos de {NIVEL_BAJO:.0f} %")
    st.write(f"Nivel alto: desde {NIVEL_ALTO:.0f} %")

if not token:
    st.error("Falta el token de InfluxDB. Configúralo en `.streamlit/secrets.toml`.")
    st.code(
        f'INFLUXDB_URL = "{DEFAULT_URL}"\n'
        f'INFLUXDB_ORG = "{DEFAULT_ORG}"\n'
        f'INFLUXDB_BUCKET = "{DEFAULT_BUCKET}"\n'
        'INFLUXDB_TOKEN = "PEGA_AQUI_UN_TOKEN_NUEVO"',
        language="toml",
    )
    st.stop()

col_refresh, col_info = st.columns([1, 3])
with col_refresh:
    actualizar = st.button("🔄 Actualizar datos", use_container_width=True)
with col_info:
    st.caption(f"Consulta: {periodo_label}. Actualiza para solicitar una consulta nueva.")

try:
    if actualizar:
        st.cache_resource.clear()
    with st.spinner("Consultando InfluxDB..."):
        df = consultar_influx(url, token, org, bucket, periodos[periodo_label])
except Exception as error:
    st.error("No fue posible consultar InfluxDB.")
    st.exception(error)
    st.info("Verifica URL, token, organización, bucket, conexión y envío de datos del ESP32.")
    st.stop()

if df.empty:
    st.warning("No hay mediciones en el periodo seleccionado. Comprueba que el ESP32 esté enviando datos.")
    st.stop()

ultima = df.iloc[-1]
humedad = ultima.get("humedad_suelo", float("nan"))
nivel = ultima.get("nivel_agua", float("nan"))
valvula = ultima.get("valvula", None)

if pd.isna(humedad) and pd.isna(nivel):
    st.error("Hay registros, pero no contienen los campos `humedad_suelo` ni `nivel_agua`.")
    st.dataframe(df.tail())
    st.stop()

hum_estado, hum_texto = interpretar_humedad(humedad)
nivel_estado, nivel_texto = interpretar_nivel(nivel)

st.subheader("📍 Estado actual del sistema")
m1, m2, m3 = st.columns(3)
m1.metric("💧 Humedad del suelo", "Sin dato" if pd.isna(humedad) else f"{humedad:.1f} %")
m2.metric("🛢️ Nivel del tanque", "Sin dato" if pd.isna(nivel) else f"{nivel:.1f} %")

if valvula is None or pd.isna(valvula):
    valvula_texto = "Sin dato"
else:
    valvula_texto = "ABIERTA" if bool(int(valvula)) else "CERRADA"
m3.metric("🚰 Válvula", valvula_texto)

if "_time" in ultima.index and pd.notna(ultima["_time"]):
    st.caption(f"Última medición registrada: {ultima['_time']}")

st.divider()
st.subheader("🧠 Interpretación de las mediciones")
c1, c2 = st.columns(2)
with c1:
    mostrar_estado("Humedad del suelo", hum_estado, hum_texto, "🌱", humedad)
with c2:
    mostrar_estado("Nivel del tanque", nivel_estado, nivel_texto, "🛢️", nivel)

st.subheader("💡 Diagnóstico conjunto")
if hum_estado == "SIN DATOS" or nivel_estado == "SIN DATOS":
    st.warning("No hay suficientes mediciones para generar un diagnóstico conjunto.")
elif hum_estado == "SECO" and nivel_estado == "BAJO":
    st.error("El suelo necesita riego, pero el tanque tiene poca agua. Rellena el tanque y verifica la disponibilidad de agua.")
elif hum_estado == "SECO":
    st.warning("El suelo está seco y el tanque no está en nivel bajo. El firmware debería solicitar riego automático según sus umbrales.")
elif nivel_estado == "BAJO":
    st.warning("El tanque está bajo. Conviene rellenarlo para mantener disponible el sistema de riego.")
elif hum_estado == "HÚMEDO":
    st.success("El suelo presenta humedad suficiente y el tanque no está bajo. No se identifica una necesidad inmediata de riego.")
else:
    st.info("El suelo presenta humedad intermedia y el tanque tiene una reserva aceptable. Continúa monitoreando.")

if valvula is not None and not pd.isna(valvula):
    if bool(int(valvula)):
        st.info("La telemetría indica que la válvula está ABIERTA. Esto no confirma por sí solo que exista caudal real de agua.")
    else:
        st.caption("La telemetría indica que la válvula está CERRADA.")

# ----------------------- Histórico -----------------------
st.divider()
st.subheader("📈 Evolución de los sensores")
disponibles = [
    campo for campo in ["humedad_suelo", "nivel_agua"]
    if campo in df.columns and df[campo].notna().any()
]
if disponibles and "_time" in df.columns:
    historico = df.dropna(subset=["_time"]).set_index("_time")[disponibles]
    historico = historico.rename(columns={
        "humedad_suelo": "Humedad del suelo (%)",
        "nivel_agua": "Nivel del tanque (%)",
    })
    st.line_chart(historico)
    st.caption("Las dos variables están expresadas en porcentaje.")
else:
    st.info("No hay datos históricos suficientes para graficar.")

with st.expander("📊 Estadísticas del periodo"):
    columnas = [c for c in ["humedad_suelo", "nivel_agua", "humedad_raw", "nivel_raw"] if c in df.columns]
    if columnas:
        st.dataframe(df[columnas].describe().T.round(2), use_container_width=True)

with st.expander("🔎 Registros recibidos desde InfluxDB"):
    visibles = [
        c for c in [
            "_time", "humedad_suelo", "nivel_agua", "estado_suelo",
            "estado_nivel", "valvula", "alerta_humedad", "alerta_nivel",
        ] if c in df.columns
    ]
    st.dataframe(df[visibles].tail(500), use_container_width=True)
    st.download_button(
        "⬇️ Descargar mediciones CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="mediciones_sistema_riego.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Sistema de monitoreo de riego con ESP32 e InfluxDB. Las interpretaciones dependen de la calibración real de los sensores.")

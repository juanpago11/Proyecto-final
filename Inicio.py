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

CAMPOS_OBJETIVO = [
    "nivel_agua",
    "humedad_suelo",
    "nivel_raw",
    "humedad_raw",
    "estado_suelo",
    "estado_nivel",
    "alerta_nivel",
    "alerta_humedad",
    "valvula",
]

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


def normalizar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Detectar columna de tiempo común en InfluxDB y Grafana
    time_candidates = ["_time", "Time", "time", "timestamp", "@timestamp"]
    col_tiempo = next((c for c in time_candidates if c in df.columns), None)
    if col_tiempo is not None:
        # Si viene en milisegundos o segundos numéricos, convertir adecuadamente
        if pd.api.types.is_numeric_dtype(df[col_tiempo]):
            max_val = df[col_tiempo].dropna().max() if not df[col_tiempo].dropna().empty else 0
            unit = "ms" if max_val > 1e12 else "s"
            df["_time"] = pd.to_datetime(df[col_tiempo], unit=unit, errors="coerce", utc=True)
        else:
            df["_time"] = pd.to_datetime(df[col_tiempo], errors="coerce", utc=True)
        df = df.dropna(subset=["_time"]).sort_values("_time")

    for col in ["nivel_agua", "humedad_suelo", "nivel_raw", "humedad_raw", "valvula"]:
        if col in df.columns:
            if df[col].dtype == object:
                limpio = (
                    df[col]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .str.strip()
                )
                df[col] = pd.to_numeric(limpio, errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def mapear_nombre_campo(nombre_col: str):
    """Intenta asociar nombres de columnas exportadas por Grafana con los campos del sistema."""
    limpio = str(nombre_col).strip()
    if limpio in CAMPOS_OBJETIVO:
        return limpio

    # Ejemplos típicos en Grafana: "tanque_planta.humedad_suelo" o "humedad_suelo {bucket=\"Sprout\"}"
    base = limpio.split("{")[0].strip()
    if "." in base:
        base = base.split(".")[-1].strip()
    if base in CAMPOS_OBJETIVO:
        return base

    for campo in CAMPOS_OBJETIVO:
        if campo in limpio.lower():
            return campo
    return None


def cargar_csv_grafana(archivo_subido) -> pd.DataFrame:
    """Lee un CSV exportado desde Grafana (formato ancho o largo) y lo normaliza."""
    df_raw = pd.read_csv(archivo_subido, sep=None, engine="python")
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    time_candidates = ["_time", "Time", "time", "timestamp", "@timestamp"]
    col_tiempo = next((c for c in time_candidates if c in df_raw.columns), None)

    # Caso 1: Formato largo (p. ej. columnas de tiempo + serie/campo + valor)
    serie_candidates = ["_field", "Metric", "metric", "Field", "field", "Series", "series", "name"]
    valor_candidates = ["_value", "Value", "value"]
    col_serie = next((c for c in serie_candidates if c in df_raw.columns), None)
    col_valor = next((c for c in valor_candidates if c in df_raw.columns), None)

    if col_tiempo and col_serie and col_valor:
        temp = df_raw[[col_tiempo, col_serie, col_valor]].copy()
        temp["_campo_mapeado"] = temp[col_serie].apply(mapear_nombre_campo)
        temp = temp.dropna(subset=["_campo_mapeado"])
        if temp.empty:
            return pd.DataFrame()
        pivote = temp.pivot_table(
            index=col_tiempo,
            columns="_campo_mapeado",
            values=col_valor,
            aggfunc="last",
        ).reset_index()
        return normalizar_dataframe(pivote)

    # Caso 2: Formato ancho (cada serie/campo en su columna)
    renombrar = {}
    for col in df_raw.columns:
        campo = mapear_nombre_campo(col)
        if campo and campo not in df_raw.columns and campo not in renombrar.values():
            renombrar[col] = campo
    df_ancho = df_raw.rename(columns=renombrar)
    return normalizar_dataframe(df_ancho)


def consultar_influx(url, token, org, bucket, periodo):
    cliente = crear_cliente(url, token, org)
    campos_flux = " or ".join([f'r["_field"] == "{c}"' for c in CAMPOS_OBJETIVO])
    flux = f"""
from(bucket: "{bucket}")
    |> range(start: -{periodo})
    |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT}")
    |> filter(fn: (r) => {campos_flux})
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
    return normalizar_dataframe(df)


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
    "de agua del tanque. Consulta datos en vivo desde InfluxDB o carga un CSV exportado desde Grafana."
)

url, org, bucket, token = obtener_configuracion()

with st.sidebar:
    st.header("⚙️ Configuración")
    fuente_datos = st.radio(
        "Fuente de datos",
        ["InfluxDB", "CSV de Grafana"],
        index=0,
    )
    st.divider()

    if fuente_datos == "InfluxDB":
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
    else:
        archivo_csv = st.file_uploader(
            "Sube el CSV exportado desde Grafana",
            type=["csv"],
            help="En Grafana: Inspect → Data → Download CSV",
        )

    st.divider()
    st.subheader("Umbrales")
    st.write(f"Humedad seca: menos de {HUMEDAD_SECA:.0f} %")
    st.write(f"Humedad adecuada: desde {HUMEDAD_HUMEDA:.0f} %")
    st.caption("Los umbrales del nivel del tanque son provisionales.")
    st.write(f"Nivel bajo: menos de {NIVEL_BAJO:.0f} %")
    st.write(f"Nivel alto: desde {NIVEL_ALTO:.0f} %")

if fuente_datos == "InfluxDB":
    if not token:
        st.error("Falta el token de InfluxDB. Configúralo en `.streamlit/secrets.toml` o cambia a **CSV de Grafana**.")
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

else:
    if not archivo_csv:
        st.info("Selecciona y sube un archivo **CSV exportado desde Grafana** en la barra lateral para analizarlo.")
        st.stop()

    try:
        with st.spinner("Procesando CSV de Grafana..."):
            df = cargar_csv_grafana(archivo_csv)
    except Exception as error:
        st.error("No fue posible leer o procesar el archivo CSV.")
        st.exception(error)
        st.stop()

if df.empty:
    st.warning("No hay mediciones válidas en la fuente seleccionada.")
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

with st.expander("🔎 Registros cargados"):
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
st.caption("Sistema de monitoreo de riego con ESP32 e InfluxDB/Grafana. Las interpretaciones dependen de la calibración real de los sensores.")

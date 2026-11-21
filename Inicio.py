import io
import re

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ============================================================
#  Flujo: CirkitDesigner (ESP32) → InfluxDB → Grafana → CSV → Streamlit
#  El CSV descargado de Grafana se sube o se pega en la app
# ============================================================
st.set_page_config(page_title="Sprout - Riego Inteligente", page_icon="💧", layout="wide")

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stAlert { margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

CANDIDATAS_TIEMPO = ["time", "_time", "timestamp", "fecha", "date", "datetime"]
METADATOS = ["result", "table", "_start", "_stop", "_measurement", "device"]

# Mismos umbrales que usa el ESP32
HUMEDAD_SECA = 20.0
HUMEDAD_HUMEDA = 55.0
NIVEL_ALERTA_ON = 16.84
NIVEL_ALERTA_OFF = 29.18


# ============================================================
#  Lectura del CSV de Grafana
# ============================================================
def limpiar_nombre(columna):
    """Grafana nombra las series como 'tanque_planta humedad_suelo {device="ESP32"}'."""
    nombre = re.sub(r"\s*\{.*\}\s*$", "", str(columna)).strip()
    return re.split(r"[ .]", nombre)[-1] if nombre else nombre


@st.cache_data
def leer_csv(contenido: bytes):
    df = pd.read_csv(io.BytesIO(contenido), sep=None, engine="python", comment="#")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # Formato largo de InfluxDB (_time, _field, _value)
    if {"_field", "_value"}.issubset(df.columns):
        df = df[df["_field"] != "_field"].copy()
        df["_value"] = pd.to_numeric(df["_value"], errors="coerce")
        df = df.pivot_table(index="_time", columns="_field", values="_value", aggfunc="mean").reset_index()
        df.columns.name = None

    # Formato largo de Grafana (Series, Time, Value)
    minus = {str(c).lower(): c for c in df.columns}
    if {"series", "time", "value"}.issubset(minus):
        df = df.pivot_table(index=minus["time"], columns=minus["series"],
                            values=minus["value"], aggfunc="mean").reset_index()
        df.columns.name = None

    df.columns = [limpiar_nombre(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df.drop(columns=[c for c in METADATOS if c in df.columns])


def preparar_datos(df):
    df = df.copy()

    col_tiempo = next(
        (c for c in df.columns if str(c).strip().lower() in CANDIDATAS_TIEMPO), None
    )
    if col_tiempo is not None:
        if pd.api.types.is_numeric_dtype(df[col_tiempo]):
            tiempo = pd.to_datetime(df[col_tiempo], unit="ms", errors="coerce", utc=True)  # epoch de Grafana
        else:
            tiempo = pd.to_datetime(df[col_tiempo], errors="coerce")
        if getattr(tiempo.dt, "tz", None) is not None:
            tiempo = tiempo.dt.tz_convert("America/Bogota").dt.tz_localize(None)
        df[col_tiempo] = tiempo
        df = df.dropna(subset=[col_tiempo]).set_index(col_tiempo).sort_index()
        df.index.name = "Time"
    else:
        df.index.name = "Registro"

    # Texto que en realidad es número (p. ej. "23,5" o "45 %")
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            texto = df[c].astype(str).str.replace(",", ".", regex=False)
            convertida = pd.to_numeric(
                texto.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False), errors="coerce"
            )
            if convertida.notna().mean() >= 0.9:
                df[c] = convertida

    numericas = df.select_dtypes(include="number").columns.tolist()
    return df, numericas, col_tiempo


def buscar(variables, nombre_exacto, claves):
    if nombre_exacto in variables:
        return variables.index(nombre_exacto)
    for i, v in enumerate(variables):
        if any(k in str(v).lower() for k in claves) and "raw" not in str(v).lower():
            return i
    return 0


def normalizar(df):
    rango = (df.max() - df.min()).replace(0, 1)
    return (df - df.min()) / rango


# ============================================================
#  Interpretación (misma lógica que el ESP32)
# ============================================================
def alerta_nivel(serie, on, off):
    """Alerta de tanque bajo con histéresis."""
    activa, salida = False, []
    for v in serie:
        if not pd.isna(v):
            if not activa and v <= on:
                activa = True
            elif activa and v >= off:
                activa = False
        salida.append(activa)
    return pd.Series(salida, index=serie.index)


def interpretar(hum, niv, seco, humedo, on, off):
    df = pd.DataFrame({"humedad_suelo (%)": hum.round(1), "nivel_agua (%)": niv.round(1)})
    df["estado_suelo"] = pd.cut(
        hum, [-np.inf, seco, humedo, np.inf],
        labels=["SECO", "INTERMEDIO", "HUMEDO"], right=False
    ).astype(str)

    bajo = alerta_nivel(niv, on, off)
    df["estado_nivel"] = np.where(bajo, "BAJO", "NORMAL")

    es_seco = df["estado_suelo"] == "SECO"
    es_humedo = df["estado_suelo"] == "HUMEDO"
    df["recomendacion"] = np.select(
        [es_seco & bajo, es_seco, es_humedo & bajo, es_humedo, bajo],
        ["Regar, pero primero llenar el tanque",
         "Regar: suelo seco",
         "No regar · llenar el tanque",
         "No regar: suelo húmedo",
         "Llenar el tanque"],
        default="Sin acción: humedad intermedia"
    )
    return df


def tendencia(serie, puntos=20):
    s = serie.dropna().iloc[-puntos:]
    if len(s) < 3:
        return None, ""
    if isinstance(s.index, pd.DatetimeIndex):
        x = np.asarray((s.index - s.index[0]).total_seconds(), dtype=float) / 3600
        unidad = "h"
    else:
        x = np.arange(len(s), dtype=float)
        unidad = "registro"
    if x[-1] <= 0:
        return None, ""
    return float(np.polyfit(x, s.to_numpy(dtype=float), 1)[0]), unidad


def texto_tendencia(p, unidad):
    if p is None:
        return "sin datos suficientes"
    flecha = "↗ subiendo" if p > 0.5 else "↘ bajando" if p < -0.5 else "→ estable"
    return f"{flecha} ({p:+.1f} %/{unidad})"


def grafico_con_umbrales(serie, titulo, umbrales, con_tiempo):
    eje_x = serie.index.name or "Registro"
    datos = serie.rename("valor").reset_index()
    tipo_x = "T" if con_tiempo else "Q"

    linea = alt.Chart(datos).mark_line().encode(
        x=alt.X(f"{eje_x}:{tipo_x}", title="Tiempo" if con_tiempo else "Registro"),
        y=alt.Y("valor:Q", title=titulo, scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip(f"{eje_x}:{tipo_x}"), alt.Tooltip("valor:Q", format=".1f")],
    )
    u = pd.DataFrame(umbrales, columns=["y", "Umbral", "color"])
    reglas = alt.Chart(u).mark_rule(strokeDash=[6, 4], size=2).encode(
        y="y:Q",
        color=alt.Color("Umbral:N", scale=alt.Scale(domain=u["Umbral"].tolist(),
                                                    range=u["color"].tolist())),
    )
    st.altair_chart((linea + reglas).interactive())


# ============================================================
#  Encabezado y barra lateral
# ============================================================
st.title("💧 Sprout · Interpretación del sistema de riego")
st.markdown("""
    **Flujo de datos:** CirkitDesigner (ESP32) → InfluxDB → Grafana → CSV → Streamlit.
    Suba o pegue el CSV descargado de Grafana: la app interpreta la **humedad del suelo**
    y el **nivel de agua del tanque** y recomienda cuándo regar.
""")

st.sidebar.header("⚙️ Umbrales (iguales al ESP32)")
hum_seco, hum_humedo = st.sidebar.slider(
    "Humedad del suelo: seco / húmedo (%)", 0.0, 100.0, (HUMEDAD_SECA, HUMEDAD_HUMEDA), 0.5
)
niv_on, niv_off = st.sidebar.slider(
    "Nivel del tanque: alerta ON / OFF (%)", 0.0, 100.0, (NIVEL_ALERTA_ON, NIVEL_ALERTA_OFF), 0.5
)


def main(df_crudo, origen):
    df1, variables, col_tiempo = preparar_datos(df_crudo)
    con_tiempo = col_tiempo is not None

    if df1.empty or not variables:
        st.error("El CSV no tiene datos numéricos.")
        return

    st.success(f"Datos de **{origen}** · **{len(df1)}** registros · variables: "
               + ", ".join(f"`{v}`" for v in variables))

    st.sidebar.header("🔌 Columnas del CSV")
    col_hum = st.sidebar.selectbox("Humedad del suelo (%)", variables,
                                   index=buscar(variables, "humedad_suelo", ["humedad", "hum"]))
    col_niv = st.sidebar.selectbox("Nivel de agua (%)", variables,
                                   index=buscar(variables, "nivel_agua", ["nivel", "agua", "level"]))

    hum = df1[col_hum].clip(0, 100)
    niv = df1[col_niv].clip(0, 100)
    interp = interpretar(hum, niv, hum_seco, hum_humedo, niv_on, niv_off)

    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        ["🌱 Interpretación", "📈 Visualización", "📊 Estadísticas", "🔍 Filtros", "🗺️ Información del Sitio"]
    )

    # ---------------- TAB 0 · Interpretación ----------------
    with tab0:
        validos = interp.dropna(subset=["humedad_suelo (%)", "nivel_agua (%)"], how="all")
        ultimo = validos.iloc[-1]
        anterior = validos.iloc[-2] if len(validos) > 1 else ultimo

        st.subheader("Estado en la última lectura")
        st.caption(f"Última lectura: {validos.index[-1]:%Y-%m-%d %H:%M}" if con_tiempo
                   else f"Última lectura: registro #{validos.index[-1]}")

        iconos = {"SECO": "🔴", "INTERMEDIO": "🟡", "HUMEDO": "🟢", "BAJO": "🟠", "NORMAL": "🟢"}
        p_h, u_h = tendencia(hum)
        p_n, u_n = tendencia(niv)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{iconos.get(ultimo['estado_suelo'], '')} Humedad · {ultimo['estado_suelo']}",
                  f"{ultimo['humedad_suelo (%)']:.1f} %",
                  f"{ultimo['humedad_suelo (%)'] - anterior['humedad_suelo (%)']:+.1f} %")
        c1.caption(f"Tendencia: {texto_tendencia(p_h, u_h)}")

        c2.metric(f"{iconos.get(ultimo['estado_nivel'], '')} Tanque · {ultimo['estado_nivel']}",
                  f"{ultimo['nivel_agua (%)']:.1f} %",
                  f"{ultimo['nivel_agua (%)'] - anterior['nivel_agua (%)']:+.1f} %")
        c2.caption(f"Tendencia: {texto_tendencia(p_n, u_n)}")
        c2.progress(int(round(ultimo["nivel_agua (%)"])) if not pd.isna(ultimo["nivel_agua (%)"]) else 0)

        if "valvula" in df1.columns:
            abierta = df1["valvula"].dropna()
            c3.metric("🚰 Válvula (servo)", "ABIERTA" if len(abierta) and abierta.iloc[-1] >= 0.5 else "CERRADA")

        rec = ultimo["recomendacion"]
        if rec.startswith("Regar, pero"):
            st.error(f"💡 {rec}: la planta necesita agua pero el tanque está bajo.")
        elif rec.startswith("Regar"):
            st.warning(f"💡 {rec}. La válvula debería abrirse automáticamente.")
        elif "tanque" in rec:
            st.warning(f"💡 {rec}.")
        else:
            st.success(f"💡 {rec}.")

        if p_n is not None and u_n == "h" and p_n < -0.1:
            horas = max(ultimo["nivel_agua (%)"] - niv_on, 0) / -p_n
            st.write(f"⏱️ Al ritmo de ese momento ({-p_n:.1f} %/h) el tanque llegaría a alerta en **~{horas:.1f} h**.")

        st.subheader("Evolución con umbrales")
        g1, g2 = st.columns(2)
        with g1:
            grafico_con_umbrales(hum, "Humedad del suelo (%)", [
                (hum_seco, f"Seco (<{hum_seco:g} %)", "#d62728"),
                (hum_humedo, f"Húmedo (≥{hum_humedo:g} %)", "#2ca02c"),
            ], con_tiempo)
        with g2:
            grafico_con_umbrales(niv, "Nivel del tanque (%)", [
                (niv_on, f"Alerta ON (≤{niv_on:g} %)", "#ff7f0e"),
                (niv_off, f"Alerta OFF (≥{niv_off:g} %)", "#1f77b4"),
            ], con_tiempo)

        st.subheader("Interpretación del periodo")
        dist_s = (interp["estado_suelo"].value_counts(normalize=True)
                  .reindex(["SECO", "INTERMEDIO", "HUMEDO"], fill_value=0) * 100)
        dist_n = (interp["estado_nivel"].value_counts(normalize=True)
                  .reindex(["BAJO", "NORMAL"], fill_value=0) * 100)

        d1, d2 = st.columns(2)
        with d1:
            st.write("**% del tiempo en cada estado del suelo**")
            st.bar_chart(dist_s.round(1))
        with d2:
            st.write("**% del tiempo en cada estado del tanque**")
            st.bar_chart(dist_n.round(1))

        h, n = hum.dropna(), niv.dropna()
        resumen = [
            f"🌱 Humedad promedio **{h.mean():.1f} %** (mín. {h.min():.1f} %, máx. {h.max():.1f} %). "
            f"Suelo **seco el {dist_s['SECO']:.0f} %** del tiempo, intermedio el "
            f"{dist_s['INTERMEDIO']:.0f} % y húmedo el {dist_s['HUMEDO']:.0f} %.",
            f"🛢️ Nivel promedio **{n.mean():.1f} %** (mín. {n.min():.1f} %). "
            f"Tanque en **alerta BAJO el {dist_n['BAJO']:.0f} %** del tiempo.",
            f"📉 Consumo acumulado del tanque: **{-n.diff().clip(upper=0).sum():.1f} %** · "
            f"recargas: **{n.diff().clip(lower=0).sum():.1f} %**.",
        ]
        if "valvula" in df1.columns:
            v = (df1["valvula"].dropna() >= 0.5).astype(int)
            resumen.append(f"🚿 La válvula se abrió **{int((v.diff() == 1).sum())}** vez/veces y estuvo "
                           f"abierta el **{v.mean() * 100:.0f} %** de los registros.")
        if dist_s["SECO"] > 30:
            resumen.append("⚠️ El suelo pasó mucho tiempo seco: revise que el servo abra o que llegue agua.")
        if dist_s["HUMEDO"] > 60:
            resumen.append("⚠️ El suelo pasó mucho tiempo húmedo: el riego podría ser excesivo.")
        if dist_n["BAJO"] > 0:
            resumen.append("⚠️ Hubo periodos con el tanque bajo: rellénelo para que el riego no falle.")

        for linea in resumen:
            st.markdown(f"- {linea}")

        st.subheader("📄 Datos interpretados")
        st.dataframe(interp)
        st.download_button(
            label="⬇️ Descargar CSV interpretado",
            data=interp.to_csv().encode("utf-8"),
            file_name="riego_interpretado.csv",
            mime="text/csv",
        )

    # ---------------- TAB 1 · Visualización ----------------
    with tab1:
        st.subheader("Visualización de Datos")
        col_a, col_b = st.columns([3, 1])
        with col_a:
            vars_graf = st.multiselect("Variables a graficar", variables,
                                       default=list(dict.fromkeys([col_hum, col_niv])))
        with col_b:
            chart_type = st.selectbox("Tipo de gráfico", ["Línea", "Área", "Barra"])

        normalizado = st.checkbox("Normalizar escalas (0–1)",
                                  help="Útil para comparar valores RAW (0–4095) con porcentajes.")

        if not vars_graf:
            st.warning("Seleccione al menos una variable.")
        else:
            datos_graf = df1[vars_graf]
            if normalizado:
                datos_graf = normalizar(datos_graf)
            if chart_type == "Línea":
                st.line_chart(datos_graf)
            elif chart_type == "Área":
                st.area_chart(datos_graf)
            else:
                st.bar_chart(datos_graf)

        if st.checkbox("Mostrar datos crudos"):
            st.dataframe(df1)

    # ---------------- TAB 2 · Estadísticas ----------------
    with tab2:
        st.subheader("Análisis Estadístico")
        st.dataframe(df1[variables].describe().T.round(2))

        var_stats = st.selectbox("Variable", variables, key="var_stats", index=variables.index(col_hum))
        serie = df1[var_stats].dropna()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Promedio", f"{serie.mean():.2f}")
        m2.metric("Máximo", f"{serie.max():.2f}")
        m3.metric("Mínimo", f"{serie.min():.2f}")
        m4.metric("Desv. estándar", f"{serie.std():.2f}")

        if len(variables) >= 2:
            st.write("#### Correlación entre variables")
            st.dataframe(df1[variables].corr().round(2))
            st.caption("Una correlación negativa entre humedad y nivel indica que el riego "
                       "está consumiendo el agua del tanque.")

    # ---------------- TAB 3 · Filtros ----------------
    with tab3:
        st.subheader("Filtros de Datos")
        var_filtro = st.selectbox("Variable a filtrar", variables, key="var_filtro")
        min_value = float(df1[var_filtro].min())
        max_value = float(df1[var_filtro].max())

        if min_value == max_value:
            st.warning(f"⚠️ Todos los valores de `{var_filtro}` son iguales: {min_value:.2f}")
            st.dataframe(df1)
        else:
            rango = st.slider(f"Rango de {var_filtro}", min_value, max_value,
                              (min_value, max_value), key=f"rango_{var_filtro}")
            filtrado = df1[df1[var_filtro].between(rango[0], rango[1])]
            st.write(f"**{len(filtrado)}** de {len(df1)} registros:")
            st.dataframe(filtrado)
            st.download_button(
                label="⬇️ Descargar datos filtrados (CSV)",
                data=filtrado.to_csv().encode("utf-8"),
                file_name=f"datos_filtrados_{var_filtro}.csv",
                mime="text/csv",
            )

    # ---------------- TAB 4 · Sitio ----------------
    with tab4:
        st.subheader("Información del Sitio de Medición")
        st.map(pd.DataFrame({"lat": [6.2006], "lon": [-75.5783]}), zoom=15)

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Ubicación")
            st.write("**Universidad EAFIT**")
            st.write("- Latitud: 6.2006")
            st.write("- Longitud: -75.5783")
            st.write("- Altitud: ~1,495 m s. n. m.")
        with col2:
            st.write("### Sistema")
            st.write("- ESP32 · sensor de humedad (GPIO32) · sensor de nivel (GPIO33)")
            st.write("- Válvula: servo en GPIO13 · LEDs rojo/amarillo/verde")
            st.write("- Datos: CirkitDesigner → InfluxDB → Grafana → CSV → Streamlit")
            st.write(f"- Registros: {len(df1)}")
            if con_tiempo and len(df1) > 0:
                st.write(f"- Desde: {df1.index.min():%Y-%m-%d %H:%M}")
                st.write(f"- Hasta: {df1.index.max():%Y-%m-%d %H:%M}")


# ============================================================
#  Cargar el CSV de Grafana: subir archivo o pegar el texto
# ============================================================
st.subheader("📥 CSV de Grafana")
modo_subir, modo_pegar = st.tabs(["📁 Subir archivo", "📋 Pegar texto"])

contenido, origen = None, None
with modo_subir:
    archivo = st.file_uploader("Arrastre aquí el CSV descargado de Grafana", type=["csv", "txt"])
    if archivo is not None:
        contenido, origen = archivo.getvalue(), archivo.name
with modo_pegar:
    texto = st.text_area("Pegue aquí el contenido del CSV", height=200,
                         placeholder='"Time","humedad_suelo","nivel_agua"\n2026-11-21 13:20:00,46.6,0')
    if texto.strip() and contenido is None:
        contenido, origen = texto.strip().encode("utf-8"), "CSV pegado"

if contenido is None:
    st.info("En Grafana: menú del panel → **Inspect → Data** → activar *Series joined by time* "
            "→ **Download CSV**. Luego súbalo o pegue su contenido aquí.")
else:
    try:
        main(leer_csv(contenido), origen)
    except Exception as e:
        st.error(f"Error al procesar el CSV: {e}")
        st.info("Verifique que el CSV de Grafana tenga las columnas humedad_suelo y nivel_agua.")

st.markdown("""
    ---
    Sprout · Sistema de riego con ESP32 + InfluxDB + Grafana
    Universidad EAFIT, Medellín, Colombia
""")

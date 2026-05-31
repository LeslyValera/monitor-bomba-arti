import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

st.set_page_config(
    page_title="ARTI · Monitor de Bomba",
    layout="wide",
    page_icon="⚙️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

header[data-testid="stHeader"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

#MainMenu, footer, div[data-testid="stToolbar"] {
    display: none !important;
    visibility: hidden !important;
}

.block-container {
    padding-top: 2rem !important;
}
            
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Diseño de las tarjetas KPI Superiores (TEMA CLARO) */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
    margin: 15px 0;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px;
    position: relative;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.kpi-card::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 4px;
    background: #3b82f6; border-radius: 0 0 10px 10px;
}
.kpi-card.warn::after { background: #f59e0b; }
.kpi-card.crit::after { background: #ef4444; }
.kpi-card.ok::after   { background: #10b981; }

.kpi-label { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: #64748b; margin-bottom: 5px; font-weight: 700; }
.kpi-value { font-family: 'Space Mono', monospace; font-size: 26px; font-weight: bold; color: #0f172a; }
.kpi-unit { font-size: 14px; color: #94a3b8; font-weight: normal; }
.kpi-delta { font-size: 12px; margin-top: 5px; color: #64748b; }
.kpi-delta.up { color: #ef4444; font-weight: bold; }
.kpi-delta.down { color: #10b981; font-weight: bold; }

/* Diseño de Telemetría (Recuadro del medio - TEMA CLARO) */
.telemetry-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 25px;
    font-family: 'Space Mono', monospace;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.telemetry-header { color: #3b82f6; font-size: 11px; letter-spacing: 0.15em; border-bottom: 1px solid #f1f5f9; padding-bottom: 10px; margin-bottom: 15px; font-weight: bold; }
.telemetry-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px dashed #f1f5f9; color: #334155; }
.telemetry-key { color: #64748b; font-weight: 600; }
.diag-value { color: #f59e0b; font-size: 18px; font-weight: bold; margin-top: 10px; }

/* Diseño del Plan de Acción (TEMA CLARO) */
.action-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 25px;
    border-left: 5px solid #3b82f6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.action-label { font-size: 11px; letter-spacing: 0.1em; color: #64748b; text-transform: uppercase; margin-bottom: 10px; font-family: 'Space Mono', monospace; font-weight: bold; }
.action-text { font-size: 15px; color: #1e293b; padding: 15px; background: #f8fafc; border-radius: 5px; line-height: 1.6; border: 1px solid #e2e8f0; }

</style>
""", unsafe_allow_html=True)

def obtener_recomendacion(falla):
    mapa = {
        'Falla Mecanica': (
            "Revisar alineación del eje, balanceo de masa e inspeccionar rodamientos. Detener operación si la vibración supera 5 mm/s.",
            "ACCIÓN CORRECTIVA INMEDIATA", "#ef4444", "fa-solid fa-triangle-exclamation"
        ),
        'Sobrecalentamiento': (
            "Verificar sistema de refrigeración del motor, ventilación y medir el aislamiento eléctrico. Reducir carga operativa.",
            "ALERTA TÉRMICA", "#f59e0b", "fa-solid fa-fire"
        ),
        'Problema de Lubricacion': (
            "Nivel de aceite bajo el mínimo seguro. Realizar cambio o rellenar fluido hidráulico antes de continuar la operación.",
            "ADVERTENCIA DE LUBRICACIÓN", "#f59e0b", "fa-solid fa-oil-can"
        ),
        'Sobrecarga Electrica': (
            "Monitorear consumo de corriente en el tablero. Verificar posible atasco mecánico o falla en el estator.",
            "ALERTA ELÉCTRICA", "#ef4444", "fa-solid fa-bolt"
        ),
        'Problema Hidraulico': (
            "Evaluar cavitación, revisar apertura de válvulas y limpiar filtros de succión.",
            "REVISIÓN HIDRÁULICA", "#3b82f6", "fa-solid fa-water"
        ),
        'Normal': (
            "La bomba opera dentro de rangos óptimos. Mantener el plan de mantenimiento preventivo rutinario.",
            "OPERACIÓN SEGURA", "#10b981", "fa-solid fa-circle-check"
        ),
    }
    return mapa.get(falla, ("Estado desconocido.", "—", "#94a3b8", "fa-solid fa-circle-info"))

try:
    df = pd.read_csv('bomba_limpio.csv')
    cols = ['Vibracion (mm/s)', 'Temperatura del motor (°C)', 'Corriente del motor (A)',
            'Presion (PSI)', 'Caudal (m³/h)', 'Nivel de Lubricacion (%)']
    X, y = df[cols], df['Tipo_Falla']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo.fit(X_train, y_train)
    accuracy_real = accuracy_score(y_test, modelo.predict(X_test)) * 100
except Exception as e:
    st.error(f"No se pudo cargar 'bomba_limpio.csv': {e}")
    st.stop()

with st.sidebar:
    st.title(":material/engineering: ARTI")
    st.caption("Monitor Predictivo")
    st.write("---")
    st.subheader(":material/tune: Control de Sensores")
    vibracion   = st.slider("Vibración (mm/s)",  0.0,  6.0,  3.5, 0.1)
    temperatura = st.slider("Temperatura (°C)",   40,  100,   70,   1)
    corriente   = st.slider("Corriente (A)",      15.0, 30.0, 21.0, 0.5)
    presion     = st.slider("Presión (PSI)",      50.0,130.0, 85.0, 0.5)
    caudal      = st.slider("Caudal (m³/h)",        50,  140,  120,   1)
    lubricacion = st.slider("Lubricación (%)",      0,  100,   75,   1)
    st.write("---")
    st.caption(f"Precisión del Modelo: {accuracy_real:.1f}%")

entrada = pd.DataFrame([{
    'Vibracion (mm/s)': vibracion, 'Temperatura del motor (°C)': temperatura,
    'Corriente del motor (A)': corriente, 'Presion (PSI)': presion,
    'Caudal (m³/h)': caudal, 'Nivel de Lubricacion (%)': lubricacion
}])

falla = modelo.predict(entrada[cols])[0]
probabilidades = modelo.predict_proba(entrada[cols])[0]
confianza = round(max(probabilidades) * 100, 1)

rec_texto, rec_titulo, rec_color, rec_icono = obtener_recomendacion(falla)

estado_class = "ok" if falla == "Normal" else ("crit" if confianza >= 85 else "warn")
estado_label = "OPERANDO" if falla == "Normal" else ("CRÍTICO" if confianza >= 85 else "ALERTA")
vib_class    = "down" if vibracion <= 3.6 else "up"
temp_class   = "down" if temperatura < 80 else "up"


st.title("Sistema de Monitoreo Predictivo")
st.markdown("Diagnóstico en tiempo real")
st.write("---")

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label"><i class="fa-solid fa-wave-square" style="color:#3b82f6; margin-right:5px;"></i> Vibración Actual</div>
        <div class="kpi-value">{vibracion}<span class="kpi-unit"> mm/s</span></div>
        <div class="kpi-delta {vib_class}">{"<i class='fa-solid fa-arrow-trend-up'></i> Por encima del umbral" if vibracion > 3.6 else "<i class='fa-solid fa-arrow-trend-down'></i> Dentro del rango seguro"}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label"><i class="fa-solid fa-temperature-half" style="color:#ef4444; margin-right:5px;"></i> Temperatura Motor</div>
        <div class="kpi-value">{temperatura}<span class="kpi-unit"> °C</span></div>
        <div class="kpi-delta {temp_class}">{"<i class='fa-solid fa-arrow-trend-up'></i> Riesgo térmico" if temperatura >= 80 else "<i class='fa-solid fa-arrow-trend-down'></i> Temperatura nominal"}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label"><i class="fa-solid fa-brain" style="color:#8b5cf6; margin-right:5px;"></i> Confianza de IA</div>
        <div class="kpi-value">{confianza}<span class="kpi-unit"> %</span></div>
        <div class="kpi-delta">Probabilidad matemática</div>
    </div>
    <div class="kpi-card {estado_class}">
        <div class="kpi-label"><i class="fa-solid fa-shield-halved" style="color:#64748b; margin-right:5px;"></i> Estado del Sistema</div>
        <div class="kpi-value" style="font-size: 22px;">{estado_label}</div>
        <div class="kpi-delta">{falla}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("---")

tab1, tab2 = st.tabs([":material/monitor_heart: DIAGNÓSTICO EN VIVO", ":material/timeline: HISTORIAL · 30 DÍAS"])

with tab1:
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown(f"""
        <div class="telemetry-box">
            <div class="telemetry-header">LECTURA DE SENSORES EN TIEMPO REAL</div>
            <div class="telemetry-row"><span class="telemetry-key"><i class="fa-solid fa-wave-square" style="width:20px;"></i> VIBRACIÓN</span><span>{vibracion} mm/s</span></div>
            <div class="telemetry-row"><span class="telemetry-key"><i class="fa-solid fa-temperature-half" style="width:20px;"></i> TEMPERATURA</span><span>{temperatura} °C</span></div>
            <div class="telemetry-row"><span class="telemetry-key"><i class="fa-solid fa-oil-can" style="width:20px;"></i> LUBRICACIÓN</span><span>{lubricacion} %</span></div>
            <div class="telemetry-row"><span class="telemetry-key"><i class="fa-solid fa-bolt" style="width:20px;"></i> CORRIENTE</span><span>{corriente} A</span></div>
            <div class="telemetry-row"><span class="telemetry-key"><i class="fa-solid fa-gauge-high" style="width:20px;"></i> PRESIÓN</span><span>{presion} PSI</span></div>
            <div class="telemetry-row"><span class="telemetry-key"><i class="fa-solid fa-water" style="width:20px;"></i> CAUDAL</span><span>{caudal} m³/h</span></div>
            <br>
            <div class="telemetry-header" style="margin-bottom:5px;">RESULTADO PREDICTIVO</div>
            <div class="diag-value">>>> {falla.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="action-card" style="border-left-color: {rec_color};">
            <div class="action-label" style="color:{rec_color};">PLAN DE ACCIÓN · {rec_titulo}</div>
            <div class="action-text">{rec_texto}</div>
        </div>
        """, unsafe_allow_html=True)
with tab2:
    sensor_sel = st.selectbox("Variable a analizar:", cols)
    colores = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']
    
    fig = px.line(df, x='Dia Prueba', y=sensor_sel, color='Tipo_Falla', markers=True, color_discrete_sequence=colores)
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_family="DM Sans", font_color="#475569",
        title=dict(text=f"Evolución: {sensor_sel}", font=dict(family="Space Mono", size=16, color="#0f172a")),
        legend=dict(bgcolor="rgba(255,255,255,0.5)", bordercolor="#e2e8f0", borderwidth=1),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1"),
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1"),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, width="stretch")
    
    st.subheader("Registro Completo de Datos")
    st.dataframe(df, hide_index=True, use_container_width=True)
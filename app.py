import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="MMEP · Bomba Sumergible",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=":material/settings:"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

#MainMenu, footer { visibility: hidden; }

.header-wrap {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 6px 0 18px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 24px;
}
.header-badge {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 11px;
    letter-spacing: 0.08em;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}
.header-title {
    font-size: 22px;
    font-weight: 600;
    color: #f0f0f0;
    margin: 0;
    letter-spacing: -0.01em;
}
.header-sub {
    font-size: 13px;
    color: rgba(255,255,255,0.4);
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 2px;
}

.estado-card {
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 4px;
}
.estado-normal {
    background: rgba(52, 199, 89, 0.08);
    border: 1px solid rgba(52, 199, 89, 0.4);
    color: #34c759;
}
.estado-alerta {
    background: rgba(255, 196, 0, 0.08);
    border: 1px solid rgba(255, 196, 0, 0.4);
    color: #ffc400;
}
.estado-critico {
    background: rgba(255, 69, 58, 0.08);
    border: 1px solid rgba(255, 69, 58, 0.4);
    color: #ff453a;
}
.estado-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 6px;
    font-family: 'IBM Plex Mono', monospace;
}
.estado-valor {
    font-size: 26px;
    font-weight: 600;
    letter-spacing: -0.02em;
}

.kpi-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: left;
}
.kpi-label {
    font-size: 11px;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 22px;
    font-weight: 600;
    color: #e8e8e8;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.02em;
}
.kpi-unit {
    font-size: 13px;
    color: rgba(255,255,255,0.35);
    font-weight: 400;
    margin-left: 3px;
}

h6 {
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: rgba(255,255,255,0.5) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    margin-bottom: 12px !important;
    margin-top: 8px !important;
}

.conf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}
.conf-label { font-size: 13px; color: rgba(255,255,255,0.6); }
.conf-pct   { font-size: 13px; font-family: 'IBM Plex Mono', monospace; color: rgba(255,255,255,0.5); }

.anom-item {
    background: rgba(255, 69, 58, 0.08);
    border-left: 3px solid #ff453a;
    border-radius: 4px;
    padding: 8px 12px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #ff8b87;
}
.anom-ok {
    background: rgba(52, 199, 89, 0.07);
    border-left: 3px solid #34c759;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 13px;
    color: #34c759;
}

.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 20px 0;
}

.resum-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.resum-label {
    font-size: 11px;
    color: rgba(255,255,255,0.35);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 6px;
}
.resum-val {
    font-size: 20px;
    font-weight: 600;
    color: #e8e8e8;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_recursos():
    modelo = joblib.load("modelo_bomba.joblib")
    df = pd.read_csv("datos_bomba_final.csv")
    return modelo, df

try:
    modelo, df_historico = cargar_recursos()
except Exception as e:
    st.error(f"Error al cargar archivos. Verifica las rutas. Detalle: {e}")
    st.stop()

with st.sidebar:
    st.markdown("### :material/dashboard: Panel de Control")
    st.markdown('###### Fuente de datos')
    st.caption("Simulación de conexión PLC/SCADA en tiempo real.")

    if 'fila_actual' not in st.session_state:
        st.session_state['fila_actual'] = df_historico.sample(1)

    if st.button("Nueva Lectura", icon=":material/sync:", use_container_width=True, type="primary"):
        st.session_state['fila_actual'] = df_historico.sample(1)

    st.markdown('###### Información del equipo')
    st.markdown("""
    | Campo | Valor |
    |---|---|
    | Equipo | Bomba Sumergible |
    | Ubicación | Mina Subterránea |
    | Modelo IA | Random Forest |
    | Versión | v2.0 |
    """)

    st.markdown('###### Umbrales operativos')
    st.caption("Percentiles 5–95 del histórico de operación.")

datos_actuales = st.session_state['fila_actual'].iloc[0]

caracteristicas = np.array([[
    datos_actuales['Caudal_Ls'],
    datos_actuales['Corriente_A'],
    datos_actuales['Temperatura_C'],
    datos_actuales['Vibracion_mm_s'],
    datos_actuales['Lubricacion_pct'],
    datos_actuales['Presion_PSI']
]])

prediccion_cruda = str(modelo.predict(caracteristicas)[0]).strip()
probabilidades   = modelo.predict_proba(caracteristicas)[0]

mapa_estados = {
    "0": "Normal", "1": "Alerta", "2": "Crítico",
    "Normal": "Normal", "Alerta": "Alerta", "Crítico": "Crítico"
}
estado_actual = mapa_estados.get(prediccion_cruda, "Desconocido")

prob_normal  = probabilidades[0] if len(probabilidades) == 3 else 1.0
prob_alerta  = probabilidades[1] if len(probabilidades) == 3 else 0.0
prob_critico = probabilidades[2] if len(probabilidades) == 3 else 0.0
indice_riesgo = prob_critico + (prob_alerta * 0.5)

try:
    imp_vals = modelo.feature_importances_
    imp_names = ["Caudal", "Corriente", "Temperatura", "Vibración", "Lubricación", "Presión"]
    df_imp = pd.DataFrame({"Sensor": imp_names, "Importancia": imp_vals})
    df_imp["Peso (%)"] = (df_imp["Importancia"] * 100).round(1)
    df_imp = df_imp.sort_values("Peso (%)", ascending=True)
    var_top = df_imp.sort_values("Peso (%)", ascending=False).iloc[0]["Sensor"]
    val_top = df_imp.sort_values("Peso (%)", ascending=False).iloc[0]["Peso (%)"]
except Exception:
    df_imp = pd.DataFrame({
        "Sensor": ["Caudal", "Corriente", "Temperatura", "Vibración", "Lubricación", "Presión"],
        "Peso (%)": [17.7, 20.7, 19.3, 15.9, 17.8, 8.6]
    }).sort_values("Peso (%)", ascending=True)
    var_top, val_top = "Corriente", 20.7

st.markdown(f"""
<div class="header-wrap">
    <div>
        <p class="header-title">Sistema de Monitoreo Predictivo — Bomba Sumergible</p>
        <p class="header-sub">Mina Subterránea &nbsp;·&nbsp; Modelo: Random Forest &nbsp;·&nbsp;
        Última lectura: {datetime.now().strftime('%d/%m/%Y &nbsp; %H:%M:%S')}</p>
    </div>
    <div style="margin-left:auto;">
        <span class="header-badge">En línea</span>
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown("###### :material/sensors: Lecturas en tiempo real")

sensores = [
    ("Caudal",       datos_actuales['Caudal_Ls'],        "L/s"),
    ("Corriente",    datos_actuales['Corriente_A'],       "A"),
    ("Temperatura",  datos_actuales['Temperatura_C'],     "°C"),
    ("Vibración",    datos_actuales['Vibracion_mm_s'],    "mm/s"),
    ("Lubricación",  datos_actuales['Lubricacion_pct'],   "%"),
    ("Presión",      datos_actuales['Presion_PSI'],       "PSI"),
]

cols_kpi = st.columns(6, gap="small")
for col, (nombre, valor, unidad) in zip(cols_kpi, sensores):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{nombre}</div>
        <div class="kpi-value">{valor:.1f}<span class="kpi-unit">{unidad}</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('###### :material/health_and_safety: Diagnóstico del sistema')

col1, col2, col3 = st.columns([1.1, 1.1, 1.8], gap="large")

with col1:
    clase_css = {"Normal": "estado-normal", "Alerta": "estado-alerta", "Crítico": "estado-critico"}.get(estado_actual, "estado-normal")
    st.markdown(f"""
    <div class="estado-card {clase_css}">
        <div class="estado-label">Estado Operativo</div>
        <div class="estado-valor">{estado_actual.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###### Índice de riesgo operativo")
    st.progress(min(int(indice_riesgo * 100), 100))
    st.caption(f"Nivel de riesgo estimado: **{indice_riesgo * 100:.1f} %**")

    st.markdown("###### Desviaciones estadísticas")
    anomalias_detectadas = []
    for sensor_col, label in [
        ('Caudal_Ls', 'Caudal'), ('Corriente_A', 'Corriente'),
        ('Temperatura_C', 'Temperatura'), ('Vibracion_mm_s', 'Vibración'),
        ('Lubricacion_pct', 'Lubricación'), ('Presion_PSI', 'Presión')
    ]:
        p5  = df_historico[sensor_col].quantile(0.05)
        p95 = df_historico[sensor_col].quantile(0.95)
        val = datos_actuales[sensor_col]
        if val < p5 or val > p95:
            anomalias_detectadas.append(label)

    if not anomalias_detectadas:
        st.markdown('<div class="anom-ok">Sin desviaciones detectadas</div>', unsafe_allow_html=True)
    else:
        for a in anomalias_detectadas:
            st.markdown(f'<div class="anom-item"> {a} fuera del rango habitual</div>', unsafe_allow_html=True)

with col2:
    st.markdown("###### Confianza del modelo")

    for label, prob, color_hex in [
        ("Normal",  prob_normal,  "#34c759"),
        ("Alerta",  prob_alerta,  "#ffc400"),
        ("Crítico", prob_critico, "#ff453a"),
    ]:
        st.markdown(f"""
        <div class="conf-row">
            <span class="conf-label">{label}</span>
            <span class="conf-pct">{prob*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(prob * 100))
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

with col3:
    st.markdown("###### Huella operativa (sensores normalizados)")
    maximos   = df_historico[['Caudal_Ls','Corriente_A','Temperatura_C','Vibracion_mm_s','Lubricacion_pct','Presion_PSI']].max()
    categorias = ['Caudal','Corriente','Temperatura','Vibración','Lubricación','Presión']
    cols_raw   = ['Caudal_Ls','Corriente_A','Temperatura_C','Vibracion_mm_s','Lubricacion_pct','Presion_PSI']
    valores_radar = [datos_actuales[c] / maximos[c] for c in cols_raw]

    color_map = {"Normal": "#34c759", "Alerta": "#ffc400", "Crítico": "#ff453a"}
    fill_map = {"Normal": "rgba(52, 199, 89, 0.10)", "Alerta": "rgba(255, 196, 0, 0.10)", "Crítico": "rgba(255, 69, 58, 0.10)"}
    
    color_radar = color_map.get(estado_actual, "#34c759")
    fondo_radar = fill_map.get(estado_actual, "rgba(52, 199, 89, 0.10)")

    fig_radar = go.Figure(data=go.Scatterpolar(
        r     = valores_radar + [valores_radar[0]],
        theta = categorias   + [categorias[0]],
        fill  = 'toself',
        fillcolor = fondo_radar,
        line  = dict(color=color_radar, width=2),
        marker= dict(size=5, color=color_radar)
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            angularaxis=dict(tickfont=dict(size=12, color='rgba(255,255,255,0.5)')),
            radialaxis=dict(visible=True, showticklabels=False, gridcolor='rgba(255,255,255,0.07)', range=[0,1])
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(t=20, b=20, l=30, r=30),
        height=240
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("###### :material/analytics: Análisis avanzado")

tab_hist, tab_pca, tab_imp = st.tabs([
    ":material/timeline: Histórico de sensores",
    ":material/view_in_ar: Mapa dimensional (PCA 3D)",
    ":material/bar_chart: Importancia de variables"
])

with tab_hist:
    c1, c2 = st.columns([3, 1], gap="large")
    with c1:
        rango_opt = st.radio(
            "Período:", ["Últimas 24 horas", "Últimos 7 días", "Últimos 30 días"],
            horizontal=True, label_visibility="collapsed"
        )
        filas_mostrar = {"Últimas 24 horas": 24, "Últimos 7 días": 168, "Últimos 30 días": 720}[rango_opt]
        sensor_elegido = st.selectbox(
            "Sensor:",
            ['Temperatura_C','Vibracion_mm_s','Caudal_Ls','Corriente_A','Presion_PSI','Lubricacion_pct'],
            label_visibility="collapsed"
        )
        df_plot = df_historico.tail(filas_mostrar).reset_index(drop=True)
        
        color_line_map = {
            "Temperatura_C": "#ff6b6b", "Vibracion_mm_s": "#ffd93d",
            "Caudal_Ls": "#4ecdc4", "Corriente_A": "#a29bfe",
            "Presion_PSI": "#74b9ff", "Lubricacion_pct": "#55efc4"
        }
        fill_line_map = {
            "Temperatura_C": "rgba(255, 107, 107, 0.06)", "Vibracion_mm_s": "rgba(255, 217, 61, 0.06)",
            "Caudal_Ls": "rgba(78, 205, 196, 0.06)", "Corriente_A": "rgba(162, 155, 254, 0.06)",
            "Presion_PSI": "rgba(116, 185, 255, 0.06)", "Lubricacion_pct": "rgba(85, 239, 196, 0.06)"
        }
        
        color_line = color_line_map.get(sensor_elegido, "#74b9ff")
        fill_line = fill_line_map.get(sensor_elegido, "rgba(116, 185, 255, 0.06)")

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot[sensor_elegido],
            mode='lines', line=dict(color=color_line, width=1.5),
            fill='tozeroy', fillcolor=fill_line
        ))
        fig_line.update_layout(
            title=dict(text=f"Evolución de {sensor_elegido.replace('_',' ')}", font=dict(size=14, color='rgba(255,255,255,0.7)')),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, color='rgba(255,255,255,0.3)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='rgba(255,255,255,0.3)'),
            margin=dict(t=40, b=20, l=10, r=10), height=300
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with c2:
        st.markdown("###### Estadísticas del período")
        serie = df_plot[sensor_elegido]
        st.metric("Mínimo",  f"{serie.min():.1f}")
        st.metric("Promedio",f"{serie.mean():.1f}")
        st.metric("Máximo",  f"{serie.max():.1f}")
        st.metric("Desv. Est.", f"{serie.std():.2f}")

with tab_pca:
    st.caption("Visualización de 500 registros aleatorios comprimidos en 3 dimensiones. El diamante negro representa la lectura actual.")
    features = ['Caudal_Ls','Corriente_A','Temperatura_C','Vibracion_mm_s','Lubricacion_pct','Presion_PSI']
    pca3 = PCA(n_components=3)
    df_muestra = df_historico.sample(500, random_state=42)
    comps_3d   = pca3.fit_transform(df_muestra[features])
    df_pca3    = pd.DataFrame(comps_3d, columns=['PC1','PC2','PC3'])
    df_pca3['Estado'] = df_muestra['Estado'].astype(str).values

    color_map_pca = {"0":"#34c759","1":"#ffc400","2":"#ff453a",
                     "Normal":"#34c759","Alerta":"#ffc400","Crítico":"#ff453a"}

    fig_3d = px.scatter_3d(
        df_pca3, x='PC1', y='PC2', z='PC3', color='Estado',
        color_discrete_map=color_map_pca, opacity=0.45,
        labels={'Estado':'Estado'}
    )
    punto_3d = pca3.transform(caracteristicas)
    fig_3d.add_scatter3d(
        x=[punto_3d[0][0]], y=[punto_3d[0][1]], z=[punto_3d[0][2]],
        mode='markers', marker=dict(size=10, color='white', symbol='diamond'),
        name='Lectura actual'
    )
    fig_3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='rgba(255,255,255,0.3)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='rgba(255,255,255,0.3)'),
            zaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='rgba(255,255,255,0.3)')
        ),
        legend=dict(font=dict(color='rgba(255,255,255,0.6)')),
        margin=dict(l=0, r=0, b=0, t=0), height=480
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with tab_imp:
    st.caption("Peso real asignado por el modelo Random Forest a cada sensor en la toma de decisiones.")
    fig_bar = px.bar(
        df_imp, x="Peso (%)", y="Sensor", orientation='h',
        text=df_imp["Peso (%)"].apply(lambda x: f"{x:.1f}%"),
        color="Peso (%)",
        color_continuous_scale=[[0,"#1a1a2e"],[0.4,"#16213e"],[0.75,"#0f3460"],[1,"#4ecdc4"]]
    )
    fig_bar.update_traces(
        textposition='outside',
        marker_line_width=0,
        textfont=dict(color='rgba(255,255,255,0.6)', size=12)
    )
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False, color='rgba(255,255,255,0.3)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', color='rgba(255,255,255,0.6)',
                   tickfont=dict(size=13)),
        coloraxis_showscale=False,
        margin=dict(t=10, b=10, l=10, r=60), height=320
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("###### :material/summarize: Resumen ejecutivo")

r1, r2, r3, r4 = st.columns(4, gap="medium")
for col, label, valor in [
    (r1, "Estado General",           estado_actual.upper()),
    (r2, "Confianza Máxima",         f"{max(probabilidades)*100:.1f} %"),
    (r3, "Anomalías Detectadas",     str(len(anomalias_detectadas))),
    (r4, "Variable más influyente",  f"{var_top} ({val_top:.1f}%)"),
]:
    col.markdown(f"""
    <div class="resum-card">
        <div class="resum-label">{label}</div>
        <div class="resum-val">{valor}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

rec_map = {
    "Normal":  ("info",    " Sin acciones requeridas. Mantener condiciones operativas actuales y continuar el monitoreo regular desde el centro de control."),
    "Alerta":  ("warning", " Se recomienda revisar las condiciones operativas. Monitorear la tendencia en el histórico y programar inspección preventiva."),
    "Crítico": ("error",   " Acción requerida de inmediato. Evaluar el estado físico del equipo. Considerar parada preventiva para evitar pérdida de continuidad operativa."),
}
tipo, mensaje = rec_map.get(estado_actual, ("info", "Estado desconocido."))
getattr(st, tipo)(mensaje)
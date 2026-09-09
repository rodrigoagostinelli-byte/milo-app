import io
import re
import html
import streamlit as st
import pandas as pd
import torch
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer

# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="MILO",
    page_icon="milo_logo.png",
    layout="wide"
)

st.logo("milo_logo.png")
# =====================================================
# ESTILO
# =====================================================

def inyectar_estilos():
    st.markdown(
        """
        <style>
            :root {
                --bg: #F8FAFC;
                --panel: #FFFFFF;
                --border: #E2E8F0;
                --text: #0F172A;
                --muted: #64748B;
                --muted-2: #94A3B8;
                --primary: #2563EB;
                --positive: #16A34A;
                --negative: #DC2626;
                --warning: #F59E0B;
                --shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
                --radius: 18px;
            }

            .stApp {
                background-color: var(--bg);
            }

            .main .block-container {
                max-width: 1460px;
                padding-top: 1.2rem;
                padding-bottom: 2rem;
            }

            section[data-testid="stSidebar"] {
                background: #FFFFFF;
                border-right: 1px solid var(--border);
            }

            section[data-testid="stSidebar"] .block-container {
                padding-top: 1rem;
            }

            .hero-card {
                background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 22px 24px;
                box-shadow: var(--shadow);
                margin-bottom: 1rem;
            }

            .hero-kicker {
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: var(--primary);
                margin-bottom: 0.45rem;
            }

            .hero-title {
                font-size: 2rem;
                line-height: 1.15;
                font-weight: 700;
                color: var(--text);
                margin: 0;
            }

            .hero-subtitle {
                font-size: 0.98rem;
                color: var(--muted);
                margin-top: 0.55rem;
                max-width: 760px;
            }

            .hero-meta-row {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 1rem;
            }

            .meta-pill {
                background: #EFF6FF;
                border: 1px solid #DBEAFE;
                color: #1D4ED8;
                border-radius: 999px;
                padding: 0.4rem 0.75rem;
                font-size: 0.82rem;
                font-weight: 600;
            }

            .section-title {
                font-size: 1.1rem;
                font-weight: 700;
                color: var(--text);
                margin-bottom: 0.2rem;
            }

            .section-subtitle {
                font-size: 0.9rem;
                color: var(--muted);
                margin-bottom: 0.9rem;
            }

            .panel-card {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 18px 20px;
                box-shadow: var(--shadow);
                margin-bottom: 1rem;
            }

            .metric-card {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 16px 18px;
                box-shadow: var(--shadow);
                min-height: 118px;
            }

            .metric-label {
                font-size: 0.78rem;
                line-height: 1.2;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.04em;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }

            .metric-value {
                font-size: 1.8rem;
                line-height: 1.1;
                color: var(--text);
                font-weight: 700;
                margin-bottom: 0.35rem;
            }

            .metric-sub {
                font-size: 0.84rem;
                color: var(--muted-2);
            }

            .metric-accent-positive {
                border-top: 3px solid var(--positive);
            }

            .metric-accent-negative {
                border-top: 3px solid var(--negative);
            }

            .metric-accent-primary {
                border-top: 3px solid var(--primary);
            }

            .metric-accent-warning {
                border-top: 3px solid var(--warning);
            }

            .summary-list {
                margin: 0;
                padding-left: 1rem;
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.7;
            }

            .highlight-card {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 18px 20px;
                box-shadow: var(--shadow);
                height: 100%;
            }

            .highlight-positive {
                border-left: 5px solid var(--positive);
            }

            .highlight-negative {
                border-left: 5px solid var(--negative);
            }

            .highlight-title {
                font-size: 0.9rem;
                font-weight: 700;
                color: var(--text);
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.8rem;
            }

            .highlight-text {
                color: var(--text);
                font-size: 1rem;
                line-height: 1.6;
                margin-bottom: 0.9rem;
            }

            .highlight-meta {
                color: var(--muted);
                font-size: 0.84rem;
                line-height: 1.5;
            }

            .comment-card {
                background: #FFFFFF;
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 14px 16px;
                box-shadow: var(--shadow);
                margin-bottom: 0.7rem;
            }

            .comment-text {
                color: var(--text);
                font-size: 0.94rem;
                line-height: 1.55;
                margin-bottom: 0.5rem;
            }

            .comment-meta {
                color: var(--muted);
                font-size: 0.78rem;
            }

            .empty-card {
                background: #FFFFFF;
                border: 1px dashed var(--border);
                border-radius: 18px;
                padding: 24px;
                color: var(--muted);
                text-align: center;
                box-shadow: var(--shadow);
            }

            .preview-label {
                font-size: 0.82rem;
                color: var(--muted);
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.4rem;
            }

            .stButton > button {
                width: 100%;
                height: 2.8rem;
                border-radius: 12px;
                border: 1px solid #1D4ED8;
                background: #2563EB;
                color: #FFFFFF;
                font-weight: 600;
            }

            .stDownloadButton > button {
                width: 100%;
                height: 2.7rem;
                border-radius: 12px;
                border: 1px solid var(--border);
                background: #FFFFFF;
                color: var(--text);
                font-weight: 600;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                margin-bottom: 1rem;
            }

            .stTabs [data-baseweb="tab"] {
                background: #FFFFFF;
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 0.5rem 0.9rem;
                color: var(--muted);
            }

            .stTabs [aria-selected="true"] {
                color: var(--text);
                border-color: #CBD5E1;
                box-shadow: var(--shadow);
            }

            [data-testid="stDataFrame"] {
                border: 1px solid var(--border);
                border-radius: 16px;
                overflow: hidden;
                background: #FFFFFF;
            }

            div[data-testid="stMetric"] {
                background: transparent;
                border: none;
                padding: 0;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def panel_titulo(titulo, subtitulo=None):
    bloque = f"<div class='section-title'>{html.escape(titulo)}</div>"
    if subtitulo:
        bloque += f"<div class='section-subtitle'>{html.escape(subtitulo)}</div>"
    st.markdown(bloque, unsafe_allow_html=True)


def tarjeta_metrica(titulo, valor, subtitulo, tono="primary"):
    tonos = {
        "primary": "metric-accent-primary",
        "positive": "metric-accent-positive",
        "negative": "metric-accent-negative",
        "warning": "metric-accent-warning",
        "neutral": "metric-accent-primary"
    }
    clase = tonos.get(tono, "metric-accent-primary")
    st.markdown(
        f"""
        <div class="metric-card {clase}">
            <div class="metric-label">{html.escape(str(titulo))}</div>
            <div class="metric-value">{html.escape(str(valor))}</div>
            <div class="metric-sub">{html.escape(str(subtitulo))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def tarjeta_highlight(titulo, texto, confianza, fuente, tono="positive"):
    clase = "highlight-positive" if tono == "positive" else "highlight-negative"
    st.markdown(
        f"""
        <div class="highlight-card {clase}">
            <div class="highlight-title">{html.escape(str(titulo))}</div>
            <div class="highlight-text">{html.escape(str(texto))}</div>
            <div class="highlight-meta">Confianza del modelo: {confianza:.1%}<br>Origen de la clasificación: {html.escape(str(fuente))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def tarjeta_comentario(texto, confianza, fuente):
    st.markdown(
        f"""
        <div class="comment-card">
            <div class="comment-text">{html.escape(str(texto))}</div>
            <div class="comment-meta">Confianza del modelo: {confianza:.1%} · Origen: {html.escape(str(fuente))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def tarjeta_vacia(texto):
    st.markdown(
        f"<div class='empty-card'>{html.escape(str(texto))}</div>",
        unsafe_allow_html=True
    )


def estilizar_figura(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#334155"
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E2E8F0",
        zeroline=False,
        title=None
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        title=None
    )
    return fig


def cargar_csv_seguro(file_obj):
    contenido = file_obj.getvalue()

    intentos = [
        {"encoding": "utf-8", "sep": None, "engine": "python"},
        {"encoding": "utf-8-sig", "sep": None, "engine": "python"},
        {"encoding": "latin-1", "sep": None, "engine": "python"},
        {"encoding": "utf-8", "sep": ";", "engine": "python"},
        {"encoding": "utf-8-sig", "sep": ";", "engine": "python"},
        {"encoding": "latin-1", "sep": ";", "engine": "python"},
        {"encoding": "utf-8", "sep": ",", "engine": "python"},
        {"encoding": "utf-8-sig", "sep": ",", "engine": "python"},
        {"encoding": "latin-1", "sep": ",", "engine": "python"}
    ]

    errores = []

    for intento in intentos:
        try:
            df_prueba = pd.read_csv(
                io.BytesIO(contenido),
                encoding=intento["encoding"],
                sep=intento["sep"],
                engine=intento["engine"]
            )

            if len(df_prueba.columns) <= 1 and intento["sep"] is None:
                # si autodetección deja una sola columna, seguimos probando delimitadores explícitos
                errores.append(f"Autodetección insuficiente con {intento['encoding']}")
                continue

            return df_prueba, intento, []

        except Exception as e:
            errores.append(f"{intento}: {str(e)}")

    # último recurso: aceptar líneas problemáticas para no bloquear toda la carga
    rescates = [
        {"encoding": "utf-8-sig", "sep": None, "engine": "python"},
        {"encoding": "latin-1", "sep": None, "engine": "python"},
        {"encoding": "utf-8-sig", "sep": ";", "engine": "python"},
        {"encoding": "latin-1", "sep": ";", "engine": "python"}
    ]

    for intento in rescates:
        try:
            df_rescate = pd.read_csv(
                io.BytesIO(contenido),
                encoding=intento["encoding"],
                sep=intento["sep"],
                engine=intento["engine"],
                on_bad_lines="skip"
            )

            return df_rescate, intento, [
                "Se omitieron filas problemáticas durante la lectura del archivo."
            ]

        except Exception as e:
            errores.append(f"rescate {intento}: {str(e)}")

    raise ValueError(
        "No fue posible interpretar el CSV. Revisá delimitador, comillas, codificación y filas rotas.\n\n"
        + "\n".join(errores[:8])
    )


inyectar_estilos()

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">Milo</div>
        <h1 class="hero-title">Milo - Sentiment Intelligence Engine</h1>
        <div class="hero-subtitle">Monitoreo de sentimiento, detección de temas y revisión de comentarios destacados en una interfaz sobria y pensada para compartir.</div>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.header("Configuración")

    file = st.file_uploader(
        "Archivo CSV",
        type=["csv"]
    )

    columna_texto = None
    error_sidebar_csv = None

    if file is not None:
        try:
            df_sidebar, _, _ = cargar_csv_seguro(file)
            columnas_disponibles = df_sidebar.columns.tolist()
            columnas_analizables = [
                c for c in columnas_disponibles
                if str(c).strip().upper() != "TONALIDAD"
            ]
            if not columnas_analizables:
                columnas_analizables = columnas_disponibles
            columna_default = (
                "Introducción"
                if "Introducción" in columnas_analizables
                else columnas_analizables[0]
            )
            columna_texto = st.selectbox(
                "Columna a analizar",
                columnas_analizables,
                index=columnas_analizables.index(columna_default)
            )
        except ValueError as e:
            error_sidebar_csv = str(e)
            st.error("No se pudo leer el CSV para listar sus columnas.")
    else:
        st.caption("Subí un CSV para habilitar la selección de columnas.")

    modelo_nombre = st.selectbox(
        "Modelo",
        [
            "BETO",
            "ROBERTUITO"
        ]
    )

    ejecutar = st.button(
        "Analizar archivo",
        use_container_width=True
    )

    st.markdown("---")
    st.caption(
        "La clasificación final sigue esta prioridad: Tonalidad > Heurística > Modelo. Si existe TONALIDAD, sólo entonces se usa para priorizar y comparar."
    )


# =====================================================
# HEURÍSTICAS
# =====================================================

patrones_positivos = [
    r"\bfascinad[oa]s?\b",
    r"\bemocionad[oa]s?\b",
    r"\bespectacular\b",
    r"\bexcelente\b",
    r"\bgenial\b",
    r"\bincre[ií]ble\b",
    r"\bimpecable\b",
    r"\bhermos[oa]s?\b",
    r"\bbuen[ií]sim[oa]s?\b",
    r"\bmuy buen[oa]s?\b",
    r"\bbien hech[oa]s?\b",
    r"\bgran trabajo\b",
    r"\bgran laburo\b",
    r"\bme gust[oó]\b",
    r"\bme encant[oó]\b",
    r"\bme encant[ao]\b",
    r"\bme fascin[oó]\b",
    r"\bme sorprend[ií][oa]\b",
    r"\bme vol[oó] la cabeza\b",
    r"\bdisfrut[eé]\b",
    r"\bdisfrut[eé] mucho\b",
    r"\brecomendad[oa]s?\b",
    r"\bsuper recomendable\b",
    r"\baltamente recomendable\b",
    r"\bgros[oa]s?\b",
    r"\buna masa\b",
    r"\buna joyita\b",
    r"\bzarpad[oa]s?\b",
    r"\btremend[oa]s?\b",
    r"\bterrible (?=pel[ií]cula|serie|tema|video|laburo|trabajo|producto|servicio)\b",
    r"\balt[oa] (?=pel[ií]cula|serie|tema|video|laburo|trabajo|producto|servicio)\b",
    r"\buna bomba\b",
    r"\bme recab[ií]\b",
    r"\bme estall[eé]\b",
    r"\bno volvimos a ilusionar\b",
    r"\bno volvim[oó] a ilusionar\b",
    r"\bno volvio a ilusionar\b",
    r"\bno volvió a ilusionar\b",
    r"\bcampe[oó]n del mundo\b",
    r"\bcampeon del mundo\b",
    r"\bcampe[oó]n\b",
    r"\bcampeon\b",
    r"\bilusionar\b",
    r"\bilusi[oó]n\b",
    r"\bmuchachos?\b",
    r"\bdiez de diez\b",
    r"\bcinco estrellas\b",
    r"\b5 estrellas\b",
]

patrones_negativos = [
    r"\bhorrible\b",
    r"\bno pasa nada\b",
    r"\bque insoportables\b",
    r"\buna cagada\b",
    r"\bno me gust[óo]\b",
    r"\baburrid[oa]s?\b",
    r"\bmal[oa]s?\b",
    r"\bp[eé]sim[oa]s?\b",
    r"\bmediocre\b",
    r"\bes una mierda\b",
    r"\bme pareci[oó] una m[ií]erd[ae4]\b",
    r"\bm[ií1]erd[ae4]\b",
    r"\bmrd\b",
    r"\bes una p[i1]j[ae]\b",
    r"\bp[i1]j[ae]\b",
    r"\bes una v[e3]rg[ae4]\b",
    r"\bque v[e3]rg[ae4]\b",
    r"\bvrg\b",
    r"\bv3rga\b",
    r"\bberga\b",
    r"\bverg4\b",
    r"\bmal[ií]sim[oa]s?\b",
    r"\bmal[oa]s?\b",
    r"\bp[eé]sim[oa]s?\b",
    r"\bmediocre\b",
    r"\baburrid[oa]s?\b",
    r"\binsoportable?s?\b",
    r"\bdecepcionante\b",
    r"\bdecepci[oó]n\b",
    r"\bdesastre\b",
    r"\bterrible\b",
    r"\blamentable\b",
    r"\bfloj[oa]s?\b",
    r"\bfallid[oa]s?\b",
    r"\bno funciona\b",
    r"\bfunciona mal\b",
    r"\bno sirve\b",
    r"\bno me gust[oó]\b",
    r"\bno me gust[óo] nada\b",
    r"\bno me convenc[ií][oó]\b",
    r"\bno lo recomiendo\b",
    r"\bno recomiendo\b",
    r"\bno pasa nada\b",
    r"\bnada especial\b",
    r"\bperdida de tiempo\b",
    r"\bp[eé]rdida de tiempo\b",
    r"\bque insoportables\b",
    r"\buna cagada\b",
    r"\bmala experiencia\b",
    r"\bp[eé]sima experiencia\b",
    r"\bmal servicio\b",
    r"\bmala atenci[oó]n\b",
    r"\btardaron mucho\b",
    r"\bmuy lento\b",
    r"\bdemasiado lento\b",
    r"\bcar[ií]sim[oa]s?\b",
    r"\bmuy caro\b",
    r"\bprecio alto\b",
    r"\bno vale la pena\b",
    r"\b0/10\b",
    r"\bcero de diez\b",
    r"\buna estrella\b",
    r"\b1 estrella\b",
]


def normalizar_tonalidad(valor):
    if pd.isnull(valor):
        return None

    valor = str(valor).strip().lower()

    mapa = {
        "pos": "POS",
        "positivo": "POS",
        "positiva": "POS",
        "positive": "POS"
    }

    return mapa.get(valor)


def normalizar_sentimiento_simple(valor):
    if pd.isnull(valor):
        return None

    valor = str(valor).strip().upper()

    if valor in {"POS", "POS*"}:
        return "POS"
    if valor in {"NEG", "NEG*"}:
        return "NEG"
    if valor == "NEU":
        return "NEU"

    return normalizar_tonalidad(valor)


def buscar_columna(df, nombre_objetivo):
    objetivo = str(nombre_objetivo).strip().upper()

    for columna in df.columns:
        if str(columna).strip().upper() == objetivo:
            return columna

    return None


def ajustar_sentimiento(texto, sentimiento_modelo, tonalidad=None):
    tonalidad_normalizada = normalizar_tonalidad(tonalidad)

    if tonalidad_normalizada is not None:
        return tonalidad_normalizada, "TONALIDAD"

    texto = str(texto).lower()

    for p in patrones_positivos:
        if re.search(p, texto):
            return "POS*", "HEURISTICA"

    for p in patrones_negativos:
        if re.search(p, texto):
            return "NEG*", "HEURISTICA"

    return sentimiento_modelo, "MODELO"


# =====================================================
# TEMAS
# =====================================================

stopwords_es = [
    "de", "la", "el", "los", "las", "un", "una",
    "que", "por", "para", "con", "sin", "del",
    "al", "es", "son", "en", "se", "me", "te",
    "le", "les", "y", "o", "u", "a", "e",
    "rt", "https", "http", "amp"
]


def limpiar_texto_temas(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"@\w+", " ", texto)
    texto = re.sub(r"#[\wáéíóúñü]+", " ", texto)
    texto = re.sub(r"[^\w\sáéíóúñü]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def obtener_top_bigramas(textos, top_n=10):
    textos = [
        limpiar_texto_temas(t)
        for t in textos
        if pd.notnull(t)
    ]

    textos = [t for t in textos if t]

    if len(textos) == 0:
        return pd.DataFrame()

    vectorizer = CountVectorizer(
        stop_words=stopwords_es,
        ngram_range=(2, 2),
        min_df=2
    )

    try:
        matriz = vectorizer.fit_transform(textos)
        frecuencias = matriz.sum(axis=0).A1
        terminos = vectorizer.get_feature_names_out()

        temas = pd.DataFrame({
            "tema": terminos,
            "frecuencia": frecuencias
        })

        return temas.sort_values(
            "frecuencia",
            ascending=False
        ).head(top_n)

    except Exception:
        return pd.DataFrame()


# =====================================================
# MODELOS
# =====================================================

@st.cache_resource
def cargar_modelo(modelo_nombre):
    modelos = {
        "BETO": "finiteautomata/beto-sentiment-analysis",
        "ROBERTUITO": "pysentimiento/robertuito-sentiment-analysis"
    }

    nombre = modelos[modelo_nombre]

    tokenizer = AutoTokenizer.from_pretrained(nombre)
    model = AutoModelForSequenceClassification.from_pretrained(nombre)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    return tokenizer, model, device


# =====================================================
# BATCH INFERENCE
# =====================================================

def analizar_batch(textos, tokenizer, model, device):
    inputs = tokenizer(
        textos,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    scores, indices = probs.max(dim=1)

    resultados = []

    for i in range(len(textos)):
        etiqueta = model.config.id2label[indices[i].item()].upper()

        if "POS" in etiqueta:
            etiqueta = "POS"
        elif "NEG" in etiqueta:
            etiqueta = "NEG"
        else:
            etiqueta = "NEU"

        resultados.append((etiqueta, scores[i].item()))

    return resultados


# =====================================================
# ESTADO
# =====================================================

if "resultado_df" not in st.session_state:
    st.session_state.resultado_df = None

if "contexto_resultado" not in st.session_state:
    st.session_state.contexto_resultado = {}


# =====================================================
# APP
# =====================================================

if file is None:
    tarjeta_vacia(
        "Cargá un archivo CSV desde la barra lateral para iniciar el análisis. La interfaz se actualizará con distribución de sentimiento, comentarios destacados, temas y detalle descargable."
    )
    st.stop()

try:
    df, lectura_csv, avisos_csv = cargar_csv_seguro(file)
except ValueError as e:
    st.error("No fue posible leer el archivo CSV cargado.")
    st.code(str(e))
    st.stop()

st.markdown("<div class='preview-label'>Archivo cargado</div>", unsafe_allow_html=True)

for aviso in avisos_csv:
    st.warning(aviso)

columna_tonalidad = buscar_columna(df, "TONALIDAD")
usa_tonalidad_archivo = columna_tonalidad is not None

meta_cols = st.columns([1.1, 1.1, 1.1, 2.2])
with meta_cols[0]:
    tarjeta_metrica(
        "Registros",
        f"{len(df):,}".replace(",", "."),
        "Filas detectadas en el archivo",
        "primary"
    )
with meta_cols[1]:
    tarjeta_metrica(
        "Columnas",
        f"{len(df.columns):,}".replace(",", "."),
        "Variables presentes en el archivo",
        "neutral"
    )
with meta_cols[2]:
    tarjeta_metrica(
        "Tonalidad",
        "Disponible" if usa_tonalidad_archivo else "No disponible",
        "Se usa sólo si la columna existe y el valor es válido",
        "warning"
    )
with meta_cols[3]:
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="section-title">Entrada y configuración</div>
            <div class="section-subtitle">Control centralizado del archivo, la columna de análisis y el modelo seleccionado.</div>
            <div class="hero-meta-row">
                <div class="meta-pill">Archivo: {html.escape(file.name)}</div>
                <div class="meta-pill">Columna: {html.escape(columna_texto)}</div>
                <div class="meta-pill">Modelo: {html.escape(modelo_nombre)}</div>
                <div class="meta-pill">Lectura CSV: {html.escape(str(lectura_csv))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with st.expander("Vista previa del archivo", expanded=False):
    st.dataframe(df.head(), use_container_width=True)

if error_sidebar_csv is not None:
    st.error("No fue posible interpretar el archivo para habilitar la selección de columnas.")
    st.code(error_sidebar_csv)
    st.stop()

if not columna_texto or columna_texto not in df.columns:
    st.error("Seleccioná una columna válida del archivo para ejecutar el análisis.")
    st.stop()

contexto_actual = {
    "file_name": file.name,
    "row_count": len(df),
    "columna_texto": columna_texto,
    "modelo_nombre": modelo_nombre
}

if ejecutar:
    tokenizer, model, device = cargar_modelo(modelo_nombre)

    sentimientos = []
    confianzas = []
    sentimientos_final = []
    fuentes_sentimiento = []
    sentimientos_post_analisis = []
    tonalidades_previas = []
    comparaciones_tonalidad = []

    textos = df[columna_texto].astype(str).tolist()
    tonalidades = (
        df[columna_tonalidad].tolist()
        if columna_tonalidad is not None else [None] * len(textos)
    )

    total = len(textos)
    batch_size = 32

    progress_bar = st.progress(0)
    status = st.empty()

    for i in range(0, total, batch_size):
        batch = textos[i:i + batch_size]
        resultados = analizar_batch(batch, tokenizer, model, device)

        for j, (sent, conf) in enumerate(resultados):
            texto = batch[j]
            tonalidad_valor = tonalidades[i + j]
            sent_ajustado, fuente = ajustar_sentimiento(
                texto,
                sent,
                tonalidad_valor
            )
            sent_post_analisis, _ = ajustar_sentimiento(
                texto,
                sent,
                None
            )
            sent_post_analisis_norm = normalizar_sentimiento_simple(sent_post_analisis)
            tonalidad_normalizada = normalizar_tonalidad(tonalidad_valor)

            if tonalidad_normalizada is None:
                comparacion_tonalidad = "Sin dato previo válido"
            elif tonalidad_normalizada == sent_post_analisis_norm:
                comparacion_tonalidad = "Coincide"
            else:
                comparacion_tonalidad = "Cambia"

            sentimientos.append(sent)
            confianzas.append(conf)
            sentimientos_final.append(sent_ajustado)
            fuentes_sentimiento.append(fuente)
            sentimientos_post_analisis.append(sent_post_analisis_norm)
            tonalidades_previas.append(tonalidad_normalizada)
            comparaciones_tonalidad.append(comparacion_tonalidad)

        progreso = min((i + batch_size) / total, 1.0)
        progress_bar.progress(progreso)
        status.text(f"Procesando {min(i + batch_size, total)} de {total}")

    status.success("Análisis completado")

    df_resultado = df.copy()
    df_resultado["sentimiento_modelo"] = sentimientos
    df_resultado["confianza"] = confianzas
    df_resultado["sentimiento_final"] = sentimientos_final
    df_resultado["fuente_sentimiento"] = fuentes_sentimiento
    df_resultado["sentimiento_post_analisis"] = sentimientos_post_analisis
    df_resultado["tonalidad_previa"] = tonalidades_previas
    df_resultado["comparacion_tonalidad"] = comparaciones_tonalidad
    df_resultado["sentimiento_dashboard"] = (
        df_resultado["sentimiento_final"]
        .replace({
            "POS*": "POS",
            "NEG*": "NEG"
        })
    )

    st.session_state.resultado_df = df_resultado
    st.session_state.contexto_resultado = contexto_actual

if st.session_state.resultado_df is None:
    tarjeta_vacia(
        "El archivo está listo. Ejecutá el análisis desde la barra lateral para generar el dashboard completo."
    )
    st.stop()

# si cambia archivo, modelo o columna después del análisis previo, avisar
if st.session_state.contexto_resultado != contexto_actual:
    st.warning(
        "La configuración actual difiere del último análisis ejecutado. Si querés refrescar resultados, volvé a ejecutar el análisis."
    )

df = st.session_state.resultado_df.copy()

# =====================================================
# AMPLIFICACIÓN SOCIAL
# =====================================================

columna_compartidos = buscar_columna(
    df,
    "COMPARTIDOS"
)

usa_compartidos = columna_compartidos is not None

if usa_compartidos:

    df["compartidos_calc"] = (
        pd.to_numeric(
            df[columna_compartidos],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )

    df["peso_social"] = (
        1 + df["compartidos_calc"]
    )

    positive_amplified = df.loc[
        df["sentimiento_dashboard"] == "POS",
        "peso_social"
    ].sum()

    negative_amplified = df.loc[
        df["sentimiento_dashboard"] == "NEG",
        "peso_social"
    ].sum()

    neutral_amplified = df.loc[
        df["sentimiento_dashboard"] == "NEU",
        "peso_social"
    ].sum()

    total_amplified = (
        positive_amplified
        + negative_amplified
        + neutral_amplified
    )

    amplified_score = (
        (
            positive_amplified
            - negative_amplified
        )
        / total_amplified
    ) * 100 if total_amplified else 0

else:

    positive_amplified = 0
    negative_amplified = 0
    neutral_amplified = 0
    total_amplified = 0
    amplified_score = 0
    
total = len(df)
positivos = df["sentimiento_dashboard"].eq("POS").sum()
negativos = df["sentimiento_dashboard"].eq("NEG").sum()
neutros = df["sentimiento_dashboard"].eq("NEU").sum()
score_global = ((positivos - negativos) / total) * 100 if total else 0
conf_prom = df["confianza"].mean() if total else 0
resumen = df["sentimiento_dashboard"].value_counts().reset_index()
resumen.columns = ["sentimiento", "cantidad"]
sentimiento_dominante = resumen.iloc[0]["sentimiento"] if len(resumen) else "N/D"
fuente_resumen = df["fuente_sentimiento"].value_counts().reset_index()
fuente_resumen.columns = ["fuente", "cantidad"]
fuente_dominante = fuente_resumen.iloc[0]["fuente"] if len(fuente_resumen) else "N/D"

hero_meta = [
    f"{len(df):,} registros analizados".replace(",", "."),
    f"Modelo activo: {modelo_nombre}",
    f"Fuente dominante: {fuente_dominante}"
]
if usa_tonalidad_archivo:
    hero_meta.append(f"Columna {columna_tonalidad} detectada")

st.markdown(
    "<div class='hero-meta-row'>" + "".join([
        f"<div class='meta-pill'>{html.escape(item)}</div>" for item in hero_meta
    ]) + "</div>",
    unsafe_allow_html=True
)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    tarjeta_metrica(
        "Comentarios",
        f"{total:,}".replace(",", "."),
        "Base total procesada",
        "primary"
    )
with k2:
    tarjeta_metrica(
        "Positivos",
        f"{positivos / total:.1%}" if total else "0.0%",
        f"{positivos:,} comentarios".replace(",", "."),
        "positive"
    )
with k3:
    tarjeta_metrica(
        "Negativos",
        f"{negativos / total:.1%}" if total else "0.0%",
        f"{negativos:,} comentarios".replace(",", "."),
        "negative"
    )
with k4:
    tarjeta_metrica(
        "Neutros",
        f"{neutros / total:.1%}" if total else "0.0%",
        f"{neutros:,} comentarios".replace(",", "."),
        "warning"
    )
with k5:
    tarjeta_metrica(
        "Score global",
        f"{score_global:.1f}",
        f"Confianza promedio del modelo: {conf_prom:.1%}",
        "primary"
    )

tab1, tab2 = st.tabs(["Resumen", "Detalle"])

with tab1:
    if usa_compartidos:
       c1, c2, c3 = st.columns([1.6, 1, 1])
    else:
        c1, c2 = st.columns([1.6, 1])

    with c1:
        panel_titulo(
            "Distribución de sentimiento",
            "Composición general del análisis final luego de aplicar prioridad de Tonalidad, heurística y modelo."
        )
        fig = px.pie(
            resumen,
            names="sentimiento",
            values="cantidad",
            hole=0.72,
            color="sentimiento",
            color_discrete_map={
                "POS": "#16A34A",
                "NEU": "#94A3B8",
                "NEG": "#DC2626"
            }
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=True)
        st.plotly_chart(estilizar_figura(fig, height=420), use_container_width=True)

    with c2:
        panel_titulo(
            "Resumen conversacional",
            "Análisis basado en volumen de publicaciones."
        )
    
        st.markdown(
            f"""
            <div class="panel-card">
                <ul class="summary-list">
                    <li>Sentimiento dominante: <strong>{html.escape(str(sentimiento_dominante))}</strong>.</li>
                    <li>Participación positiva: <strong>{positivos / total:.1%}</strong>.</li>
                    <li>Participación negativa: <strong>{negativos / total:.1%}</strong>.</li>
                    <li>Fuente dominante de clasificación: <strong>{html.escape(str(fuente_dominante))}</strong>.</li>
                    <li>Confianza promedio del modelo: <strong>{conf_prom:.1%}</strong>.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
        fig_fuente = px.bar(
            fuente_resumen,
            x="cantidad",
            y="fuente",
            orientation="h",
            color="fuente",
            color_discrete_map={
                "MODELO": "#2563EB",
                "HEURISTICA": "#F59E0B",
                "TONALIDAD": "#7C3AED"
            }
        )
    
        fig_fuente.update_layout(showlegend=False)
    
        st.plotly_chart(
            estilizar_figura(fig_fuente, height=240),
            use_container_width=True
        )
    
    # =====================================================
    # RESUMEN AMPLIFICADO
    # =====================================================
    
    if usa_compartidos:
    
        amp_dist = {
            "POS": positive_amplified,
            "NEG": negative_amplified,
            "NEU": neutral_amplified
        }
    
        sentimiento_amplificado_dominante = max(
            amp_dist,
            key=amp_dist.get
        )
    
        positive_amplified_pct = (
            positive_amplified / total_amplified
        ) if total_amplified else 0
    
        negative_amplified_pct = (
            negative_amplified / total_amplified
        ) if total_amplified else 0
    
        neutral_amplified_pct = (
            neutral_amplified / total_amplified
        ) if total_amplified else 0
    
        with c3:
    
            panel_titulo(
                "Resumen amplificado",
                "Análisis basado en publicaciones y compartidos."
            )
    
            st.markdown(
                f"""
                <div class="panel-card">
                    <ul class="summary-list">
                        <li>Sentimiento dominante: <strong>{sentimiento_amplificado_dominante}</strong>.</li>
                        <li>Impacto positivo: <strong>{positive_amplified_pct:.1%}</strong>.</li>
                        <li>Impacto negativo: <strong>{negative_amplified_pct:.1%}</strong>.</li>
                        <li>Impacto neutro: <strong>{neutral_amplified_pct:.1%}</strong>.</li>
                        <li>Amplified Score: <strong>{amplified_score:.1f}</strong>.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    
        # =====================================================
        # AMPLIFICACION SOCIAL
        # =====================================================
    
        if usa_compartidos:
    
            panel_titulo(
                "Conversación vs Amplificación",
                "Comparación entre volumen de publicaciones y volumen total propagado."
            )
    
            comparativo_amp = pd.DataFrame({
                "sentimiento": ["POS", "NEG", "NEU"],
                "Conversación": [
                    positivos,
                    negativos,
                    neutros
                ],
                "Amplificación": [
                    positive_amplified,
                    negative_amplified,
                    neutral_amplified
                ]
            })
    
            comparativo_amp = comparativo_amp.melt(
                id_vars="sentimiento",
                var_name="tipo",
                value_name="valor"
            )
    
            fig_amp = px.bar(
                comparativo_amp,
                x="sentimiento",
                y="valor",
                color="tipo",
                barmode="group"
            )
    
            st.plotly_chart(
                estilizar_figura(
                    fig_amp,
                    height=420
                ),
                use_container_width=True
            )
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    
        panel_titulo(
            "Comentarios representativos",
            "Selección automática del comentario positivo y del comentario negativo con mayor confianza del modelo."
        )
    
        h1, h2 = st.columns(2)
        positivos_df = (
            df[df["sentimiento_dashboard"] == "POS"]
            .sort_values("confianza", ascending=False)
        )
        negativos_df = (
            df[df["sentimiento_dashboard"] == "NEG"]
            .sort_values("confianza", ascending=False)
        )

        with h1:
            if len(positivos_df):
                mejor = positivos_df.iloc[0]
                tarjeta_highlight(
                    "Comentario positivo destacado",
                    mejor[columna_texto],
                    mejor["confianza"],
                    mejor["fuente_sentimiento"],
                    "positive"
                )
            else:
                tarjeta_vacia("No se detectaron comentarios positivos para destacar.")
    
        with h2:
            if len(negativos_df):
                peor = negativos_df.iloc[0]
                tarjeta_highlight(
                    "Comentario crítico destacado",
                    peor[columna_texto],
                    peor["confianza"],
                    peor["fuente_sentimiento"],
                    "negative"
                )
            else:
                tarjeta_vacia("No se detectaron comentarios negativos para destacar.")
    
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    panel_titulo(
        "Temas detectados",
        "Bigramas frecuentes sobre comentarios positivos y negativos, con limpieza básica de texto y un umbral mínimo de repetición."
    )

    t1, t2 = st.columns(2)
    with t1:
        temas_pos = obtener_top_bigramas(
            df[df["sentimiento_dashboard"] == "POS"][columna_texto]
        )
        if len(temas_pos):
            fig_pos = px.bar(
                temas_pos,
                x="frecuencia",
                y="tema",
                orientation="h",
                color_discrete_sequence=["#16A34A"]
            )
            fig_pos.update_layout(yaxis=dict(categoryorder="total ascending"), showlegend=False)
            st.plotly_chart(estilizar_figura(fig_pos, height=420), use_container_width=True)
        else:
            tarjeta_vacia("No hay suficientes comentarios positivos para detectar temas de forma estable.")

    with t2:
        temas_neg = obtener_top_bigramas(
            df[df["sentimiento_dashboard"] == "NEG"][columna_texto]
        )
        if len(temas_neg):
            fig_neg = px.bar(
                temas_neg,
                x="frecuencia",
                y="tema",
                orientation="h",
                color_discrete_sequence=["#DC2626"]
            )
            fig_neg.update_layout(yaxis=dict(categoryorder="total ascending"), showlegend=False)
            st.plotly_chart(estilizar_figura(fig_neg, height=420), use_container_width=True)
        else:
            tarjeta_vacia("No hay suficientes comentarios negativos para detectar temas de forma estable.")

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        panel_titulo(
            "Comentarios positivos destacados",
            "Registros ordenados por confianza del modelo dentro del grupo positivo."
        )
        if len(positivos_df):
            for _, row in positivos_df.head(5).iterrows():
                tarjeta_comentario(
                    row[columna_texto],
                    row["confianza"],
                    row["fuente_sentimiento"]
                )
        else:
            tarjeta_vacia("No hay comentarios positivos destacados para mostrar.")

    with d2:
        panel_titulo(
            "Comentarios críticos destacados",
            "Registros ordenados por confianza del modelo dentro del grupo negativo."
        )
        if len(negativos_df):
            for _, row in negativos_df.head(5).iterrows():
                tarjeta_comentario(
                    row[columna_texto],
                    row["confianza"],
                    row["fuente_sentimiento"]
                )
        else:
            tarjeta_vacia("No hay comentarios críticos destacados para mostrar.")

with tab2:
    panel_titulo(
        "Detalle completo",
        "Filtrado interactivo sobre el dataset procesado y exportación del resultado visible."
    )
    # =====================================================
    # PUBLICACIONES MÁS AMPLIFICADAS
    # =====================================================

    if usa_compartidos:

        panel_titulo(
            "Publicaciones más amplificadas",
            "Ranking de contenidos según cantidad de compartidos."
        )

        columnas_top = [
            columna_texto,
            columna_compartidos,
            "sentimiento_dashboard",
            "confianza"
        ]

        columnas_top = [
            c for c in columnas_top
            if c in df.columns
        ]

        st.dataframe(
            df.sort_values(
                ["compartidos_calc", "confianza"],
                ascending=[False, False]
            )[columnas_top].head(20),
            use_container_width=True
        )

    # =====================================================
    # COMPARACION CON TONALIDAD
    # =====================================================
    if usa_tonalidad_archivo:
        panel_titulo(
            "Comparación con TONALIDAD previa",
            "Si la columna existe, se compara el dato previo contra el resultado nuevo generado por modelo + heurística, sin alterar el dashboard final."
        )
        df_comparable = df[df["tonalidad_previa"].notna()].copy()
        total_comparable = len(df_comparable)
        coincidencias = df_comparable["comparacion_tonalidad"].eq("Coincide").sum()
        cambios = df_comparable["comparacion_tonalidad"].eq("Cambia").sum()

        ccomp1, ccomp2, ccomp3 = st.columns(3)
        with ccomp1:
            tarjeta_metrica(
                "Base comparable",
                f"{total_comparable:,}".replace(",", "."),
                "Registros con TONALIDAD válida",
                "primary"
            )
        with ccomp2:
            tarjeta_metrica(
                "Coincidencia",
                f"{(coincidencias / total_comparable):.1%}" if total_comparable else "0.0%",
                f"{coincidencias:,} alineados".replace(",", "."),
                "positive"
            )
        with ccomp3:
            tarjeta_metrica(
                "Cambios",
                f"{cambios:,}".replace(",", "."),
                "Previo vs post análisis",
                "negative"
            )

        if total_comparable:
            resumen_comparacion = (
                df["comparacion_tonalidad"]
                .fillna("Sin dato previo válido")
                .value_counts()
                .rename_axis("estado")
                .reset_index(name="cantidad")
            )
            fig_comparacion = px.bar(
                resumen_comparacion,
                x="estado",
                y="cantidad",
                color="estado",
                color_discrete_map={
                    "Coincide": "#16A34A",
                    "Cambia": "#DC2626",
                    "Sin dato previo válido": "#94A3B8"
                }
            )
            fig_comparacion.update_layout(showlegend=False)
            st.plotly_chart(estilizar_figura(fig_comparacion, height=320), use_container_width=True)

            columnas_comparacion = [
                columna_texto,
                "tonalidad_previa",
                "sentimiento_post_analisis",
                "comparacion_tonalidad",
                "fuente_sentimiento",
                "confianza"
            ]
            columnas_comparacion = [c for c in columnas_comparacion if c in df.columns]
            st.dataframe(
                df[df["comparacion_tonalidad"] == "Cambia"][columnas_comparacion].head(50),
                use_container_width=True
            )

    if usa_tonalidad_archivo:
        f1, f2, f3, f4 = st.columns(4)
    else:
        f1, f2, f3 = st.columns(3)
        f4 = None
    with f1:
        filtro_sentimiento = st.multiselect(
            "Sentimiento",
            options=sorted(df["sentimiento_dashboard"].unique()),
            default=sorted(df["sentimiento_dashboard"].unique())
        )
    with f2:
        filtro_fuente = st.multiselect(
            "Origen",
            options=sorted(df["fuente_sentimiento"].unique()),
            default=sorted(df["fuente_sentimiento"].unique())
        )
    with f3:
        confianza_min = st.slider(
            "Confianza mínima del modelo",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01
        )
    if f4 is not None:
        with f4:
            opciones_comparacion = sorted(df["comparacion_tonalidad"].fillna("Sin dato previo válido").unique())
            filtro_comparacion = st.multiselect(
                "Comparación",
                options=opciones_comparacion,
                default=opciones_comparacion
            )
    else:
        filtro_comparacion = None

    mascara = (
        df["sentimiento_dashboard"].isin(filtro_sentimiento)
        & df["fuente_sentimiento"].isin(filtro_fuente)
        & (df["confianza"] >= confianza_min)
    )
    if filtro_comparacion is not None:
        mascara = mascara & df["comparacion_tonalidad"].fillna("Sin dato previo válido").isin(filtro_comparacion)

    df_filtrado = df[mascara]

    st.dataframe(df_filtrado, use_container_width=True)

    csv = df_filtrado.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Descargar resultado",
        csv,
        "resultado.csv",
        "text/csv",
        use_container_width=True
    )

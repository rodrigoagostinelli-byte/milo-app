import streamlit as st
import pandas as pd
import torch
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import re

# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="Sentiment MILO",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sentiment MILO")
st.caption(
    "Análisis automático de sentimiento sobre comentarios y opiniones."
)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("⚙️ Configuración")

    file = st.file_uploader(
        "Subí un CSV",
        type=["csv"]
    )

    columna_texto = st.text_input(
        "Nombre de la columna de texto",
        "Introducción"
    )

    modelo_nombre = st.selectbox(
        "Modelo",
        [
            "BETO",
            "ROBERTUITO"
        ]
    )

# =====================================================
# HEURÍSTICAS
# =====================================================

patrones_positivos = [
    r"fascinad[oa]s?",
    r"emocionad[oa]s?",
    r"espectacular",
    r"muy buen[oa]s?",
    r"disfrut[éeé]?",
    r"genial",
    r"gros[oa]s?",
    r"bien hech[oa]s?",
    r"me encant[oó]",
    r"me vol[oó] la cabeza",
    r"excelente",
    r"me sorprend[ií][oa]",
    r"de la concha de la lora",
    r"una masa",
    r"una joyita",
    r"alt[oa] (?=pel[ií]cula|serie|tema|video|laburo)",
    r"zarpad[oa]s?",
    r"una bomb[aa]",
    r"me recab[ií]",
    r"me estall[éeé]",
    r"no volvim[oó] a ilusionar",
    r"no volvio a ilusionar",
    r"campe[oó]n del mundo",
    r"campeon del mundo",
    r"muchacho",
    r"campe[oó]n",
    r"campeon",
    r"ilusionar",
    r"ilusion"
]

patrones_negativos = [
    r"horrible",
    r"no pasa nada",
    r"que insoportables",
    r"una cagada",
    r"no me gust[óo]",
    r"aburrid[oa]s?",
    r"mal[oa]s?",
    r"p[eé]sim[oa]s?",
    r"mediocre",
    r"es una mierda",
    r"me pareci[oó] una m[ií]erd[ae4]",
    r"m[ií1]erd[ae4]",
    r"mrd",
    r"es una p[i1]j[ae]",
    r"p[i1]j[ae]",
    r"es una v[e3]rg[ae4]",
    r"que v[e3]rg[ae4]",
    r"vrg",
    r"v3rga",
    r"berga",
    r"verg4"
]


def normalizar_tonalidad(valor):

    if pd.isnull(valor):
        return None

    valor = str(valor).strip().lower()

    mapa = {
        "pos": "POS",
        "positivo": "POS",
        "positiva": "POS",
        "positive": "POS",
        "neg": "NEG",
        "negativo": "NEG",
        "negativa": "NEG",
        "negative": "NEG",
        "neu": "NEU",
        "neutro": "NEU",
        "neutra": "NEU",
        "neutral": "NEU"
    }

    return mapa.get(valor)


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


def obtener_top_bigramas(textos, top_n=10):

    textos = [
        str(t)
        for t in textos
        if pd.notnull(t)
    ]

    if len(textos) == 0:
        return pd.DataFrame()

    vectorizer = CountVectorizer(
        stop_words=stopwords_es,
        ngram_range=(2, 2),
        min_df=2
    )

    try:

        matriz = vectorizer.fit_transform(
            textos
        )

        frecuencias = (
            matriz.sum(axis=0)
            .A1
        )

        terminos = (
            vectorizer
            .get_feature_names_out()
        )

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

    model = AutoModelForSequenceClassification.from_pretrained(
        nombre
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    return tokenizer, model, device


# =====================================================
# BATCH INFERENCE
# =====================================================

def analizar_batch(
    textos,
    tokenizer,
    model,
    device
):

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

    probs = torch.nn.functional.softmax(
        logits,
        dim=1
    )

    scores, indices = probs.max(dim=1)

    resultados = []

    for i in range(len(textos)):

        etiqueta = model.config.id2label[
            indices[i].item()
        ]

        etiqueta = etiqueta.upper()

        if "POS" in etiqueta:
            etiqueta = "POS"

        elif "NEG" in etiqueta:
            etiqueta = "NEG"

        else:
            etiqueta = "NEU"

        resultados.append(
            (
                etiqueta,
                scores[i].item()
            )
        )

    return resultados


# =====================================================
# APP
# =====================================================

if file is not None:

    df = pd.read_csv(file)

    st.subheader("Vista previa")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    if columna_texto not in df.columns:

        st.error(
            f"No existe la columna '{columna_texto}'"
        )

    else:

        st.success("✅ Columna encontrada")

        usar_tonalidad = "Tonalidad" in df.columns

        if usar_tonalidad:
            st.info("ℹ️ Se detectó la columna Tonalidad: tendrá prioridad sobre el modelo y la heurística cuando su valor sea válido.")

        if st.button(
            "Analizar archivo",
            use_container_width=True
        ):

            tokenizer, model, device = cargar_modelo(
                modelo_nombre
            )

            sentimientos = []
            confianzas = []
            sentimientos_final = []
            fuentes_sentimiento = []

            textos = (
                df[columna_texto]
                .astype(str)
                .tolist()
            )

            tonalidades = (
                df["Tonalidad"].tolist()
                if usar_tonalidad else [None] * len(textos)
            )

            total = len(textos)

            batch_size = 32

            progress_bar = st.progress(0)

            status = st.empty()

            for i in range(
                0,
                total,
                batch_size
            ):

                batch = textos[
                    i:i + batch_size
                ]

                resultados = analizar_batch(
                    batch,
                    tokenizer,
                    model,
                    device
                )

                for j, (sent, conf) in enumerate(
                    resultados
                ):

                    texto = batch[j]
                    tonalidad_valor = tonalidades[i + j]

                    sent_ajustado, fuente = (
                        ajustar_sentimiento(
                            texto,
                            sent,
                            tonalidad_valor
                        )
                    )

                    sentimientos.append(sent)
                    confianzas.append(conf)
                    sentimientos_final.append(
                        sent_ajustado
                    )
                    fuentes_sentimiento.append(
                        fuente
                    )

                progreso = min(
                    (i + batch_size) / total,
                    1.0
                )

                progress_bar.progress(
                    progreso
                )

                status.text(
                    f"Procesando {min(i + batch_size, total)} de {total}"
                )

            status.success(
                "✅ Análisis completado"
            )

            # =====================================================
            # RESULTADOS
            # =====================================================

            df["sentimiento_modelo"] = sentimientos
            df["confianza"] = confianzas
            df["sentimiento_final"] = sentimientos_final
            df["fuente_sentimiento"] = fuentes_sentimiento

            # Dashboard limpio
            df["sentimiento_dashboard"] = (
                df["sentimiento_final"]
                .replace({
                    "POS*": "POS",
                    "NEG*": "NEG"
                })
            )

            # =====================================================
            # KPIs
            # =====================================================

            total = len(df)

            positivos = (
                df["sentimiento_dashboard"]
                .eq("POS")
                .sum()
            )

            negativos = (
                df["sentimiento_dashboard"]
                .eq("NEG")
                .sum()
            )

            neutros = (
                df["sentimiento_dashboard"]
                .eq("NEU")
                .sum()
            )

            score_global = (
                (positivos - negativos)
                / total
            ) * 100

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "😊 Positivos",
                    f"{positivos/total:.1%}"
                )

            with c2:
                st.metric(
                    "😐 Neutros",
                    f"{neutros/total:.1%}"
                )

            with c3:
                st.metric(
                    "😡 Negativos",
                    f"{negativos/total:.1%}"
                )

            with c4:
                st.metric(
                    "📈 Score",
                    f"{score_global:.1f}"
                )

            # =====================================================
            # TABS
            # =====================================================

            tab1, tab2 = st.tabs(
                [
                    "📊 Dashboard",
                    "📋 Detalle"
                ]
            )

            # =====================================================
            # DASHBOARD
            # =====================================================

            with tab1:

                st.markdown("---")

                col1, col2 = st.columns(
                    [2, 1]
                )

                with col1:

                    resumen = (
                        df["sentimiento_dashboard"]
                        .value_counts()
                        .reset_index()
                    )

                    resumen.columns = [
                        "sentimiento",
                        "cantidad"
                    ]

                    fig = px.pie(
                        resumen,
                        names="sentimiento",
                        values="cantidad",
                        hole=0.70,
                        color="sentimiento",
                        color_discrete_map={
                            "POS": "#22C55E",
                            "NEU": "#94A3B8",
                            "NEG": "#EF4444"
                        }
                    )

                    fig.update_layout(
                        height=450,
                        margin=dict(
                            l=10,
                            r=10,
                            t=10,
                            b=10
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                with col2:

                    st.subheader(
                        "📋 Resumen Ejecutivo"
                    )

                    st.metric(
                        "Positivos",
                        f"{positivos/total:.1%}"
                    )

                    st.metric(
                        "Neutros",
                        f"{neutros/total:.1%}"
                    )

                    st.metric(
                        "Negativos",
                        f"{negativos/total:.1%}"
                    )

                    st.metric(
                        "Score Global",
                        f"{score_global:.1f}"
                    )

                # =====================================================
                # INSIGHTS
                # =====================================================

                st.markdown("---")

                st.subheader(
                    "💡 Insights automáticos"
                )

                conf_prom = (
                    df["confianza"]
                    .mean()
                )

                sentimiento_dominante = (
                    resumen.iloc[0]["sentimiento"]
                )

                st.info(
                    f"""
                    • Sentimiento dominante: {sentimiento_dominante}

                    • {positivos/total:.1%} de los comentarios fueron positivos

                    • {negativos/total:.1%} de los comentarios fueron negativos

                    • Confianza promedio del modelo: {conf_prom:.1%}
                    """
                )

                # =====================================================
                # ORIGEN DE CLASIFICACIÓN
                # =====================================================

                st.markdown("---")

                st.subheader(
                    "🧭 Origen de la clasificación"
                )

                fuente_resumen = (
                    df["fuente_sentimiento"]
                    .value_counts()
                    .reset_index()
                )
                fuente_resumen.columns = [
                    "fuente",
                    "cantidad"
                ]

                st.dataframe(
                    fuente_resumen,
                    use_container_width=True,
                    hide_index=True
                )

                # =====================================================
                # HIGHLIGHTS
                # =====================================================

                st.markdown("---")

                st.subheader(
                    "⭐ Highlights"
                )

                col1, col2 = st.columns(2)

                with col1:

                    positivos_df = (
                        df[
                            df["sentimiento_dashboard"] == "POS"
                        ]
                        .sort_values(
                            "confianza",
                            ascending=False
                        )
                    )

                    if len(positivos_df):

                        mejor = positivos_df.iloc[0]

                        st.success(
                            f"""
                            🏆 Highlight Positivo

                            {mejor[columna_texto]}

                            Confianza: {mejor['confianza']:.1%}
                            Fuente: {mejor['fuente_sentimiento']}
                            """
                        )
                    else:
                        st.info("No se detectaron comentarios positivos para destacar.")

                with col2:

                    negativos_df = (
                        df[
                            df["sentimiento_dashboard"] == "NEG"
                        ]
                        .sort_values(
                            "confianza",
                            ascending=False
                        )
                    )

                    if len(negativos_df):

                        peor = negativos_df.iloc[0]

                        st.error(
                            f"""
                            🚨 Principal Crítica

                            {peor[columna_texto]}

                            Confianza: {peor['confianza']:.1%}
                            Fuente: {peor['fuente_sentimiento']}
                            """
                        )
                    else:
                        st.info("No se detectaron comentarios negativos para destacar.")

                # =====================================================
                # TEMAS
                # =====================================================

                st.markdown("---")

                st.subheader(
                    "🧠 Temas Detectados"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        "### 😊 Temas Positivos"
                    )

                    temas_pos = obtener_top_bigramas(
                        df[
                            df["sentimiento_dashboard"] == "POS"
                        ][columna_texto]
                    )

                    if len(temas_pos):

                        fig_pos = px.bar(
                            temas_pos,
                            x="frecuencia",
                            y="tema",
                            orientation="h",
                            color_discrete_sequence=["#22C55E"]
                        )

                        fig_pos.update_layout(
                            height=400,
                            yaxis=dict(
                                categoryorder="total ascending"
                            ),
                            margin=dict(
                                l=10,
                                r=10,
                                t=10,
                                b=10
                            )
                        )

                        st.plotly_chart(
                            fig_pos,
                            use_container_width=True
                        )
                    else:
                        st.info("No hay suficientes comentarios positivos para detectar temas.")

                with col2:

                    st.markdown(
                        "### 😡 Temas Negativos"
                    )

                    temas_neg = obtener_top_bigramas(
                        df[
                            df["sentimiento_dashboard"] == "NEG"
                        ][columna_texto]
                    )

                    if len(temas_neg):

                        fig_neg = px.bar(
                            temas_neg,
                            x="frecuencia",
                            y="tema",
                            orientation="h",
                            color_discrete_sequence=["#EF4444"]
                        )

                        fig_neg.update_layout(
                            height=400,
                            yaxis=dict(
                                categoryorder="total ascending"
                            ),
                            margin=dict(
                                l=10,
                                r=10,
                                t=10,
                                b=10
                            )
                        )

                        st.plotly_chart(
                            fig_neg,
                            use_container_width=True
                        )
                    else:
                        st.info("No hay suficientes comentarios negativos para detectar temas.")

                # =====================================================
                # POS / NEG DESTACADOS
                # =====================================================

                st.markdown("---")

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader(
                        "😡 Comentarios negativos destacados"
                    )

                    negativos_df = (
                        df[
                            df["sentimiento_dashboard"] == "NEG"
                        ]
                        .sort_values(
                            "confianza",
                            ascending=False
                        )
                    )

                    for _, row in (
                        negativos_df
                        .head(5)
                        .iterrows()
                    ):

                        st.error(
                            f"{row[columna_texto]}\n\n"
                            f"Confianza: {row['confianza']:.1%}"
                        )

                with col2:

                    st.subheader(
                        "😊 Comentarios positivos destacados"
                    )

                    positivos_df = (
                        df[
                            df["sentimiento_dashboard"] == "POS"
                        ]
                        .sort_values(
                            "confianza",
                            ascending=False
                        )
                    )

                    for _, row in (
                        positivos_df
                        .head(5)
                        .iterrows()
                    ):

                        st.success(
                            f"{row[columna_texto]}\n\n"
                            f"Confianza: {row['confianza']:.1%}"
                        )

            # =====================================================
            # DETALLE
            # =====================================================

            with tab2:

                st.subheader(
                    "📋 Detalle completo"
                )

                filtro = st.multiselect(
                    "Filtrar sentimiento",
                    options=sorted(
                        df["sentimiento_dashboard"]
                        .unique()
                    ),
                    default=sorted(
                        df["sentimiento_dashboard"]
                        .unique()
                    )
                )

                df_filtrado = (
                    df[
                        df["sentimiento_dashboard"]
                        .isin(filtro)
                    ]
                )

                st.dataframe(
                    df_filtrado,
                    use_container_width=True
                )

                csv = (
                    df_filtrado
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                st.download_button(
                    "📥 Descargar resultado",
                    csv,
                    "resultado.csv",
                    "text/csv",
                    use_container_width=True
                )
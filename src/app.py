from pathlib import Path
from textwrap import dedent

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.predict import COLUNAS_MODELO, prever_probabilidade


# ==================================================
# CONFIGURAÇÃO DA PÁGINA
# ==================================================
st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CAMINHOS DO PROJETO
# ==================================================
ROOT = Path(__file__).resolve().parents[1]

CAMINHO_DADOS = (
    ROOT
    / "data"
    / "processed"
    / "clientes_tratados.csv"
)

CAMINHO_MODELO_WARD = (
    ROOT
    / "models"
    / "clustering"
    / "modelo_hierarquico_ward.pkl"
)


# ==================================================
# CONFIGURAÇÕES
# ==================================================
ORDEM_RISCO = {
    "Baixo": 0,
    "Moderado": 1,
    "Alto": 2,
    "Crítico": 3,
}

CORES_RISCO = {
    "Baixo": "#22C55E",
    "Moderado": "#EAB308",
    "Alto": "#F97316",
    "Crítico": "#DC2626",
}


# ==================================================
# FUNÇÃO PARA RENDERIZAR HTML CORRETAMENTE
# ==================================================
def renderizar_html(conteudo: str) -> None:
    st.html(
        dedent(conteudo).strip()
    )

# ==================================================
# ESTILO VISUAL
# ==================================================
renderizar_html(
    """
    <style>
        .stApp,
        [data-testid="stAppViewContainer"] {
            background-color: #F4F7FC;
            color: #0F172A;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #0F172A 0%,
                #172554 100%
            );
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {
            color: #FFFFFF !important;
        }

        /* Banner */
        .hero {
            padding: 2.3rem 2.5rem;
            border-radius: 24px;
            background: linear-gradient(
                135deg,
                #1E3A8A 0%,
                #2563EB 55%,
                #06B6D4 100%
            );
            margin-bottom: 1.8rem;
            box-shadow: 0 18px 45px rgba(30, 64, 175, 0.22);
        }

        .hero h1 {
            margin: 0;
            color: #FFFFFF !important;
            font-size: 2.65rem;
            line-height: 1.15;
        }

        .hero p {
            max-width: 900px;
            margin-top: 1rem;
            margin-bottom: 0;
            color: #E0F2FE !important;
            font-size: 1.05rem;
            line-height: 1.7;
        }

        .badge {
            display: inline-block;
            padding: 0.38rem 0.85rem;
            margin-bottom: 1rem;
            border-radius: 999px;
            background-color: rgba(255, 255, 255, 0.18);
            color: #FFFFFF !important;
            font-size: 0.82rem;
            font-weight: 700;
        }

        /* Títulos */
        .section-title {
            margin-top: 1.6rem;
            margin-bottom: 1rem;
            color: #0F172A !important;
            font-size: 1.45rem;
            font-weight: 800;
        }

        /* Cards iniciais */
        .info-card {
            min-height: 185px;
            padding: 1.5rem;
            border: 1px solid #DCE4F0;
            border-radius: 18px;
            background-color: #FFFFFF;
            box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06);
        }

        .info-card h3 {
            margin-top: 0;
            margin-bottom: 0.8rem;
            color: #1E3A8A !important;
            font-size: 1.4rem;
        }

        .info-card p {
            margin-bottom: 0;
            color: #475569 !important;
            line-height: 1.65;
        }

        /* Fluxo */
        .flow-card {
            min-height: 390px;
            padding: 1.6rem;
            border: 1px solid #DCE4F0;
            border-radius: 18px;
            background-color: #FFFFFF;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        .flow-card h3 {
            margin-top: 0;
            margin-bottom: 1.2rem;
            color: #1E3A8A !important;
        }

        .flow-step {
            margin: 0.45rem 0;
            padding: 0.75rem 1rem;
            border-radius: 10px;
            background-color: #EFF6FF;
            color: #1E3A8A !important;
            font-weight: 700;
            text-align: center;
        }

        .flow-arrow {
            color: #64748B !important;
            font-size: 1.1rem;
            font-weight: bold;
            text-align: center;
        }

        /* Observação */
        .model-note {
            padding: 1.1rem 1.3rem;
            border-left: 5px solid #2563EB;
            border-radius: 14px;
            background-color: #EAF2FF;
            color: #1E3A8A !important;
            line-height: 1.65;
        }

        /* Cards de análise */
        .analysis-card {
            min-height: 285px;
            padding: 1.5rem;
            border: 1px solid #DCE4F0;
            border-radius: 18px;
            background-color: #FFFFFF;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        .analysis-card h3 {
            margin-top: 0;
            color: #0F172A !important;
        }

        .analysis-card p,
        .analysis-card li {
            color: #475569 !important;
            line-height: 1.65;
        }

        .analysis-card ul {
            padding-left: 1.3rem;
        }

        .card-blue {
            border-top: 6px solid #2563EB;
        }

        .card-orange {
            border-top: 6px solid #F97316;
        }

        .alert-yellow {
            margin-top: 1rem;
            padding: 0.9rem 1rem;
            border-radius: 12px;
            background-color: #FEF9C3;
            color: #854D0E !important;
            font-weight: 600;
        }

        .alert-red {
            margin-top: 1rem;
            padding: 0.9rem 1rem;
            border-radius: 12px;
            background-color: #FEE2E2;
            color: #B91C1C !important;
            font-weight: 600;
        }

        /* Métricas */
        div[data-testid="stMetric"] {
            min-height: 112px;
            padding: 1rem 1.1rem;
            border: 1px solid #DCE4F0;
            border-radius: 16px;
            background-color: #FFFFFF !important;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] *,
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricLabel"] div,
        [data-testid="stMetricLabel"] span {
            color: #475569 !important;
            opacity: 1 !important;
            font-weight: 600 !important;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] *,
        [data-testid="stMetricValue"] div,
        [data-testid="stMetricValue"] span {
            color: #0F172A !important;
            opacity: 1 !important;
        }

        [data-testid="stMetricDelta"],
        [data-testid="stMetricDelta"] * {
            color: #64748B !important;
            opacity: 1 !important;
        }

        /* Upload */
        [data-testid="stFileUploaderDropzone"] {
            background-color: #FFFFFF;
            border: 1px dashed #94A3B8;
            border-radius: 14px;
        }

        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] span {
            color: #475569 !important;
            opacity: 1 !important;
        }

        /* Tabelas */
        [data-testid="stDataFrame"] {
            overflow: hidden;
            border: 1px solid #DCE4F0;
            border-radius: 14px;
        }

        /* Botão principal */
        .stButton > button[kind="primary"] {
            border: none;
            border-radius: 10px;
            background: linear-gradient(
                90deg,
                #2563EB,
                #06B6D4
            );
            color: #FFFFFF;
            font-weight: 700;
        }

        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(
                90deg,
                #1D4ED8,
                #0891B2
            );
            color: #FFFFFF;
        }

        /* Rodapé */
        .footer {
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid #DCE4F0;
            color: #64748B !important;
            font-size: 0.9rem;
            text-align: center;
        }
    </style>
    """
)


# ==================================================
# FUNÇÕES DE NEGÓCIO
# ==================================================
def classificar_risco(probabilidade: float) -> str:
    if probabilidade <= 0.30:
        return "Baixo"

    if probabilidade <= 0.60:
        return "Moderado"

    if probabilidade <= 0.80:
        return "Alto"

    return "Crítico"


def criar_features(dados: pd.DataFrame) -> pd.DataFrame:
    resultado = dados.copy()

    resultado["atraso_critico"] = (
        resultado["atraso_entrega_dias"] >= 2
    ).astype(int)

    resultado["problema_complexo"] = (
        (resultado["contatos_atendimento"] >= 2)
        & (resultado["tempo_resolucao_dias"] >= 3)
    ).astype(int)

    resultado["reclamacao_recorrente"] = (
        resultado["numero_reclamacoes"] >= 3
    ).astype(int)

    return resultado


def recomendar_acao(linha: pd.Series) -> str:
    atraso = linha["atraso_critico"] == 1
    problema = linha["problema_complexo"] == 1
    reclamacao = linha["reclamacao_recorrente"] == 1
    risco = linha["nivel_risco"]

    if atraso and problema and reclamacao:
        return (
            "Atendimento sênior e plano completo "
            "de recuperação"
        )

    if reclamacao and problema:
        return (
            "Atendimento especializado e acompanhamento "
            "até a resolução"
        )

    if atraso and reclamacao:
        return (
            "Contato prioritário e compensação "
            "pela falha logística"
        )

    if atraso and problema:
        return (
            "Atendimento especializado e acompanhamento "
            "da entrega"
        )

    if reclamacao:
        return (
            "Analisar a causa recorrente e realizar "
            "contato prioritário"
        )

    if atraso:
        return (
            "Pedido de desculpas e benefício "
            "no próximo frete"
        )

    if problema:
        return "Encaminhar para atendimento especializado"

    if risco == "Crítico":
        return "Contato preventivo imediato"

    if risco == "Alto":
        return "Acompanhamento preventivo"

    if risco == "Moderado":
        return (
            "Monitoramento ativo e comunicação "
            "de relacionamento"
        )

    return "Monitoramento padrão"


def validar_arquivo(dados: pd.DataFrame) -> None:
    if dados.empty:
        raise ValueError("O arquivo enviado está vazio.")

    colunas_ausentes = [
        coluna
        for coluna in COLUNAS_MODELO
        if coluna not in dados.columns
    ]

    if colunas_ausentes:
        raise ValueError(
            "Colunas ausentes: "
            + ", ".join(colunas_ausentes)
        )


def processar_classificacao(
    dados: pd.DataFrame,
) -> pd.DataFrame:
    resultado = criar_features(dados)

    resultado["prob_detrator"] = prever_probabilidade(
        resultado
    ).clip(0, 1)

    resultado["nivel_risco"] = (
        resultado["prob_detrator"]
        .apply(classificar_risco)
    )

    resultado["quantidade_problemas"] = resultado[
        [
            "atraso_critico",
            "problema_complexo",
            "reclamacao_recorrente",
        ]
    ].sum(axis=1)

    resultado["valor_em_risco"] = (
        resultado["prob_detrator"]
        * resultado["valor_pedido"]
    ).round(2)

    resultado["desconto_sugerido"] = (
        resultado["valor_em_risco"] * 0.08
    ).clip(upper=80).round(2)

    resultado["acao_recomendada"] = resultado.apply(
        recomendar_acao,
        axis=1,
    )

    resultado["ordem_risco"] = (
        resultado["nivel_risco"]
        .map(ORDEM_RISCO)
    )

    resultado = resultado.sort_values(
        by=[
            "ordem_risco",
            "quantidade_problemas",
            "valor_em_risco",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    return resultado.drop(columns="ordem_risco")


def converter_csv(dados: pd.DataFrame) -> bytes:
    return dados.to_csv(
        index=False,
        encoding="utf-8-sig",
    ).encode("utf-8-sig")


def formatar_valor_resumido(valor: float) -> str:
    if valor >= 1_000_000:
        return f"R$ {valor / 1_000_000:.2f} mi"

    if valor >= 1_000:
        return f"R$ {valor / 1_000:.1f} mil"

    return f"R$ {valor:.2f}"


def formatar_moeda(valor: float) -> str:
    valor_formatado = f"{valor:,.2f}"

    valor_formatado = (
        valor_formatado
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    return f"R$ {valor_formatado}"


# ==================================================
# CARREGAMENTO DOS SEGMENTOS
# ==================================================
@st.cache_data
def carregar_segmentos() -> pd.DataFrame:
    if not CAMINHO_DADOS.exists():
        raise FileNotFoundError(
            f"Base não encontrada: {CAMINHO_DADOS}"
        )

    if not CAMINHO_MODELO_WARD.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {CAMINHO_MODELO_WARD}"
        )

    dados = pd.read_csv(CAMINHO_DADOS)
    modelo_ward = joblib.load(CAMINHO_MODELO_WARD)

    dados = criar_features(dados)

    X = dados.drop(columns="nps")

    X_treino, _ = train_test_split(
        X,
        test_size=0.20,
        random_state=42,
    )

    dados_segmentados = X_treino.copy()

    dados_segmentados["nps"] = dados.loc[
        X_treino.index,
        "nps",
    ]

    labels = modelo_ward.labels_

    if len(labels) != len(dados_segmentados):
        raise ValueError(
            "A quantidade de rótulos do modelo não corresponde "
            "à quantidade de clientes reconstruída."
        )

    dados_segmentados["cluster"] = labels

    variaveis_criticidade = [
        "atraso_entrega_dias",
        "contatos_atendimento",
        "tempo_resolucao_dias",
        "numero_reclamacoes",
    ]

    perfil = (
        dados_segmentados
        .groupby("cluster")[variaveis_criticidade]
        .mean()
    )

    indice_criticidade = (
        perfil
        .rank(pct=True)
        .mean(axis=1)
    )

    cluster_alta = indice_criticidade.idxmax()
    cluster_menor = indice_criticidade.idxmin()

    nomes = {
        cluster_alta: "Alta criticidade",
        cluster_menor: "Menor criticidade",
    }

    dados_segmentados["segmento"] = (
        dados_segmentados["cluster"]
        .map(nomes)
    )

    dados_segmentados["classe_nps"] = pd.cut(
        dados_segmentados["nps"],
        bins=[
            -float("inf"),
            6,
            8,
            float("inf"),
        ],
        labels=[
            "Detrator",
            "Neutro",
            "Promotor",
        ],
    )

    return dados_segmentados


# ==================================================
# GRÁFICOS
# ==================================================
def criar_grafico_risco(
    resultado: pd.DataFrame,
):
    distribuicao = (
        resultado["nivel_risco"]
        .value_counts()
        .reindex(
            [
                "Baixo",
                "Moderado",
                "Alto",
                "Crítico",
            ],
            fill_value=0,
        )
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))

    barras = ax.bar(
        distribuicao.index,
        distribuicao.values,
        color=[
            CORES_RISCO[nivel]
            for nivel in distribuicao.index
        ],
        width=0.65,
    )

    ax.set_title(
        "Distribuição dos clientes por nível de risco",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Quantidade")

    ax.spines[
        ["top", "right", "left"]
    ].set_visible(False)

    ax.grid(
        axis="y",
        alpha=0.15,
    )

    ax.bar_label(
        barras,
        padding=4,
        fontsize=10,
        fontweight="bold",
    )

    plt.tight_layout()

    return fig


def criar_grafico_segmentos(
    dados_segmentados: pd.DataFrame,
):
    distribuicao = (
        dados_segmentados["segmento"]
        .value_counts()
        .reindex(
            [
                "Alta criticidade",
                "Menor criticidade",
            ]
        )
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    barras = ax.bar(
        distribuicao.index,
        distribuicao.values,
        color=[
            "#F97316",
            "#2563EB",
        ],
        width=0.6,
    )

    ax.set_title(
        "Clientes por segmento",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Quantidade")

    ax.spines[
        ["top", "right", "left"]
    ].set_visible(False)

    ax.grid(
        axis="y",
        alpha=0.15,
    )

    ax.bar_label(
        barras,
        padding=4,
        fontweight="bold",
    )

    plt.xticks(rotation=4)
    plt.tight_layout()

    return fig


def criar_grafico_nps(
    dados_segmentados: pd.DataFrame,
):
    distribuicao = pd.crosstab(
        dados_segmentados["segmento"],
        dados_segmentados["classe_nps"],
        normalize="index",
    ).mul(100)

    distribuicao = distribuicao.reindex(
        [
            "Alta criticidade",
            "Menor criticidade",
        ]
    )

    ordem_classes = [
        classe
        for classe in [
            "Detrator",
            "Neutro",
            "Promotor",
        ]
        if classe in distribuicao.columns
    ]

    distribuicao = distribuicao[
        ordem_classes
    ]

    cores = {
        "Detrator": "#DC2626",
        "Neutro": "#EAB308",
        "Promotor": "#22C55E",
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))

    distribuicao.plot(
        kind="bar",
        stacked=True,
        color=[
            cores[coluna]
            for coluna in distribuicao.columns
        ],
        ax=ax,
    )

    ax.set_title(
        "Composição do NPS por segmento",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Percentual")
    ax.set_ylim(0, 100)

    ax.legend(
        title="Classe de NPS",
        frameon=False,
    )

    ax.spines[
        ["top", "right", "left"]
    ].set_visible(False)

    ax.grid(
        axis="y",
        alpha=0.15,
    )

    plt.xticks(rotation=4)
    plt.tight_layout()

    return fig


# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("## 📊 Customer Intelligence")

    st.caption(
        "Análise preditiva e segmentação de clientes"
    )

    st.divider()

    pagina = st.radio(
        "Navegação",
        [
            "Visão geral",
            "Predição de detratores",
            "Segmentação de clientes",
            "Sobre o projeto",
        ],
    )

    st.divider()

    st.markdown("### Modelos utilizados")

    st.markdown(
        """
**Classificação**

Gradient Boosting

**Segmentação**

Kernel PCA + Ward
        """
    )

    st.divider()

    st.caption(
        "Projeto de Ciência de Dados aplicado "
        "à satisfação de clientes."
    )


# ==================================================
# VISÃO GERAL
# ==================================================
if pagina == "Visão geral":
    renderizar_html(
        """
        <div class="hero">
            <span class="badge">
                Machine Learning aplicado ao negócio
            </span>

            <h1>Customer Intelligence</h1>

            <p>
                Uma aplicação para identificar clientes com maior
                risco de insatisfação, estimar impacto financeiro
                e analisar segmentos com diferentes níveis de
                criticidade operacional.
            </p>
        </div>
        """
    )

    renderizar_html(
        """
        <div class="section-title">
            O que esta aplicação entrega
        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        renderizar_html(
            """
            <div class="info-card">
                <h3>🎯 Predição de risco</h3>

                <p>
                    Estima a probabilidade de cada cliente
                    se tornar detrator e classifica o risco
                    em quatro níveis.
                </p>
            </div>
            """
        )

    with col2:
        renderizar_html(
            """
            <div class="info-card">
                <h3>💰 Priorização financeira</h3>

                <p>
                    Combina a probabilidade prevista com o
                    valor do pedido para identificar clientes
                    prioritários.
                </p>
            </div>
            """
        )

    with col3:
        renderizar_html(
            """
            <div class="info-card">
                <h3>🧩 Segmentação</h3>

                <p>
                    Apresenta os grupos de menor e alta
                    criticidade encontrados pelo modelo
                    hierárquico.
                </p>
            </div>
            """
        )

    renderizar_html(
        """
        <div class="section-title">
            Fluxo da solução
        </div>
        """
    )

    fluxo1, fluxo2 = st.columns(2)

    with fluxo1:
        renderizar_html(
            """
            <div class="flow-card">
                <h3>🎯 Modelo supervisionado</h3>

                <div class="flow-step">
                    Dados do cliente
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    Pré-processamento
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    Gradient Boosting
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    Probabilidade de detrator
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    Risco, prioridade e ação
                </div>
            </div>
            """
        )

    with fluxo2:
        renderizar_html(
            """
            <div class="flow-card">
                <h3>🧩 Modelo não supervisionado</h3>

                <div class="flow-step">
                    Dados tratados
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    Kernel PCA
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    Clusterização Hierárquica
                </div>

                <div class="flow-arrow">↓</div>

                <div class="flow-step">
                    Menor ou alta criticidade
                </div>
            </div>
            """
        )


# ==================================================
# PREDIÇÃO DE DETRATORES
# ==================================================
elif pagina == "Predição de detratores":
    renderizar_html(
        """
        <div class="hero">
            <span class="badge">
                Modelo supervisionado
            </span>

            <h1>Predição de detratores</h1>

            <p>
                Envie uma base em CSV para identificar
                clientes com maior risco e gerar uma fila
                de priorização para ações de retenção.
            </p>
        </div>
        """
    )

    renderizar_html(
        """
        <div class="section-title">
            Envio dos dados
        </div>
        """
    )

    arquivo = st.file_uploader(
        "Selecione um arquivo CSV",
        type=["csv"],
        help=(
            "O arquivo deve conter as mesmas colunas "
            "utilizadas durante o treinamento."
        ),
    )

    if arquivo is None:
        st.info(
            "Envie um arquivo CSV para iniciar a análise."
        )

    else:
        try:
            dados = pd.read_csv(arquivo)

            validar_arquivo(dados)

            with st.expander(
                "Visualizar dados enviados",
                expanded=False,
            ):
                st.dataframe(
                    dados.head(15),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    f"{len(dados)} registros encontrados."
                )

            executar = st.button(
                "🚀 Executar análise",
                type="primary",
                use_container_width=True,
            )

            if executar:
                with st.spinner(
                    "Analisando clientes e calculando prioridades..."
                ):
                    resultado = processar_classificacao(
                        dados
                    )

                st.success(
                    "Análise concluída com sucesso."
                )

                total = len(resultado)

                risco_medio = resultado[
                    "prob_detrator"
                ].mean()

                criticos = (
                    resultado["nivel_risco"]
                    == "Crítico"
                ).sum()

                valor_risco = resultado[
                    "valor_em_risco"
                ].sum()

                problemas_medios = resultado[
                    "quantidade_problemas"
                ].mean()

                renderizar_html(
                    """
                    <div class="section-title">
                        Resumo executivo
                    </div>
                    """
                )

                col1, col2, col3, col4, col5 = (
                    st.columns(5)
                )

                col1.metric(
                    "Clientes analisados",
                    f"{total:,}".replace(",", "."),
                )

                col2.metric(
                    "Risco médio",
                    f"{risco_medio:.1%}",
                )

                col3.metric(
                    "Clientes críticos",
                    criticos,
                )

                col4.metric(
                    "Valor em risco",
                    formatar_valor_resumido(
                        valor_risco
                    ),
                )

                col5.metric(
                    "Problemas por cliente",
                    f"{problemas_medios:.1f}",
                )

                st.markdown("<br>", unsafe_allow_html=True)

                grafico_col, resumo_col = st.columns(
                    [2, 1]
                )

                with grafico_col:
                    st.pyplot(
                        criar_grafico_risco(
                            resultado
                        ),
                        use_container_width=True,
                    )

                with resumo_col:
                    nivel_dominante = (
                        resultado["nivel_risco"]
                        .value_counts()
                        .idxmax()
                    )

                    percentual_critico = (
                        criticos / total
                        if total > 0
                        else 0
                    )

                    renderizar_html(
                        f"""
                        <div class="analysis-card card-orange">
                            <h3>📌 Leitura rápida</h3>

                            <p>
                                <strong>Nível predominante:</strong>
                                {nivel_dominante}
                            </p>

                            <p>
                                <strong>Clientes críticos:</strong>
                                {percentual_critico:.1%}
                            </p>

                            <p>
                                <strong>Valor priorizado:</strong>
                                {formatar_moeda(valor_risco)}
                            </p>

                            <p>
                                <strong>Recomendação:</strong>
                                concentrar os esforços nos clientes
                                críticos com múltiplos problemas.
                            </p>
                        </div>
                        """
                    )

                renderizar_html(
                    """
                    <div class="section-title">
                        Fila de priorização
                    </div>
                    """
                )

                tabela_resultado = resultado.copy()

                tabela_resultado[
                    "probabilidade_percentual"
                ] = (
                    tabela_resultado[
                        "prob_detrator"
                    ] * 100
                ).round(2)

                colunas_exibicao = [
                    "probabilidade_percentual",
                    "nivel_risco",
                    "quantidade_problemas",
                    "valor_pedido",
                    "valor_em_risco",
                    "desconto_sugerido",
                    "acao_recomendada",
                ]

                st.dataframe(
                    tabela_resultado[
                        colunas_exibicao
                    ],
                    use_container_width=True,
                    hide_index=True,
                    height=520,
                    column_config={
                        "probabilidade_percentual": (
                            st.column_config.ProgressColumn(
                                "Probabilidade de detrator",
                                min_value=0,
                                max_value=100,
                                format="%.1f%%",
                            )
                        ),
                        "nivel_risco": (
                            st.column_config.TextColumn(
                                "Nível de risco"
                            )
                        ),
                        "quantidade_problemas": (
                            st.column_config.NumberColumn(
                                "Problemas",
                                format="%d",
                            )
                        ),
                        "valor_pedido": (
                            st.column_config.NumberColumn(
                                "Valor do pedido",
                                format="R$ %.2f",
                            )
                        ),
                        "valor_em_risco": (
                            st.column_config.NumberColumn(
                                "Valor em risco",
                                format="R$ %.2f",
                            )
                        ),
                        "desconto_sugerido": (
                            st.column_config.NumberColumn(
                                "Desconto sugerido",
                                format="R$ %.2f",
                            )
                        ),
                        "acao_recomendada": (
                            st.column_config.TextColumn(
                                "Ação recomendada",
                                width="large",
                            )
                        ),
                    },
                )

                st.download_button(
                    "⬇️ Baixar análise completa",
                    data=converter_csv(resultado),
                    file_name=(
                        "resultado_analise_clientes.csv"
                    ),
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                )

        except Exception as erro:
            st.error(
                "Não foi possível processar o arquivo: "
                f"{erro}"
            )


# ==================================================
# SEGMENTAÇÃO
# ==================================================
elif pagina == "Segmentação de clientes":
    renderizar_html(
        """
        <div class="hero">
            <span class="badge">
                Modelo não supervisionado
            </span>

            <h1>Segmentação de clientes</h1>

            <p>
                Análise dos dois perfis encontrados pelo
                Kernel PCA e pela Clusterização Hierárquica
                com método Ward.
            </p>
        </div>
        """
    )

    renderizar_html(
        """
        <div class="model-note">
            O modelo hierárquico apresenta os grupos
            encontrados durante o treinamento. Os segmentos
            representam níveis relativos de criticidade e
            não uma separação perfeita entre clientes
            satisfeitos e insatisfeitos.
        </div>
        """
    )

    try:
        dados_segmentados = carregar_segmentos()

        total = len(dados_segmentados)

        alta = (
            dados_segmentados["segmento"]
            == "Alta criticidade"
        ).sum()

        menor = (
            dados_segmentados["segmento"]
            == "Menor criticidade"
        ).sum()

        nps_alta = (
            dados_segmentados.loc[
                dados_segmentados["segmento"]
                == "Alta criticidade",
                "nps",
            ]
            .mean()
        )

        nps_menor = (
            dados_segmentados.loc[
                dados_segmentados["segmento"]
                == "Menor criticidade",
                "nps",
            ]
            .mean()
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = (
            st.columns(5)
        )

        col1.metric(
            "Clientes analisados",
            f"{total:,}".replace(",", "."),
        )

        col2.metric(
            "Alta criticidade",
            alta,
        )

        col3.metric(
            "Menor criticidade",
            menor,
        )

        col4.metric(
            "NPS médio — alta",
            f"{nps_alta:.2f}",
        )

        col5.metric(
            "NPS médio — menor",
            f"{nps_menor:.2f}",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        grafico1, grafico2 = st.columns(2)

        with grafico1:
            st.pyplot(
                criar_grafico_segmentos(
                    dados_segmentados
                ),
                use_container_width=True,
            )

        with grafico2:
            st.pyplot(
                criar_grafico_nps(
                    dados_segmentados
                ),
                use_container_width=True,
            )

        renderizar_html(
            """
            <div class="section-title">
                Perfil médio dos segmentos
            </div>
            """
        )

        colunas_perfil = [
            "valor_pedido",
            "atraso_entrega_dias",
            "contatos_atendimento",
            "tempo_resolucao_dias",
            "numero_reclamacoes",
            "nps",
        ]

        perfil = (
            dados_segmentados
            .groupby("segmento")[
                colunas_perfil
            ]
            .mean()
            .round(2)
        )

        st.dataframe(
            perfil,
            use_container_width=True,
            column_config={
                "valor_pedido": (
                    st.column_config.NumberColumn(
                        "Valor médio do pedido",
                        format="R$ %.2f",
                    )
                ),
                "atraso_entrega_dias": (
                    st.column_config.NumberColumn(
                        "Atraso médio",
                        format="%.2f dias",
                    )
                ),
                "contatos_atendimento": (
                    st.column_config.NumberColumn(
                        "Contatos médios",
                        format="%.2f",
                    )
                ),
                "tempo_resolucao_dias": (
                    st.column_config.NumberColumn(
                        "Tempo médio de resolução",
                        format="%.2f dias",
                    )
                ),
                "numero_reclamacoes": (
                    st.column_config.NumberColumn(
                        "Reclamações médias",
                        format="%.2f",
                    )
                ),
                "nps": (
                    st.column_config.NumberColumn(
                        "NPS médio",
                        format="%.2f",
                    )
                ),
            },
        )

        renderizar_html(
            """
            <div class="section-title">
                Interpretação dos grupos
            </div>
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            renderizar_html(
                """
                <div class="analysis-card card-blue">
                    <h3>🟦 Menor criticidade</h3>

                    <p>
                        Clientes com indicadores operacionais
                        relativamente mais favoráveis.
                    </p>

                    <ul>
                        <li>menor incidência de atrasos;</li>
                        <li>menos contatos com atendimento;</li>
                        <li>menor número de reclamações;</li>
                        <li>NPS médio superior.</li>
                    </ul>

                    <div class="alert-yellow">
                        O segmento ainda possui clientes
                        detratores e deve permanecer em
                        monitoramento.
                    </div>
                </div>
                """
            )

        with col2:
            renderizar_html(
                """
                <div class="analysis-card card-orange">
                    <h3>🟧 Alta criticidade</h3>

                    <p>
                        Clientes com maior concentração de
                        problemas operacionais e insatisfação.
                    </p>

                    <ul>
                        <li>maior atraso médio;</li>
                        <li>mais contatos com atendimento;</li>
                        <li>maior recorrência de reclamações;</li>
                        <li>menor NPS médio.</li>
                    </ul>

                    <div class="alert-red">
                        Esse grupo deve receber prioridade
                        nas ações de recuperação e
                        acompanhamento.
                    </div>
                </div>
                """
            )

    except Exception as erro:
        st.error(
            "Não foi possível carregar os segmentos: "
            f"{erro}"
        )


# ==================================================
# SOBRE O PROJETO
# ==================================================
else:
    renderizar_html(
        """
        <div class="hero">
            <span class="badge">
                Projeto de portfólio
            </span>

            <h1>Sobre a solução</h1>

            <p>
                Projeto de Ciência de Dados que integra
                análise exploratória, classificação,
                clusterização e aplicação dos resultados
                ao contexto de negócio.
            </p>
        </div>
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        renderizar_html(
            """
            <div class="analysis-card card-blue">
                <h3>🎯 Modelo de classificação</h3>

                <p>
                    O Gradient Boosting estima a probabilidade
                    de um cliente ser detrator.
                </p>

                <p>
                    <strong>Principais resultados gerados:</strong>
                </p>

                <ul>
                    <li>probabilidade de detrator;</li>
                    <li>nível de risco;</li>
                    <li>valor em risco;</li>
                    <li>recomendação de ação;</li>
                    <li>desconto sugerido.</li>
                </ul>
            </div>
            """
        )

    with col2:
        renderizar_html(
            """
            <div class="analysis-card card-orange">
                <h3>🧩 Modelo de clusterização</h3>

                <p>
                    O Kernel PCA e o método Ward identificam
                    clientes com diferentes níveis de criticidade.
                </p>

                <p>
                    <strong>Segmentos encontrados:</strong>
                </p>

                <ul>
                    <li>clientes de menor criticidade;</li>
                    <li>clientes de alta criticidade.</li>
                </ul>
            </div>
            """
        )

    renderizar_html(
        """
        <div class="section-title">
            Tecnologias utilizadas
        </div>
        """
    )

    st.code(
        """Python
Pandas
NumPy
Scikit-learn
Matplotlib
Streamlit
Joblib""",
        language="text",
    )

    renderizar_html(
        """
        <div class="model-note">
            Os resultados desta aplicação funcionam como
            apoio à tomada de decisão. Recomendações
            financeiras e comerciais devem ser validadas
            de acordo com as regras e políticas da empresa.
        </div>
        """
    )


# ==================================================
# RODAPÉ
# ==================================================
renderizar_html(
    """
    <div class="footer">
        Customer Intelligence • Ciência de Dados aplicada
        à satisfação e retenção de clientes
    </div>
    """
)
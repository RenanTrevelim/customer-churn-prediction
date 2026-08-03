from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split

from src.predict import (
    COLUNAS_MODELO,
    prever_probabilidade,
)


# ================================
# Configuração da página
# ================================
st.set_page_config(
    page_title="Análise de Satisfação",
    page_icon="📊",
    layout="wide",
)


ROOT = Path(__file__).resolve().parents[1]


# ================================
# Funções auxiliares
# ================================
def classificar_risco(probabilidade):
    if probabilidade <= 0.30:
        return "Baixo"

    if probabilidade <= 0.60:
        return "Moderado"

    if probabilidade <= 0.80:
        return "Alto"

    return "Crítico"


def recomendar_acao(risco):
    if risco == "Crítico":
        return "Contato imediato e atendimento prioritário"

    if risco == "Alto":
        return "Acompanhamento preventivo"

    if risco == "Moderado":
        return "Monitoramento ativo"

    return "Monitoramento padrão"


def converter_csv(dados):
    return dados.to_csv(
        index=False
    ).encode("utf-8-sig")


def validar_colunas(dados):
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


# ================================
# Título
# ================================
st.title("📊 Análise de Satisfação de Clientes")

st.write(
    "Aplicação para prever clientes detratores "
    "e visualizar os segmentos encontrados "
    "pelo modelo de clusterização."
)


aba_classificacao, aba_clusterizacao = st.tabs(
    [
        "Predição de detratores",
        "Segmentação de clientes",
    ]
)


# ==================================================
# ABA 1 - MODELO SUPERVISIONADO
# ==================================================
with aba_classificacao:
    st.header("Predição de detratores")

    st.write(
        "Envie um arquivo CSV para calcular "
        "a probabilidade de cada cliente ser detrator."
    )

    arquivo = st.file_uploader(
        "Selecione um arquivo CSV",
        type=["csv"],
    )

    if arquivo is not None:
        try:
            dados = pd.read_csv(arquivo)

            if dados.empty:
                st.warning("O arquivo está vazio.")
                st.stop()

            validar_colunas(dados)

            st.subheader("Prévia dos dados")

            st.dataframe(
                dados.head(),
                use_container_width=True,
                hide_index=True,
            )

            if st.button(
                "Executar previsão",
                type="primary",
            ):
                with st.spinner(
                    "Gerando previsões..."
                ):
                    resultado = dados.copy()

                    resultado["prob_detrator"] = (
                        prever_probabilidade(dados)
                    )

                    resultado["nivel_risco"] = (
                        resultado["prob_detrator"]
                        .apply(classificar_risco)
                    )

                    resultado["valor_em_risco"] = (
                        resultado["prob_detrator"]
                        * resultado["valor_pedido"]
                    ).round(2)

                    resultado["acao_recomendada"] = (
                        resultado["nivel_risco"]
                        .apply(recomendar_acao)
                    )

                st.success(
                    "Previsão concluída."
                )

                total_clientes = len(resultado)

                risco_medio = resultado[
                    "prob_detrator"
                ].mean()

                clientes_criticos = (
                    resultado["nivel_risco"]
                    == "Crítico"
                ).sum()

                valor_em_risco = resultado[
                    "valor_em_risco"
                ].sum()

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Clientes analisados",
                    total_clientes,
                )

                col2.metric(
                    "Risco médio",
                    f"{risco_medio:.1%}",
                )

                col3.metric(
                    "Clientes críticos",
                    clientes_criticos,
                )

                col4.metric(
                    "Valor em risco",
                    f"R$ {valor_em_risco:,.2f}",
                )

                st.subheader(
                    "Distribuição por risco"
                )

                distribuicao = (
                    resultado["nivel_risco"]
                    .value_counts()
                )

                st.bar_chart(distribuicao)

                st.subheader(
                    "Resultado completo"
                )

                st.dataframe(
                    resultado[
                        [
                            "prob_detrator",
                            "nivel_risco",
                            "valor_pedido",
                            "valor_em_risco",
                            "acao_recomendada",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Baixar resultado",
                    data=converter_csv(resultado),
                    file_name="resultado_clientes.csv",
                    mime="text/csv",
                )

        except Exception as erro:
            st.error(
                f"Erro ao processar o arquivo: {erro}"
            )

    else:
        st.info(
            "Envie um arquivo CSV para começar."
        )


# ==================================================
# ABA 2 - MODELO DE CLUSTERIZAÇÃO
# ==================================================
with aba_clusterizacao:
    st.header("Segmentação de clientes")

    st.write(
        "Esta área mostra os dois grupos encontrados "
        "pelo modelo hierárquico com método Ward."
    )

    st.caption(
        "O modelo Ward não prevê novos clientes. "
        "Aqui são apresentados os grupos formados "
        "durante o treinamento."
    )

    try:
        caminho_dados = (
            ROOT
            / "data"
            / "processed"
            / "clientes_tratados.csv"
        )

        caminho_modelo = (
            ROOT
            / "models"
            / "clustering"
            / "modelo_hierarquico_ward.pkl"
        )

        dados = pd.read_csv(caminho_dados)

        modelo_ward = joblib.load(
            caminho_modelo
        )

        # Criação das mesmas variáveis
        dados["atraso_critico"] = (
            dados["atraso_entrega_dias"] >= 2
        ).astype(int)

        dados["problema_complexo"] = (
            (dados["contatos_atendimento"] >= 2)
            & (dados["tempo_resolucao_dias"] >= 3)
        ).astype(int)

        dados["reclamacao_recorrente"] = (
            dados["numero_reclamacoes"] >= 3
        ).astype(int)

        X = dados.drop(
            columns="nps"
        )

        # Mesma divisão usada no notebook
        X_treino, _ = train_test_split(
            X,
            test_size=0.20,
            random_state=42,
        )

        dados_segmentos = X_treino.copy()

        dados_segmentos["nps"] = dados.loc[
            X_treino.index,
            "nps",
        ]

        dados_segmentos["cluster"] = (
            modelo_ward.labels_
        )

        # Descobre qual grupo tem mais problemas
        perfil = (
            dados_segmentos
            .groupby("cluster")[
                [
                    "atraso_entrega_dias",
                    "contatos_atendimento",
                    "tempo_resolucao_dias",
                    "numero_reclamacoes",
                ]
            ]
            .mean()
        )

        cluster_critico = (
            perfil.mean(axis=1).idxmax()
        )

        nomes = {
            cluster_critico: (
                "Clientes de alta criticidade"
            )
        }

        for cluster in dados_segmentos[
            "cluster"
        ].unique():
            if cluster != cluster_critico:
                nomes[cluster] = (
                    "Clientes de menor criticidade"
                )

        dados_segmentos["segmento"] = (
            dados_segmentos["cluster"]
            .map(nomes)
        )

        total = len(dados_segmentos)

        alta_criticidade = (
            dados_segmentos["segmento"]
            == "Clientes de alta criticidade"
        ).sum()

        menor_criticidade = (
            dados_segmentos["segmento"]
            == "Clientes de menor criticidade"
        ).sum()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Clientes analisados",
            total,
        )

        col2.metric(
            "Alta criticidade",
            alta_criticidade,
        )

        col3.metric(
            "Menor criticidade",
            menor_criticidade,
        )

        st.subheader(
            "Quantidade por segmento"
        )

        st.bar_chart(
            dados_segmentos[
                "segmento"
            ].value_counts()
        )

        st.subheader(
            "Perfil médio"
        )

        perfil_segmentos = (
            dados_segmentos
            .groupby("segmento")[
                [
                    "valor_pedido",
                    "atraso_entrega_dias",
                    "contatos_atendimento",
                    "tempo_resolucao_dias",
                    "numero_reclamacoes",
                    "nps",
                ]
            ]
            .mean()
            .round(2)
        )

        st.dataframe(
            perfil_segmentos,
            use_container_width=True,
        )

        st.subheader(
            "Clientes utilizados na análise"
        )

        st.dataframe(
            dados_segmentos[
                [
                    "segmento",
                    "nps",
                    "valor_pedido",
                    "atraso_entrega_dias",
                    "contatos_atendimento",
                    "tempo_resolucao_dias",
                    "numero_reclamacoes",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    except Exception as erro:
        st.error(
            f"Erro ao carregar os segmentos: {erro}"
        )
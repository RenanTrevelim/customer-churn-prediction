from pathlib import Path

import joblib
import pandas as pd


# Pasta principal do projeto
ROOT = Path(__file__).resolve().parents[1]


# Caminhos dos arquivos salvos
MODEL_PATH = ROOT / "models" / "supervised" / "modelo_final.pkl"

PREPROCESSOR_PATH = (
    ROOT
    / "models"
    / "preprocessing"
    / "pre_processamento.pkl"
)


# Colunas necessárias para realizar a previsão
COLUNAS_MODELO = [
    "idade_cliente",
    "regiao_cliente",
    "tempo_cliente_meses",
    "valor_pedido",
    "quantidade_itens",
    "valor_desconto",
    "parcelas_pagamento",
    "tempo_entrega_dias",
    "atraso_entrega_dias",
    "valor_frete",
    "tentativas_entrega",
    "contatos_atendimento",
    "tempo_resolucao_dias",
    "numero_reclamacoes",
]


def criar_features(dados):
    """
    Cria as mesmas variáveis utilizadas
    durante o treinamento do modelo.
    """

    dados = dados[COLUNAS_MODELO].copy()

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

    return dados


def prever_probabilidade(dados):
    """
    Retorna a probabilidade de o cliente
    ser detrator.
    """

    modelo = joblib.load(MODEL_PATH)
    pre_processamento = joblib.load(PREPROCESSOR_PATH)

    dados_modelo = criar_features(dados)

    dados_tratados = pre_processamento.transform(
        dados_modelo
    )

    probabilidades = modelo.predict_proba(
        dados_tratados
    )

    return probabilidades[:, 1]
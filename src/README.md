# Interface Streamlit

Esta pasta contém a aplicação web desenvolvida com Streamlit para apresentar os resultados do projeto de forma interativa e acessível.

A interface reúne os dois principais componentes construídos durante o projeto:

- modelo supervisionado para previsão de clientes detratores;
- análise dos segmentos encontrados pelo modelo de clusterização.

---

## Estrutura

```text
src/
├── app.py
├── predict.py
└── README.md
```

### `app.py`

Arquivo principal da aplicação.

Responsável por:

- configurar a interface do Streamlit;
- criar o menu lateral;
- organizar as páginas da aplicação;
- receber arquivos CSV;
- exibir métricas, gráficos e tabelas;
- apresentar os resultados da classificação;
- apresentar a análise dos segmentos;
- disponibilizar o download dos resultados.

### `predict.py`

Responsável por carregar os artefatos da modelagem supervisionada e gerar as probabilidades de detrator.

O fluxo utilizado é:

```text
dados de entrada
→ criação das features
→ pré-processamento
→ modelo supervisionado
→ probabilidade de detrator
```

---

## Funcionalidades da aplicação

### Visão geral

Apresenta o objetivo da solução e resume os dois fluxos de Machine Learning utilizados no projeto.

A página destaca:

- predição de risco;
- priorização financeira;
- segmentação de clientes;
- fluxo do modelo supervisionado;
- fluxo do modelo não supervisionado.

---

### Predição de detratores

Permite enviar um arquivo CSV com novos registros de clientes.

A aplicação utiliza o modelo supervisionado exportado para calcular:

- probabilidade de o cliente ser detrator;
- nível de risco;
- quantidade de problemas identificados;
- valor financeiro em risco;
- desconto sugerido;
- ação recomendada.

Os níveis de risco utilizados são:

```text
Baixo
Moderado
Alto
Crítico
```

Os resultados são organizados em uma fila de priorização, considerando o nível de risco, a quantidade de problemas e o valor em risco.

Também é possível baixar o resultado completo em formato CSV.

---

### Segmentação de clientes

Apresenta os dois grupos encontrados pelo modelo de clusterização hierárquica com método Ward.

Os segmentos são interpretados como:

- clientes de menor criticidade;
- clientes de alta criticidade.

A página apresenta:

- quantidade de clientes por segmento;
- NPS médio por grupo;
- composição das classes de NPS;
- perfil médio dos segmentos;
- interpretação dos grupos;
- principais diferenças operacionais.

O modelo hierárquico é utilizado para analisar os grupos encontrados durante o treinamento. Como ele não possui o método `predict`, essa parte da aplicação não atribui novos clientes aos clusters.

---

### Sobre o projeto

Apresenta uma visão resumida dos modelos utilizados, principais resultados gerados e tecnologias aplicadas.

---

## Modelos utilizados

### Classificação supervisionada

```text
Gradient Boosting
```

O modelo estima a probabilidade de um cliente ser detrator.

Artefatos utilizados:

```text
models/supervised/modelo_final.pkl
models/preprocessing/pre_processamento.pkl
```

### Clusterização

```text
Kernel PCA + Clusterização Hierárquica com Ward
```

A análise de segmentação utiliza os grupos encontrados durante o treinamento.

Artefato utilizado:

```text
models/clustering/modelo_hierarquico_ward.pkl
```

A base utilizada para reconstruir a análise é:

```text
data/processed/clientes_tratados.csv
```

---

## Arquivo de entrada

O arquivo CSV utilizado na página de predição deve conter as seguintes colunas:

```text
idade_cliente
regiao_cliente
tempo_cliente_meses
valor_pedido
quantidade_itens
valor_desconto
parcelas_pagamento
tempo_entrega_dias
atraso_entrega_dias
valor_frete
tentativas_entrega
contatos_atendimento
tempo_resolucao_dias
numero_reclamacoes
```

A coluna `nps` não é necessária para realizar novas previsões.

---

## Engenharia de atributos

Antes da previsão, a aplicação recria as mesmas variáveis utilizadas durante o treinamento:

```text
atraso_critico
problema_complexo
reclamacao_recorrente
```

Essas features representam:

- atrasos relevantes na entrega;
- problemas que exigiram mais contatos e maior tempo de resolução;
- recorrência de reclamações.

---

## Tecnologias utilizadas

```text
Python
Streamlit
Pandas
NumPy
Scikit-learn
Matplotlib
Joblib
```

---

## Instalação

Na raiz do projeto, crie e ative o ambiente virtual.

### Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### Linux ou macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Execução

Execute a aplicação a partir da raiz do projeto:

```bash
python -m streamlit run src/app.py
```

O Streamlit abrirá a aplicação no navegador.

O endereço local normalmente será:

```text
http://localhost:8501
```

---

## Fluxo da aplicação

### Classificação

```text
upload do CSV
→ validação das colunas
→ criação das features
→ pré-processamento
→ modelo Gradient Boosting
→ probabilidade de detrator
→ nível de risco
→ recomendação de ação
→ download do resultado
```

### Segmentação

```text
clientes_tratados.csv
→ recriação das features
→ mesma divisão utilizada no notebook
→ rótulos salvos no modelo Ward
→ interpretação dos segmentos
→ métricas e gráficos
```

---

## Observações

A aplicação funciona como uma camada de apresentação e apoio à decisão.

As recomendações de desconto, priorização e ação possuem caráter analítico e devem ser adaptadas às políticas comerciais e operacionais da empresa.

O modelo hierárquico é utilizado apenas para análise dos grupos já encontrados durante o treinamento, pois não permite prever diretamente o segmento de novos clientes.
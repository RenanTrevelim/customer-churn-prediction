# Customer Intelligence — Predição de Detratores e Segmentação de Clientes

Projeto de Ciência de Dados aplicado à satisfação de clientes em um contexto de e-commerce.

A solução combina **análise exploratória, engenharia de atributos, Machine Learning supervisionado, otimização com Optuna, interpretabilidade com SHAP, clusterização, aplicação em Streamlit e containerização com Docker** para identificar clientes com maior risco de insatisfação e grupos com diferentes níveis de criticidade operacional.

---

## Visão geral

O projeto foi desenvolvido para responder a duas perguntas principais:

> Qual é a probabilidade de um cliente ser detrator?

> Quais perfis de clientes apresentam padrões semelhantes de comportamento e criticidade?

Para isso, foram construídas duas abordagens complementares:

- **Classificação supervisionada:** estima o risco individual de um cliente ser detrator;
- **Clusterização:** identifica grupos de clientes com padrões operacionais semelhantes.

Os resultados foram integrados a uma aplicação Streamlit para exploração das previsões, segmentos e recomendações de negócio.

---

## Aplicação

A aplicação apresenta de forma integrada o fluxo das duas abordagens desenvolvidas no projeto.

<img width="1902" height="887" alt="Visão geral da aplicação" src="https://github.com/user-attachments/assets/6e94ef9e-804c-4227-931f-d54e8a5bcba2" />

---

## Objetivos

- compreender os principais fatores associados à insatisfação;
- analisar o comportamento do NPS;
- identificar padrões relacionados a entrega, atendimento e reclamações;
- criar atributos relevantes para modelagem;
- comparar diferentes algoritmos de Machine Learning;
- otimizar e validar o modelo supervisionado;
- interpretar as previsões com Feature Importance e SHAP;
- estimar risco individual e valor financeiro em risco;
- segmentar clientes por criticidade operacional;
- disponibilizar os resultados em uma aplicação interativa;
- garantir reprodutibilidade da aplicação com Docker.

---

## Arquitetura da solução

```text
Dados brutos
    ↓
Limpeza e validação
    ↓
Análise exploratória
    ↓
Engenharia de atributos
    ↓
Pré-processamento
    ↓
VarianceThreshold
    ↓
┌─────────────────────────────┬─────────────────────────────┐
│ Modelo supervisionado       │ Modelo não supervisionado   │
│ XGBoost + Optuna            │ Kernel PCA + Ward           │
│                             │                             │
│ Probabilidade de detrator   │ Segmentos de criticidade    │
│ Nível de risco              │ Perfil dos grupos           │
│ Valor em risco              │ Relação com o NPS           │
│ Ação recomendada            │ Interpretação operacional   │
└─────────────────────────────┴─────────────────────────────┘
    ↓
Aplicação Streamlit
    ↓
Container Docker
```

---

## Estrutura do projeto

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── nps_clientes.csv
│   ├── processed/
│   │   └── clientes_tratados.csv
│   └── README.md
│
├── models/
│   ├── supervised/
│   │   └── modelo_optuna.pkl
│   ├── clustering/
│   │   ├── kernel_pca_clusterizacao.pkl
│   │   └── modelo_hierarquico_ward.pkl
│   ├── preprocessing/
│   │   ├── pre_processamento.pkl
│   │   └── pre_processamento_clusterizacao.pkl
│   └── README.md
│
├── notebooks/
│   ├── 01_entendimento_e_preparacao_dos_dados.ipynb
│   ├── 02_analise_exploratoria_dos_dados.ipynb
│   ├── 03_modelo_machine_learning.ipynb
│   ├── 04_modelos_clusterização.ipynb
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── predict.py
│   └── README.md
│
├── docs/
│   └── images/
│       ├── streamlit-visao-geral.png
│       ├── streamlit-predicao.png
│       └── streamlit-segmentacao.png
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## Notebooks

### 01 — Entendimento e preparação dos dados

Responsável pela leitura, validação e tratamento inicial da base.

Principais etapas:

- análise da estrutura dos dados;
- verificação dos tipos das variáveis;
- identificação de valores ausentes e duplicados;
- validação da consistência;
- preparação da base tratada.

---

### 02 — Análise exploratória dos dados

Responsável pela investigação dos principais padrões associados à satisfação.

Foram analisados:

- distribuição do NPS;
- relação entre satisfação, entrega e atendimento;
- impacto de atrasos e reclamações;
- correlação entre as variáveis;
- padrões utilizados posteriormente na engenharia de atributos.

Os resultados indicaram maior associação da insatisfação com fatores relacionados a **atrasos na entrega, número de reclamações, contatos com atendimento e tempo de resolução**.

---

### 03 — Modelagem supervisionada

Responsável pela construção do modelo de classificação de detratores.

Algoritmos avaliados:

- Regressão Logística;
- Árvore de Decisão;
- Random Forest;
- KNN;
- SVM;
- Gaussian Naive Bayes;
- Gradient Boosting;
- XGBoost.

O fluxo incluiu:

- definição da variável-alvo;
- engenharia de atributos;
- pré-processamento;
- Feature Selection;
- comparação de modelos;
- otimização de hiperparâmetros;
- validação cruzada;
- Feature Importance;
- interpretabilidade com SHAP;
- aplicação das probabilidades ao negócio.

---

### 04 — Clusterização

Responsável pela identificação de grupos de clientes com diferentes níveis de criticidade operacional.

Algoritmos avaliados:

- K-Means;
- DBSCAN;
- Clusterização Hierárquica.

Técnicas de redução de dimensionalidade:

- PCA;
- Kernel PCA.

A combinação entre **Kernel PCA + Clusterização Hierárquica com método Ward** foi selecionada como solução final.

---

## Engenharia de atributos

Foram criadas três variáveis derivadas com base nos padrões encontrados durante a análise exploratória.

### `atraso_critico`

Indica entregas com atraso igual ou superior a dois dias.

### `problema_complexo`

Representa situações com múltiplos contatos com atendimento e maior tempo de resolução.

### `reclamacao_recorrente`

Identifica clientes com três ou mais reclamações.

Essas variáveis sintetizam comportamentos operacionais associados à insatisfação.

---

## Pré-processamento e Feature Selection

O pipeline supervisionado utiliza:

- `StandardScaler` para variáveis numéricas;
- `OneHotEncoder` para `regiao_cliente`;
- `ColumnTransformer` para aplicação das transformações;
- `VarianceThreshold` para verificação de features com baixa variabilidade.

Após o pré-processamento, nenhuma das 20 features apresentou variância inferior ao limite definido, portanto todas foram preservadas.

O pipeline foi exportado com `Joblib` para manter consistência entre treinamento e inferência.

---

# Modelagem supervisionada

A variável-alvo foi definida como:

```text
1 → detrator
0 → não detrator
```

Clientes com **NPS ≤ 6** foram considerados detratores.

Como a base apresenta desbalanceamento entre as classes, a avaliação priorizou métricas como **F1 Macro, Recall, F1-score, ROC-AUC e matriz de confusão**.

---

## Otimização do modelo

Após a comparação inicial dos classificadores, o XGBoost avançou para uma etapa adicional de otimização.

Foram comparadas:

- `RandomizedSearchCV` com 20 iterações;
- Optuna com 20 trials;
- Optuna com 200 trials.

O **XGBoost otimizado com Optuna em 200 trials** apresentou o melhor desempenho geral.

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,8380 |
| Recall | 0,9297 |
| F1-score | 0,8947 |
| F1 Macro | 0,7720 |
| ROC-AUC | 0,8881 |

A escolha considerou principalmente o equilíbrio entre as classes, o alto recall e os ganhos em F1 Macro e ROC-AUC.

---

## Validação cruzada

O modelo final foi avaliado com validação cruzada estratificada de 5 folds.

| Métrica | Média |
|---|---:|
| Accuracy | 0,8375 |
| Precision | 0,8560 |
| Recall | 0,9379 |
| F1-score | 0,8953 |
| ROC-AUC | 0,8774 |

Os baixos desvios-padrão observados indicaram comportamento consistente entre diferentes divisões dos dados.

---

## Feature Importance e SHAP

A importância das variáveis foi inicialmente analisada com `feature_importances_` do XGBoost.

Entre as principais features, destacaram-se:

- `numero_reclamacoes`;
- `atraso_entrega_dias`;
- `contatos_atendimento`;
- `tempo_resolucao_dias`;
- `problema_complexo`;
- `valor_pedido`.

Para aprofundar a interpretação do modelo, também foi aplicado **SHAP (SHapley Additive exPlanations)** utilizando `TreeExplainer`.

### Análise global

Foram utilizados:

- Summary Plot;
- Summary Plot em barras;
- Dependence Plot.

Essas visualizações permitiram analisar importância, magnitude, direção do impacto e comportamento de features específicas.

### Análise local

Foram utilizados:

- Waterfall Plot;
- Force Plot.

Essas visualizações permitiram explicar previsões individuais e identificar quais características aumentaram ou reduziram a saída do modelo.

O SHAP complementou a Feature Importance ao mostrar não apenas **quais variáveis são relevantes**, mas também **como e em qual direção elas influenciam as previsões**.

---

# Aplicação das previsões ao negócio

As probabilidades geradas pelo XGBoost foram transformadas em indicadores para apoio à tomada de decisão.

A solução permite:

- estimar a probabilidade de detrator;
- classificar clientes por nível de risco;
- contabilizar problemas operacionais;
- estimar valor financeiro em risco;
- sugerir ações de retenção;
- sugerir descontos;
- criar uma fila de priorização.

### Níveis de risco

```text
Baixo
Moderado
Alto
Crítico
```

### Valor em risco

O indicador é calculado por:

```text
probabilidade de detrator × valor do pedido
```

O valor funciona como uma medida de priorização e não representa uma perda financeira garantida.

---

## Predição de detratores na aplicação

A aplicação recebe uma base em CSV, executa o pipeline de pré-processamento e utiliza o modelo final para gerar uma visão executiva dos clientes analisados.

Entre os resultados apresentados estão:

- risco médio;
- quantidade de clientes críticos;
- distribuição por nível de risco;
- valor financeiro em risco;
- quantidade de problemas por cliente;
- fila de priorização;
- ação recomendada;
- desconto sugerido.

<img width="1905" height="894" alt="Predição de detratores" src="https://github.com/user-attachments/assets/3b9cc992-6c8d-4d7c-b063-2cfe7ffc603b" />

---

# Clusterização

A análise não supervisionada foi utilizada para identificar grupos de clientes sem utilizar diretamente o NPS na formação dos clusters.

### K-Means

Apresentou resultados estatisticamente interessantes, principalmente quando combinado ao Kernel PCA, porém alguns grupos apresentaram perfis médios pouco distintos.

### DBSCAN

Não identificou uma estrutura de densidade suficientemente consistente, apresentando configurações com excesso de ruído, clusters pequenos ou baixo Silhouette Score.

### Clusterização Hierárquica

Foram comparados:

```text
single
complete
average
ward
```

O **Ward combinado com Kernel PCA** apresentou melhor equilíbrio entre qualidade estatística, distribuição dos clientes e interpretação de negócio.

---

## Segmentos identificados

Foram identificados dois grupos principais.

### Menor criticidade

Apresenta, em média:

- menor atraso;
- menos contatos com atendimento;
- menor número de reclamações;
- NPS mais elevado.

### Alta criticidade

Apresenta, em média:

- maiores atrasos;
- mais contatos com atendimento;
- maior recorrência de reclamações;
- menor NPS;
- maior concentração de detratores.

O segmento de menor criticidade ainda possui clientes insatisfeitos, portanto os grupos representam **criticidade relativa**, e não uma separação direta entre satisfeitos e insatisfeitos.

---

## Relação dos segmentos com o NPS

O NPS não participou da formação dos clusters e foi utilizado posteriormente para auxiliar na interpretação dos grupos.

| Segmento | NPS médio | Detratores |
|---|---:|---:|
| Menor criticidade | 5,86 | 52,46% |
| Alta criticidade | 3,29 | 90,68% |

O grupo de alta criticidade apresentou concentração significativamente maior de detratores e problemas operacionais.

---

## Segmentação na aplicação

A aplicação apresenta uma visão comparativa dos grupos encontrados durante o treinamento.

São exibidos:

- quantidade de clientes por segmento;
- NPS médio;
- distribuição entre detratores, neutros e promotores;
- perfil operacional médio;
- principais diferenças entre os grupos.

<img width="1906" height="893" alt="Segmentação de clientes" src="https://github.com/user-attachments/assets/330196b3-8be5-48af-9b68-9c6405d26635" />

---

# Relação entre os modelos

As duas abordagens possuem objetivos complementares.

O modelo supervisionado responde:

> Qual é a probabilidade deste cliente ser detrator?

A clusterização responde:

> Quais clientes apresentam padrões semelhantes de comportamento e criticidade?

Assim, a solução combina:

- **visão individual:** probabilidade prevista pelo XGBoost;
- **visão comportamental:** segmentos encontrados pela clusterização.

---

# Aplicação Streamlit

A aplicação está disponível em `src/app.py` e possui quatro áreas principais.

### Visão geral

Apresenta o objetivo da solução e o fluxo dos modelos.

### Predição de detratores

Permite enviar um arquivo CSV e gerar:

- probabilidade de detrator;
- nível de risco;
- valor em risco;
- quantidade de problemas;
- desconto sugerido;
- ação recomendada;
- fila de priorização;
- download dos resultados.

### Segmentação de clientes

Apresenta:

- quantidade de clientes por segmento;
- NPS médio;
- composição das classes de NPS;
- perfil médio dos grupos;
- principais diferenças operacionais.

### Sobre o projeto

Resume modelos, tecnologias e principais características da solução.

---

## Arquivo de entrada para predição

O CSV deve conter:

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

A coluna `nps` não é necessária para novas previsões.

---

# Artefatos exportados

### Modelo supervisionado

```text
models/supervised/modelo_optuna.pkl
```

### Clusterização

```text
models/clustering/kernel_pca_clusterizacao.pkl
models/clustering/modelo_hierarquico_ward.pkl
```

### Pré-processamento

```text
models/preprocessing/pre_processamento.pkl
models/preprocessing/pre_processamento_clusterizacao.pkl
```

> O `AgglomerativeClustering` não possui método `predict`, portanto o modelo Ward é utilizado como artefato analítico dos grupos formados durante o treinamento.

---

# Tecnologias

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Optuna
- SHAP
- Plotly
- SciPy
- Joblib
- Streamlit
- Docker
- Jupyter Notebook
- Git
- GitHub

---

# Instalação

Clone o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
cd customer-churn-prediction
```

Crie o ambiente virtual:

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# Execução

## Notebooks

Ordem recomendada:

```text
01 → Entendimento e preparação
02 → Análise exploratória
03 → Modelagem supervisionada
04 → Clusterização
```

## Streamlit

Na raiz do projeto:

```bash
python -m streamlit run src/app.py
```

A aplicação será disponibilizada em:

```text
http://localhost:8501
```

---

# Execução com Docker

O projeto também pode ser executado em um container Docker, evitando a necessidade de configurar manualmente o ambiente Python e suas dependências.

### Build da imagem

Na raiz do projeto:

```bash
docker build -t customer-churn-prediction .
```

### Execução do container

```bash
docker run --rm -p 8501:8501 customer-churn-prediction
```

Após a inicialização, acesse:

```text
http://localhost:8501
```

O container utiliza **Python 3.12**, instala as dependências descritas no `requirements.txt` e inicializa automaticamente a aplicação Streamlit.

---

# Limitações

- os dados não apresentam separação natural perfeita entre os segmentos;
- os Silhouette Scores da clusterização são moderados;
- o segmento de menor criticidade ainda possui detratores;
- regras de desconto e ações recomendadas são simulações analíticas;
- o valor em risco não representa perda financeira garantida;
- o modelo hierárquico não realiza inferência direta para novos clientes;
- Feature Importance e SHAP explicam o comportamento do modelo, mas não estabelecem causalidade;
- o desempenho deve ser monitorado caso o perfil dos dados mude.

---

# Próximos passos

Possíveis evoluções:

- testes automatizados;
- validação mais robusta dos arquivos de entrada;
- explicações SHAP individuais diretamente na aplicação;
- monitoramento de drift;
- rastreamento de experimentos com MLflow;
- criação de pipelines automatizados;
- API com FastAPI;
- integração com banco de dados;
- deploy em cloud;
- pipeline CI/CD;
- dashboard gerencial;
- validação financeira das estratégias de retenção.

---

# Conclusão

O projeto integra diferentes etapas de um fluxo de Ciência de Dados aplicado à experiência do cliente:

- preparação e análise dos dados;
- engenharia e seleção de atributos;
- modelagem supervisionada;
- otimização com Optuna;
- interpretabilidade com Feature Importance e SHAP;
- clusterização;
- aplicação interativa com Streamlit;
- containerização com Docker.

O **XGBoost otimizado com Optuna** apresentou boa capacidade de identificação de detratores e comportamento consistente durante a validação.

A análise de interpretabilidade reforçou a importância de fatores relacionados a **reclamações, atrasos e atendimento**, enquanto a clusterização identificou grupos com diferentes níveis de criticidade operacional.

A aplicação Streamlit transforma os resultados dos modelos em uma camada de consumo voltada à tomada de decisão, enquanto o Docker garante um ambiente padronizado e reprodutível para execução da solução.

A combinação dessas abordagens cria uma solução analítica capaz de apoiar estratégias de **retenção, priorização de atendimento e melhoria da experiência do consumidor**.

---

## Autor

Desenvolvido por **Renan Trevelim**.

Projeto criado para estudo, portfólio e aplicação prática de técnicas de Ciência de Dados, Machine Learning e análise de negócio.
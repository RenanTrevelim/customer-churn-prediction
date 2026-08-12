# Customer Intelligence — Predição de Detratores e Segmentação de Clientes

Projeto de Ciência de Dados aplicado à satisfação de clientes em um contexto de e-commerce, combinando análise exploratória, engenharia de atributos, seleção de features, Machine Learning supervisionado, clusterização e aplicação interativa com Streamlit.

A solução utiliza informações relacionadas ao perfil do cliente, pedidos, entrega, atendimento e reclamações para gerar indicadores úteis à retenção, priorização de clientes e melhoria da experiência do consumidor.

---

## Visão geral

O projeto foi desenvolvido para responder a duas perguntas principais:

> Qual é a probabilidade de um cliente se tornar detrator?

> Quais perfis de clientes apresentam padrões semelhantes de comportamento e criticidade operacional?

Para isso, foram construídas duas abordagens complementares:

* **Classificação supervisionada**, para estimar o risco individual de um cliente ser detrator;
* **Clusterização**, para identificar grupos com diferentes níveis de criticidade operacional.

Os resultados foram integrados a uma aplicação Streamlit, permitindo explorar previsões, segmentos e recomendações de negócio.

---

## Objetivos do projeto

Os principais objetivos são:

* compreender os fatores associados à insatisfação dos clientes;
* analisar o comportamento do NPS;
* identificar padrões relacionados a entrega, atendimento e reclamações;
* criar atributos relevantes para a modelagem;
* aplicar técnicas de seleção e análise de features;
* comparar diferentes algoritmos de Machine Learning;
* estimar a probabilidade de um cliente se tornar detrator;
* classificar clientes por nível de risco;
* estimar o valor financeiro em risco;
* gerar recomendações de retenção;
* segmentar clientes por nível de criticidade;
* disponibilizar os resultados em uma aplicação interativa.

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
Feature Selection
VarianceThreshold
    ↓
┌─────────────────────────────┬─────────────────────────────┐
│ Modelo supervisionado       │ Modelo não supervisionado   │
│ XGBoost                     │ Kernel PCA + Ward           │
│                             │                             │
│ Probabilidade de detrator   │ Segmentos de criticidade    │
│ Nível de risco              │ Perfil médio dos grupos     │
│ Valor em risco              │ Relação com o NPS           │
│ Ação recomendada            │ Interpretação operacional   │
└─────────────────────────────┴─────────────────────────────┘
    ↓
Aplicação Streamlit
```

---

## Estrutura do projeto

```text
customer-intelligence/
│
├── data/
│   ├── raw/
│   │   └── nps_clientes.csv
│   │
│   ├── processed/
│   │   └── clientes_tratados.csv
│   │
│   └── README.md
│
├── models/
│   ├── supervised/
│   │   └── modelo_final.pkl
│   │
│   ├── clustering/
│   │   ├── kernel_pca_clusterizacao.pkl
│   │   └── modelo_hierarquico_ward.pkl
│   │
│   ├── preprocessing/
│   │   ├── pre_processamento.pkl
│   │   └── pre_processamento_clusterizacao.pkl
│   │
│   └── README.md
│
├── notebooks/
│   ├── 01_entendimento_e_preparacao_dos_dados.ipynb
│   ├── 02_analise_exploratoria_dos_dados.ipynb
│   ├── 03_modelos_supervisionados.ipynb
│   ├── 04_modelos_clusterizacao.ipynb
│   └── README.md
│
├── src/
│   ├── app.py
│   ├── predict.py
│   └── README.md
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Dados

A pasta `data/` está organizada em duas camadas.

### `raw/`

Contém a base original, preservada sem alterações.

```text
nps_clientes.csv
```

### `processed/`

Contém a base após as etapas de limpeza e preparação.

```text
clientes_tratados.csv
```

A separação entre dados brutos e processados contribui para rastreabilidade e reprodutibilidade do projeto.

---

## Notebooks

### 01. Entendimento e preparação dos dados

Responsável pela leitura, validação e tratamento inicial da base.

Principais etapas:

* análise da estrutura dos dados;
* verificação dos tipos das variáveis;
* identificação de valores ausentes;
* verificação de registros duplicados;
* validação da consistência da base;
* preparação dos dados para as etapas seguintes.

```text
01_entendimento_e_preparacao_dos_dados.ipynb
```

---

### 02. Análise exploratória dos dados

Responsável pela investigação dos padrões relacionados à satisfação dos clientes.

Principais análises:

* distribuição das variáveis;
* comportamento do NPS;
* relação entre satisfação, entrega e atendimento;
* impacto de atrasos e reclamações;
* análise de correlação entre variáveis numéricas e NPS;
* identificação de padrões relevantes para a modelagem;
* apoio à criação de novas variáveis.

```text
02_analise_exploratoria_dos_dados.ipynb
```

---

### 03. Modelos supervisionados

Responsável pela construção e avaliação dos modelos de classificação voltados à identificação de clientes detratores.

Principais etapas:

* definição da variável-alvo;
* engenharia de atributos;
* divisão entre treino e teste;
* pré-processamento;
* Feature Selection;
* treinamento e comparação de classificadores;
* otimização de hiperparâmetros;
* validação cruzada;
* análise da importância das features;
* seleção do modelo final;
* aplicação das probabilidades ao contexto de negócio.

Algoritmos avaliados:

* Regressão Logística;
* Árvore de Decisão;
* Random Forest;
* K-Nearest Neighbors;
* Support Vector Machine;
* Gaussian Naive Bayes;
* Gradient Boosting;
* XGBoost.

```text
03_modelos_supervisionados.ipynb
```

---

### 04. Modelos de clusterização

Responsável pela segmentação dos clientes utilizando aprendizado não supervisionado.

Algoritmos avaliados:

* K-Means;
* DBSCAN;
* Clusterização Hierárquica.

Técnicas de redução de dimensionalidade:

* PCA;
* Kernel PCA.

A combinação entre **Kernel PCA** e **Clusterização Hierárquica com método Ward** foi selecionada como solução final por produzir grupos mais coerentes e interpretáveis para o contexto de negócio.

```text
04_modelos_clusterizacao.ipynb
```

---

## Engenharia de atributos

Durante a análise e modelagem, foram criadas três variáveis derivadas para representar situações operacionais relevantes.

### `atraso_critico`

Identifica entregas com atraso igual ou superior a dois dias.

### `problema_complexo`

Identifica clientes com múltiplos contatos com o atendimento e maior tempo de resolução.

### `reclamacao_recorrente`

Identifica clientes com três ou mais reclamações registradas.

Esses atributos sintetizam padrões identificados durante a análise exploratória e permitem representar situações de insatisfação de forma mais direta.

---

## Pré-processamento e Feature Selection

O fluxo de preparação dos dados supervisionados inclui:

* padronização das variáveis numéricas com `StandardScaler`;
* codificação da variável `regiao_cliente` com `OneHotEncoder`;
* aplicação das transformações com `ColumnTransformer`;
* aplicação do método `VarianceThreshold`;
* preservação das mesmas transformações entre treino, teste e inferência.

O `VarianceThreshold` foi utilizado após o pré-processamento para verificar a existência de features com baixa variabilidade.

Nenhuma das 20 features apresentou variância inferior ao limite definido, portanto todas foram preservadas para a modelagem.

O pré-processamento final foi exportado com `Joblib` para garantir consistência durante novas previsões.

---

## Modelagem supervisionada

A variável-alvo foi definida para identificar clientes detratores com base no NPS.

Foram considerados detratores os clientes com nota igual ou inferior a 6.

### Modelo selecionado

O **XGBoost** foi selecionado como modelo final após comparação de métricas, otimização de hiperparâmetros, análise das matrizes de confusão, curvas ROC e validação cruzada.

Resultados médios obtidos na validação cruzada:

| Métrica   | Resultado aproximado |
| --------- | -------------------: |
| Accuracy  |               0,8300 |
| Precision |               0,8560 |
| Recall    |               0,9264 |
| F1-score  |               0,8898 |
| ROC-AUC   |               0,8734 |

O recall superior a **92%** foi especialmente relevante para o objetivo do projeto, pois indica boa capacidade de identificação dos clientes pertencentes à classe positiva.

Os baixos desvios observados durante a validação cruzada também indicaram comportamento consistente entre diferentes divisões dos dados.

### Importância das features

Após a escolha do modelo final, foi analisado o atributo `feature_importances_` do XGBoost.

Entre as variáveis com maior contribuição para as previsões, destacaram-se:

* `reclamacao_recorrente`;
* `numero_reclamacoes`;
* `atraso_entrega_dias`;
* `atraso_critico`;
* `contatos_atendimento`;
* `tempo_resolucao_dias`.

Os resultados reforçam os padrões encontrados durante a análise exploratória, principalmente em relação ao impacto de atrasos, reclamações recorrentes e dificuldades de atendimento.

---

## Aplicação do modelo supervisionado

As probabilidades geradas pelo XGBoost foram transformadas em indicadores para apoio à decisão.

A solução permite:

* estimar a probabilidade de detrator;
* classificar clientes por nível de risco;
* calcular a quantidade de problemas operacionais;
* estimar o valor financeiro em risco;
* recomendar ações de recuperação;
* sugerir descontos;
* criar uma fila de priorização.

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

Esse valor não representa uma perda financeira garantida, mas funciona como uma medida de priorização.

As recomendações consideram o nível de risco e a presença de fatores como atraso crítico, problema complexo e reclamação recorrente.

---

## Clusterização

A análise não supervisionada foi desenvolvida para complementar a classificação individual.

O objetivo foi identificar grupos com diferentes níveis de criticidade operacional, sem utilizar diretamente o NPS na formação dos clusters.

### K-Means

O K-Means apresentou bons valores de Silhouette Score, principalmente quando combinado com Kernel PCA.

Entretanto, parte dos grupos apresentou perfis médios semelhantes, reduzindo a clareza da segmentação para o negócio.

### DBSCAN

O DBSCAN não encontrou uma estrutura de densidade consistente.

Dependendo dos parâmetros, o algoritmo apresentou situações como:

* grande quantidade de registros classificados como ruído;
* muitos clusters pequenos;
* concentração excessiva em um único grupo;
* Silhouette Score próximo de zero ou negativo.

Por esse motivo, não foi selecionado.

### Clusterização Hierárquica

Foram comparados os métodos:

```text
single
complete
average
ward
```

Algumas configurações apresentaram Silhouette Score superior, mas produziram grupos extremamente desbalanceados.

O método **Ward**, combinado com Kernel PCA, apresentou melhor equilíbrio entre qualidade estatística, distribuição dos grupos e interpretação de negócio.

---

## Segmentos finais

Os clientes foram divididos em dois grupos principais.

### Clientes de menor criticidade

Apresentaram, em média:

* menor atraso;
* menos contatos com o atendimento;
* menor número de reclamações;
* NPS médio mais elevado.

Esse grupo ainda possui clientes detratores, portanto a interpretação representa menor criticidade relativa e não ausência de problemas.

### Clientes de alta criticidade

Apresentaram, em média:

* maior atraso;
* mais contatos com o atendimento;
* maior recorrência de reclamações;
* menor NPS médio;
* maior concentração de detratores.

Esse grupo representa clientes prioritários para acompanhamento e ações de recuperação.

---

## Relação dos segmentos com o NPS

O NPS não foi utilizado na formação dos clusters.

Ele foi analisado posteriormente como forma de validar a interpretação dos segmentos.

| Segmento          | NPS médio | Detratores |
| ----------------- | --------: | ---------: |
| Menor criticidade |      5,86 |     52,46% |
| Alta criticidade  |      3,29 |     90,68% |

Apesar de a clusterização não produzir separação perfeita entre clientes satisfeitos e insatisfeitos, foi possível identificar um grupo significativamente mais vulnerável e com maior concentração de problemas operacionais.

---

## Relação entre os modelos

As duas abordagens possuem objetivos complementares.

O modelo supervisionado responde:

> Qual é a probabilidade de este cliente se tornar detrator?

A clusterização responde:

> Quais clientes apresentam padrões semelhantes de comportamento e criticidade?

A combinação permite analisar o problema em dois níveis:

* **individual**, por meio da probabilidade prevista pelo XGBoost;
* **comportamental**, por meio dos segmentos encontrados.

---

## Aplicação Streamlit

A pasta `src/` contém a aplicação interativa desenvolvida para apresentação e utilização dos resultados.

A interface possui quatro páginas.

### Visão geral

Apresenta o objetivo da solução, principais entregas e fluxo dos modelos.

### Predição de detratores

Permite enviar um arquivo CSV e gerar:

* probabilidade de detrator;
* nível de risco;
* valor em risco;
* quantidade de problemas;
* desconto sugerido;
* ação recomendada;
* fila de priorização;
* download do resultado completo.

### Segmentação de clientes

Apresenta:

* quantidade de clientes em cada segmento;
* NPS médio por grupo;
* composição das classes de NPS;
* perfil médio dos segmentos;
* principais diferenças operacionais.

### Sobre o projeto

Resume os modelos utilizados, tecnologias, resultados e limitações da solução.

---

## Arquivo de entrada da aplicação

O arquivo CSV enviado para a área de predição deve conter:

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

## Artefatos exportados

### Modelo supervisionado

```text
models/supervised/modelo_final.pkl
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

> O `AgglomerativeClustering` não possui método `predict`. Por isso, o modelo Ward é utilizado na aplicação como artefato analítico dos grupos formados durante o treinamento.

---

## Tecnologias utilizadas

* Python;
* Pandas;
* NumPy;
* Matplotlib;
* Seaborn;
* Scikit-learn;
* XGBoost;
* LightGBM;
* SciPy;
* Joblib;
* Streamlit;
* Jupyter Notebook;
* Git;
* GitHub.

---

## Instalação

Clone o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
```

Acesse a pasta:

```bash
cd customer-intelligence
```

Crie o ambiente virtual com Python 3.12.

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

Atualize o `pip`:

```bash
python -m pip install --upgrade pip
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Execução dos notebooks

A ordem recomendada é:

```text
01 → Entendimento e preparação dos dados
02 → Análise exploratória
03 → Modelos supervisionados
04 → Modelos de clusterização
```

---

## Execução da aplicação

Na raiz do projeto, execute:

```bash
python -m streamlit run src/app.py
```

A aplicação será aberta em:

```text
http://localhost:8501
```

---

## Limitações

Algumas limitações devem ser consideradas:

* os dados não apresentam separação natural perfeita entre os grupos;
* os valores de Silhouette Score permaneceram relativamente baixos;
* o segmento de menor criticidade ainda possui clientes detratores;
* as regras de desconto e ação são simulações analíticas;
* o valor em risco não representa perda financeira garantida;
* o modelo hierárquico não possui inferência direta para novos clientes;
* a importância das features não representa relação causal;
* as recomendações precisam ser validadas de acordo com as regras reais da empresa;
* o desempenho dos modelos deve ser monitorado caso o perfil dos dados mude.

---

## Possíveis melhorias

Evoluções futuras do projeto:

* criar testes automatizados;
* adicionar validação mais completa dos arquivos enviados;
* incluir análise individual de clientes na aplicação;
* testar estratégias adicionais de Feature Selection;
* comparar `feature_importances_` com Permutation Importance ou SHAP;
* testar redução de features e reavaliar o desempenho do XGBoost;
* criar um classificador indutivo para novos segmentos;
* implementar monitoramento de drift;
* adicionar rastreamento de experimentos com MLflow;
* criar pipelines automatizados;
* containerizar a aplicação com Docker;
* desenvolver uma API com FastAPI;
* integrar a solução a um banco de dados;
* publicar a aplicação em ambiente cloud;
* desenvolver um dashboard gerencial em Power BI;
* validar financeiramente as estratégias de retenção.

---

## Aplicações de negócio

A solução pode apoiar áreas como atendimento, logística, retenção, experiência do consumidor, relacionamento e gestão de reclamações.

Entre os principais usos estão:

* priorização de clientes críticos;
* monitoramento preventivo;
* personalização de ações de recuperação;
* redução da recorrência de problemas;
* análise do impacto operacional sobre o NPS;
* construção de campanhas de retenção;
* acompanhamento do valor financeiro em risco;
* apoio à tomada de decisão.

---

## Conclusão

Este projeto demonstra uma abordagem completa de Ciência de Dados aplicada à satisfação de clientes, integrando:

* preparação de dados;
* análise exploratória;
* engenharia de atributos;
* seleção e análise de features;
* Machine Learning supervisionado;
* clusterização;
* interpretação de negócio;
* aplicação interativa.

O **XGBoost** foi selecionado como modelo supervisionado final, apresentando boa capacidade de identificação de clientes com risco de insatisfação e desempenho consistente durante a validação cruzada.

A análise de importância das features também reforçou os padrões encontrados durante o EDA, destacando principalmente fatores relacionados a reclamações, atrasos e atendimento.

A clusterização complementa a previsão individual ao identificar grupos com diferentes níveis de criticidade operacional.

A integração das duas abordagens cria uma base analítica para estratégias de retenção, priorização de atendimento e melhoria contínua da experiência do cliente.

---

## Autor

Desenvolvido por **Renan Trevelim**.

Projeto criado para estudo, portfólio e aplicação prática de técnicas de Ciência de Dados, Machine Learning e análise de negócio.

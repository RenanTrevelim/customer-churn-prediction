# Customer Intelligence — Predição de Detratores e Segmentação de Clientes

Projeto completo de Ciência de Dados aplicado à satisfação de clientes, combinando análise exploratória, engenharia de atributos, Machine Learning supervisionado, clusterização e uma aplicação interativa desenvolvida com Streamlit.

A solução foi construída para transformar informações relacionadas ao perfil do cliente, pedidos, entrega, atendimento e reclamações em indicadores úteis para priorização, retenção e melhoria da experiência do consumidor.

---

## Visão geral

A insatisfação de clientes pode estar associada a diferentes fatores operacionais, como atrasos na entrega, recorrência de reclamações, excesso de contatos com o atendimento e tempo elevado de resolução.

Este projeto foi desenvolvido para responder a duas perguntas complementares:

> Qual é a probabilidade de um cliente se tornar detrator?

> Quais perfis de clientes apresentam padrões semelhantes de comportamento e criticidade operacional?

Para responder a essas perguntas, foram desenvolvidas duas abordagens:

- **Classificação supervisionada**, para estimar o risco individual de um cliente ser detrator;
- **Clusterização**, para identificar grupos com diferentes níveis de criticidade operacional.

Os resultados foram integrados a uma interface em Streamlit, permitindo explorar previsões, métricas, gráficos, segmentos e recomendações de negócio.

---

## Objetivos do projeto

Os principais objetivos são:

- compreender os fatores associados à insatisfação dos clientes;
- analisar o comportamento do NPS;
- identificar padrões relacionados a entrega, atendimento e reclamações;
- criar atributos relevantes para a modelagem;
- comparar diferentes algoritmos de classificação;
- estimar a probabilidade de um cliente se tornar detrator;
- classificar clientes por nível de risco;
- estimar o valor financeiro em risco;
- gerar recomendações de retenção;
- segmentar clientes por nível de criticidade;
- disponibilizar os resultados em uma aplicação interativa.

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
┌─────────────────────────────┬─────────────────────────────┐
│ Modelo supervisionado       │ Modelo não supervisionado   │
│ Gradient Boosting           │ Kernel PCA + Ward           │
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

Essa base é utilizada nas etapas de entendimento, validação, tratamento e análise exploratória.

### `processed/`

Contém a base após as etapas de limpeza e preparação.

```text
clientes_tratados.csv
```

Essa versão é utilizada nos notebooks de modelagem supervisionada, clusterização e na aplicação Streamlit.

> A separação entre dados brutos e processados garante rastreabilidade e reprodutibilidade.

---

## Notebooks

### 01. Entendimento e preparação dos dados

Responsável pela leitura, validação e tratamento inicial da base.

Principais etapas:

- análise da estrutura dos dados;
- verificação dos tipos das variáveis;
- identificação de valores ausentes;
- verificação de registros duplicados;
- validação da consistência da base;
- preparação dos dados para as etapas seguintes.

Arquivo:

```text
01_entendimento_e_preparacao_dos_dados.ipynb
```

---

### 02. Análise exploratória dos dados

Responsável pela investigação dos principais padrões presentes na base.

Principais análises:

- distribuição das variáveis;
- comportamento do NPS;
- relação entre satisfação, entrega e atendimento;
- impacto de atrasos e reclamações;
- identificação de padrões relevantes para a modelagem;
- apoio à criação de novas variáveis.

Arquivo:

```text
02_analise_exploratoria_dos_dados.ipynb
```

---

### 03. Modelos supervisionados

Responsável pela construção de modelos de classificação voltados à identificação de clientes detratores.

Principais etapas:

- definição da variável-alvo;
- engenharia de atributos;
- divisão entre treino e teste;
- pré-processamento;
- treinamento de diferentes classificadores;
- comparação de métricas;
- otimização de hiperparâmetros;
- validação cruzada;
- seleção do modelo final;
- aplicação das probabilidades ao contexto de negócio.

Algoritmos avaliados:

- Regressão Logística;
- Árvore de Decisão;
- Random Forest;
- K-Nearest Neighbors;
- Support Vector Machine;
- Gaussian Naive Bayes;
- Gradient Boosting.

Arquivo:

```text
03_modelos_supervisionados.ipynb
```

---

### 04. Modelos de clusterização

Responsável pela segmentação de clientes por meio de técnicas de aprendizado não supervisionado.

Algoritmos avaliados:

- K-Means;
- DBSCAN;
- Clusterização Hierárquica.

Técnicas de redução de dimensionalidade:

- PCA;
- Kernel PCA.

Critérios de avaliação:

- Silhouette Score;
- distribuição dos clientes entre os clusters;
- equilíbrio dos grupos;
- interpretação dos perfis;
- utilidade para o negócio.

A combinação entre **Kernel PCA** e **Clusterização Hierárquica com método Ward** foi selecionada como solução final.

Arquivo:

```text
04_modelos_clusterizacao.ipynb
```

---

## Engenharia de atributos

Durante a modelagem, foram criadas três variáveis derivadas para representar situações operacionais relevantes.

### `atraso_critico`

Identifica entregas com atraso igual ou superior a dois dias.

### `problema_complexo`

Identifica clientes que realizaram múltiplos contatos e tiveram maior tempo de resolução.

### `reclamacao_recorrente`

Identifica clientes com três ou mais reclamações registradas.

Esses atributos ajudam os modelos a representar padrões de insatisfação de maneira mais direta e interpretável.

---

## Pré-processamento

O fluxo de preparação dos dados inclui:

- padronização das variáveis numéricas com `StandardScaler`;
- codificação da variável `regiao_cliente` com `OneHotEncoder`;
- aplicação das transformações com `ColumnTransformer`;
- preservação do mesmo pré-processamento entre treino, teste e aplicação;
- exportação dos objetos com `Joblib`.

Na clusterização, também foram utilizadas técnicas de redução de dimensionalidade para diminuir a quantidade de variáveis e facilitar a identificação dos grupos.

---

## Modelagem supervisionada

A variável-alvo foi definida para identificar clientes detratores com base no NPS.

O projeto considerou como detratores os clientes com nota igual ou inferior a 6.

### Modelo selecionado

O **Gradient Boosting** foi selecionado como modelo final por apresentar bom equilíbrio entre as métricas e maior capacidade de identificar clientes detratores.

Resultados obtidos na validação cruzada:

| Métrica | Resultado aproximado |
|---|---:|
| Accuracy | 0,8315 |
| Precision | 0,8591 |
| Recall | 0,9244 |
| F1-score | 0,8905 |
| ROC-AUC | 0,8726 |

A seleção priorizou especialmente o **recall**, pois deixar de identificar um cliente detrator pode representar perda de receita, recorrência de reclamações e piora na experiência.

---

## Aplicação do modelo supervisionado

As probabilidades geradas pelo modelo foram transformadas em uma camada de apoio à decisão.

A solução permite:

- estimar a probabilidade de detrator;
- classificar clientes por nível de risco;
- calcular a quantidade de problemas operacionais;
- estimar o valor financeiro em risco;
- recomendar ações de recuperação;
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

Esse valor não representa uma perda financeira garantida, mas funciona como uma medida de prioridade.

### Recomendações

As ações sugeridas consideram:

- nível de risco;
- atraso crítico;
- problema complexo;
- reclamação recorrente;
- combinação entre diferentes problemas.

---

## Clusterização

A análise não supervisionada foi desenvolvida para complementar o modelo de classificação.

O objetivo não foi reproduzir diretamente as classes tradicionais do NPS, mas identificar perfis de clientes com diferentes níveis de criticidade operacional.

---

### K-Means

O K-Means apresentou os maiores valores de Silhouette Score entre os modelos avaliados, principalmente quando combinado com Kernel PCA.

Entretanto, parte dos grupos apresentou perfis médios semelhantes, reduzindo a clareza da segmentação para o negócio.

---

### DBSCAN

O DBSCAN não encontrou uma estrutura de densidade consistente.

Dependendo dos parâmetros utilizados, o algoritmo:

- classificou grande parte dos registros como ruído;
- criou muitos clusters pequenos;
- concentrou praticamente todos os clientes em um único grupo;
- apresentou Silhouette Score próximo de zero ou negativo.

Por esse motivo, o modelo foi descartado.

---

### Clusterização Hierárquica

Foram comparados os métodos:

```text
single
complete
average
ward
```

Algumas configurações apresentaram Silhouette Score superior, mas geraram grupos extremamente desbalanceados, chegando a isolar apenas uma observação.

O método **Ward**, combinado com Kernel PCA, produziu uma divisão mais equilibrada e coerente com o objetivo do projeto.

---

## Segmentos finais

Os clientes foram divididos em dois grupos:

### Clientes de menor criticidade

Apresentaram, em média:

- menor atraso;
- menos contatos com o atendimento;
- menor número de reclamações;
- NPS médio mais elevado.

Esse grupo ainda possui clientes detratores, portanto a interpretação é relativa e não representa ausência de problemas.

### Clientes de alta criticidade

Apresentaram, em média:

- maior atraso;
- mais contatos com o atendimento;
- maior recorrência de reclamações;
- menor NPS médio;
- maior concentração de detratores.

Esse grupo deve receber prioridade nas ações de recuperação e acompanhamento.

---

## Relação dos segmentos com o NPS

O NPS não foi utilizado na formação dos clusters.

Ele foi incluído posteriormente apenas para validar a interpretação dos grupos.

| Segmento | NPS médio | Detratores |
|---|---:|---:|
| Menor criticidade | 5,86 | 52,46% |
| Alta criticidade | 3,29 | 90,68% |

Os resultados mostram que a clusterização não separa perfeitamente clientes satisfeitos e insatisfeitos.

Ainda assim, o modelo conseguiu identificar um grupo significativamente mais vulnerável, com menor satisfação e maior concentração de problemas operacionais.

---

## Relação entre os modelos

As duas abordagens são complementares.

O modelo supervisionado responde:

> Qual é a probabilidade de este cliente se tornar detrator?

A clusterização responde:

> Quais clientes apresentam padrões semelhantes de comportamento e criticidade?

A combinação permite analisar o problema em dois níveis:

- **individual**, por meio da probabilidade prevista;
- **comportamental**, por meio dos segmentos encontrados.

---

## Aplicação Streamlit

A pasta `src/` contém uma aplicação interativa para apresentação dos resultados.

A interface possui quatro páginas.

### Visão geral

Apresenta:

- objetivo da solução;
- principais entregas;
- fluxo do modelo supervisionado;
- fluxo do modelo não supervisionado.

### Predição de detratores

Permite enviar um arquivo CSV e gerar:

- probabilidade de detrator;
- nível de risco;
- valor em risco;
- quantidade de problemas;
- desconto sugerido;
- ação recomendada;
- fila de priorização;
- download do resultado completo.

### Segmentação de clientes

Apresenta:

- quantidade de clientes em cada segmento;
- NPS médio por grupo;
- composição das classes de NPS;
- perfil médio dos segmentos;
- interpretação dos grupos;
- principais diferenças operacionais.

### Sobre o projeto

Resume:

- modelos utilizados;
- resultados gerados;
- tecnologias aplicadas;
- limitações da solução.

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

- Python;
- Pandas;
- NumPy;
- Matplotlib;
- Seaborn;
- Scikit-learn;
- SciPy;
- Joblib;
- Streamlit;
- Jupyter Notebook;
- Git;
- GitHub.

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

Crie o ambiente virtual com Python 3.12:

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

A aplicação será aberta no endereço:

```text
http://localhost:8501
```

---

## Limitações

Algumas limitações devem ser consideradas:

- os dados não apresentam separação natural perfeita entre os grupos;
- os valores de Silhouette Score permaneceram baixos;
- o segmento de menor criticidade ainda possui clientes detratores;
- as regras de desconto e ação são simulações analíticas;
- o valor em risco não representa perda financeira garantida;
- o modelo hierárquico não possui inferência direta para novos clientes;
- as recomendações precisam ser validadas de acordo com as regras reais da empresa;
- o desempenho deve ser monitorado caso o perfil dos dados mude.

---

## Possíveis melhorias

Evoluções futuras do projeto:

- criar testes automatizados;
- adicionar validação mais completa dos arquivos enviados;
- incluir análise individual de clientes na aplicação;
- criar um classificador indutivo para novos segmentos;
- implementar monitoramento de drift;
- adicionar rastreamento de experimentos com MLflow;
- criar pipelines automatizados;
- containerizar a aplicação com Docker;
- desenvolver uma API com FastAPI;
- integrar a solução a um banco de dados;
- publicar a aplicação em ambiente cloud;
- desenvolver um dashboard gerencial em Power BI;
- validar financeiramente as estratégias de retenção.

---

## Aplicações de negócio

A solução pode apoiar áreas como:

- atendimento ao cliente;
- experiência do consumidor;
- logística;
- retenção;
- relacionamento;
- marketing;
- gestão de reclamações;
- operações;
- planejamento comercial.

Possíveis usos:

- priorização de clientes críticos;
- monitoramento preventivo;
- personalização de ações de recuperação;
- redução da recorrência de problemas;
- análise do impacto operacional sobre o NPS;
- construção de campanhas de retenção;
- acompanhamento do valor financeiro em risco;
- apoio à tomada de decisão.

---

## Conclusão

Este projeto demonstra uma abordagem completa de Ciência de Dados aplicada à satisfação de clientes.

Mais do que treinar modelos, a solução conecta:

- preparação de dados;
- análise exploratória;
- engenharia de atributos;
- Machine Learning;
- clusterização;
- interpretação de negócio;
- aplicação interativa.

O modelo supervisionado permite estimar o risco individual de insatisfação, enquanto a clusterização identifica grupos com diferentes níveis de criticidade operacional.

A integração das duas abordagens oferece uma visão mais ampla da experiência do cliente e cria uma base analítica para estratégias de retenção, priorização de atendimento e melhoria contínua.

---

## Autor

Desenvolvido por **Renan Trevelim**.

Projeto criado para estudo, portfólio e aplicação prática de técnicas de Ciência de Dados, Machine Learning e análise de negócio.
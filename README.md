# Customer Satisfaction Analytics

Projeto de Ciência de Dados voltado à análise de satisfação de clientes, identificação de detratores e segmentação por nível de criticidade operacional.

A solução combina análise exploratória, engenharia de atributos, modelos supervisionados e técnicas de clusterização para transformar dados de pedidos, entrega, atendimento e reclamações em informações acionáveis para o negócio.

> **Status do projeto:** em desenvolvimento.  
> Os notebooks de análise, modelagem supervisionada e clusterização já foram estruturados. A pasta `src/` e algumas etapas de operacionalização ainda estão em evolução.

---

## Visão geral

A insatisfação de clientes pode estar relacionada a diferentes fatores, como atrasos na entrega, recorrência de reclamações, tempo de resolução e quantidade de contatos com o atendimento.

Este projeto busca responder a duas perguntas complementares:

> Qual é a probabilidade de um cliente se tornar detrator?

> Quais perfis de clientes apresentam padrões semelhantes de comportamento e criticidade?

Para isso, foram desenvolvidas duas abordagens:

- **modelagem supervisionada**, para estimar o risco de um cliente ser detrator;
- **clusterização**, para identificar grupos com diferentes níveis de criticidade operacional.

---

## Objetivos

Os principais objetivos do projeto são:

- compreender os fatores associados à insatisfação dos clientes;
- analisar o comportamento do NPS;
- identificar clientes com maior probabilidade de se tornarem detratores;
- comparar diferentes algoritmos de Machine Learning;
- criar níveis de risco para apoiar a priorização de atendimentos;
- segmentar clientes de acordo com padrões operacionais;
- gerar recomendações de negócio orientadas pelos resultados;
- estruturar uma solução reproduzível e organizada para portfólio.

---

## Estrutura do projeto

```text
customer-satisfaction-analytics/
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
│   └── em desenvolvimento
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Dados

A pasta `data/` está organizada em duas camadas.

### `raw/`

Contém os dados originais, preservados sem alterações.

Arquivo principal:

```text
nps_clientes.csv
```

Essa base é utilizada para entendimento, validação, tratamento e análise exploratória.

### `processed/`

Contém os dados após as etapas de limpeza, validação e preparação.

Arquivo principal:

```text
clientes_tratados.csv
```

Essa base é utilizada nos notebooks de análise exploratória, modelos supervisionados e clusterização.

> Os dados originais permanecem inalterados para garantir rastreabilidade e reprodutibilidade.

---

## Notebooks

### 01. Entendimento e preparação dos dados

Responsável pela leitura, validação e preparação inicial da base.

Principais etapas:

- análise da estrutura dos dados;
- verificação dos tipos das variáveis;
- identificação de valores ausentes e duplicados;
- tratamento e padronização das informações;
- preparação da base para as etapas seguintes.

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

Foram comparados diferentes algoritmos de Machine Learning, considerando métricas adequadas ao desbalanceamento das classes.

Principais etapas:

- definição da variável-alvo;
- engenharia de atributos;
- divisão entre treino e teste;
- pré-processamento dos dados;
- treinamento de diferentes classificadores;
- otimização de hiperparâmetros;
- validação cruzada;
- seleção do modelo final;
- aplicação das probabilidades previstas ao contexto de negócio.

O modelo final foi utilizado para:

- estimar a probabilidade de um cliente ser detrator;
- classificar clientes por nível de risco;
- calcular um indicador de valor em risco;
- recomendar ações de retenção;
- priorizar clientes para acompanhamento.

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

A combinação entre **Kernel PCA** e **Clusterização Hierárquica com método Ward** foi selecionada como solução final por produzir dois grupos mais coerentes e interpretáveis:

- clientes de menor criticidade;
- clientes de alta criticidade.

Arquivo:

```text
04_modelos_clusterizacao.ipynb
```

---

## Engenharia de atributos

Durante o desenvolvimento foram criadas variáveis derivadas para resumir situações operacionais relevantes:

```text
atraso_critico
problema_complexo
reclamacao_recorrente
```

Esses atributos foram construídos a partir de informações relacionadas a:

- atraso na entrega;
- quantidade de contatos com o atendimento;
- tempo de resolução;
- recorrência de reclamações.

O objetivo foi facilitar a identificação de padrões associados à insatisfação e à criticidade operacional.

---

## Pré-processamento

O fluxo de preparação dos dados inclui:

- separação entre variáveis numéricas e categóricas;
- padronização com `StandardScaler`;
- codificação da região com `OneHotEncoder`;
- aplicação das transformações por meio de `ColumnTransformer`;
- salvamento dos artefatos de pré-processamento com `Joblib`.

Nos modelos de clusterização, também foram comparadas duas representações dos dados:

- PCA tradicional;
- Kernel PCA com kernel RBF.

---

## Modelagem supervisionada

A variável-alvo foi definida para identificar clientes detratores a partir do NPS.

Foram avaliados diferentes algoritmos de classificação, incluindo:

- Regressão Logística;
- Árvore de Decisão;
- Random Forest;
- K-Nearest Neighbors;
- Support Vector Machine;
- Naive Bayes;
- Gradient Boosting.

Os modelos com melhor desempenho passaram por otimização de hiperparâmetros e validação cruzada.

### Modelo selecionado

O **Gradient Boosting** foi escolhido como modelo final por apresentar bom equilíbrio entre as métricas e maior capacidade de identificar clientes detratores.

A seleção considerou especialmente:

- recall da classe detratora;
- F1-score;
- F1 Macro;
- ROC-AUC;
- estabilidade na validação cruzada;
- impacto dos erros no contexto de negócio.

### Aplicação ao negócio

As probabilidades previstas pelo modelo foram utilizadas para:

- classificar clientes em níveis de risco;
- criar uma fila de priorização;
- estimar valor financeiro em risco;
- sugerir ações de recuperação;
- apoiar estratégias de retenção.

Os níveis considerados foram:

```text
Baixo
Moderado
Alto
Crítico
```

---

## Clusterização

A análise não supervisionada foi desenvolvida para complementar o modelo de classificação.

O objetivo não foi reproduzir diretamente as categorias de NPS, mas identificar grupos com diferentes padrões de criticidade operacional.

### K-Means

O K-Means apresentou os maiores valores de Silhouette Score entre os modelos avaliados, especialmente quando combinado com Kernel PCA.

Entretanto, alguns clusters apresentaram perfis parcialmente semelhantes, reduzindo a utilidade prática da segmentação.

### DBSCAN

O DBSCAN não encontrou uma estrutura de densidade consistente.

Dependendo da configuração, o modelo:

- classificou grande parte dos registros como ruído;
- criou muitos clusters pequenos;
- agrupou quase todos os registros em um único grupo;
- apresentou Silhouette Score próximo de zero ou negativo.

Por esse motivo, não foi selecionado.

### Clusterização Hierárquica

Foram comparados os métodos:

```text
single
complete
average
ward
```

Algumas configurações apresentaram Silhouette Score elevado, mas formaram grupos extremamente desbalanceados.

A combinação entre **Kernel PCA** e **Ward** apresentou a divisão mais coerente para o objetivo do projeto.

### Segmentos finais

Os dois grupos foram interpretados como:

- **Clientes de menor criticidade**
- **Clientes de alta criticidade**

O segmento de alta criticidade apresentou, em média:

- maior atraso;
- maior número de contatos;
- maior tempo de resolução;
- maior recorrência de reclamações;
- menor NPS;
- maior concentração de detratores.

---

## Principais resultados

### Modelagem supervisionada

A análise supervisionada permitiu construir um modelo capaz de estimar o risco de um cliente se tornar detrator.

Além da previsão, os resultados foram convertidos em uma camada de aplicação ao negócio, permitindo:

- classificar clientes por nível de risco;
- priorizar atendimentos;
- recomendar ações de recuperação;
- estimar o valor financeiro em risco;
- apoiar estratégias de retenção.

### Clusterização

A análise não supervisionada permitiu identificar dois segmentos com diferentes níveis de criticidade operacional.

O grupo de alta criticidade concentrou:

- maior atraso médio;
- maior número de contatos com o atendimento;
- maior tempo de resolução;
- maior recorrência de reclamações;
- menor NPS médio;
- maior proporção de detratores.

O grupo de menor criticidade apresentou indicadores relativamente mais favoráveis, embora ainda concentrasse clientes insatisfeitos.

A segmentação não produziu uma separação perfeita entre clientes satisfeitos e insatisfeitos. Ainda assim, conseguiu identificar um grupo significativamente mais vulnerável e prioritário para acompanhamento.

---

## Relação entre os modelos

As duas abordagens desenvolvidas são complementares.

O modelo supervisionado responde:

> Qual é a probabilidade de um cliente se tornar detrator?

A clusterização responde:

> Quais perfis de clientes apresentam padrões semelhantes de comportamento e criticidade?

A combinação das duas abordagens oferece uma visão mais completa da experiência do cliente, unindo previsão individual e segmentação comportamental.

---

## Aplicação ao negócio

Os resultados podem apoiar áreas como:

- atendimento ao cliente;
- experiência do consumidor;
- logística;
- relacionamento;
- retenção;
- marketing;
- gestão de reclamações.

Possíveis aplicações:

- priorização de clientes críticos;
- monitoramento preventivo;
- personalização de ações de recuperação;
- redução da recorrência de problemas;
- direcionamento de campanhas de relacionamento;
- construção de dashboards operacionais;
- acompanhamento de clientes com maior risco;
- apoio à tomada de decisão comercial.

---

## Artefatos gerados

Os modelos e objetos de transformação foram exportados para a pasta `models/`.

### Modelos supervisionados

```text
modelo_final.pkl
```

### Clusterização

```text
kernel_pca_clusterizacao.pkl
modelo_hierarquico_ward.pkl
```

### Pré-processamento

```text
pre_processamento.pkl
pre_processamento_clusterizacao.pkl
```

> O `AgglomerativeClustering` não possui método `predict`. Por isso, o modelo hierárquico salvo funciona como artefato analítico da segmentação atual. A atribuição de novos clientes exigirá uma estratégia adicional.

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
- Jupyter Notebook;
- Git;
- GitHub.

---

## Instalação

Clone o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
```

Acesse a pasta do projeto:

```bash
cd customer-satisfaction-analytics
```

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente no Linux ou macOS:

```bash
source .venv/bin/activate
```

No Windows:

```bash
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Inicie o Jupyter Notebook:

```bash
jupyter notebook
```

---

## Ordem de execução

A execução recomendada segue a ordem numérica dos notebooks:

```text
01 → Entendimento e preparação dos dados
02 → Análise exploratória
03 → Modelos supervisionados
04 → Modelos de clusterização
```

As etapas posteriores utilizam dados, transformações e decisões construídas nos notebooks anteriores.

---

## Limitações

Algumas limitações devem ser consideradas:

- os clusters apresentaram sobreposição, refletida pelos valores de Silhouette Score;
- o grupo de menor criticidade ainda possui clientes detratores;
- a segmentação representa diferenças relativas entre os perfis;
- as regras de ação e desconto possuem caráter analítico e precisam de validação comercial;
- o modelo hierárquico não permite atribuição direta de novos registros;
- a solução ainda não foi transformada em uma aplicação ou API;
- a pasta `src/` ainda está em desenvolvimento.

---

## Próximas etapas

As próximas evoluções previstas incluem:

- estruturar os scripts reutilizáveis na pasta `src/`;
- criar funções para pré-processamento e inferência;
- desenvolver uma API com FastAPI;
- criar um dashboard com Streamlit ou Power BI;
- implementar testes automatizados;
- adicionar monitoramento das métricas;
- criar um modelo auxiliar para atribuir novos clientes aos segmentos;
- containerizar a solução com Docker;
- melhorar a documentação técnica;
- preparar uma estratégia de deploy.

---

## Considerações finais

Este projeto demonstra uma abordagem completa de Ciência de Dados aplicada à satisfação de clientes.

Mais do que treinar modelos, o trabalho conecta preparação de dados, análise exploratória, engenharia de atributos, Machine Learning, clusterização e interpretação de negócio.

A modelagem supervisionada permite estimar o risco individual de insatisfação. A clusterização complementa essa visão ao identificar perfis com diferentes níveis de criticidade operacional.

O resultado é uma solução analítica capaz de apoiar estratégias de retenção, priorização de atendimento, monitoramento preventivo e melhoria da experiência do cliente.

---

## Autor

Desenvolvido por **Renan Trevelim**.

Projeto criado para estudo, portfólio e aplicação prática de técnicas de Ciência de Dados e Machine Learning.
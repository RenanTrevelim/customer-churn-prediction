# Análise de Satisfação e Segmentação de Clientes

Este projeto apresenta uma solução de Ciência de Dados aplicada à análise de satisfação de clientes em um contexto de e-commerce.

O trabalho foi estruturado em quatro notebooks, cobrindo desde o entendimento e preparação dos dados até a construção de modelos supervisionados e não supervisionados.

A proposta combina **análise exploratória, engenharia de atributos, seleção de features, Machine Learning, interpretabilidade com SHAP e clusterização**, transformando dados de perfil, pedidos, entrega e atendimento em informações úteis para retenção, priorização de clientes e melhoria da experiência.

---

## Objetivo do projeto

O projeto busca compreender os fatores associados à insatisfação dos clientes e desenvolver soluções capazes de apoiar duas frentes complementares:

- identificar clientes com maior probabilidade de se tornarem detratores;
- segmentar clientes de acordo com padrões de comportamento e criticidade operacional.

Com isso, os dados históricos podem ser utilizados para apoiar áreas como atendimento, relacionamento, logística e experiência do cliente.

---

# Estrutura dos notebooks

## 01. Entendimento e preparação dos dados

Responsável pela leitura, validação e tratamento inicial da base.

### Principais etapas

- análise da estrutura dos dados;
- verificação dos tipos das variáveis;
- identificação de valores ausentes e duplicados;
- tratamento e padronização das informações;
- preparação da base para as etapas seguintes.

### Arquivo

```text
01_entendimento_e_preparacao_dos_dados.ipynb
```

---

## 02. Análise exploratória dos dados

Responsável pela investigação dos principais padrões presentes na base e pela identificação dos fatores associados à satisfação dos clientes.

### Principais análises

- distribuição das variáveis;
- comportamento do NPS;
- proporção de detratores e promotores;
- relação entre satisfação, entrega e atendimento;
- impacto de atrasos e reclamações;
- análise de correlação com o NPS;
- identificação de padrões relevantes para modelagem;
- apoio à criação de novas variáveis.

A análise exploratória indicou maior associação da insatisfação com fatores relacionados a **atrasos na entrega, quantidade de reclamações, contatos com atendimento e tempo de resolução**.

### Arquivo

```text
02_analise_exploratoria_dos_dados.ipynb
```

---

## 03. Modelos supervisionados

Responsável pela construção de modelos de classificação voltados à identificação de clientes detratores.

Como a base apresenta desbalanceamento entre as classes, a avaliação foi orientada principalmente por métricas como **F1 Macro, Recall, F1-score, ROC-AUC e matriz de confusão**.

### Principais etapas

- definição da variável-alvo;
- engenharia de atributos;
- divisão entre treino e teste;
- pré-processamento dos dados;
- verificação de features com baixa variabilidade;
- treinamento e comparação de diferentes classificadores;
- otimização de hiperparâmetros;
- comparação entre `RandomizedSearchCV` e Optuna;
- validação cruzada;
- avaliação por matriz de confusão e curva ROC;
- análise de importância das features;
- interpretabilidade com SHAP;
- seleção do modelo final;
- aplicação das probabilidades previstas ao contexto de negócio.

---

### Feature Selection

Após o pré-processamento, foi aplicado o `VarianceThreshold` como etapa de verificação de variáveis com baixa variabilidade.

Como nenhuma das 20 features apresentou variância inferior ao limite estabelecido, todas foram mantidas para as etapas seguintes.

Essa etapa foi utilizada principalmente como uma verificação inicial, evitando a remoção prematura de variáveis potencialmente relevantes.

---

### Otimização e modelo final

Após a comparação inicial dos algoritmos, o `XGBoost` avançou para uma etapa adicional de otimização.

Foram comparadas três estratégias:

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

A escolha considerou principalmente o equilíbrio entre as classes, o alto recall para detratores, o ganho em F1 Macro e a melhor capacidade de discriminação apresentada pela ROC-AUC.

---

### Validação cruzada

O modelo final foi submetido a validação cruzada estratificada com 5 folds para verificar sua estabilidade.

Resultados médios:

| Métrica | Média |
|---|---:|
| Accuracy | 0,8375 |
| Precision | 0,8560 |
| Recall | 0,9379 |
| F1-score | 0,8953 |
| ROC-AUC | 0,8774 |

Os baixos desvios-padrão observados indicaram comportamento consistente entre diferentes divisões dos dados.

---

### Feature Importance

Após a seleção do modelo final, foi analisada a importância das variáveis utilizando `feature_importances_` do XGBoost.

Entre as principais features, destacaram-se:

- `numero_reclamacoes`;
- `atraso_entrega_dias`;
- `contatos_atendimento`;
- `tempo_resolucao_dias`;
- `problema_complexo`;
- `valor_pedido`.

Esses resultados reforçaram os padrões observados durante a análise exploratória, principalmente em relação a reclamações, atrasos e dificuldades no atendimento.

---

### Interpretabilidade com SHAP

Além da Feature Importance tradicional, foi utilizado o **SHAP (SHapley Additive exPlanations)** para aprofundar a interpretação do modelo.

Como o modelo final é baseado em árvores, foi utilizado o `TreeExplainer`.

A análise foi dividida em duas perspectivas:

#### Análise global

Foram utilizados:

- `Summary Plot`;
- `Summary Plot` em barras;
- `Dependence Plot`.

Essas visualizações permitiram identificar:

- quais variáveis possuem maior impacto global;
- a magnitude das contribuições;
- a direção do impacto sobre a previsão;
- como diferentes valores de uma feature alteram o comportamento do modelo.

Os resultados reforçaram a importância de variáveis relacionadas a reclamações, atrasos e atendimento.

#### Análise local

Foram utilizados:

- `Waterfall Plot`;
- `Force Plot`.

Essas visualizações permitiram explicar previsões individuais, mostrando quais características de um determinado cliente aumentaram ou reduziram sua probabilidade prevista de ser detrator.

O SHAP complementou a análise tradicional ao permitir compreender não apenas **quais features são importantes**, mas também **como e em qual direção elas influenciam as previsões**.

---

### Aplicação do modelo final

As probabilidades previstas pelo modelo foram utilizadas para criar uma camada de aplicação ao negócio.

A solução permite:

- estimar a probabilidade de um cliente ser detrator;
- classificar clientes por nível de risco;
- calcular um indicador de valor financeiro em risco;
- recomendar ações de retenção;
- priorizar clientes para acompanhamento;
- apoiar decisões de atendimento e relacionamento.

### Arquivo

```text
03_modelos_supervisionados.ipynb
```

---

## 04. Modelos de clusterização

Responsável pela segmentação dos clientes por meio de técnicas de aprendizado não supervisionado.

### Algoritmos avaliados

- K-Means;
- DBSCAN;
- Clusterização Hierárquica.

### Técnicas de redução de dimensionalidade

- PCA;
- Kernel PCA.

### Critérios de avaliação

Os modelos foram analisados considerando:

- Silhouette Score;
- distribuição dos clientes entre os clusters;
- equilíbrio dos grupos;
- interpretação dos perfis;
- utilidade para o negócio.

### Modelo selecionado

A combinação entre **Kernel PCA e Clusterização Hierárquica com método Ward** foi selecionada como solução final.

Foram identificados dois grupos principais:

- clientes de menor criticidade;
- clientes de alta criticidade.

O grupo de alta criticidade apresentou piores indicadores relacionados a atraso, atendimento, reclamações e satisfação.

### Arquivo

```text
04_modelos_clusterizacao.ipynb
```

---

# Principais resultados

## Modelagem supervisionada

O modelo final baseado em **XGBoost + Optuna** apresentou alta capacidade de identificação de clientes detratores e bom equilíbrio geral entre as classes.

Além da previsão, a combinação entre Feature Importance e SHAP permitiu identificar os fatores que mais influenciam o comportamento do modelo.

Os principais fatores estiveram relacionados a:

- quantidade de reclamações;
- atrasos na entrega;
- tempo de resolução;
- contatos com atendimento;
- complexidade dos problemas.

As probabilidades previstas foram transformadas em informações úteis para:

- classificação de risco;
- priorização de atendimentos;
- recomendação de ações de recuperação;
- estimativa de valor financeiro em risco;
- apoio a estratégias de retenção.

---

## Clusterização

A análise não supervisionada identificou dois segmentos com diferentes níveis de criticidade operacional.

O grupo de **alta criticidade** concentrou:

- maior atraso médio;
- maior número de contatos com atendimento;
- maior tempo de resolução;
- maior recorrência de reclamações;
- menor NPS médio;
- maior proporção de detratores.

Já o grupo de **menor criticidade** apresentou indicadores mais favoráveis, embora ainda concentrasse parte dos clientes insatisfeitos.

A segmentação não representa uma separação direta entre clientes satisfeitos e insatisfeitos, mas permite identificar grupos com diferentes perfis de risco e criticidade.

---

# Relação entre os modelos

As abordagens supervisionada e não supervisionada são complementares.

O modelo supervisionado responde:

> Qual é a probabilidade de um cliente ser detrator?

A clusterização responde:

> Quais clientes apresentam padrões semelhantes de comportamento e criticidade?

A combinação dessas abordagens oferece uma visão mais completa da experiência do cliente, unindo **previsão individual, interpretabilidade e segmentação comportamental**.

---

# Aplicação ao negócio

Os resultados podem apoiar diferentes áreas da empresa, como:

- atendimento ao cliente;
- experiência do consumidor;
- logística;
- relacionamento;
- retenção;
- marketing;
- gestão de reclamações.

Entre as possíveis aplicações estão:

- priorização de clientes críticos;
- monitoramento preventivo;
- personalização de ações de recuperação;
- redução da recorrência de problemas;
- direcionamento de campanhas de relacionamento;
- construção de dashboards e sistemas de apoio à decisão.

---

# Organização do projeto

```text
notebooks/
│
├── 01_entendimento_e_preparacao_dos_dados.ipynb
├── 02_analise_exploratoria_dos_dados.ipynb
├── 03_modelos_supervisionados.ipynb
├── 04_modelos_clusterizacao.ipynb
└── README.md
```

---

# Conclusão

Os quatro notebooks formam um fluxo integrado de Ciência de Dados, partindo da preparação e exploração dos dados até a construção de soluções preditivas e de segmentação.

A modelagem supervisionada permitiu estimar o risco individual de insatisfação, enquanto o SHAP adicionou uma camada de transparência às previsões. A clusterização complementou essa análise ao identificar grupos de clientes com diferentes níveis de criticidade operacional.

O resultado é uma solução que integra **análise exploratória, engenharia de atributos, Machine Learning, otimização, validação, interpretabilidade e clusterização**, com foco na geração de informações acionáveis para retenção e melhoria da experiência do cliente.
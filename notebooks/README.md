# Análise de Satisfação e Segmentação de Clientes

Este projeto apresenta uma solução completa de Ciência de Dados aplicada à análise de satisfação de clientes em um contexto de e-commerce.

O trabalho foi estruturado em quatro notebooks, cobrindo desde o entendimento e a preparação dos dados até a construção de modelos supervisionados e não supervisionados.

A proposta combina análise exploratória, engenharia de atributos, Machine Learning e clusterização para transformar dados de perfil, pedido, entrega e atendimento em informações úteis para retenção, priorização de clientes e melhoria da experiência.

---

## Objetivo do projeto

O principal objetivo é compreender os fatores associados à insatisfação dos clientes e desenvolver soluções capazes de apoiar duas frentes complementares:

- identificar clientes com maior probabilidade de se tornarem detratores;
- segmentar clientes de acordo com padrões de comportamento e criticidade operacional.

Com isso, o projeto busca transformar dados históricos em informações acionáveis para as áreas de atendimento, relacionamento, logística e experiência do cliente.

---

## Estrutura dos notebooks

### 01. Entendimento e preparação dos dados

Responsável pela leitura, validação e tratamento inicial da base.

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

## 02. Análise exploratória dos dados

Responsável pela investigação dos principais padrões presentes na base.

### Principais análises

- distribuição das variáveis;
- comportamento do NPS;
- relação entre satisfação, entrega e atendimento;
- impacto de atrasos e reclamações;
- identificação de padrões relevantes para a modelagem;
- apoio à criação de novas variáveis.

### Arquivo

```text
02_analise_exploratoria_dos_dados.ipynb
```

## 03. Modelos supervisionados

Responsável pela construção de modelos de classificação voltados à identificação de clientes detratores.

Foram comparados diferentes algoritmos de Machine Learning, considerando métricas adequadas ao desbalanceamento das classes.

### Principais etapas

- definição da variável-alvo;
- engenharia de atributos;
- divisão entre treino e teste;
- pré-processamento dos dados;
- treinamento de diferentes classificadores;
- otimização de hiperparâmetros;
- validação cruzada;
- seleção do modelo final;
- aplicação das probabilidades previstas ao contexto de negócio.

### Aplicação do modelo final

O modelo final foi utilizado para:

- estimar a probabilidade de um cliente ser detrator;
- classificar clientes por nível de risco;
- calcular um indicador de valor em risco;
- recomendar ações de retenção;
- priorizar clientes para acompanhamento.

### Arquivo

```text
03_modelos_supervisionados.ipynb
```

## 04. Modelos de clusterização

Responsável pela segmentação de clientes por meio de técnicas de aprendizado não supervisionado.

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

A combinação entre Kernel PCA e Clusterização Hierárquica com método Ward foi selecionada como solução final por produzir dois grupos mais coerentes e interpretáveis:

- clientes de menor criticidade;
- clientes de alta criticidade.

### Arquivo

```text
04_modelos_clusterizacao.ipynb
```

## Principais resultados

### Modelagem supervisionada

A análise supervisionada permitiu construir um modelo capaz de estimar o risco de um cliente se tornar detrator.

Além da previsão, o resultado foi convertido em uma camada de aplicação ao negócio, permitindo:

- classificar clientes por nível de risco;
- priorizar atendimentos;
- recomendar ações de recuperação;
- estimar o valor financeiro em risco;
- apoiar estratégias de retenção.

### Clusterização

A análise não supervisionada permitiu identificar dois segmentos com diferentes níveis de criticidade operacional.

O grupo de **alta criticidade** concentrou:

- maior atraso médio;
- maior número de contatos com o atendimento;
- maior tempo de resolução;
- maior recorrência de reclamações;
- menor NPS médio;
- maior proporção de detratores.

Já o grupo de **menor criticidade** apresentou indicadores relativamente mais favoráveis, embora ainda concentrasse clientes insatisfeitos.

A segmentação não produziu uma separação perfeita entre clientes satisfeitos e insatisfeitos, mas conseguiu identificar um grupo significativamente mais vulnerável e prioritário para acompanhamento.

---

## Relação entre os modelos

As duas abordagens desenvolvidas no projeto são complementares.

O modelo supervisionado responde:

> Qual é a probabilidade de um cliente se tornar detrator?

A clusterização responde:

> Quais perfis de clientes apresentam padrões semelhantes de comportamento e criticidade?

Essa combinação oferece uma visão mais completa da experiência do cliente, unindo previsão individual e segmentação comportamental.

---

## Aplicação ao negócio

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
- construção de dashboards operacionais.

---

## Organização do projeto

```text
notebooks/
│
├── 01_entendimento_e_preparacao_dos_dados.ipynb
├── 02_analise_exploratoria_dos_dados.ipynb
├── 03_modelos_supervisionados.ipynb
├── 04_modelos_clusterizacao.ipynb
└── README.md
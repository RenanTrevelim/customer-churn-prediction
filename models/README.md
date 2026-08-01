# Artefatos de modelagem

Esta pasta reúne os modelos e objetos de transformação gerados durante as etapas de Machine Learning e clusterização.

## Estrutura

### `supervised/`

Contém o modelo supervisionado utilizado para estimar a probabilidade de um cliente ser detrator.

Arquivo:

```text
modelo_final.pkl
```

### `clustering/`

Contém os artefatos utilizados na segmentação de clientes.

Arquivos:

```text
kernel_pca_clusterizacao.pkl
modelo_hierarquico_ward.pkl
```

O Kernel PCA é responsável pela transformação não linear dos dados, enquanto o modelo hierárquico com método Ward foi utilizado para formar os segmentos finais.

### `preprocessing/`

Contém os objetos responsáveis pelas transformações aplicadas antes dos modelos.

Arquivos:

```text
pre_processamento.pkl
pre_processamento_clusterizacao.pkl
```

Esses artefatos garantem que novos dados sejam submetidos às mesmas etapas de padronização, codificação e preparação utilizadas durante o desenvolvimento.

## Observação

O modelo hierárquico foi salvo como artefato analítico. Como o `AgglomerativeClustering` não possui o método `predict`, ele não permite atribuir diretamente novos clientes aos clusters existentes.

> Os artefatos devem ser carregados na mesma ordem em que foram utilizados durante o treinamento: pré-processamento, redução de dimensionalidade e modelo.
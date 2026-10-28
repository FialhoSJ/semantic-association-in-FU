# Arquitetura e protocolo científico

## 1. Decisões antes da implementação semântica

1. **Unidade de esquecimento.** O pedido deve apontar um par de conceitos e uma população de referência. “Esquecer associação” será operacionalizado como reduzir a resposta a exemplos da combinação alvo, mantendo a resposta a cada conceito isolado e a pares vizinhos. A definição precisa de sucesso será fixada antes de treinar SAFU.
2. **Dados identificáveis.** CIFAR-10 serve para validar FL, mas não mede a hipótese central. O benchmark futuro precisa de exemplos com anotações multiatributo, pares alvo e controles em que cada atributo apareça separadamente. Separar combinações entre treino e teste ajuda a distinguir memorização de generalização.
3. **Descoberta não é esquecimento.** Apriori encontra coocorrências em transações escolhidas; protótipos agregados podem apagar a informação de coocorrência. A construção de transações será auditada em nível de exemplo ou contexto, antes de agregá-las. Lift alto sozinho não demonstra associação no modelo.
4. **Unlearning mensurável.** Mudança de embedding ou queda de score não equivale a remoção da influência do dado. O protocolo inclui retraining em dados retidos, utilidade dos conceitos, amostras fora da distribuição e testes de privacidade quando pertinentes. Resultados negativos são aceitos.
5. **Justiça de custo.** O tempo, FLOPs e bytes de SRE, SPG, SAI, CSL e STL devem entrar no custo de SAFU. Comunicação de protótipos pode custar mais que a economia de adapters. O orçamento total e o hardware serão os mesmos nas comparações.
6. **Privacidade.** Dados brutos permanecem nos clientes. Isso não torna protótipos ou gradientes automaticamente seguros. Uma futura versão terá análise de vazamento, minimização de mensagens e mecanismo de proteção explícito quando exigido.

## 2. Fluxo e limites de fase

```text
Client Data → Local Model → Semantic Representation → Prototype
→ Semantic Transaction → Apriori → Association Graph → Forget Request
→ Critical Layers → Sparse Adapter → Semantic Transformation
→ Global Aggregation → Evaluation
```

A Fase 1 implementa o caminho `Dataset → Dirichlet partitions → Client local SGD → FedAvg → Test metrics`. O modelo expõe `forward_features` para a fase 2. O servidor usa atualizações do cliente independentes, com `state_dict` completo e peso por número de exemplos. SAI, CSL, SSA, STL, FedProx e qualquer métrica de forgetting não têm implementação ainda.

## 3. Contratos atuais no código

| Componente | Entrada | Saída | Regra |
| --- | --- | --- | --- |
| `load_config` | caminho YAML | `ExperimentConfig` imutável | rejeita chaves desconhecidas e parâmetros inválidos |
| `load_data` | `DatasetConfig`, seed | `DataBundle(train, test, train_labels, num_classes)` | test não participa do treino |
| `dirichlet_partition` | labels `ndarray[N]`, K, α, seed, mínimo | `list[ndarray[int64]]` | cobertura exata, sem índices duplicados, seed reproduzível |
| `build_model` | `ModelConfig`, classes | `BaseModel` | `forward_features: Tensor[B,3,32,32] → Tensor[B,D]`; `forward → Tensor[B,C]` |
| `train_client` | id, modelo global, dataset, índices, config, device, seed | `ClientUpdate(id, num_samples, state, train_loss, duration)` | copia modelo; não muda o global |
| `fedavg` | lista de `ClientUpdate` | `State = dict[str, Tensor]` | média de tensores float ponderada por exemplos; buffers inteiros copiados do maior cliente |
| `FederatedServer.run_round` | número da rodada | linha de métricas | seleção sem reposição; avalia o global após agregação |
| `evaluate` | modelo, dataset de teste, batch, device | `dict[test_loss, test_accuracy, test_precision_macro, test_recall_macro, test_f1_macro]` | perda e accuracy por exemplo; métricas macro por classe |

No `metrics.csv`, `communication_bytes` soma os bytes de todos os tensores do modelo enviados a cada cliente e recebidos de volta. `local_train_seconds` soma tempo de treino dos clientes em execução sequencial. `round_seconds` mede a rodada inteira; as duas métricas não devem ser somadas.

## 4. Contratos planejados, ainda sem implementação

Estes tipos definem as fronteiras futuras; não se deve registrar resultados das fases 2–10 até existir um benchmark apropriado.

| Tipo / interface | Campos de entrada | Saída prevista |
| --- | --- | --- |
| `SemanticExample` | `sample_id: str`, `client_id: int`, `concept_ids: frozenset[str]`, `context_id: str`, `split: str` | anotação local de conceitos por contexto; IDs pseudônimos |
| `RepresentationExtractor.extract` | `BaseModel`, `Dataset`, `sample_ids`, `device` | `EmbeddingBatch(sample_ids, client_id, vectors: Tensor[N,D], concept_ids)` local |
| `PrototypeGenerator.fit` | `EmbeddingBatch`, estratégia e seed | `PrototypeSet(prototype_ids, centroids: Tensor[M,D], counts, provenance)` |
| `TransactionBuilder.build` | `SemanticExample`, `PrototypeSet`, threshold | `SemanticTransaction(context_id, item_ids: frozenset[str], weight)`; manter vínculo de coocorrência |
| `AssociationMiner.mine` | transações, `min_support`, `min_confidence`, `min_lift` | `AssociationGraph(nodes, edges)`; cada aresta tem contagens e métricas, além de intervalo de incerteza |
| `UnlearningRequest` | `target_pair: tuple[str,str]`, `preserve_concepts`, `preserve_pairs`, `scope`, `request_id` | pedido validado; par não direcionado, salvo protocolo explícito |
| `LayerSelector.select` | modelo, dados alvo/controle, pedido, métrica, K | `LayerSelection(names, scores, cost)`; scores calculados com dados de treino apenas |
| `AdapterFactory.attach` | modelo congelado, camadas, orçamento de parâmetros, seed | modelo com adapters treináveis e `AdapterManifest` |
| `TransformationLearner.fit_local` | modelo, pedido, dados alvo/retidos, referência global, pesos das perdas | `AdapterUpdate`, perdas por termo, exemplos usados, tempo |
| `AdapterAggregator.aggregate` | updates, manifestos compatíveis, pesos | estado global do adapter; modelo base intacto |
| `UnlearningEvaluator.evaluate` | modelo original, modelo retrained, modelo alterado, benchmark fixo | scores de alvo, conceitos, pares vizinhos, drift, utilidade, custo e intervalos |

Os contratos distinguem **metadados internos ao cliente** de mensagens permitidas ao servidor. Um `EmbeddingBatch` individual não deve ser enviado automaticamente. O protocolo de comunicação futura deve permitir apenas um `SemanticMessage` versionado com `sender_id`, `round`, `payload_type`, `payload`, `byte_count`, política de privacidade e proveniência. O custo medido deve incluir todos os estágios e os metadados.

## 5. Definições candidatas para estudos futuros

Para transações ponderadas `T`, `support(A,B) = Σ_t w_t·1[A,B∈t] / Σ_t w_t`; `confidence(A→B) = support(A,B)/support(A)`; `lift(A,B) = support(A,B)/(support(A)support(B))`. Denominadores nulos exigem regra explícita. Essas são métricas de coocorrência, não de causalidade ou de “esquecimento”. Thresholds são hiperparâmetros de descoberta e devem ser definidos usando validação, nunca teste.

`L_forget` deverá atacar o comportamento da **combinação** alvo, enquanto `L_preserve` compara representações e desempenho de conceitos isolados e pares vizinhos. `L_semantic` controla a direção da transformação; `L_local` e `L_global` controlam consistência no cliente e entre clientes; `L_sparse` penaliza uso de parâmetros. Não há fórmula final validada. Combinar perdas com pesos exige normalização e ablação. O SUS = eficácia × preservação é uma métrica exploratória; ambos os termos, faixas e direção precisam de pré-registro.

## 6. Plano de comparação

O protocolo final comparará retraining sem os exemplos da associação alvo, FUCRT-like, atualização semântica de todo o modelo e SAFU. O procedimento de retraining precisa esclarecer se a remoção é de exemplos, rótulos ou somente associação. Uma comparação de “association-level” com retraining por remoção de todos os exemplos alvo pode ter estimandos diferentes; por isso ambos os alvos operacionais serão reportados separadamente.

Para cada método: mesmos clientes, partições, seeds, modelo base, budget de otimização, avaliações cegas no mesmo conjunto, repetições e intervalos. Registrar accuracy/F1 global, score e drift da combinação alvo, conceito A, conceito B, pares vizinhos, parâmetros treináveis, tempo, memória, FLOPs aproximados e bytes. Rodar α = 0.1, 0.5 e 1.0. As ablações retirarão SAI, CSL, adapter, `L_preserve` e `L_global`, variando K, sparsity e thresholds. “Sem SAI” precisará de um mecanismo alternativo de seleção de associação previamente especificado.

## 7. Árvore do projeto

```text
semantic-federated-unlearning/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/experiments/{phase1_smoke,cifar10_fedavg}.yaml
├── docs/architecture.md
├── scripts/train_fl.py
├── src/safu/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── federated.py
│   ├── reproducibility.py
│   └── cli.py
├── tests/{test_partition,test_federated,test_smoke}.py
└── results/.gitkeep
```

Módulos futuros serão adicionados conforme a fase: `semantic/{representation,prototypes,transactions,association}`, `unlearning/{request,csl,adapters,transformation,losses}`, `communication`, `evaluation` e `baselines`. Criar arquivos vazios para esses módulos sugeriria uma capacidade inexistente; os contratos acima são o ponto de integração.

## 8. MVP e critérios de saída da Fase 1

O MVP de infraestrutura é: uma configuração YAML válida treina FedAvg em 2 rodadas sintéticas; o teste verifica artefatos e payload; o particionamento não perde nem duplica exemplos em α = 0.1, 0.5 e 1.0; FedAvg pondera por amostras; CIFAR-10 executa com o mesmo código. Dados sintéticos são só um teste de engenharia. Nenhum resultado gerado nesta fase sustenta a hipótese de unlearning.

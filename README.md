# SAFU — Semantic Association Federated Unlearning

Repositório experimental para investigar **se** é possível enfraquecer uma associação semântica específica em um modelo federado, preservando os conceitos envolvidos e suas outras relações. A hipótese de maior preservação e menor custo **ainda não foi testada**.

## Fluxo planejado

```text
Client Data → Local Model → Semantic Representation → Prototype
→ Semantic Transaction → Apriori → Association Graph → Forget Request
→ Critical Layers → Sparse Adapter → Semantic Transformation
→ Global Aggregation → Evaluation
```

Somente `Client Data → Local Model → Global Aggregation → Evaluation` está implementado. Os contratos para as demais etapas estão em [docs/architecture.md](docs/architecture.md).

## Estado das fases

| Fase | Estado | Entrega |
| --- | --- | --- |
| 1 — FL baseline | Implementada e validada no teste sintético; CIFAR-10 real pendente | CIFAR-10, teste sintético, Dirichlet Non-IID, SimpleCNN, FedAvg, avaliação e resultados |
| 2 — SRE/SPG | Planejada | Embeddings e protótipos |
| 3 — Apriori/SAI | Planejada | Transações, métricas e grafo |
| 4 — Forget request | Planejada | Associação alvo e conjunto de preservação |
| 5 — CSL | Planejada | Sensibilidade por associação |
| 6 — SSA | Planejada | Adapters esparsos |
| 7 — STL | Planejada | Transformação específica |
| 8 — Losses | Planejada | Perdas modulares |
| 9 — FU federado | Planejada | Agregação de adapters |
| 10 — Avaliação | Planejada | Baselines, ablações e custos comparáveis |

## Execução

Requer Python 3.10 ou superior e uma versão de PyTorch disponível para seu sistema. Ambiente virtual recomendado.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest
safu-train --config configs/experiments/phase1_smoke.yaml
safu-train --config configs/experiments/cifar10_fedavg.yaml
```

O teste sintético verifica somente a integração do pipeline. Para não baixar dados, altere `dataset.download` para `false` após colocar o CIFAR-10 em `data/`. O exemplo CIFAR-10 de 100 rodadas pode ser caro; reduza as rodadas durante desenvolvimento.

Validação local: 6 testes passaram com Python 3.14.4 e PyTorch 2.15.0 em CPU. Duas execuções da configuração sintética com a mesma seed produziram métricas idênticas, desconsiderando duração. CIFAR-10 real ainda não foi executado.

Cada execução gera `results/<name>/<timestamp>_seed<seed>/` com `config.yaml`, `partitions.json`, `metrics.csv`, `summary.json`, `logs.txt` e `checkpoints/final.pt`. `summary.json` registra commit Git quando existe um commit. A contagem de comunicação inclui envio e retorno de um `state_dict` completo para cada cliente selecionado; é uma estimativa de payload sem protocolo, cabeçalhos, compressão ou retransmissão.

## Limites científicos atuais

- CIFAR-10 contém rótulos de classe simples, não anotações de associações entre conceitos. Não é válido inferir `german_shepherd ↔ police` de suas classes. A Fase 2 requer um benchmark composicional com pares positivos e negativos, anotações independentes e divisão de treino/teste por combinação.
- Coocorrência frequente pode refletir frequência marginal ou viés do dataset. `support`, `confidence` e `lift` serão candidatos de descoberta, não prova causal de que o modelo aprendeu a associação.
- Enfraquecer um score de associação não garante remoção da influência dos dados. A avaliação deve comparar com retraining e ataques apropriados, sem prometer esquecimento exato.
- Protótipos e estatísticas podem vazar informação; a futura comunicação semântica precisará de análise de ameaça e, se aplicável, proteção de privacidade.
- O custo de identificar camadas e minerar associações deve entrar na comparação total. Todas as baselines usarão o mesmo split, seed, orçamento de treino e hardware.

## TODO

- [ ] Validar a Fase 1 em CIFAR-10 real e, se disponível, em GPU.
- [ ] Criar benchmark de associações anotadas com controles de frequência e pares negativos.
- [ ] Implementar FedProx como algoritmo FL alternativo antes dos estudos comparativos.
- [ ] Implementar fases 2–10 em ordem, com testes de contratos e critérios de sucesso/fracasso.
- [ ] Adicionar ResNet-18, baselines de unlearning e grid de α = 0.1, 0.5, 1.0.
- [ ] Medir custo real de CPU/GPU, memória e FLOPs; manter payload estimado separado de bytes transmitidos na rede.

Referências metodológicas: [FUCRT (ICCV 2025)](https://arxiv.org/abs/2410.06848) e [FUSED (CVPR 2025)](https://arxiv.org/abs/2502.20709). O projeto é uma proposta própria e não uma reimplementação desses métodos.

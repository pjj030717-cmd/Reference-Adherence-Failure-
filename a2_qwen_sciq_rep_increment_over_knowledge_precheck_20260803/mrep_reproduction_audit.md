# mrep_reproduction_audit.md

## M_rep（true-prefix representation risk score）

- 来源：D4-Q1 冻结 `scripts/_frozen/probe.npz`（train-only 重建，非新提取）。
- 规格：layer 18 × R_end；StandardScaler(train)；LogisticRegression(C=0.01, max_iter=2000, balanced)。
- dev 只读复现（D2-R1 dev hidden states + D2-R1 dev SS 标签）：

| 指标 | 复现 | 记录 | 容差 |
|---|---|---|---|
| AUROC | 0.913887 | 0.9138872915468661 | 0.0001 |
| AUPRC | 0.963219 | 0.9632189056523878 | 0.0001 |

- 结论：冻结 M_rep 在 dev 上精确复现，继承成立。

## B_surface（上下文参照，D2-R1 冻结）

| 指标 | 复现 | 记录 | 容差 |
|---|---|---|---|
| AUROC | 0.620759 | 0.6207590569292697 | 0.0001 |
| AUPRC | 0.818192 | 0.8181915706506249 | 0.0001 |

- B_surface 仅为上下文参照，不参与最终标签判定。

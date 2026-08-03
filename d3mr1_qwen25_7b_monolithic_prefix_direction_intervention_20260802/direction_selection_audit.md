# Direction selection audit

Selected: **V_logit@C=0.01**

| method | AUROC | AUPRC | balacc | Spearman(D2-R1,desc) |
|---|---|---|---|---|
| V_lda | 0.899788 | 0.969153 | 0.829715 | None |
| V_logit@C=0.001 | 0.921658 | 0.976829 | 0.834626 | None |
| V_logit@C=0.01 | 0.922287 | 0.977663 | 0.853713 | None |
| V_logit@C=0.1 | 0.920707 | 0.977186 | 0.853713 | None |
| V_mean | 0.890577 | 0.964658 | 0.803787 | None |

Reason: max OOF AUPRC 0.977663; if within 0.005 of runner-up then AUROC tiebreak.
sigma_z_train = 2.803511

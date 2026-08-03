# inheritance_audit.md

## E0 / E0-R1 → E0-R2 继承审计（只读）

| 项 | 状态 |
|---|---|
| E0 label | ✓ |
| E0-R1 label | ✓ |
| E0 failure 16<20 | ✓ |
| E0-R1 failure dev color=4 | ✓ |
| relation universe == 16 | ✓ |
| relation names match official | ✓ |
| total retained 14077 | ✓ |
| split counts 8446/2815/2816 | ✓ |
| train covers 16/16 | ✓ |
| train max share <=0.25 | ✓ |
| dev covers 16/16 | ✓ |
| dev max share <=0.25 | ✓ |
| final_reserve covers 16/16 | ✓ |
| final_reserve max share <=0.25 | ✓ |
| E0-R1 train min matches | ✓ |
| E0-R1 dev min matches | ✓ |
| E0-R1 final_reserve min matches | ✓ |
| no cross-split overlap | ✓ |
| donor audit rows | ✓ |
| donors same split | ✓ |
| donors answer differs (r_o!=r_s) | ✓ |
| template contract rows | ✓ |
| c_o != c_s all templates | ✓ |
| four-cell rows | ✓ |
| four-cell shared contract | ✓ |
| TT0 canonical+sha | ✓ |
| TT1 canonical+sha | ✓ |
| TT2 canonical+sha | ✓ |
| E0 no judge | ✓ |
| E0 no inference | ✓ |
| E0-R1 no judge | ✓ |
| E0-R1 no inference | ✓ |

## 每 split relation 覆盖统计（额外记录）

| split | relation 数 | 最小类样本数 | 最小类 | 最大类占比 | 最大类 |
|---|---|---|---|---|---|
| train | 16 | 17 | color | 0.142908 | screenwriter |
| dev | 16 | 4 | color | 0.148845 | director |
| final_reserve | 16 | 13 | color | 0.142401 | screenwriter |

## 结论

E0 与 E0-R1 的全部有效工件可唯一核验。`color` 类未删除、未重采样、未重新切分。

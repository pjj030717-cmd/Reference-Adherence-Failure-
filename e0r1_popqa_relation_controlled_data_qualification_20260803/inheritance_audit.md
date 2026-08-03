# inheritance_audit.md

## E0 → E0-R1 继承审计

| 项 | 状态 |
|---|---|
| E0 final_label | ✓ |
| E0 failure is relation count | ✓ |
| E0 revision | ✓ |
| E0 test.tsv sha256 | ✓ |
| E0 README sha256 | ✓ |
| E0 schema fields | ✓ |
| E0 14,267 rows | ✓ |
| R1-R6 in E0 script | ✓ |
| R7/R8 in E0 script | ✓ |
| split seed 20260816 | ✓ |
| 60/20/20 | ✓ |
| per-group RNG | ✓ |
| donor same-split same-relation | ✓ |
| candidates sorted by sgid | ✓ |
| answer differs | ✓ |
| E0 funnel R6=189 | ✓ |
| E0 funnel R4=1 | ✓ |
| TT0 canonical+sha | ✓ |
| TT1 canonical+sha | ✓ |
| TT2 canonical+sha | ✓ |
| E0 spec templates | ✓ |
| E0 judge_loaded False | ✓ |
| E0 inference False | ✓ |
| E0 blind packet absent | ✓ |

## 结论

E0-R1 仅修正 relation 覆盖门槛（见 `protocol_amendment_e0_to_e0r1.md`）；数据源、过滤、split、donor、
四格、模板全部逐字继承 E0。

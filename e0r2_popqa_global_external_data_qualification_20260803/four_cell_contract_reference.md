# four_cell_contract_reference.md

## 四格合同（E0/E0-R1 已核验，E0-R2 引用）

对每个保留 group（q = question，r_o = source obj，r_s = donor obj，c_o/c_s = 模板渲染）：

| cell | Reference | Candidate | 协议下正确 verdict |
|---|---|---|---|
| OO | r_o | c_o | Accept |
| OS | r_o | c_s | Reject |
| SO | r_s | c_o | Reject |
| SS | r_s | c_s | Accept |

机械合同（已核验）：
- r_o != r_s（规范化后）
- c_o != c_s（每个模板）
- 四格共享同一 q、r_o、r_s、c_o、c_s
- donor 与 source 同一 split、同一 prop、不同 source_group_id

逐行审计见 E0-R1 `four_cell_contract_audit.csv` 与 `candidate_template_contract_audit.csv`。

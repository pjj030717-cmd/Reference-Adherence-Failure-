# data_contract_reference.md

## 参考来源

本文件引用 E0 / E0-R1 已唯一恢复并核验的数据契约，不重新下载或重跑构造管道。

- dataset：`akariasai/PopQA`
- revision：`098765c79ea10a2cb19c828324e33281b8336ec0`
- test.tsv SHA256：`9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b`
- README.md SHA256：`bb04b56bc87a3b2865cc2e2a1649ba6c766a7a44dcba5a53170fbfc72c0da9f0`
- 记录总数：14,267
- schema：question / obj（canonical answer）/ prop（relation）/ id（official id）
- source_group_id = SHA256(NFKC(q) ∥ '\x00' ∥ NFKC(obj) ∥ '\x00' ∥ NFKC(prop) ∥ '\x00' ∥ NFKC(str(id)))

详细契约见 E0 `source_data_contract.md` 与 E0-R1 `source_access_audit.md`。

# model_access_audit.md

| 文件 | SHA256 |
|---|---|
| config.json | 7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c |
| tokenizer.json | c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 |
| tokenizer_config.json | 5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583 |
| vocab.json | ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 |
| merges.txt | 599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3 |
| model.safetensors.index.json | 624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028 |

## 模型访问范围
- 仅加载 `Qwen/Qwen2.5-7B-Instruct`（revision a09a3545…）
- 读取评分：train 587 groups + dev 195 groups（T0 四格）
- 禁止读取/评分/缓存/提取：final-reserve 197 groups（未触碰）

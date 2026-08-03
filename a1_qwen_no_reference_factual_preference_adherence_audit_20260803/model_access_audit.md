# model_access_audit.md

## 模型

- Qwen/Qwen2.5-7B-Instruct，本地 `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`
- BF16 / eval / inference_mode / batch_size=1
- 与 D1/E1 完全一致；未加载任何额外 Judge。

## 边界

- 未读取 hidden state；未训练 Probe；无 hook/intervention。

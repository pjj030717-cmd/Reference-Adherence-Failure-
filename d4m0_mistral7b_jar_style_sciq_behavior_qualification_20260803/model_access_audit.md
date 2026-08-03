# model_access_audit.md

## 模型
- 路径：`/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3`
- 架构：MistralForCausalLM（Mistral-7B-Instruct-v0.3）
- revision（本地）：`not-available-locally`
- 加载：BF16、`model.eval()`、`torch.inference_mode()`、`batch_size=1`
- 加载耗时：5.1s

## 文件哈希（SHA256，加载前计算）
| 文件 | SHA256 |
|---|---|
| `config.json` | `affafc6478ec0fd07a32f0ca57aa2fc57743f4d17d6730f86a96ac24d1507f99` |
| `tokenizer.json` | `e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49` |
| `tokenizer_config.json` | `0533dec9cfe319163801b6618d0f3ec9cfa126b6288e3df5deca6e32acb09cd2` |
| `model.safetensors.index.json` | `e489ba553b87cde188d921b1a8283c2e0b9d33d635b88147d96ff0fcd6250016` |

## 显存
- 加载前峰值：0.00 GB
- 评分后峰值：14.55 GB

## 访问结论
- 模型在本地、BF16 正常加载；无下载、无替换。

# tokenization_audit.md

## 加载

- 仅加载 D1 固定 revision 的 Qwen tokenizer（`/root/autodl-tmp/models/Qwen2.5-7B-Instruct`）。
- 未加载 `AutoModelForCausalLM`，无任何前向、logits、hidden-state 或生成。

## 模板 tokenization（probe answer 渲染）

对每个模板与 6 个 probe answer，记录渲染后的完整 token ids 与 token 数（见 `template_rendering_audit.csv`）。

## 关键检查

- `" A"`/`" B"` continuation token ids 不在本阶段计算（不加载模型）；模板渲染本身不涉及 continuation。
- 模板中的 `<answer>` 占位符在 Qwen tokenizer 下为多 token；渲染后整体 token 数随答案变化。

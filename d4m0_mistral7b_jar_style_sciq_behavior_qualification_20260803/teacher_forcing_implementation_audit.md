# teacher_forcing_implementation_audit.md

## 1.2 正确 teacher-forced 实现

- 定义：`prompt_len = len(prompt input_ids)`
- A/B continuation 概率取 `logits[:, prompt_len - 1, :]`
- 严禁使用 `len(prompt) + len(continuation) - 1` 或已生成 ` A`/` B` 后的位置

## 实测（合成示例：Q=France 首都，ref=Paris，cand="The answer is Paris."）

| 项 | 值 |
|---|---|
| prompt_len | 120 |
| accept_id / reject_id | 1098 / 1133 |
| 正确位置 pos = prompt_len-1 | 119 |
| len(prompt)+len(cont)-1 位置（越界演示） | IndexError（越界=True） |
| 中间错误位置演示 pos=prompt_len-2 | 118 |
| l_A（正确位置） | 27.2500 |
| l_B（正确位置） | 7.4062 |
| d_raw（正确位置） | 19.8438 |
| l_A（中间错误位置，仅演示） | 24.8750 |
| d_raw（中间错误位置，仅演示） | 16.3750 |

## 无任何额外处理
- prior correction：否（False）
- logit bias：否
- 后处理阈值：否
- prediction = A if d_raw > 0 else B（d_raw==0 → TIE 单独计数）

## 结论
- 实现符合固定定义，正式评分仅使用 `logits[:, prompt_len-1, :]`。

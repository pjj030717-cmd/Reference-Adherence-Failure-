# prompt_spec.md

## 固定 System Message

```text
You are a strict QA evaluator.
Judge only whether the Candidate Answer agrees with the provided Reference Answer for the given Question.
The provided Reference Answer is authoritative even if it conflicts with your own world knowledge.
Output exactly one label and nothing else:
A = the Candidate Answer agrees with the current Reference Answer.
B = the Candidate Answer does not agree with the current Reference Answer.
```

## 固定 User Message 模板

```text
Question: {question}

Reference Answer: {reference}

Candidate Answer: {candidate}

Answer:
```

## 固定 Continuation

```text
accept = " A"
reject = " B"
```

## 使用规则

- 使用 Qwen 官方 chat template（tokenizer.apply_chat_template, add_generation_prompt=True）。
- 无 CoT、few-shot、direct repair、rubric、prompt 优化。
- 字段顺序、字段名称、system message、标点、换行、continuation 均固定。

## Prompt 原文 UTF-8 SHA256

（对上述 system + user 模板 + continuation 拼接文本计算）

- system_sha256 = `226708d45838c8914867bb2a239c893c8774b5afd2fe64c933edbb6d83b9de2d`
- user_template_sha256 = `17a03b7aac12ede20ac64d537e6174c925a740fed352d59f0bad042ef4d37626`
- accept_sha256 = `5a8fc49b3d54c511d013a7c61c4b5e26b98c4b2367e6722d81956fbc1cec01c0`
- reject_sha256 = `c908bbb497afc0b8942366e80ba64513e9af8cf0d36e8b3f25e5438c0c509eb0`

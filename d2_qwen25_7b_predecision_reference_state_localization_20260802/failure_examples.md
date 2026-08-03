# failure_examples.md

## SS cell（reference=r_s, candidate=c_s, expected=A）中的 Judge 错拒示例（dev）

### 错拒（predicted=B，y_SS_error=1）示例

| group_id(前12) | question | reference (r_s) | candidate (c_s) | d_raw |
|---|---|---|---|---|
| 5af325bc6e29 | The seven bones of the ankle are called the what? | oxygen | The answer is oxygen. | -17.434 |
| f37d2a4996b7 | Typically, what feature of an angiosperm has four main parts known as … | raise sea levels | The answer is raise sea levels. | -18.844 |
| 10d1ca49154e | Exotic species, also known as invasive or non-native species often cau… | nervous system | The answer is nervous system. | -15.438 |
| 510f55edebf3 | What's the name for an organic compound in which halogen atoms are sub… | reactions | The answer is reactions. | -14.484 |
| 55bfeb38bb86 | What do we call the phase of nuclear division, in which one nucleus di… | raise sea levels | The answer is raise sea levels. | -19.266 |
| 7d17895daf33 | What part of the plant absorbs water? | metamorphosis | The answer is metamorphosis. | -11.094 |

### 正确（predicted=A，y_SS_error=0）示例

| group_id(前12) | question | reference (r_s) | candidate (c_s) | d_raw |
|---|---|---|---|---|
| d7bf00bba92d | What is the term for a partial degradation of glucose without the use … | myoglobin | The answer is myoglobin. | 10.984 |
| ad6dc70331b4 | Polymers can disassemble by the reverse process called what? | light | The answer is light. | 11.594 |

## R_end 定位样例（token_span_mapping_audit.md 中有 60 组逐条明细）

R_end = Reference Answer 正文最后一个非空白 token；C_end = Candidate Answer 正文最后一个非空白 token；D_pos = prompt_len-1。
定位基于 chat-rendered prompt 的 offset mapping，不依赖字符串猜测或固定下标。

# failure_examples.md

## 说明

D1-L 是 Candidate 表达稳健性资格门；失败示例为各模板下 SS 格（reference=r_s、candidate=render(r_s)）被错误 Reject 的示例（group 前缀脱敏，至多 10 条）。

- **T4 / 05677026**  d_raw=-18.47  predicted=B
  - Q: What is the creation of a new species called?
  - ref: gills | cand: gills is the requested answer. This response gives the answe
- **T4 / 06efe48e**  d_raw=-18.53  predicted=B
  - Q: What kind of muscle cells have a single nucleus and are spindle-shaped?
  - ref: fiber optics | cand: fiber optics is the requested answer. This response gives th
- **T4 / 0a10ac31**  d_raw=-9.62  predicted=B
  - Q: The radioactive gas radon and uv radiation are culprits in different types of wh
  - ref: alteration | cand: alteration is the requested answer. This response gives the 
- **T4 / 0e5e53a1**  d_raw=-16.55  predicted=B
  - Q: Purple loosestrife is a european wildflower that was introduced to which contine
  - ref: intercellular | cand: intercellular is the requested answer. This response gives t
- **T4 / 0e6720dd**  d_raw=-17.07  predicted=B
  - Q: Which element has the highest electronegativity value?
  - ref: chemosynthesis | cand: chemosynthesis is the requested answer. This response gives 
- **T4 / 0f7d3b6b**  d_raw=-2.50  predicted=B
  - Q: What is the simplest life cycle?
  - ref: suspension feeders | cand: suspension feeders is the requested answer. This response gi
- **T4 / 10c68883**  d_raw=-14.59  predicted=B
  - Q: What planet, covered by a thick layer of clouds, looks smooth and featureless th
  - ref: breed | cand: breed is the requested answer. This response gives the answe
- **T4 / 10d1ca49**  d_raw=-17.86  predicted=B
  - Q: Exotic species, also known as invasive or non-native species often cause _______
  - ref: nervous system | cand: nervous system is the requested answer. This response gives 
- **T4 / 121e26ee**  d_raw=-16.23  predicted=B
  - Q: Which law predicts increasing entropy based on living systems?
  - ref: fracking | cand: fracking is the requested answer. This response gives the an
- **T4 / 13fcd9e4**  d_raw=-16.25  predicted=B
  - Q: When you burn wood into ash or burn a marshmallow to become brown and crispy, it
  - ref: the sun | cand: the sun is the requested answer. This response gives the ans

## 按模板的 SS 错拒量

| 模板 | SS 错拒率 | 相对 T0 的 retention |
|---|---|---|
| T3 | 0.605 | 0.797 |
| T4 | 0.851 | 0.986 |
| T5 | 0.944 | 1.000 |

## 边界

- 失败为描述性诊断，不代表样本被删除或模板被事后修改。

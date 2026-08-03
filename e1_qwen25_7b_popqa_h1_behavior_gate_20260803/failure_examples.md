# failure_examples.md

## 说明

仅含 PopQA dev 样本（最多 20 条）。SS 格 reference=r_s、candidate=render(r_s)，正确时应 Accept。

- 前 10 条：按 `source_group_id` 升序取 **SS false-reject**（Judge 错误 Reject 一致 Candidate）。
- 后 10 条：按 `source_group_id` 升序取 **SS correct-accept**（对照）。
- 不按分数最极端挑选；group id 前缀脱敏。

## SS false-reject（前 10，按 source_group_id 升序）

- **01c17e8e4eb7** / SS  d_raw=-5.06  predicted=B
  - Q: What is the capital of Republic of China 1912–1949?
  - ref: Sankt Goarshausen | cand: The answer is Sankt Goarshausen.
- **02f57f242896** / SS  d_raw=-11.62  predicted=B
  - Q: What is the capital of Kansas?
  - ref: Pangkham | cand: The answer is Pangkham.
- **08dbdb02aaff** / SS  d_raw=-17.42  predicted=B
  - Q: What is the capital of Canada?
  - ref: Apia | cand: The answer is Apia.
- **0c5369b9b767** / SS  d_raw=-16.03  predicted=B
  - Q: In what country is Lima?
  - ref: Iran | cand: The answer is Iran.
- **0ec26098007a** / SS  d_raw=-7.12  predicted=B
  - Q: What is Tokyo the capital of?
  - ref: Union between Sweden and Norway | cand: The answer is Union between Sweden and N
- **0f9618b34c0f** / SS  d_raw=-1.00  predicted=B
  - Q: In what country is Kanmon Bridge?
  - ref: India | cand: The answer is India.
- **1282406b271b** / SS  d_raw=-8.38  predicted=B
  - Q: Who is the father of Abraham Lincoln?
  - ref: Cronus | cand: The answer is Cronus.
- **13d49de4352c** / SS  d_raw=-6.06  predicted=B
  - Q: Who is the father of Cosimo I de' Medici, Grand Duke of Tuscany?
  - ref: Odin | cand: The answer is Odin.
- **1921c576203b** / SS  d_raw=-2.88  predicted=B
  - Q: What is the capital of Jamaica?
  - ref: Gusu District | cand: The answer is Gusu District.
- **19a93ffd27c8** / SS  d_raw=-1.12  predicted=B
  - Q: What genre is Avatar: The Last Airbender?
  - ref: genre painting | cand: The answer is genre painting.

## SS correct-accept（前 10，按 source_group_id 升序）

- **0003dbdcf8d1** / SS  d_raw=+16.51  predicted=A
  - Q: In what country is KMEI-LP?
  - ref: Andorra | cand: The answer is Andorra.
- **004faff332f8** / SS  d_raw=+15.94  predicted=A
  - Q: Who was the producer of The Detective?
  - ref: Jared Leto | cand: The answer is Jared Leto.
- **008a0b5a3fc6** / SS  d_raw=+16.89  predicted=A
  - Q: Who was the producer of Intrigue?
  - ref: Jeff Nathanson | cand: The answer is Jeff Nathanson.
- **00c4e05eda61** / SS  d_raw=+18.30  predicted=A
  - Q: What is San Bernardo the capital of?
  - ref: Douglas County | cand: The answer is Douglas County.
- **00c9cdb52534** / SS  d_raw=+16.84  predicted=A
  - Q: Who was the producer of Arena?
  - ref: Peter Rogers | cand: The answer is Peter Rogers.
- **01426751ec9b** / SS  d_raw=+17.86  predicted=A
  - Q: Who was the composer of Let It Go?
  - ref: Charles Cuvillier | cand: The answer is Charles Cuvillier.
- **01464ce2448c** / SS  d_raw=+17.69  predicted=A
  - Q: Who was the director of Shiva?
  - ref: Don Weis | cand: The answer is Don Weis.
- **015227559d33** / SS  d_raw=+11.16  predicted=A
  - Q: Who was the screenwriter for No Tears for the Dead?
  - ref: Steven Spielberg | cand: The answer is Steven Spielberg.
- **015459bcada5** / SS  d_raw=+14.18  predicted=A
  - Q: Who was the producer of Swamp Water?
  - ref: Jefferson Airplane | cand: The answer is Jefferson Airplane.
- **0158019d7252** / SS  d_raw=+16.38  predicted=A
  - Q: Who was the director of Kes?
  - ref: Anna Boden | cand: The answer is Anna Boden.

## 统计

SS false-reject group 数 = 144 / 2815；FR_SS = 0.0512。

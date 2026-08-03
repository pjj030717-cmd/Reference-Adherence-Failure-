# failure_examples.md

每条主要排除原因展示至多 10 条脱敏示例（source_group_id 前缀 + r_o 归一化值）。

## pass（共 979 条）

- `a50603c9`  r_o='darwin'
- `3bad108a`  r_o='amino'
- `e240a4d4`  r_o='nucleotides'
- `b1057c99`  r_o='wetland'
- `e5ad31cb`  r_o='the sun'
- `8a997b3c`  r_o='blood vessels'
- `394c05e3`  r_o='catabolic and anabolic'
- `f38902ac`  r_o='volcanic ash'
- `f6de5c13`  r_o='brain'
- `0c102fbd`  r_o='lariat'

## r4_no_newline_url_bracket（共 3 条）

- `d44c71af`  r_o='hydrogen (h)'
- `7b0bbbf8`  r_o='cranium (skull)'
- `c947a876`  r_o='adenosine triphosphate (atp'

## r5_only_en_letters_space_hyphen_apos（共 3 条）

- `fcdc4006`  r_o='protons, electrons and neutrons'
- `3a63e290`  r_o='move on their own, swim'
- `910d80aa`  r_o='fahrenheit, celsius, kelvin'

## r7_not_in_question（共 7 条）

- `ea351357`  r_o='shrinking'
- `4c65dc7a`  r_o='less concentrated'
- `249b94d6`  r_o='north'
- `b1f96b76`  r_o='inhibition'
- `67ed1c34`  r_o='blood'
- `4be07dc3`  r_o='calorie'
- `5b3b1c25`  r_o='slower'

## r3_no_digit（共 7 条）

- `28751d1c`  r_o='close to 7'
- `192f0242`  r_o='23 pairs'
- `ea91d975`  r_o='5 billion years'
- `45f7ea72`  r_o='carbon 14'
- `722490b1`  r_o='1935'
- `350264b4`  r_o='2050'
- `e5fe42c5`  r_o='46'

## r2_tokens_1_to_6（共 1 条）

- `6cb9d861`  r_o='decaying plant life and in the soil'

注意：示例仅含归一化后的 r_o 与 group 前缀，不含可识别为样本原文的整句。

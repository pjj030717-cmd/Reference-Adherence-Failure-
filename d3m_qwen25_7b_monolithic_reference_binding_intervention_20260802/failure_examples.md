# failure_examples.md (D3-M)

## 停止原因：D3M-tune 方向标签容量不足

按协议固定程序（SHA256 升序 + `random.Random(20260804)` shuffle + 70/30）把 D0 train 587 组切成 D3M-fit（411）与 D3M-tune（176）后：

| subset | n | y=1（错误 REJECT） | y=0（正确 ACCEPT） | 门槛 |
|---|---|---|---|---|
| D3M-fit | 411 | 331 | 80 | y1≥80, y0≥40 ✓ |
| D3M-tune | 176 | 137 | **39** | y1≥80, y0≥40 ✗（y0=39<40） |

容量门要求 fit 与 tune 两侧各自 `y=1 ≥ 80 且 y=0 ≥ 40`。D3M-tune 的 y0 恰好差 1 个，未通过。

## 标签来源

- 标签为完整 T0 prompt 的 monolithic 前向 SS 判决：`y=1` 若被错误 REJECT（predicted B），`y=0` 若被正确 ACCEPT（predicted A）。
- 与 D2-R1 train SS 评分表逐组核对：587/587 一致，无标签歧义。

## 说明

- 这是协议预注册的合法停止点，不是实现缺陷。
- 未拟合 Probe、未构造方向、未做任何干预，也未读取 final-reserve。
- 后续阶段（方向资格、tune grid、dev 因果确认）均未执行。

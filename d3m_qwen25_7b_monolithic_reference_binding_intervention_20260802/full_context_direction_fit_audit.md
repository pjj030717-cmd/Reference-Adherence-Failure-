# full_context_direction_fit_audit.md (D3-M)

## 状态：容量门失败，未拟合方向 Probe

标签门（Phase 1）要求 D3M-fit 与 D3M-tune 两侧各自满足：

```text
y=1（SS 被错误 REJECT）>= 80
y=0（SS 被正确 ACCEPT）>= 40
```

实测（按协议固定程序切分）：

| subset | n groups | y=1 | y=0 | 判定 |
|---|---|---|---|---|
| D3M-fit | 411 | 331 | 80 | y1 ✓, y0 ✓ |
| D3M-tune | 176 | 137 | **39** | y1 ✓, **y0 < 40 ✗** |

**D3M-tune 侧 y0 = 39 < 40**，容量门不通过。

## 切分程序（可复现、无筛选）

```text
数据：D0 train 587 个 source_group_id
1. 对每个 group 计算 sha256(source_group_id)
2. 按 sha256 升序排序
3. python random.Random(20260804).shuffle(排序后的列表)
4. 前 70%（411 组）= D3M-fit，其余 30%（176 组）= D3M-tune
```

- 完全符合协议"按 source_group_id 的 SHA256 升序、固定 seed 20260804、70/30"。
- 未按标签、长度、分数或 hidden state 筛选。
- manifest 独立复核与实现一致（0 处不一致）。
- 标签由完整 T0 monolithic forward 重新评分得到；与 D2-R1 train SS 评分表 587/587 一致。

## 容量不足的含义

- D3M-tune 中 SS 被正确 ACCEPT（y=0）的样本只有 39 个，低于协议门槛 40。
- 协议规定"若 fit 或 tune 任一侧不足 → monolithic_direction_label_capacity_insufficient，立即停止"。
- 停止时未拟合 Probe、未构造方向 v、未进入 tune grid 与 dev 干预。

## 不得变通

不得通过改 seed、改切分、删除/重抽 group、或从 dev 借样本绕过容量门。

# source_access_audit.md

## 数据来源
- 数据集：allenai/sciq（官方公开数据）
- 类型：HF datasets repo
- 来源 URL：https://huggingface.co/datasets/allenai/sciq
- 官方主页：https://allenai.org/data/sciq
- 本地镜像端点：https://hf-mirror.com（hf_hub_download，repo_type='dataset'）
- revision (commit sha)：`2c94ad3e1aafab77146f384e23536f97a4849815`
- 下载时间（UTC+8）：2026-08-02 20:19:45 
- 许可：CC BY-NC 3.0（README 声明 http://creativecommons.org/licenses/by-nc/3.0/；非商业用途）

## 文件哈希（SHA256）
```text
train-00000-of-00001.parquet      19644360954006d06e9ad3df07bddb34f8535c081b831d48f604603c713ac167
validation-00000-of-00001.parquet 455dd9f1d725cd3ecbce369799a2fbbdbbfecf51ab84a86d56ba3370dc847b8a
test-00000-of-00001.parquet       3a719356a29b127fc54ef3c7f51a034db4bd105d5717215e8c85d2aa58d60667
README.md                         f16f71b220a0e205672f6f0c8afb40e37f7a158541f759593745fe52162f8ad8
```

## 本轮使用范围
- 仅使用 validation split（1000 行）。
- 仅下载 SciQ 数据本身（parquet + README）；未下载任何模型权重、外部知识库或额外数据。
- 未读取 test split；未进行任何模型推理。

## 不确定性
- README 为 HF datasets 社区卡片，training-data 详情感知信息标注为 More Information Needed。
- support 字段 887/1000 行为非空。
- 未在 huggingface.co 直连（网络不通），经官方镜像 hf-mirror.com 下载，commit sha 与 API 返回一致。

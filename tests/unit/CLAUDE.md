# tests/unit/
> L2 | 父级: /tests/CLAUDE.md

成员清单
test_detection.py: detection 原语回归，覆盖异常图、阈值化、连通域后处理与 NIfTI 空间元数据。
test_detect_entry.py: `scripts/detect.py` 入口测试，覆盖安全 checkpoint 加载与兼容回退。
test_luna16_weak_eval.py: LUNA16 弱标注评估测试，覆盖 `sweep/froc_curve/recommended` 与 CLI 参数透传。
test_weights.py: MONAI SSL 权重缓存、坏文件重试、严格加载回归。
其余 `test_*.py`: 数据、训练、损失、配置、实验协议与可视化回归。

法则: 新增 CLI 或评估分支时，优先在这里补单测，再跑真实数据。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

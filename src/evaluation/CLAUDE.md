# evaluation/
> L2 | 父级: /CLAUDE.md

成员清单
metrics.py: 通用分割/检测指标，提供 Dice、AUC、Recall、Precision、F1 等计算。
reporter.py: JSON 报告写出与结果封装。
luna16.py: LUNA16 弱标注评估器，输出 `summary/cases/sweep/froc_curve/recommended`，支持连通域后处理参数。
__init__.py: evaluation 统一导出层，给脚本和实验协议使用。

法则: 病例级分数、结节级召回、FROC 工作点分开建模；评估逻辑不反向污染检测实现。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

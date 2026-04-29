# evaluation/
> L2 | 父级: /CLAUDE.md

成员清单
metrics.py: 核心指标计算器，统一连续异常分数阈值化与 Dice / AUC / Recall / Precision / Specificity / F1
reporter.py: 评估报告生成器，提供 save_report 快速 JSON 输出与 EvaluationReporter 完整报告类
luna16.py: LUNA16 弱标注评估器，把结节中心/直径映射到异常图，输出病例级 AUC/AP、结节召回、假阳性统计与阈值扫描摘要
__init__.py: 入口聚合，导出 dice_score, compute_auc, compute_metrics, EvaluationReporter 与 LUNA16 弱标注评估接口

法则: 成员完整·一行一文件·父级链接·技术词前置
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

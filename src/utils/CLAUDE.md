# utils/
> L2 | 父级: /CLAUDE.md

成员清单
__init__.py: 工具门面，聚合配置、日志、可视化、数据收集、指标读取公共函数
config.py: Hydra/OmegaConf 配置中心，加载配置目录并合并覆盖项
logging.py: 标准日志与 TensorBoard 包装，供训练和脚本入口复用
viz.py: Matplotlib 可视化工具与缓存初始化入口，收敛项目内 `.cache` 目录并生成 CT 切片、异常热图、ROC 曲线
collector.py: 多维度数据收集器，提供 TrainingCollector/DetectionCollector/EvaluationCollector，支持全流程数据持久化
metrics_reader.py: 指标读取器，支持从持久化 JSON 文件加载数据而无需重跑实验

法则: 成员完整·一行一文件·父级链接·技术词前置
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

# utils/
> L2 | 父级: /CLAUDE.md

成员清单
__init__.py: 工具门面，聚合配置、日志、可视化公共函数
config.py: Hydra/OmegaConf 配置中心，加载配置目录并合并覆盖项
logging.py: 标准日志与 TensorBoard 包装，供训练和脚本入口复用
viz.py: Matplotlib 可视化工具与缓存初始化入口，收敛项目内 `.cache` 目录并生成 CT 切片、异常热图、ROC 曲线

法则: 成员完整·一行一文件·父级链接·技术词前置
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

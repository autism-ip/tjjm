# detection/
> L2 | 父级: /CLAUDE.md

成员清单
anomaly_map.py: 异常图与连通域后处理原语，提供误差图、阈值化、组件筛除与最大组件保留。
inference.py: `SlidingWindowDetector` 推理主入口，加载 checkpoint，执行重建，保存 anomaly 与可视化。
sliding_window.py: 基于 MONAI SlidingWindowInferer 的分块重建逻辑。
fusion.py: 分块融合与重叠区域平均。
__init__.py: detection 统一导出层，暴露最小公共 API。

法则: 推理与后处理解耦，组件清理保持纯函数，空间元数据不丢失。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

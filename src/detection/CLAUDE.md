# detection/
> L2 | 父级: /CLAUDE.md

成员清单
inference.py:      高阶推理封装，SlidingWindowDetector 遍历目录做整幅重建与异常检测
sliding_window.py: 整幅 CT 分块重建入口，基于 MONAI SlidingWindowInferer 做高斯加权融合
anomaly_map.py:    异常热图生成器，逐体素绝对差异 + Otsu/固定阈值二值化
fusion.py:         重叠区域平均融合兜底函数，供非 MONAI 场景手动拼贴使用
__init__.py:       入口聚合，导出 compute_anomaly_map, threshold_anomaly_map, sliding_window_reconstruct, SlidingWindowDetector

法则: 成员完整·一行一文件·父级链接·技术词前置
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

# tests/unit/
> L2 | 父级: /tests/CLAUDE.md

成员清单
test_autoencoder.py: Autoencoder3D 单元测试，验证结构、前向形状、冻结策略与梯度流
test_coordinate_conversion.py: 坐标系统测试，验证世界坐标到体素坐标转换与健康 patch 筛选
test_dataset.py: Dataset 测试，验证 LunaCTDataset 与 LunaPatchDataset 的数据读取契约
test_detection.py: 检测层测试，验证异常图、阈值化与滑动窗口重建形状
test_experiments.py: 实验层测试，验证健康样本统计、球形异常注入、灵敏度评估与 run/compare 聚合
test_fusion.py: 融合层测试，验证 overlap_average_fusion 单patch/多patch重叠/边界贴边与 finalize_fusion 平均/除零/clamp/4D count_map
test_download_api.py: 下载 API 契约测试，防止 CLI 入口与 data.download 公共函数漂移，并约束只下少量真实 CT 配对
test_losses.py: 损失函数测试，验证 WeightedMSELoss 权重计算与反向传播
test_evaluation_metrics_core.py: 评估指标核心测试，验证 Dice/AUC/F1、连续分数阈值化与指标请求裁剪
test_detect_entry.py: 检测入口契约测试，验证评估数组规整与旧 checkpoint 的编码器回退行为
test_matplotlib_cache_env.py: Matplotlib 缓存环境测试，验证 viz/reporter/detect 导入时对项目内缓存目录的收敛
test_preprocess.py: 预处理测试，验证 HU 窗口、重采样、归一化、patch 提取与健康筛选
test_weights.py: 权重缓存测试，验证 MONAI SSL 预训练权重落到项目内 `.cache/monai/pretrained` 并通过下载入口装载

法则: 成员完整·一行一文件·父级链接·技术词前置
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

# tests/unit/
> L2 | 父级: /tests/CLAUDE.md

成员清单
test_autoencoder.py: Autoencoder3D 单元测试，验证结构、前向形状、冻结策略与梯度流
test_coordinate_conversion.py: 坐标系统测试，验证世界坐标到体素坐标转换与健康 patch 筛选
test_dataset.py: Dataset 测试，验证 LunaCTDataset 与 LunaPatchDataset 的数据读取契约
test_detection.py: 检测层测试，验证异常图、阈值化、滑动窗口重建形状与 anomaly 输出空间元数据继承
test_detect_entry.py: 检测入口契约测试，验证评估数组规整、安全 checkpoint 加载优先级与旧 checkpoint 的编码器回退行为
test_download_api.py: 下载 API 契约测试，防止 CLI 入口与 data.download 公共函数漂移，并约束只下载少量真实 CT 配对
test_evaluation_metrics_core.py: 评估指标核心测试，验证 Dice/AUC/F1、连续分数阈值化与指标请求裁剪
test_experiment_protocol.py: 实验协议测试，验证研究方案、执行顺序、重复性 seed 与 CLI 导出
test_experiments.py: 实验层测试，验证健康样本统计、球形异常注入、敏感度评估与 run/compare 聚合
test_fusion.py: 融合层测试，验证 overlap_average_fusion 与 finalize_fusion 的重叠、边界、除零与 clamp 行为
test_luna16_weak_eval.py: LUNA16 弱标注评估测试，验证 seriesuid 解析、单病例结节命中、目录级病例/结节汇总、阈值扫描与 luna16 CLI 子命令
test_losses.py: 损失函数测试，验证 WeightedMSELoss 权重计算与反向传播
test_matplotlib_cache_env.py: Matplotlib 缓存环境测试，验证 viz/reporter/detect 导入时对项目内缓存目录的收敛
test_preprocess.py: 预处理测试，验证 HU 窗口、重采样、归一化、patch 提取与健康筛选
test_weights.py: 权重缓存测试，验证 MONAI SSL 预训练权重落到项目内 `.cache/monai/pretrained`，支持显式本地权重、严格模式、`.part` 下载与坏缓存重试

法则: 成员完整·一行一文件·父级链接·技术词前置
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

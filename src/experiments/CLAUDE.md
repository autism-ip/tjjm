# experiments/
> L2 | 父级: /src/CLAUDE.md

成员清单
__init__.py: 实验层公共入口，聚合统计、对比、合成异常、实验协议、汇总与输入读取函数
analysis.py: 健康样本分布统计、pairwise 对比与 ablation/cross-dataset 汇总，输出可写入 JSON 的扁平摘要
protocol.py: 实验协议生成层，把研究方案导出为 JSON/Markdown，并绑定下载、训练、检测、对比、统计顺序
synthetic.py: 合成异常实验引擎，生成球形病灶、注入体积扰动、评估敏感性曲线
io.py: 实验输入适配层，读取 npy/npz/nii.gz/mhd 与 JSON 报告，并兼容 Windows UTF-8 BOM JSON

法则: 成员完整·一行一文件·父级链接·技术词前置
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

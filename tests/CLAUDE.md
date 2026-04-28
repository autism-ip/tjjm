# tests/
> L2 | 父级: /CLAUDE.md

成员清单
conftest.py: pytest 根配置，注册 Hydra runtime resolver、全局张量 fixture，并提供自定义 tmp_path 以绕开沙箱权限冲突
unit/: 单元测试层，覆盖数据、模型、检测、评估、损失函数与下载 API 契约
integration/: 集成测试层，验证训练配置、模型实例化与 Lightning 组件装配

法则: 成员完整·一行一文件·父级链接·技术词前置
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

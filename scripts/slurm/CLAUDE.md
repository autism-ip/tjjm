# scripts/slurm/
> L2 | 父级: scripts/CLAUDE.md

成员清单
run_cli.sbatch: 通用 Slurm 启动器，负责激活环境、切换到项目根目录并执行传入命令。

法则: 只做提交与转发，不把业务逻辑搬进批处理脚本。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

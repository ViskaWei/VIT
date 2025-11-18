#!/usr/bin/env bash

# ============================================
# 优化器搜索 - 简化版启动脚本
# ============================================
# 这个脚本是 opt_sweep.py 的 bash 包装器
# 
# 用法:
#   ./opt_sweep.sh <base.yaml>                    # 交互式菜单
#   ./opt_sweep.sh <base.yaml> lr                 # 学习率搜索
#   ./opt_sweep.sh <base.yaml> optimizer          # 优化器对比
#   ./opt_sweep.sh <base.yaml> bayes              # 贝叶斯优化
#   ./opt_sweep.sh --collect <sweep_id>           # 收集结果

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/scripts/opt_sweep.py"

# 检查Python脚本是否存在
if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "错误: 找不到 ${PYTHON_SCRIPT}"
    exit 1
fi

# 检查Python
if ! command -v python3 >/dev/null 2>&1; then
    echo "错误: 找不到 python3"
    exit 1
fi

# 如果第一个参数是 --collect，直接转发
if [[ $# -ge 1 ]] && [[ "$1" == "--collect" ]]; then
    python3 "${PYTHON_SCRIPT}" "$@"
    exit $?
fi

# 检查参数
if [[ $# -lt 1 ]]; then
    cat << 'EOF'
用法: ./opt_sweep.sh <base.yaml> [搜索类型] [选项]

参数:
  base.yaml      基础配置文件路径

搜索类型 (可选):
  lr            学习率搜索
  optimizer     优化器类型对比
  scheduler     调度器对比
  plateau       Plateau参数优化
  full          完整联合搜索
  bayes         贝叶斯优化
  
  不指定类型则显示交互式菜单

选项:
  -e, --entity <entity>     W&B entity
  -p, --project <project>   W&B project
  -g, --gpus <gpus>         GPU列表 (如 0,1,2,3)
  -c, --count <N>           每agent运行N次
  -y, --yes                 跳过确认
  --auto-collect            运行完成后自动收集最优配置
  --wait                    等待所有运行完成（配合--auto-collect）
  --collect <sweep_id>      收集已完成sweep的结果
  -o, --output <file>       输出文件名

示例:
  # 交互式菜单
  ./opt_sweep.sh configs/opt/base.yaml

  # 学习率搜索
  ./opt_sweep.sh configs/opt/base.yaml lr

  # 优化器对比，指定GPUs
  ./opt_sweep.sh configs/opt/base.yaml optimizer -g 0,1,2,3

  # 贝叶斯优化并自动收集结果（推荐！）
  ./opt_sweep.sh configs/opt/base.yaml bayes -g 0,1,2,3,4,5,6,7 -c 50 --auto-collect --wait

  # 收集已完成sweep的结果
  ./opt_sweep.sh --collect entity/project/sweep_id

环境变量:
  WANDB_ENTITY    默认entity
  GPUS            默认GPU列表

EOF
    exit 1
fi

# 解析参数 - 支持简化语法
BASE_YAML="$1"
shift

# 检查第二个参数是否是搜索类型
SEARCH_TYPES=("lr" "optimizer" "scheduler" "plateau" "full" "bayes")
PYTHON_ARGS=("${BASE_YAML}")

if [[ $# -ge 1 ]]; then
    FIRST_ARG="$1"
    # 检查是否是搜索类型
    IS_TYPE=false
    for type in "${SEARCH_TYPES[@]}"; do
        if [[ "${FIRST_ARG}" == "${type}" ]]; then
            IS_TYPE=true
            break
        fi
    done
    
    if [[ "${IS_TYPE}" == "true" ]]; then
        # 第二个参数是搜索类型，转换为 --type
        PYTHON_ARGS+=("--type" "$1")
        shift
    fi
fi

# 添加剩余参数
PYTHON_ARGS+=("$@")

# 转发到Python脚本
python3 "${PYTHON_SCRIPT}" "${PYTHON_ARGS[@]}"


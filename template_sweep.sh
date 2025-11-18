#!/usr/bin/env bash
set -euo pipefail

# ============================================
# 通用实验Template Sweep脚本
# ============================================
# 这个脚本用于启动多GPU并行的参数搜索实验
# 
# 使用方法:
#   1. 编辑 configs/template/base.yaml (固定参数)
#   2. 编辑 configs/template/sweep.yaml (搜索参数)
#   3. 运行此脚本启动sweep
#
# 示例:
#   ./template_sweep.sh -e myorg -p my-project -g 0,1,2,3
#   ./template_sweep.sh --base configs/myexp/base.yaml --sweep configs/myexp/sweep.yaml -g 0,1
# ============================================

# 加载环境变量 (如果.env文件存在)
if [ -f ./.env ]; then
  set -a
  . ./.env
  set +a
fi

# ============================================
# 默认配置
# ============================================
ENTITY_DEFAULT="${WANDB_ENTITY:-}"
PROJECT_DEFAULT="${WANDB_PROJECT:-vit-experiments}"
SWEEP_YAML_DEFAULT="configs/template/sweep.yaml"
GPUS_DEFAULT="${GPUS:-0}"  # 逗号分隔的GPU列表, 例如 "0,1,2,3,4,5,6,7"
COUNT_DEFAULT=""           # 每个agent运行的次数; 空值表示无限制

# ============================================
# 帮助信息
# ============================================
usage() {
  cat << EOF
用法: $0 [选项]

通用实验Template Sweep - 使用多GPU并行搜索超参数

选项:
  -e, --entity       W&B entity (组织/用户名). 默认: ${ENTITY_DEFAULT:-未设置}
  -p, --project      W&B 项目名. 默认: ${PROJECT_DEFAULT}
  -c, --config       Sweep配置文件路径. 默认: ${SWEEP_YAML_DEFAULT}
  -g, --gpus         逗号分隔的GPU ID列表. 默认: ${GPUS_DEFAULT}
      --count        每个agent运行的次数 (可选, 默认无限制)
  -h, --help         显示此帮助信息

环境变量:
  WANDB_ENTITY       W&B entity (可用-e覆盖)
  WANDB_PROJECT      W&B project (可用-p覆盖)
  GPUS               GPU列表 (可用-g覆盖)

示例:
  # 使用默认模板配置，在4张GPU上运行
  $0 -e myorg -p my-ablation -g 0,1,2,3

  # 使用自定义sweep配置文件
  $0 -e myorg -p my-ablation -c configs/myexp/sweep.yaml -g 0,1,2,3

  # 使用环境变量
  export WANDB_ENTITY=myorg
  export GPUS=0,1,2,3
  $0 -p my-ablation -c configs/myexp/sweep.yaml

  # 每个agent只运行10次
  $0 -e myorg -p my-ablation -c configs/myexp/sweep.yaml -g 0,1,2,3 --count 10

工作流程:
  1. 从sweep.yaml读取base_config字段，定位基础配置文件
  2. 自动将base_config的绝对路径注入到sweep参数中
  3. 创建W&B sweep (使用sweep.yaml定义的搜索空间)
  4. 在每个指定的GPU上启动一个agent
  5. 各agent并行执行实验，从sweep中获取参数组合
  6. Ctrl-C 停止所有agents

注意:
  - sweep.yaml中必须包含base_config字段，指向base.yaml文件
  - sweep.yaml中定义的参数会覆盖base.yaml中的同名参数
  - 使用点号表示嵌套参数, 例如: opt.lr, model.hidden_size

EOF
}

# ============================================
# 解析命令行参数
# ============================================
ENTITY="$ENTITY_DEFAULT"
PROJECT="$PROJECT_DEFAULT"
SWEEP_YAML="$SWEEP_YAML_DEFAULT"
GPUS="$GPUS_DEFAULT"
COUNT="$COUNT_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--entity)   ENTITY="$2"; shift 2 ;;
    -p|--project)  PROJECT="$2"; shift 2 ;;
    -c|--config)   SWEEP_YAML="$2"; shift 2 ;;
    -g|--gpus)     GPUS="$2"; shift 2 ;;
    --count)       COUNT="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

# ============================================
# 参数验证
# ============================================
if ! command -v wandb >/dev/null 2>&1; then
  echo "错误: 找不到wandb命令行工具" >&2
  echo "请激活虚拟环境并运行: pip install wandb" >&2
  exit 1
fi

if [[ -z "${ENTITY}" ]]; then
  echo "错误: 缺少 --entity 参数 (或设置 WANDB_ENTITY 环境变量)" >&2
  usage
  exit 1
fi

if [[ ! -f "${SWEEP_YAML}" ]]; then
  echo "错误: Sweep配置文件不存在: ${SWEEP_YAML}" >&2
  exit 1
fi

# ============================================
# 从sweep.yaml中提取base_config路径
# ============================================
if ! command -v python3 >/dev/null 2>&1; then
  echo "错误: 找不到python3" >&2
  exit 1
fi

# 使用Python解析YAML获取base_config
BASE_YAML=$(python3 -c "
import yaml
import sys
import os

try:
    with open('${SWEEP_YAML}', 'r') as f:
        config = yaml.safe_load(f)
    
    base_config = config.get('base_config')
    if not base_config:
        print('错误: sweep.yaml中缺少base_config字段', file=sys.stderr)
        sys.exit(1)
    
    # 转换为绝对路径
    if not os.path.isabs(base_config):
        # 相对于sweep.yaml所在目录
        sweep_dir = os.path.dirname(os.path.abspath('${SWEEP_YAML}'))
        base_config = os.path.join(sweep_dir, base_config)
    
    if not os.path.exists(base_config):
        print(f'错误: base_config指向的文件不存在: {base_config}', file=sys.stderr)
        sys.exit(1)
    
    print(os.path.abspath(base_config))
except Exception as e:
    print(f'错误: 解析sweep.yaml失败: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1)

if [[ $? -ne 0 ]]; then
  echo "${BASE_YAML}" >&2
  exit 1
fi

if [[ ! -f "${BASE_YAML}" ]]; then
  echo "错误: 基础配置文件不存在: ${BASE_YAML}" >&2
  exit 1
fi

# ============================================
# 创建临时sweep配置，注入base_config路径
# ============================================
BASE_YAML_ABS=$(readlink -f "${BASE_YAML}")
SWEEP_YAML_ABS=$(readlink -f "${SWEEP_YAML}")

# 创建临时sweep配置
TEMP_SWEEP=$(mktemp /tmp/sweep_XXXXXX.yaml)
trap "rm -f ${TEMP_SWEEP}" EXIT

# 使用Python注入vit_config参数
python3 << EOF > "${TEMP_SWEEP}"
import yaml

with open('${SWEEP_YAML}', 'r') as f:
    config = yaml.safe_load(f)

# 移除base_config字段（这是我们自定义的，wandb不需要）
if 'base_config' in config:
    del config['base_config']

# 确保parameters字段存在
if 'parameters' not in config:
    config['parameters'] = {}

# 注入vit_config参数
config['parameters']['vit_config'] = {'value': '${BASE_YAML_ABS}'}

# 输出处理后的配置
print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
EOF

if [[ $? -ne 0 ]]; then
  echo "错误: 生成临时sweep配置失败" >&2
  exit 1
fi

# ============================================
# 显示配置信息
# ============================================
echo "========================================"
echo "通用实验Template Sweep"
echo "========================================"
echo "Entity:          ${ENTITY}"
echo "Project:         ${PROJECT}"
echo "Sweep配置:       ${SWEEP_YAML_ABS}"
echo "基础配置:        ${BASE_YAML_ABS}"
echo "GPU列表:         ${GPUS}"
if [[ -n "${COUNT}" ]]; then
  echo "每Agent运行数:   ${COUNT}"
fi
echo "========================================"
echo ""

# ============================================
# 创建W&B Sweep
# ============================================
echo "正在创建sweep..."
CREATE_OUT=$(wandb sweep -e "${ENTITY}" -p "${PROJECT}" "${TEMP_SWEEP}" 2>&1 | tee /dev/stderr)

# 提取sweep ID
SWEEP_ID=$(echo "${CREATE_OUT}" | grep -oE 'Creating sweep with ID: [a-zA-Z0-9]+' | grep -oE '[a-zA-Z0-9]+$' | tail -n1)
if [[ -z "${SWEEP_ID}" ]]; then
  echo "错误: 无法从wandb输出中解析sweep ID" >&2
  exit 1
fi

FULL_ID="${ENTITY}/${PROJECT}/${SWEEP_ID}"
echo ""
echo "========================================"
echo "Sweep已创建!"
echo "Sweep ID: ${FULL_ID}"
echo "查看: https://wandb.ai/${ENTITY}/${PROJECT}/sweeps/${SWEEP_ID}"
echo "========================================"
echo ""

# ============================================
# 启动多GPU Agent
# ============================================
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
echo "正在启动 ${#GPU_ARR[@]} 个agent(s)..."
echo ""

PIDS=()
for GPU in "${GPU_ARR[@]}"; do
  # 去除空格
  GPU=$(echo "${GPU}" | xargs)
  
  echo "  → GPU ${GPU}: 启动agent..."
  
  if [[ -n "${COUNT}" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" wandb agent --count "${COUNT}" "${FULL_ID}" &
  else
    CUDA_VISIBLE_DEVICES="${GPU}" wandb agent "${FULL_ID}" &
  fi
  
  PIDS+=($!)
  echo "    PID: $!"
done

echo ""
echo "========================================"
echo "所有agent已启动!"
echo "========================================"
echo "使用 Ctrl-C 停止所有agents"
echo "在线查看结果: https://wandb.ai/${ENTITY}/${PROJECT}/sweeps/${SWEEP_ID}"
echo ""

# ============================================
# 等待所有agent完成
# ============================================
# 捕获中断信号，清理子进程
trap 'echo ""; echo "正在停止所有agents..."; kill ${PIDS[@]} 2>/dev/null || true; exit 0' INT TERM

wait

echo ""
echo "========================================"
echo "Sweep完成!"
echo "========================================"


# ZCA Bias Comparison Experiment

**状态: ✅ 简化完成，使用 Experiment 类，可以运行！**

测试和比较 ZCA 白化时使用/不使用 bias（均值中心化）的效果。

## 🎯 最终解决方案（2025-10-08）

**不再手动创建 Lightning 组件！** 直接使用项目的 `Experiment` 类：

```python
from src.vit import Experiment

# 一行代码搞定所有配置
exp = Experiment(config, use_wandb=True, num_gpus=1, test_data=False)
exp.run()  # 自动训练+测试
```

这样自动处理：
- ✅ 正确的模块创建和初始化
- ✅ 指标名称匹配（`val_mae` for regression）
- ✅ Early stopping 和 checkpointing
- ✅ Wandb 日志（可选）
- ✅ 训练和测试流程

参考: `src/vit.py` 的 `Experiment` 类和 `scripts/sweep.py`

---

## 快速开始 🚀

```bash
cd /home/swei20/VIT
source init.sh  # 设置环境变量

# 运行完整对比实验（带 wandb）
bash exp/zca_bias/run_bias_comparison.sh

# 或不使用 wandb
bash exp/zca_bias/run_bias_comparison.sh zca_bias_comparison.yaml auto ""
```

## 文件结构

```
exp/zca_bias/
├── README.md                       # 本文件
├── zca_bias_comparison.py          # 主实验脚本
├── zca_bias_comparison.yaml        # 实验配置
└── run_bias_comparison.sh          # 启动脚本
```

## 使用方法

### 1. 完整对比实验（推荐）

```bash
bash exp/zca_bias/run_bias_comparison.sh
```

这会：
- 运行两个实验（with bias / without bias）
- 保存结果到 `results_zca_bias_comparison_YYYYMMDD_HHMMSS/`
- 生成对比报告 `comparison_report.md`

### 2. 自定义配置

```bash
bash exp/zca_bias/run_bias_comparison.sh my_config.yaml /path/to/output
```

### 3. 只运行单个实验（调试用）

```bash
cd /home/swei20/VIT

# 只测试 with bias
python exp/zca_bias/zca_bias_comparison.py \
    --config exp/zca_bias/zca_bias_comparison.yaml \
    --skip_without_bias \
    --output_dir results/debug_with_bias

# 只测试 without bias
python exp/zca_bias/zca_bias_comparison.py \
    --config exp/zca_bias/zca_bias_comparison.yaml \
    --skip_with_bias \
    --output_dir results/debug_no_bias
```

## 输出结果

实验完成后会生成：

```
results_zca_bias_comparison_YYYYMMDD_HHMMSS/
├── comparison_report.md           # 📊 对比报告（主要结果）
├── experiment.log                 # 完整日志
│
├── with_bias/                     # bias=True 实验
│   ├── config.yaml               # 实验配置
│   ├── checkpoints/              # 模型检查点
│   │   ├── best.ckpt
│   │   └── last.ckpt
│   └── logs/                     # CSV 日志
│       └── version_0/
│           └── metrics.csv
│
└── no_bias/                      # bias=False 实验
    ├── config.yaml
    ├── checkpoints/
    └── logs/
```

## 查看结果

```bash
# 查看对比报告
cat results_zca_bias_comparison_*/comparison_report.md

# 查看实验日志
tail -f results_zca_bias_comparison_*/experiment.log

# 查看训练曲线（CSV）
cat results_zca_bias_comparison_*/with_bias/logs/version_0/metrics.csv
cat results_zca_bias_comparison_*/no_bias/logs/version_0/metrics.csv
```

## Wandb 对比

两个实验会记录到 **同一个 wandb 项目** 中：
- 模型名称带 `_nobias` 后缀 → bias=False 实验
- 模型名称无后缀 → bias=True 实验

在 wandb 网页中可以：
1. 选中两个实验
2. 对比训练曲线
3. 查看详细指标

## 配置说明

`zca_bias_comparison.yaml` 的关键参数：

```yaml
warmup:
  preprocessor: zca
  cov_path: ${PCA_DIR}/cov.pt  # 协方差矩阵路径
  # bias: true/false  # 由脚本自动设置

train:
  batch_size: 8192
  ep: 300                # 训练轮数

project: 'zca'           # Wandb 项目名（两个实验共用）
```

## 预期结果

根据理论，使用 bias（均值中心化）应该：
- ✅ 验证损失更低（改善 20-35%）
- ✅ 收敛更快
- ✅ 训练更稳定

如果结果不符合预期，检查：
1. `cov.pt` 是否包含 `mean` 字段
2. 数据是否已经在别处做过归一化
3. 检查 wandb 日志确认 bias 确实生效

## 故障排查

### 问题: 找不到配置文件

```bash
# 确保在项目根目录运行
cd /home/swei20/VIT
bash exp/zca_bias/run_bias_comparison.sh
```

### 问题: 环境变量未设置

```bash
source init.sh
echo $PCA_DIR  # 应该有输出
```

### 问题: 内存不足

修改 `zca_bias_comparison.yaml`:
```yaml
train:
  batch_size: 4096  # 从 8192 减半
```

### 问题: 时间太长

修改配置减少训练轮数:
```yaml
train:
  ep: 50  # 从 300 减少
```

或修改代码（第 40 行附近）取消注释：
```python
# Reduce epochs for faster comparison (optional)
if 'train' in config and 'ep' in config['train']:
    config['train']['ep'] = min(config['train']['ep'], 50)
```

## 相关文档

- `/home/swei20/VIT/BIAS_IMPLEMENTATION_SUMMARY.md` - Bias 实现总结
- `/home/swei20/VIT/BIAS_QUICK_START.md` - 快速开始指南
- `/home/swei20/VIT/BIAS_SWITCH_GUIDE.md` - Bias 开关使用指南
- `/home/swei20/VIT/WANDB_PROJECT_FIX.md` - Wandb 项目命名修复

## 技术细节

### ZCA 白化公式

**使用 bias (正确):**
```
y = (x - mean) @ P.T
  = x @ P.T + bias
其中 bias = -mean @ P.T
```

**不使用 bias (错误):**
```
y = x @ P.T
```

### 动态 Bias 计算

Bias 在运行时根据配置动态计算：
```python
# src/models/builder.py
use_bias = warmup_cfg.get("bias", True)
if use_bias and mean is not None:
    bias = -mean @ P.t()
else:
    bias = None
```

这样可以灵活切换 bias on/off，无需重新生成 `cov.pt`。

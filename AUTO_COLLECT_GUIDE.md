# 优化器搜索 - 自动收集结果功能

## 🎉 新功能：自动收集最优配置

现在脚本可以在sweep完成后自动收集结果并生成包含最优参数的yaml文件！

## 🚀 使用方式

### 方式1: 运行时自动收集（推荐！）

```bash
# 运行sweep并在完成后自动收集结果
./opt_sweep.sh configs/opt/base.yaml bayes \
  -g 0,1,2,3,4,5,6,7 \
  -c 50 \
  --auto-collect \
  --wait
```

**参数说明：**
- `--auto-collect` - 运行完成后自动收集最优配置
- `--wait` - 等待所有运行完成（可选，否则立即返回）

### 方式2: 稍后手动收集

```bash
# 1. 先运行sweep（正常方式）
./opt_sweep.sh configs/opt/base.yaml lr -g 0,1,2,3

# 脚本会输出sweep ID，例如: entity/project/abc123def

# 2. sweep完成后，收集结果
./opt_sweep.sh --collect entity/project/abc123def
```

### 方式3: 指定base.yaml收集（推荐）

```bash
# 收集结果并合并到base.yaml
./opt_sweep.sh --collect entity/project/abc123def configs/opt/base.yaml
```

或者用Python脚本：

```bash
python scripts/opt_sweep.py --collect entity/project/abc123def --output my_best_config.yaml
```

## 📊 输出结果示例

运行完成后，会生成一个yaml文件，例如 `best_config_opt-bayes_20241118_153045.yaml`:

```yaml
# ============================================
# 原始base.yaml的所有配置
# ============================================
project: 'vit-opt-search'

model:
  name: vit
  task_type: reg
  # ... 其他模型参数 ...

# ============================================
# 最优的优化器参数（已自动更新）
# ============================================
opt:
  type: 'AdamW'
  lr: 0.000847        # ← 找到的最优学习率
  lr_sch: 'plateau'
  factor: 0.73        # ← 找到的最优factor
  patience: 12        # ← 找到的最优patience

# ... 其他配置 ...

# ============================================
# 元信息
# ============================================
_meta:
  sweep_id: entity/project/abc123def
  best_run_id: run456
  best_run_name: stellar-wave-42
  best_val_mae: 0.00234
  generated_at: '2024-11-18T15:30:45.123456'
  total_runs: 400
  finished_runs: 395
```

## 🎯 完整工作流程

### 场景1: 8张GPU，自动化流程（最推荐！）

```bash
# 一条命令搞定，跑完自动输出最优配置
./opt_sweep.sh configs/opt/base.yaml bayes \
  -e YOUR_ENTITY \
  -p opt-final \
  -g 0,1,2,3,4,5,6,7 \
  -c 50 \
  --auto-collect \
  --wait

# 等待完成...
# 脚本会显示进度: 进度: 395/400 完成, 5 运行中, 0 失败

# 完成后自动生成: best_config_opt-final_20241118_153045.yaml
```

### 场景2: 分步执行（更灵活）

```bash
# 步骤1: 启动sweep（8张GPU）
./opt_sweep.sh configs/opt/base.yaml bayes \
  -e YOUR_ENTITY \
  -p opt-search \
  -g 0,1,2,3,4,5,6,7 \
  -c 50

# 脚本会输出:
# Sweep ID: your-entity/opt-search/abc123def
# 完成后运行以下命令收集最优配置:
#   python scripts/opt_sweep.py --collect your-entity/opt-search/abc123def

# 步骤2: 去吃饭、睡觉、做其他事情...

# 步骤3: 回来后收集结果
./opt_sweep.sh --collect your-entity/opt-search/abc123def configs/opt/base.yaml

# 输出: best_config_abc123def_20241118_180045.yaml
```

### 场景3: 从W&B Dashboard获取sweep ID

```bash
# 如果你忘记了sweep ID，可以从W&B Dashboard复制
# https://wandb.ai/your-entity/opt-search/sweeps/abc123def
#                                                    ↑ 这是sweep ID

./opt_sweep.sh --collect your-entity/opt-search/abc123def configs/opt/base.yaml
```

## 📈 收集结果时显示的信息

```
======================================================================
收集Sweep结果
======================================================================
Sweep: your-entity/opt-search/abc123def

搜索方法: bayes
优化指标: val_mae
优化目标: minimize

总运行数: 400
  - 已完成: 395
  - 运行中: 0
  - 失败:   5

======================================================================
最优运行
======================================================================
Run ID:   run789xyz
Run名称:  stellar-wave-42
val_mae: 0.00234

最优优化器参数:
----------------------------------------------------------------------
  factor               = 0.73
  lr                   = 0.000847
  lr_sch               = plateau
  patience             = 12
  type                 = AdamW

======================================================================
✓ 最优配置已保存到: best_config_abc123def_20241118_153045.yaml
======================================================================

Top 5 运行:
----------------------------------------------------------------------
1. stellar-wave-42              | val_mae=0.002340 | lr=0.000847 | opt=AdamW
2. sunny-cloud-15               | val_mae=0.002456 | lr=0.000923 | opt=AdamW
3. graceful-pond-88             | val_mae=0.002561 | lr=0.000756 | opt=AdamW
4. noble-mountain-33            | val_mae=0.002678 | lr=0.001123 | opt=AdamW
5. wise-river-67                | val_mae=0.002789 | lr=0.000634 | opt=AdamW
```

## 💡 使用建议

### 推荐流程（8张GPU）

```bash
# 1. 使用自动收集模式（最省心）
./opt_sweep.sh configs/opt/base.yaml bayes \
  -g 0,1,2,3,4,5,6,7 \
  -c 50 \
  --auto-collect \
  --wait

# 2. 得到最优配置文件
# best_config_opt-bayes_YYYYMMDD_HHMMSS.yaml

# 3. 直接使用最优配置训练
python scripts/train.py --config best_config_opt-bayes_20241118_153045.yaml
```

### 快速测试流程

```bash
# 小规模测试（1张GPU，3次运行）
./opt_sweep.sh configs/opt/base.yaml lr \
  -g 0 \
  -c 3 \
  --auto-collect \
  --wait

# 快速验证系统工作正常
```

### 大规模搜索流程

```bash
# 贝叶斯优化，8张GPU，每张50次 = 400次实验
./opt_sweep.sh configs/opt/base.yaml bayes \
  -g 0,1,2,3,4,5,6,7 \
  -c 50 \
  --auto-collect \
  --wait \
  -y  # 跳过确认

# 预计时间: 8-12小时（取决于每次实验时长）
# 完成后自动输出最优配置
```

## 🔧 高级选项

### 指定输出文件名

```bash
./opt_sweep.sh --collect entity/project/sweep_id \
  configs/opt/base.yaml \
  -o my_optimal_config.yaml
```

### 只输出优化器参数（不合并base.yaml）

```bash
./opt_sweep.sh --collect entity/project/sweep_id
# 不指定base.yaml，只输出opt参数
```

### 中断等待后继续

```bash
# 启动自动收集
./opt_sweep.sh configs/opt/base.yaml bayes --auto-collect --wait

# 按 Ctrl+C 中断等待

# 脚本会提示:
# 用户中断等待
# 你可以稍后运行以下命令收集结果:
#   python scripts/opt_sweep.py --collect entity/project/sweep_id

# 稍后手动收集
./opt_sweep.sh --collect entity/project/sweep_id
```

## 📝 输出文件包含的信息

生成的yaml文件包含：

1. **完整的配置** - 如果指定了base.yaml，包含所有原始配置
2. **最优参数** - opt部分已更新为找到的最优值
3. **元信息** - sweep ID、最优run ID、指标值、生成时间等
4. **可直接使用** - 可以直接用于训练

## ✅ 优势

✅ **全自动** - 一条命令，从搜索到输出最优配置  
✅ **实时监控** - 显示进度和运行状态  
✅ **Top N** - 显示前5名最优运行  
✅ **完整信息** - 包含sweep元信息便于追溯  
✅ **即用配置** - 输出的yaml可以直接用于训练  
✅ **灵活使用** - 支持自动收集或手动收集  

## 🎉 立即试用

```bash
# 最简单的完整流程（8张GPU）
./opt_sweep.sh configs/opt/base.yaml bayes \
  -e YOUR_ENTITY \
  -g 0,1,2,3,4,5,6,7 \
  -c 50 \
  --auto-collect \
  --wait

# 坐等结果，喝杯咖啡☕
# 完成后得到 best_config_*.yaml
```

Happy Optimizing! 🚀


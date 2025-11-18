# 优化器搜索脚本 - 快速测试指南

## ✅ 脚本已修复！

现在可以使用简化语法了：

```bash
./opt_sweep.sh configs/opt/base.yaml bayes -e YOUR_ENTITY -g 0,1,2,3,4,5,6,7
```

## 🧪 快速测试

### 1. 检查环境

```bash
# 确保已安装wandb
pip install wandb

# 或在虚拟环境中
conda activate your-env
pip install wandb
```

### 2. 设置环境变量

```bash
# 在 .env 文件中
export WANDB_ENTITY=viskawei-johns-hopkins-university
export GPUS=0,1,2,3,4,5,6,7
```

### 3. 测试命令（单GPU，1次运行）

```bash
./opt_sweep.sh configs/opt/base.yaml lr -e viskawei-johns-hopkins-university -g 0 -c 1 --yes
```

### 4. 完整8GPU自动化流程

```bash
./opt_sweep.sh configs/opt/base.yaml bayes \
  -e viskawei-johns-hopkins-university \
  -p opt-bayes-test \
  -g 0,1,2,3,4,5,6,7 \
  -c 50 \
  --auto-collect \
  --wait
```

## 📋 支持的所有语法

### 简化语法（推荐）

```bash
# 学习率搜索
./opt_sweep.sh configs/opt/base.yaml lr

# 优化器对比
./opt_sweep.sh configs/opt/base.yaml optimizer

# 调度器对比
./opt_sweep.sh configs/opt/base.yaml scheduler

# Plateau参数
./opt_sweep.sh configs/opt/base.yaml plateau

# 完整搜索
./opt_sweep.sh configs/opt/base.yaml full

# 贝叶斯优化
./opt_sweep.sh configs/opt/base.yaml bayes
```

### 完整语法

```bash
./opt_sweep.sh configs/opt/base.yaml bayes \
  -e YOUR_ENTITY \
  -p PROJECT_NAME \
  -g 0,1,2,3,4,5,6,7 \
  -c 50 \
  --auto-collect \
  --wait \
  -o output.yaml
```

### 收集结果

```bash
# 收集指定sweep的结果
./opt_sweep.sh --collect entity/project/sweep_id configs/opt/base.yaml

# 或只输出优化器参数
./opt_sweep.sh --collect entity/project/sweep_id
```

## 🎯 完整8GPU工作流

```bash
# 步骤1: 设置环境变量
export WANDB_ENTITY=viskawei-johns-hopkins-university

# 步骤2: 运行贝叶斯优化（8张GPU，每张50次 = 400次实验）
./opt_sweep.sh configs/opt/base.yaml bayes \
  -g 0,1,2,3,4,5,6,7 \
  -c 50 \
  --auto-collect \
  --wait

# 步骤3: 等待完成...
# 进度: 395/400 完成, 5 运行中, 0 失败

# 步骤4: 自动输出
# ✓ 最优配置已保存到: best_config_opt-bayes_20241118_153045.yaml

# 步骤5: 使用最优配置训练
# python scripts/train.py --config best_config_opt-bayes_20241118_153045.yaml
```

## 🐛 常见问题

### Q: `error: unrecognized arguments: bayes`

A: 已修复！更新后的脚本会自动将 `bayes` 转换为 `--type bayes`

### Q: `错误: 找不到wandb命令行工具`

A: 安装wandb:
```bash
pip install wandb
wandb login
```

### Q: 如何测试脚本是否正常？

A: 运行快速测试（1个GPU，1次运行）:
```bash
./opt_sweep.sh configs/opt/base.yaml lr -e YOUR_ENTITY -g 0 -c 1 --yes
```

### Q: 如何查看所有选项？

A: 
```bash
./opt_sweep.sh configs/opt/base.yaml --help
# 或
python3 scripts/opt_sweep.py --help
```

## 📊 预期输出

成功运行后，你会看到：

```
======================================================================
搜索类型: 贝叶斯优化
基础配置: configs/opt/base.yaml
Entity:   viskawei-johns-hopkins-university
Project:  opt-bayes
GPUs:     0,1,2,3,4,5,6,7
每Agent:  50 次
自动收集: 是
======================================================================

确认运行? [Y/n]: y

启动sweep...

正在创建sweep...
Creating sweep with ID: abc123def
Sweep ID: viskawei-johns-hopkins-university/opt-bayes/abc123def
查看: https://wandb.ai/viskawei-johns-hopkins-university/opt-bayes/sweeps/abc123def

正在启动 8 个agent(s)...
  → GPU 0: 启动agent...
    PID: 12345
  → GPU 1: 启动agent...
    PID: 12346
  ...

所有agent已启动!
使用 Ctrl-C 停止所有agents

等待所有运行完成...
进度: 395/400 完成, 5 运行中, 0 失败

所有运行已完成!

======================================================================
收集Sweep结果
======================================================================
最优运行
======================================================================
Run ID:   run789xyz
Run名称:  stellar-wave-42
val_mae: 0.00234

最优优化器参数:
----------------------------------------------------------------------
  type                 = AdamW
  lr                   = 0.000847
  lr_sch               = plateau
  factor               = 0.73
  patience             = 12

======================================================================
✓ 最优配置已保存到: best_config_opt-bayes_20241118_153045.yaml
======================================================================
```

## 🎉 现在可以使用了！

```bash
# 一条命令搞定所有事情
./opt_sweep.sh configs/opt/base.yaml bayes \
  -e viskawei-johns-hopkins-university \
  -g 0,1,2,3,4,5,6,7 \
  -c 50 \
  --auto-collect \
  --wait
```

Happy Experimenting! 🚀


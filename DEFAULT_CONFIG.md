# 优化器搜索 - 默认配置说明

## 🎯 新的默认值

现在脚本已设置以下默认值，可以直接使用最简单的命令：

### 默认参数

```bash
-e viskawei-johns-hopkins-university  # W&B entity
-g 1,2,3,4,5,6,7                      # 7张GPU (跳过GPU 0)
-c 50                                  # 每个agent运行50次
--auto-collect                         # 自动收集结果 (启用)
--wait                                 # 等待完成 (启用)
```

## 🚀 超简单使用

### 最简单的命令（使用所有默认值）

```bash
# 只需指定 base.yaml 和搜索类型！
./opt_sweep.sh configs/opt/base.yaml bayes
```

这等同于：

```bash
./opt_sweep.sh configs/opt/base.yaml bayes \
  -e viskawei-johns-hopkins-university \
  -g 1,2,3,4,5,6,7 \
  -c 50 \
  --auto-collect \
  --wait
```

### 使用交互式菜单（最推荐）

```bash
# 连搜索类型都不用指定！
./opt_sweep.sh configs/opt/base.yaml
```

然后选择你想要的搜索类型（比如选择 [6] 贝叶斯优化）。

## 📋 覆盖默认值

如果需要修改某些参数：

```bash
# 使用不同的GPU
./opt_sweep.sh configs/opt/base.yaml bayes -g 0,1,2,3

# 改变运行次数
./opt_sweep.sh configs/opt/base.yaml bayes -c 100

# 禁用自动收集
./opt_sweep.sh configs/opt/base.yaml bayes --no-auto-collect

# 不等待完成
./opt_sweep.sh configs/opt/base.yaml bayes --no-wait

# 使用不同的entity
./opt_sweep.sh configs/opt/base.yaml bayes -e other-entity

# 组合多个覆盖
./opt_sweep.sh configs/opt/base.yaml bayes -g 0 -c 10 --no-wait
```

## 🎯 常见使用场景

### 场景1: 完整的7GPU搜索（默认设置）

```bash
# 一条命令，自动完成所有操作
./opt_sweep.sh configs/opt/base.yaml bayes
```

**预期**:
- 在GPU 1-7上启动7个agents
- 每个agent运行50次 → 总共350次实验
- 全部完成后自动收集最优配置
- 输出: `best_config_*.yaml`

### 场景2: 快速测试（单GPU，少量运行）

```bash
./opt_sweep.sh configs/opt/base.yaml lr -g 1 -c 3
```

### 场景3: 大规模搜索（更多运行）

```bash
./opt_sweep.sh configs/opt/base.yaml bayes -c 100
# 7 × 100 = 700次实验
```

### 场景4: 使用所有8张GPU

```bash
./opt_sweep.sh configs/opt/base.yaml bayes -g 0,1,2,3,4,5,6,7
# 8 × 50 = 400次实验
```

### 场景5: 只启动sweep，不等待

```bash
./opt_sweep.sh configs/opt/base.yaml bayes --no-wait
# 启动后立即返回，稍后手动收集结果
```

## 🔧 环境变量覆盖

在 `.env` 文件中设置：

```bash
# 覆盖默认entity
WANDB_ENTITY=my-org

# 覆盖默认GPU
GPUS=0,1,2,3

# 这些会覆盖脚本中的默认值
```

## 📊 默认值的优势

✅ **命令更简洁** - 不需要每次都输入长长的参数  
✅ **适合团队** - 统一的默认配置  
✅ **快速测试** - 一条命令就能开始实验  
✅ **灵活覆盖** - 需要时仍可自定义参数  

## 🎉 快速开始示例

### 完整工作流（最简单）

```bash
# 1. 准备配置
cat configs/opt/base.yaml  # 检查配置

# 2. 运行sweep（使用所有默认值）
./opt_sweep.sh configs/opt/base.yaml bayes

# 等待提示 "确认运行? [Y/n]:"
# 按 Enter 或输入 y

# 3. 等待完成（自动）
# 进度: 345/350 完成, 5 运行中, 0 失败

# 4. 自动输出最优配置（自动）
# ✓ 最优配置已保存到: best_config_opt-bayes_20241118_153045.yaml

# 5. 使用最优配置
# 完成！
```

### 跳过确认（完全自动）

```bash
./opt_sweep.sh configs/opt/base.yaml bayes -y
# 直接开始，无需确认
```

## 📝 对比

### 之前（需要指定所有参数）

```bash
./opt_sweep.sh configs/opt/base.yaml bayes \
  -e viskawei-johns-hopkins-university \
  -p opt-bayes \
  -g 1,2,3,4,5,6,7 \
  -c 50 \
  --auto-collect \
  --wait
```

### 现在（使用默认值）

```bash
./opt_sweep.sh configs/opt/base.yaml bayes
```

**节省了 90% 的输入！** 🎉

## 🔍 查看当前默认值

```bash
./opt_sweep.sh configs/opt/base.yaml --help
```

会显示所有参数的默认值。

## 💡 小贴士

- **GPU 0** 被跳过（默认使用1-7），通常是为了保留给其他任务
- **50次运行** 是一个平衡值：足够找到好参数，又不会太慢
- **自动收集** 让你不用记sweep ID
- **等待完成** 确保你能立即得到结果

---

**现在只需要一条命令就能开始实验了！** 🚀

```bash
./opt_sweep.sh configs/opt/base.yaml bayes
```


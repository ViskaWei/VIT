# 优化器搜索 - 统一脚本方案

## 🎯 一个脚本解决所有问题

现在你只需要一个脚本 + 一个 base.yaml 就能运行所有优化器搜索实验！

## 📦 文件

```
/home/swei20/VIT/
├── opt_sweep.sh              # Bash启动脚本 ⭐
├── scripts/
│   └── opt_sweep.py          # Python核心脚本
└── configs/
    └── opt/
        └── base.yaml          # 你的基础配置
```

## 🚀 使用方法

### 方式1: 交互式菜单（最简单）

```bash
./opt_sweep.sh configs/opt/base.yaml
```

会显示菜单让你选择要运行的实验：

```
╔════════════════════════════════════════════════════════════╗
║          优化器超参数搜索 - 选择搜索类型                    ║
╚════════════════════════════════════════════════════════════╝

  [1] 学习率搜索
      └─ 搜索最优学习率 (7个值)

  [2] 优化器类型对比
      └─ 对比 Adam vs AdamW vs SGD

  [3] 学习率调度器对比
      └─ 对比不同的LR调度策略

  [4] Plateau调度器参数优化
      └─ 精细调节 factor 和 patience

  [5] 完整联合搜索
      └─ 随机搜索所有参数

  [6] 贝叶斯优化
      └─ 智能搜索最优参数组合

  [0] 退出

请选择 [0-6]:
```

### 方式2: 直接指定搜索类型

```bash
# 学习率搜索
./opt_sweep.sh configs/opt/base.yaml lr

# 优化器对比
./opt_sweep.sh configs/opt/base.yaml optimizer

# 调度器对比
./opt_sweep.sh configs/opt/base.yaml scheduler

# Plateau参数优化
./opt_sweep.sh configs/opt/base.yaml plateau

# 完整搜索
./opt_sweep.sh configs/opt/base.yaml full

# 贝叶斯优化
./opt_sweep.sh configs/opt/base.yaml bayes
```

### 方式3: 带完整选项

```bash
./opt_sweep.sh configs/opt/base.yaml optimizer \
  -e YOUR_ENTITY \
  -p my-optimizer-search \
  -g 0,1,2,3 \
  -c 50
```

## 📋 命令行选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-e, --entity` | W&B entity | `$WANDB_ENTITY` |
| `-p, --project` | W&B project | `opt-<类型>` |
| `-g, --gpus` | GPU列表 | `$GPUS` 或 `0,1,2,3` |
| `-c, --count` | 每agent运行次数 | 无限制 |
| `-y, --yes` | 跳过确认 | 交互式 |

## 🎯 6种搜索类型

| 类型 | 关键词 | 说明 | 组合数 |
|------|--------|------|--------|
| 学习率搜索 | `lr` | 7个学习率值 | 7 |
| 优化器对比 | `optimizer` | Adam/AdamW/SGD × 3个LR | 9 |
| 调度器对比 | `scheduler` | 4个调度器 × 4个LR | 16 |
| Plateau优化 | `plateau` | factor × patience | 16 |
| 完整搜索 | `full` | 随机搜索所有参数 | 自定义 |
| 贝叶斯优化 | `bayes` | 智能搜索 | 自定义 |

## 💡 使用示例

### 示例1: 使用baseline.yaml做学习率搜索

```bash
./opt_sweep.sh configs/exp/att_clp/baseline.yaml lr -e myorg -g 0,1,2,3
```

### 示例2: 交互式选择

```bash
./opt_sweep.sh configs/opt/base.yaml
# 然后选择 [1] 学习率搜索
```

### 示例3: 贝叶斯优化（推荐）

```bash
./opt_sweep.sh configs/opt/base.yaml bayes \
  -e myorg \
  -p opt-bayes-final \
  -g 0,1,2,3,4,5,6,7 \
  -c 50
```

### 示例4: 快速测试

```bash
./opt_sweep.sh configs/opt/base.yaml lr -g 0 -c 3 -y
```

## 🔄 工作流程

1. **脚本读取你的 base.yaml**
2. **根据选择的类型自动生成 sweep 配置**
3. **创建临时 sweep.yaml 文件**
4. **调用 template_sweep.sh 运行**
5. **清理临时文件**

**你不需要手动创建任何 sweep.yaml 文件！**

## ⚙️ 环境变量配置

在 `.env` 文件中设置默认值：

```bash
WANDB_ENTITY=your-org
GPUS=0,1,2,3
```

然后只需要：

```bash
./opt_sweep.sh configs/opt/base.yaml lr
```

## 📊 内置搜索配置

### 1. 学习率搜索 (`lr`)
```python
opt.lr: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
```

### 2. 优化器对比 (`optimizer`)
```python
opt.type: ['Adam', 'AdamW', 'SGD']
opt.lr: [1e-4, 1e-3, 1e-2]
```

### 3. 调度器对比 (`scheduler`)
```python
opt.lr_sch: ['plateau', 'cosine', 'step', 'none']
opt.lr: [1e-4, 5e-4, 1e-3, 5e-3]
```

### 4. Plateau优化 (`plateau`)
```python
opt.lr_sch: 'plateau'
opt.factor: [0.5, 0.7, 0.8, 0.9]
opt.patience: [5, 10, 15, 20]
opt.lr: 0.001
```

### 5. 完整搜索 (`full`)
```python
method: random
opt.type: ['Adam', 'AdamW', 'SGD']
opt.lr: log_uniform(1e-5, 1e-2)
opt.lr_sch: ['plateau', 'cosine', 'step', 'none']
opt.factor: [0.5, 0.7, 0.8, 0.9]
opt.patience: [5, 10, 15, 20]
```

### 6. 贝叶斯优化 (`bayes`)
```python
method: bayes
opt.type: 'AdamW'
opt.lr: log_uniform(1e-5, 1e-2)
opt.lr_sch: ['plateau', 'cosine']
opt.factor: uniform(0.5, 0.95)
opt.patience: int_uniform(5, 25)
```

## 🎓 推荐流程

### 新手流程

```bash
# 1. 先搜索学习率
./opt_sweep.sh configs/opt/base.yaml lr

# 2. 对比优化器
./opt_sweep.sh configs/opt/base.yaml optimizer

# 3. 对比调度器
./opt_sweep.sh configs/opt/base.yaml scheduler
```

### 高级用户

```bash
# 直接贝叶斯优化
./opt_sweep.sh configs/opt/base.yaml bayes -g 0,1,2,3,4,5,6,7 -c 50
```

## 📝 Python API

也可以直接使用Python脚本：

```bash
python3 scripts/opt_sweep.py configs/opt/base.yaml \
  --type lr \
  --entity myorg \
  --project my-lr-search \
  --gpus 0,1,2,3
```

## 🔧 自定义搜索空间

如果你想修改搜索空间，编辑 `scripts/opt_sweep.py` 中的 `SWEEP_CONFIGS` 字典：

```python
SWEEP_CONFIGS = {
    "lr": {
        "parameters": {
            "opt.lr": {
                "values": [1e-4, 1e-3, 1e-2]  # 改成你想要的值
            }
        }
    },
    # ...
}
```

## ✨ 优势

✅ **只需要一个输入** - 你的 base.yaml  
✅ **不需要创建任何 sweep.yaml** - 自动生成  
✅ **交互式菜单** - 友好的用户界面  
✅ **6种预配置实验** - 开箱即用  
✅ **完全自动化** - 一条命令搞定  
✅ **支持所有选项** - entity, project, gpus, count  

## 🎉 立即开始

```bash
# 最简单的使用方式
./opt_sweep.sh configs/opt/base.yaml

# 或者用你自己的baseline.yaml
./opt_sweep.sh configs/exp/att_clp/baseline.yaml

# 选择一个实验类型，然后坐等结果！
```

---

**总结：一个脚本 + 一个 base.yaml = 所有优化器搜索实验！** 🚀


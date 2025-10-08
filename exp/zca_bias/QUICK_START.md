# ZCA Bias 对比实验 - 快速开始

## ✅ 简化完成！使用 Experiment 类

**核心改进:**
- ✅ 使用 `Experiment` 类（参考 `src/vit.py`）
- ✅ 自动处理所有 Lightning 配置
- ✅ 指标名称自动匹配
- ✅ 支持 Wandb 日志

详见 `BUGFIX_LOG.md`

---

## 一行命令运行 🚀

```bash
cd /home/swei20/VIT && source init.sh && bash exp/zca_bias/run_bias_comparison.sh
```

## 使用选项

```bash
# 带 wandb（默认）
bash exp/zca_bias/run_bias_comparison.sh

# 不使用 wandb
bash exp/zca_bias/run_bias_comparison.sh zca_bias_comparison.yaml auto ""

# 自定义输出目录
bash exp/zca_bias/run_bias_comparison.sh zca_bias_comparison.yaml ./my_results --use_wandb
```

---

```bash
cd /home/swei20/VIT && source init.sh && bash exp/zca_bias/run_bias_comparison.sh
```

## 步骤详解

### 1. 进入项目目录
```bash
cd /home/swei20/VIT
```

### 2. 设置环境变量
```bash
source init.sh
```

这会设置：
- `$PCA_DIR` - 协方差矩阵路径
- `$TRAIN_DIR` - 训练数据路径
- `$VAL_DIR` - 验证数据路径
- `$TEST_DIR` - 测试数据路径

### 3. 运行实验
```bash
bash exp/zca_bias/run_bias_comparison.sh
```

按 `y` 确认后开始训练。

## 预期时间 ⏱️

- 完整实验（300 epochs × 2）: 约 **4-8 小时**
- 快速测试（50 epochs × 2）: 约 **1-2 小时**

## 查看结果 📊

实验完成后：

```bash
# 查看对比报告
cat results_zca_bias_comparison_*/comparison_report.md

# 示例输出：
# ==========================================
# COMPARISON RESULTS
# ==========================================
# 
# Validation Loss:
#   With bias:    0.012300
#   Without bias: 0.018900
#   Improvement:  +35.45% ✓
# 
# ✓ CONCLUSION: WITH BIAS is better - Mean centering improves performance
```

## 查看 Wandb

1. 打开 wandb 项目（项目名称在配置文件的 `project` 字段）
2. 找到两个实验：
   - `vit_model_r32` - with bias
   - `vit_model_r32_nobias` - without bias
3. 选中两个实验，对比训练曲线

## 只运行快速测试（调试用）

如果只想验证功能，不需要完整训练：

```bash
# 编辑配置文件，减少训练轮数
vim exp/zca_bias/zca_bias_comparison.yaml
# 修改 train.ep: 300 -> train.ep: 5

# 然后运行
bash exp/zca_bias/run_bias_comparison.sh
```

或者直接在 Python 脚本中修改（第 52 行附近）：

```python
# Reduce epochs for faster comparison (optional)
if 'train' in config and 'ep' in config['train']:
    config['train']['ep'] = min(config['train']['ep'], 5)  # 只训练 5 轮
```

## 只运行单个实验

```bash
cd /home/swei20/VIT

# 只测试 with bias
python exp/zca_bias/zca_bias_comparison.py \
    --config exp/zca_bias/zca_bias_comparison.yaml \
    --skip_without_bias \
    --output_dir results/test_with_bias

# 只测试 without bias  
python exp/zca_bias/zca_bias_comparison.py \
    --config exp/zca_bias/zca_bias_comparison.yaml \
    --skip_with_bias \
    --output_dir results/test_no_bias
```

## 故障排查 🔧

### 问题 1: 环境变量未设置

```bash
$ bash exp/zca_bias/run_bias_comparison.sh
ERROR: PCA_DIR not set. Please run: source init.sh
```

**解决方案:**
```bash
source init.sh
```

### 问题 2: 找不到 cov.pt 文件

```bash
FileNotFoundError: [Errno 2] No such file or directory: '.../cov.pt'
```

**解决方案:**
```bash
# 检查文件是否存在
ls -lh $PCA_DIR/cov.pt

# 如果不存在，需要先生成
python scripts/prepro/calculate_cov.py \
    --data_path $TRAIN_DIR/dataset.h5 \
    --output $PCA_DIR/cov.pt
```

### 问题 3: GPU 内存不足

```bash
RuntimeError: CUDA out of memory
```

**解决方案:** 减小 batch size
```bash
vim exp/zca_bias/zca_bias_comparison.yaml
# 修改 train.batch_size: 8192 -> 4096 或更小
```

### 问题 4: 脚本路径错误

```bash
python: can't open file '/home/swei20/VIT/test_bias_comparison.py'
```

**解决方案:** 确保在项目根目录运行
```bash
cd /home/swei20/VIT
bash exp/zca_bias/run_bias_comparison.sh
```

## 自定义配置

编辑 `exp/zca_bias/zca_bias_comparison.yaml`:

```yaml
# 修改训练轮数（减少时间）
train:
  batch_size: 8192
  ep: 50  # 从 300 减少到 50

# 修改模型大小（减少内存）
model:
  hidden_size: 16      # 从 32 减少到 16
  num_hidden_layers: 1 # 从 2 减少到 1

# 修改 ZCA 参数
warmup:
  preprocessor: zca
  r: 16  # 从 32 减少到 16（低秩维度）
```

## 预期结果

根据理论和之前的实验，使用 bias（均值中心化）应该：

✅ **验证损失** 改善 20-35%  
✅ **测试指标** 全面提升  
✅ **训练稳定性** 更好的收敛曲线  
✅ **白化效果** 数据均值接近 0

如果结果不符合预期，参考 `exp/zca_bias/README.md` 中的故障排查部分。

# num_labels 与 data.param 冲突修复说明

## 问题描述

之前在配置文件中，`model.num_labels` 和 `data.param` 存在冲突：
- 配置文件中手动指定了 `num_labels: 1` 或 `num_labels: 3`
- 但 `data.param` 可能是 `log_g`（1个参数）或 `T_eff, log_g, M_H`（3个参数）
- 这导致模型输出维度与实际标签数量不匹配

## 解决方案

### 1. 修改代码逻辑 (`src/models/builder.py`)

**修改内容：**
- 对于回归任务（`task_type: reg`），完全忽略配置文件中的 `model.num_labels`
- 自动从 `data.param` 推导出正确的 `num_labels`：
  - 单个参数（如 `log_g`）→ `num_labels = 1`
  - 多个参数（如 `T_eff, log_g, M_H`）→ `num_labels = 3`
  - 列表形式参数（如 `["T_eff", "log_g"]`）→ `num_labels = 2`
- 如果检测到冲突，打印警告信息并使用从 `data.param` 推导的值

**代码变更：**
```python
# 之前的逻辑：优先使用配置文件中的 num_labels
num_labels = int(m.get("num_labels", 1) or 1)
if task in ("reg", "regression"):
    p = d.get("param", None)
    if isinstance(p, str) and len(p) > 0:
        plist = [x.strip() for x in p.split(",") if x.strip()]
        if len(plist) >= 1:
            num_labels = len(plist)
    ...

# 修改后的逻辑：忽略配置文件，只依赖 data.param
if task in ("reg", "regression"):
    p = d.get("param", None)
    num_labels = 1  # default
    
    if isinstance(p, str) and len(p) > 0:
        plist = [x.strip() for x in p.split(",") if x.strip()]
        if len(plist) >= 1:
            num_labels = len(plist)
    elif isinstance(p, (list, tuple)) and len(p) > 0:
        num_labels = len(p)
    
    # 检测并警告冲突
    config_num_labels = m.get("num_labels")
    if config_num_labels is not None and int(config_num_labels) != num_labels:
        print(f"Warning: model.num_labels={config_num_labels} conflicts with data.param (which implies {num_labels} labels). Using {num_labels} from data.param.")
    
    # 始终使用从 data.param 推导的值
    m["num_labels"] = num_labels
```

### 2. 更新配置文件

**修改范围：**
更新了所有回归任务的配置文件（共 31 个文件），包括：

**主要配置文件：**
- `configs/opt/base.yaml`
- `configs/template/base.yaml`
- `configs/exp/att_clp/baseline.yaml`
- `configs/volta/run.yaml`
- `configs/vit.yaml`
- `configs/exp/zca/base.yaml`

**实验配置文件：**
- `configs/exp/att_clp/add_*.yaml` (5 个文件)
- `configs/exp/zca/*.yaml` (3 个文件)
- `configs/volta/*.yaml` (8 个文件)
- 其他 VIT 配置文件 (10+ 个文件)

**修改方式：**
```yaml
# 之前
model:
  name: vit
  task_type: reg
  num_labels: 1    # 或 num_labels: 3
  image_size: 4096
  ...

# 修改后
model:
  name: vit
  task_type: reg
  # num_labels is automatically derived from data.param (no need to specify)
  image_size: 4096
  ...
```

## 测试验证

创建了测试脚本验证以下场景：
1. ✓ 单参数：`param: log_g` → `num_labels = 1`
2. ✓ 多参数：`param: T_eff, log_g, M_H` → `num_labels = 3`
3. ✓ 冲突检测：`num_labels: 3` + `param: log_g` → 警告并使用 `num_labels = 1`
4. ✓ 列表形式：`param: ["T_eff", "log_g"]` → `num_labels = 2`
5. ✓ 无参数：默认 `num_labels = 1`
6. ✓ 分类任务：保持原有逻辑不变

所有测试通过！

## 使用说明

### 对于新配置文件

不再需要手动指定 `model.num_labels`，只需在 `data.param` 中指定要预测的参数：

```yaml
model:
  task_type: reg
  # num_labels 会自动推导，无需指定

data:
  param: log_g              # 自动推导 num_labels = 1
  # 或
  param: T_eff, log_g, M_H  # 自动推导 num_labels = 3
```

### 对于现有配置文件

如果配置文件中仍有 `num_labels` 指定：
- 如果值与 `data.param` 匹配：正常工作，但建议移除以避免冗余
- 如果值不匹配：会打印警告并使用从 `data.param` 推导的值

## 影响范围

- **代码变更：** 仅修改 `src/models/builder.py` 中的 `get_vit_config` 函数
- **配置变更：** 更新了 31 个回归任务配置文件
- **向后兼容：** 完全兼容，即使配置文件中仍有 `num_labels` 也能正确工作
- **分类任务：** 不受影响，仍使用配置文件中的 `num_labels`

## 总结

此修复确保了：
1. **消除冲突：** `model.num_labels` 不再与 `data.param` 冲突
2. **自动推导：** 回归任务的 `num_labels` 自动从 `data.param` 推导
3. **简化配置：** 配置文件更简洁，减少手动维护的参数
4. **错误检测：** 自动检测并警告配置不一致
5. **向后兼容：** 不破坏现有代码和工作流程


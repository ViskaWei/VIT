#!/usr/bin/env python3
"""
优化器超参数搜索脚本

用法:
    python opt_sweep.py <base.yaml> [选项]
    
示例:
    python opt_sweep.py configs/exp/att_clp/baseline.yaml --type lr
    python opt_sweep.py configs/opt/base.yaml --type optimizer --gpus 0,1,2,3
    
    # 收集已完成sweep的结果
    python opt_sweep.py --collect <entity>/<project>/<sweep_id>
"""

import argparse
import yaml
import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime

# 预定义的搜索配置模板
SWEEP_CONFIGS = {
    "lr": {
        "name": "学习率搜索",
        "description": "搜索最优学习率 (7个值)",
        "method": "grid",
        "parameters": {
            "opt.lr": {
                "values": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
            }
        }
    },
    
    "optimizer": {
        "name": "优化器类型对比",
        "description": "对比 Adam vs AdamW vs SGD",
        "method": "grid",
        "parameters": {
            "opt.type": {
                "values": ["Adam", "AdamW", "SGD"]
            },
            "opt.lr": {
                "values": [1e-4, 1e-3, 1e-2]
            }
        }
    },
    
    "scheduler": {
        "name": "学习率调度器对比",
        "description": "对比不同的LR调度策略",
        "method": "grid",
        "parameters": {
            "opt.lr_sch": {
                "values": ["plateau", "cosine", "step", "none"]
            },
            "opt.lr": {
                "values": [1e-4, 5e-4, 1e-3, 5e-3]
            }
        }
    },
    
    "plateau": {
        "name": "Plateau调度器参数优化",
        "description": "精细调节 factor 和 patience",
        "method": "grid",
        "parameters": {
            "opt.lr_sch": {
                "value": "plateau"
            },
            "opt.factor": {
                "values": [0.5, 0.7, 0.8, 0.9]
            },
            "opt.patience": {
                "values": [5, 10, 15, 20]
            },
            "opt.lr": {
                "value": 0.001
            }
        }
    },
    
    "full": {
        "name": "完整联合搜索",
        "description": "随机搜索所有参数",
        "method": "random",
        "parameters": {
            "opt.type": {
                "values": ["Adam", "AdamW", "SGD"]
            },
            "opt.lr": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2
            },
            "opt.lr_sch": {
                "values": ["plateau", "cosine", "step", "none"]
            },
            "opt.factor": {
                "values": [0.5, 0.7, 0.8, 0.9]
            },
            "opt.patience": {
                "values": [5, 10, 15, 20]
            }
        }
    },
    
    "bayes": {
        "name": "贝叶斯优化",
        "description": "智能搜索最优参数组合",
        "method": "bayes",
        "parameters": {
            "opt.type": {
                "value": "AdamW"
            },
            "opt.lr": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2
            },
            "opt.lr_sch": {
                "values": ["plateau", "cosine"]
            },
            "opt.factor": {
                "distribution": "uniform",
                "min": 0.5,
                "max": 0.95
            },
            "opt.patience": {
                "distribution": "int_uniform",
                "min": 5,
                "max": 25
            }
        }
    }
}


def create_sweep_config(base_yaml_path, sweep_type, metric_name="val_mae", metric_goal="minimize"):
    """创建sweep配置"""
    
    if sweep_type not in SWEEP_CONFIGS:
        raise ValueError(f"未知的搜索类型: {sweep_type}")
    
    template = SWEEP_CONFIGS[sweep_type]
    
    # 获取base.yaml的绝对路径
    base_yaml_abs = os.path.abspath(base_yaml_path)
    
    sweep_config = {
        "program": "scripts/sweep.py",
        "method": template["method"],
        "metric": {
            "name": metric_name,
            "goal": metric_goal
        },
        "parameters": {
            "vit_config": {
                "value": base_yaml_abs
            }
        }
    }
    
    # 添加搜索参数
    sweep_config["parameters"].update(template["parameters"])
    
    return sweep_config


def show_menu():
    """显示交互式菜单"""
    print("\n" + "="*70)
    print("          优化器超参数搜索 - 选择搜索类型")
    print("="*70)
    print()
    
    for i, (key, config) in enumerate(SWEEP_CONFIGS.items(), 1):
        print(f"  [{i}] {config['name']}")
        print(f"      └─ {config['description']}")
        print()
    
    print(f"  [0] 退出")
    print()
    print("="*70)
    
    while True:
        try:
            choice = input("请选择 [0-{}]: ".format(len(SWEEP_CONFIGS)))
            choice_num = int(choice)
            if 0 <= choice_num <= len(SWEEP_CONFIGS):
                return choice_num
            else:
                print(f"请输入 0-{len(SWEEP_CONFIGS)} 之间的数字")
        except (ValueError, KeyboardInterrupt):
            print("\n退出")
            return 0


def collect_sweep_results(sweep_path, output_file=None, base_yaml=None):
    """收集sweep结果并生成最优参数配置"""
    
    try:
        import wandb
    except ImportError:
        print("错误: 需要安装wandb库")
        print("运行: pip install wandb")
        return 1
    
    print("\n" + "="*70)
    print("收集Sweep结果")
    print("="*70)
    print(f"Sweep: {sweep_path}")
    
    # 初始化API
    api = wandb.Api()
    
    try:
        sweep = api.sweep(sweep_path)
    except Exception as e:
        print(f"\n错误: 无法访问sweep: {e}")
        return 1
    
    # 获取sweep信息
    print(f"\n搜索方法: {sweep.config.get('method', 'unknown')}")
    print(f"优化指标: {sweep.config.get('metric', {}).get('name', 'unknown')}")
    print(f"优化目标: {sweep.config.get('metric', {}).get('goal', 'unknown')}")
    
    # 获取所有运行
    runs = list(sweep.runs)
    
    if not runs:
        print("\n错误: 该sweep没有任何运行记录")
        return 1
    
    print(f"\n总运行数: {len(runs)}")
    
    # 统计运行状态
    finished = sum(1 for r in runs if r.state == "finished")
    running = sum(1 for r in runs if r.state == "running")
    failed = sum(1 for r in runs if r.state in ["failed", "crashed"])
    
    print(f"  - 已完成: {finished}")
    print(f"  - 运行中: {running}")
    print(f"  - 失败:   {failed}")
    
    if finished == 0:
        print("\n警告: 还没有完成的运行")
        return 1
    
    # 找到最优运行
    metric_name = sweep.config.get('metric', {}).get('name', 'val_mae')
    goal = sweep.config.get('metric', {}).get('goal', 'minimize')
    
    best_run = sweep.best_run()
    
    if not best_run:
        print("\n错误: 无法找到最优运行")
        return 1
    
    print("\n" + "="*70)
    print("最优运行")
    print("="*70)
    print(f"Run ID:   {best_run.id}")
    print(f"Run名称:  {best_run.name}")
    print(f"{metric_name}: {best_run.summary.get(metric_name, 'N/A')}")
    
    # 提取最优参数
    best_config = best_run.config
    
    # 过滤出opt相关的参数
    opt_params = {}
    for key, value in best_config.items():
        if key.startswith('opt.'):
            param_name = key.replace('opt.', '')
            opt_params[param_name] = value
    
    print("\n最优优化器参数:")
    print("-" * 70)
    for key, value in sorted(opt_params.items()):
        print(f"  {key:20s} = {value}")
    
    # 生成输出配置
    if base_yaml and os.path.exists(base_yaml):
        # 读取base配置
        with open(base_yaml, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # 更新opt参数
        if 'opt' not in base_config:
            base_config['opt'] = {}
        
        for key, value in opt_params.items():
            base_config['opt'][key] = value
        
        output_config = base_config
    else:
        # 只输出opt参数
        output_config = {'opt': opt_params}
    
    # 添加元信息
    output_config['_meta'] = {
        'sweep_id': sweep_path,
        'best_run_id': best_run.id,
        'best_run_name': best_run.name,
        f'best_{metric_name}': best_run.summary.get(metric_name),
        'generated_at': datetime.now().isoformat(),
        'total_runs': len(runs),
        'finished_runs': finished
    }
    
    # 确定输出文件名
    if not output_file:
        sweep_id = sweep_path.split('/')[-1]
        output_file = f"best_config_{sweep_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    
    # 写入文件
    with open(output_file, 'w') as f:
        yaml.dump(output_config, f, default_flow_style=False, allow_unicode=True)
    
    print("\n" + "="*70)
    print(f"✓ 最优配置已保存到: {output_file}")
    print("="*70)
    
    # 显示Top 5运行
    print("\nTop 5 运行:")
    print("-" * 70)
    
    # 获取所有已完成的运行并排序
    finished_runs = [r for r in runs if r.state == "finished" and metric_name in r.summary]
    
    if goal == "minimize":
        finished_runs.sort(key=lambda r: r.summary.get(metric_name, float('inf')))
    else:
        finished_runs.sort(key=lambda r: r.summary.get(metric_name, float('-inf')), reverse=True)
    
    for i, run in enumerate(finished_runs[:5], 1):
        metric_value = run.summary.get(metric_name, 'N/A')
        lr = run.config.get('opt.lr', 'N/A')
        opt_type = run.config.get('opt.type', 'N/A')
        print(f"{i}. {run.name[:30]:30s} | {metric_name}={metric_value:.6f} | lr={lr} | opt={opt_type}")
    
    return 0


def run_sweep(base_yaml, sweep_type, entity, project, gpus, count=None, interactive=True, 
              auto_collect=False, wait_completion=False):
    """运行sweep"""
    
    # 验证base.yaml存在
    if not os.path.exists(base_yaml):
        print(f"错误: 找不到文件 {base_yaml}")
        return 1
    
    # 创建sweep配置
    sweep_config = create_sweep_config(base_yaml, sweep_type)
    
    # 创建临时sweep配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sweep_config, f, default_flow_style=False, allow_unicode=True)
        temp_sweep_file = f.name
    
    sweep_id = None
    
    try:
        # 显示配置信息
        print("\n" + "="*70)
        print(f"搜索类型: {SWEEP_CONFIGS[sweep_type]['name']}")
        print(f"基础配置: {base_yaml}")
        print(f"Entity:   {entity}")
        print(f"Project:  {project}")
        print(f"GPUs:     {gpus}")
        if count:
            print(f"每Agent:  {count} 次")
        if auto_collect:
            print(f"自动收集: 是")
        print("="*70)
        
        if interactive:
            confirm = input("\n确认运行? [Y/n]: ")
            if confirm.lower() not in ['', 'y', 'yes']:
                print("已取消")
                return 0
        
        # 构建命令
        script_dir = Path(__file__).parent.parent
        sweep_script = script_dir / "template_sweep.sh"
        
        cmd = [
            str(sweep_script),
            "-e", entity,
            "-p", project,
            "-c", temp_sweep_file,
            "-g", gpus
        ]
        
        if count:
            cmd.extend(["--count", str(count)])
        
        print("\n启动sweep...")
        print(f"命令: {' '.join(cmd)}\n")
        
        # 运行命令并捕获输出以获取sweep ID
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # 实时显示输出并查找sweep ID
        for line in process.stdout:
            print(line, end='')
            # 查找sweep ID
            if 'Sweep ID:' in line:
                # 格式: "Sweep ID: entity/project/sweep_id"
                parts = line.split('Sweep ID:')
                if len(parts) > 1:
                    sweep_id = parts[1].strip()
        
        process.wait()
        
        if process.returncode != 0:
            return process.returncode
        
        # 如果需要自动收集结果
        if auto_collect and sweep_id:
            print("\n" + "="*70)
            print("准备收集结果...")
            print("="*70)
            
            if wait_completion:
                print("\n等待所有运行完成...")
                print("(你可以按 Ctrl+C 取消等待，稍后手动收集结果)")
                
                try:
                    import wandb
                    api = wandb.Api()
                    
                    while True:
                        try:
                            sweep = api.sweep(sweep_id)
                            runs = list(sweep.runs)
                            
                            finished = sum(1 for r in runs if r.state == "finished")
                            running = sum(1 for r in runs if r.state == "running")
                            failed = sum(1 for r in runs if r.state in ["failed", "crashed"])
                            
                            print(f"\r进度: {finished}/{len(runs)} 完成, {running} 运行中, {failed} 失败", end='')
                            
                            if running == 0 and finished > 0:
                                print("\n\n所有运行已完成!")
                                break
                            
                            time.sleep(30)  # 每30秒检查一次
                            
                        except KeyboardInterrupt:
                            print("\n\n用户中断等待")
                            print("你可以稍后运行以下命令收集结果:")
                            print(f"  python scripts/opt_sweep.py --collect {sweep_id}")
                            return 0
                        
                except ImportError:
                    print("\n警告: 需要安装wandb库才能自动等待完成")
                    print("你可以稍后手动收集结果:")
                    print(f"  python scripts/opt_sweep.py --collect {sweep_id}")
                    return 0
            
            # 收集结果
            print("\n")
            output_file = f"best_config_{project}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            return collect_sweep_results(sweep_id, output_file, base_yaml)
        
        elif sweep_id:
            print("\n" + "="*70)
            print("Sweep已启动!")
            print("="*70)
            print(f"\n完成后运行以下命令收集最优配置:")
            print(f"  python scripts/opt_sweep.py --collect {sweep_id}")
            print(f"\n或者:")
            print(f"  ./opt_sweep.sh --collect {sweep_id}")
            print()
        
        return 0
        
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_sweep_file)
        except:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="优化器超参数搜索脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式菜单
  python opt_sweep.py configs/opt/base.yaml
  
  # 直接运行学习率搜索
  python opt_sweep.py configs/opt/base.yaml --type lr
  
  # 运行优化器对比
  python opt_sweep.py configs/opt/base.yaml --type optimizer --gpus 0,1,2,3
  
  # 贝叶斯优化并自动收集结果
  python opt_sweep.py configs/opt/base.yaml --type bayes --gpus 0,1,2,3,4,5,6,7 --count 50 --auto-collect
  
  # 收集已完成sweep的结果
  python opt_sweep.py --collect entity/project/sweep_id
  
可用的搜索类型:
  lr         - 学习率搜索
  optimizer  - 优化器类型对比
  scheduler  - 调度器对比
  plateau    - Plateau参数优化
  full       - 完整联合搜索
  bayes      - 贝叶斯优化
        """
    )
    
    parser.add_argument("base_yaml", nargs='?', help="基础配置文件路径 (base.yaml)")
    parser.add_argument("--type", "-t", choices=list(SWEEP_CONFIGS.keys()),
                        help="搜索类型 (不指定则显示菜单)")
    parser.add_argument("--entity", "-e", 
                        default=os.environ.get("WANDB_ENTITY", "viskawei-johns-hopkins-university"),
                        help="W&B entity (默认: viskawei-johns-hopkins-university)")
    parser.add_argument("--project", "-p",
                        help="W&B project名称 (默认: opt-<type>)")
    parser.add_argument("--gpus", "-g",
                        default=os.environ.get("GPUS", "1,2,3,4,5,6,7"),
                        help="GPU列表 (默认: 1,2,3,4,5,6,7)")
    parser.add_argument("--count", "-c", type=int,
                        default=50,
                        help="每个agent运行次数 (默认: 50)")
    parser.add_argument("--metric", 
                        default="val_mae",
                        help="优化指标名称 (默认: val_mae)")
    parser.add_argument("--goal",
                        choices=["minimize", "maximize"],
                        default="minimize",
                        help="优化目标 (默认: minimize)")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="跳过确认")
    parser.add_argument("--auto-collect", action="store_true",
                        default=True,
                        help="运行完成后自动收集最优配置 (默认: 启用)")
    parser.add_argument("--no-auto-collect", action="store_false", dest="auto_collect",
                        help="禁用自动收集")
    parser.add_argument("--wait", action="store_true",
                        default=True,
                        help="等待所有运行完成 (默认: 启用)")
    parser.add_argument("--no-wait", action="store_false", dest="wait",
                        help="不等待运行完成")
    parser.add_argument("--collect", metavar="SWEEP_ID",
                        help="收集指定sweep的最优配置 (格式: entity/project/sweep_id)")
    parser.add_argument("--output", "-o",
                        help="输出文件名 (默认: best_config_<sweep_id>_<timestamp>.yaml)")
    
    args = parser.parse_args()
    
    # 收集结果模式
    if args.collect:
        if not args.base_yaml:
            print("提示: 未指定base.yaml，将只输出优化器参数")
            base_yaml = None
        else:
            base_yaml = args.base_yaml
        
        return collect_sweep_results(args.collect, args.output, base_yaml)
    
    # 运行sweep模式
    if not args.base_yaml:
        parser.print_help()
        return 1
    
    # 验证entity (现在有默认值，所以不会失败)
    if not args.entity:
        print("错误: 必须指定 --entity 或设置 WANDB_ENTITY 环境变量")
        print("提示: 默认值已设置为 viskawei-johns-hopkins-university")
        return 1
    
    # 交互式菜单或直接运行
    if args.type:
        sweep_type = args.type
    else:
        choice = show_menu()
        if choice == 0:
            return 0
        sweep_type = list(SWEEP_CONFIGS.keys())[choice - 1]
    
    # 设置项目名称
    if not args.project:
        args.project = f"opt-{sweep_type}"
    
    # 运行sweep
    return run_sweep(
        args.base_yaml,
        sweep_type,
        args.entity,
        args.project,
        args.gpus,
        args.count,
        interactive=not args.yes,
        auto_collect=args.auto_collect,
        wait_completion=args.wait
    )


if __name__ == "__main__":
    sys.exit(main())


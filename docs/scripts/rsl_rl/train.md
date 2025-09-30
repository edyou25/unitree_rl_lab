# train.py 代码解读文档

## 概述

`train.py` 是基于 Isaac Lab 和 RSL-RL 框架的强化学习训练脚本，主要用于训练机器人控制策略。该脚本集成了 Isaac Sim 物理仿真器、Gymnasium 环境接口和 RSL-RL 算法库。

## 文件结构分析

### 1. import

- `AppLauncher` 用于启动 Isaac Sim 仿真器
- `cli_args` 处理命令行参数


### 2. 参数

- `--video`: 是否录制训练视频
- `--video_length`: 视频长度（步数）
- `--video_interval`: 录制视频的间隔
- `--num_envs`: 并行仿真环境数量
- `--task`: 任务名称
- `--seed`: 随机种子
- `--max_iterations`: 最大训练迭代次数
- `--distributed`: 是否启用分布式训练

### 3. Isaac Sim 启动器配置

启动 Isaac Sim 仿真应用

```python
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

### 4. RSL-RL 版本检查

```python
RSL_RL_VERSION = "2.3.1" ## required >= 2.3.3
```

实际版本：
```python
>>> print(metadata.version("rsl-rl-lib"))
```

```sheel
2.3.3
```

### 5. PyTorch 后端优化

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
```

**功能说明：**
- 启用 TensorFloat-32 (TF32) 以提高训练速度
- 关闭确定性计算以提升性能
- 关闭 cuDNN 基准测试以节省内存

### 6. 主训练函数

```python
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
```

**功能说明：**
- 使用 Hydra 装饰器管理配置
- 支持多种环境配置类型：
  - `ManagerBasedRLEnvCfg`: 基于管理器的 RL 环境
  - `DirectRLEnvCfg`: 直接 RL 环境
  - `DirectMARLEnvCfg`: 直接多智能体 RL 环境

#### 6.1 配置覆盖

```python
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
agent_cfg.max_iterations = (
    args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
)
```

**功能说明：**
- 用命令行参数覆盖配置文件设置
- 灵活调整环境数量和训练迭代次数

#### 6.2 种子设置

```python
env_cfg.seed = agent_cfg.seed
env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
```

**功能说明：**
- 统一设置环境和智能体的随机种子
- 配置仿真设备（CPU/GPU）

#### 6.3 分布式训练配置

```python
if args_cli.distributed:
    env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    agent_cfg.device = f"cuda:{app_launcher.local_rank}"
    
    seed = agent_cfg.seed + app_launcher.local_rank
    env_cfg.seed = seed
    agent_cfg.seed = seed
```

**功能说明：**
- 为每个 GPU 分配独立的设备
- 为不同进程设置不同的随机种子以增加多样性

#### 6.4 日志目录配置

```python
log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
log_root_path = os.path.abspath(log_root_path)
log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if agent_cfg.run_name:
    log_dir += f"_{agent_cfg.run_name}"
log_dir = os.path.join(log_root_path, log_dir)
```

**功能说明：**
- 创建层级化的日志目录结构
- 使用时间戳确保每次运行的唯一性
- 支持自定义运行名称

#### 6.5 环境创建和配置

```python
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

if isinstance(env.unwrapped, DirectMARLEnv):
    env = multi_agent_to_single_agent(env)
```

**功能说明：**
- 使用 Gymnasium 接口创建仿真环境
- 根据是否录制视频设置渲染模式
- 支持多智能体环境转换为单智能体

#### 6.6 检查点恢复

```python
if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
```

**功能说明：**
- 支持从检查点恢复训练
- 支持知识蒸馏算法的模型加载

#### 6.7 视频录制配置

```python
if args_cli.video:
    video_kwargs = {
        "video_folder": os.path.join(log_dir, "videos", "train"),
        "step_trigger": lambda step: step % args_cli.video_interval == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
```

**功能说明：**
- 配置视频录制参数
- 按指定间隔触发录制
- 使用 Gymnasium 的 RecordVideo 包装器

#### 6.8 环境包装

```python
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
```

**功能说明：**
- 使用 RSL-RL 专用的环境包装器
- 支持动作裁剪功能

#### 6.9 训练器创建和配置

```python
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
runner.add_git_repo_to_log(__file__)
```

**功能说明：**
- 创建 RSL-RL 的在线策略训练器
- 记录 Git 仓库状态用于实验追踪

#### 6.10 配置保存

```python
dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
export_deploy_cfg(env.unwrapped, log_dir)
```

**功能说明：**
- 保存环境和智能体配置为 YAML 和 Pickle 格式
- 导出部署配置
- 复制环境配置文件到日志目录

#### 6.11 训练执行

```python
runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
```

**功能说明：**
- 开始强化学习训练
- `init_at_random_ep_len=True` 表示从随机回合长度开始初始化

## 关键特性

### 1. 多环境支持
- 支持基于管理器和直接控制的环境
- 支持单智能体和多智能体环境

### 2. 分布式训练
- 支持多GPU/多节点训练
- 自动处理设备分配和种子设置

### 3. 实验管理
- 完整的日志记录系统
- Git 状态追踪
- 配置文件保存和恢复

### 4. 视频录制
- 训练过程可视化
- 可配置的录制间隔和长度

### 5. 模型恢复
- 支持从检查点恢复训练
- 支持知识蒸馏模式

## 使用示例

```bash
# 基本训练
python train.py --task Isaac-Velocity-Flat-Anymal-D-v0

# 分布式训练
python train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --distributed

# 录制视频训练
python train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --video

# 自定义参数训练
python train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 4096 --max_iterations 1000
```

## 依赖项

- Isaac Lab: 物理仿真和机器人环境
- RSL-RL: 强化学习算法库
- Gymnasium: 环境接口标准
- Hydra: 配置管理系统
- PyTorch: 深度学习框架

## 注意事项

1. 需要先启动 Isaac Sim 仿真器
2. 分布式训练需要 RSL-RL 版本 ≥ 2.3.1
3. 录制视频会自动启用摄像头功能
4. 所有配置都会被保存到日志目录中
5. 支持通过命令行参数覆盖配置文件设置
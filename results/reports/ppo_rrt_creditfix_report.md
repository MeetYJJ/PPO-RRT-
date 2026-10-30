# PPO-RRT* 真实交互 credit assignment 修复报告

本轮将 `ppo_rrt_experiment.py` 迁移到新的 `DecisionTransition`。每个策略动作控制最多 10 次 RRT* 扩展；动作在窗口开始时从 Actor 分布采样一次，并立即保存 action、old log probability 和 old value。窗口内每次采样都使用同一个动作对应的采样器。窗口结束后根据实际窗口事件写入一条 transition。

## 修复内容

原错误是回合结束后重新生成动作并构造 transition，导致训练动作不一定是控制 RRT* 的动作。现在动作在 `WindowController._start()` 采样，在 `sample()` 中直接执行，在 `after_iteration()` 中根据实际扩展数、碰撞检查数、首条可行路径和路径改善写入 rollout。

窗口 transition 使用实际 `duration`，PPO GAE 使用 `gamma ** duration` 的窗口折扣；真实终止窗口不 bootstrap，窗口截断使用下一状态价值。每个窗口只写入一次 transition。

## 实际运行

命令：

```bash
source /home/nvidia/miniconda3/etc/profile.d/conda.sh
conda activate visionyolo
cd ppo_rrtstar_validation
python -m pytest tests -q
python scripts/ppo_rrt_experiment.py
```

测试结果：`7 passed`。训练 smoke 运行 6 轮，每轮 simple/medium/hard 三个真实 RRT* episode；共写入 92 个窗口 transition。窗口持续时间分布实际包含少于 10 次的提前终止窗口，所有窗口 duration 均在 1--10 之间。模型保存为 `models/ppo_rrt_planner_creditfix.pt`。

训练日志中 policy loss、value loss、entropy 和 mean duration 均为有限值；策略熵约 1.384，说明 smoke 阶段仍接近四动作均匀分布，不能解释为已经学到稳定策略。

## 输出

- 原始窗口 CSV：`results/raw/ppo_rrt_creditfix_smoke.csv`
- 模型：`models/ppo_rrt_planner_creditfix.pt`
- 旧结果 `ppo_planner_comparison.csv` 和旧 checkpoint 均未覆盖。

本轮证明的是动作—采样窗口—奖励—GAE—PPO 更新—保存加载链路已经连通，不是 PPO 性能结论。进入正式训练前仍应增加窗口级 ratio 审计、动作执行日志和独立评估脚本。

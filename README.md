# PPO 引导 SE(3)-RRT* 验证工程

纯 Python 几何仿真，支持 NumPy/PyTorch 等可选依赖；核心首版实现仅依赖标准库。状态为位置加单位四元数，距离 `sqrt(||dp||²+(lR*theta)²)`，`lR=0.5`。长方体包络用最大半径做保守球近似，边按插值离散检查。

运行：
```
cd ppo_rrtstar_validation
python -m pytest tests -v
python scripts/train_ppo.py
python scripts/run_benchmark.py
python scripts/analyze_results.py
```

benchmark.csv 为实际运行数据；当前快速配置使用 3 难度×5 算法×3 种子。PPO 接口和采样模式已独立，后续可替换为 Torch PPO 更新；报告不将失败路径计作零代价。

在未安装 pytest 的环境中，可运行 `python -m compileall src scripts` 和 README 中的 benchmark；安装 requirements 后执行完整 pytest。完整实验配置给出 50 seeds，但运行时间取决于机器，快速配置用于阻断性验证。

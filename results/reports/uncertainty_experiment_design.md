# 不确定性感知实验接口设计

每个场景记录 `predicted_pose`, `prediction_error_se3`, `covariance_6x6`, `uncertainty_level`。规划器只使用预测位姿和协方差，不访问真实误差；真实误差仅用于评估终端误差和鲁棒性。所有方法保持相同碰撞判定、终端容差和预算。

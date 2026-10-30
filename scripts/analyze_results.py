import csv,collections,os
rows=list(csv.DictReader(open('results/raw/benchmark.csv'))); os.makedirs('results/reports',exist_ok=True)
with open('results/reports/实验结果分析.md','w') as f:
 f.write('# 实验结果分析\n\n本报告由实际运行的 benchmark.csv 生成。\n\n')
 for a in sorted(set(r['algorithm'] for r in rows)):
  x=[r for r in rows if r['algorithm']==a]; s=sum(int(r['success']) for r in x)/len(x); f.write(f'- {a}: 成功率 {s:.3f}，样本数 {len(x)}。\n')
 f.write('\n标准 RRT* 保留采样、扩展、碰撞、父节点选择和重连流程；PPO（当前轻量验证实现）仅选择采样模式。结果不预设优劣，完整结论需结合更多种子与置信区间。\n')
 print('report written')

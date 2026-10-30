import csv,os,statistics
import matplotlib.pyplot as plt
rows=list(csv.DictReader(open('results/raw/baseline_validation.csv'))); algs=['rrtstar','fixed_mixed','soft_adaptive','faroni_inspired']; labels=['RRT*','Fixed mixed','Soft adaptive','Faroni-inspired']
os.makedirs('results/figures',exist_ok=True)
for diff in ['simple','medium','hard','all']:
 x=[]
 for a in algs:
  z=[r for r in rows if r['algorithm']==a and (diff=='all' or r['difficulty']==diff)]; x.append(sum(int(r['success']) for r in z)/len(z))
 plt.figure(figsize=(7,4)); plt.bar(labels,x); plt.ylim(0,1); plt.ylabel('success rate'); plt.title(diff); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig('results/figures/baseline_'+diff+'.png',dpi=160); plt.close()
print('baseline figures generated')

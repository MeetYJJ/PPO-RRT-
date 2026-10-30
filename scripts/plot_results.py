import csv,os,math
import matplotlib.pyplot as plt
rows=list(csv.DictReader(open('results/raw/benchmark.csv'))); os.makedirs('results/figures',exist_ok=True)
algs=sorted(set(r['algorithm'] for r in rows)); rates=[sum(int(r['success']) for r in rows if r['algorithm']==a)/sum(1 for r in rows if r['algorithm']==a) for a in algs]
plt.figure(figsize=(7,4)); plt.bar(algs,rates); plt.ylabel('success rate'); plt.ylim(0,1); plt.xticks(rotation=25); plt.tight_layout(); plt.savefig('results/figures/success_rate.png',dpi=150); plt.close()
for key,label in [('cost','path cost'),('extensions','extensions')]:
 vals=[]
 for a in algs:
  x=[float(r[key]) for r in rows if r['algorithm']==a and r[key]!='nan']; vals.append(sum(x)/len(x) if x else float('nan'))
 plt.figure(figsize=(7,4)); plt.bar(algs,vals); plt.ylabel(label); plt.xticks(rotation=25); plt.tight_layout(); plt.savefig('results/figures/'+key+'.png',dpi=150); plt.close()
print('figures written')

"""Independent-seed learning probe; never overwrites an existing run."""
import sys, json, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ppo_rrt_experiment import WindowController
from src.environment import Env
from src.collision_checker import CollisionChecker
from src.rrt_star import RRTStar
from src.rollout import RolloutBuffer
from src.ppo_torch import PPOUpdater
from src.metrics import write_csv

def episode(agent, difficulty, seed, uniform=False):
    env=Env.demo(difficulty,seed)
    buffer=RolloutBuffer()
    controller=WindowController(env,agent,buffer)
    planner=RRTStar(env,CollisionChecker(env),seed,max_iter=120,
                    goal_bias=0, sampler=controller.sample,
                    iteration_callback=controller.after_iteration)
    start=time.perf_counter()
    planner.run()
    elapsed=time.perf_counter()-start
    assert sum(t.duration for t in buffer.data)==len(planner.history)
    with torch.no_grad():
        states=torch.tensor([t.obs for t in buffer.data],dtype=torch.float32)
        logits,_=agent.net(states)
        lp=torch.distributions.Categorical(logits=logits).log_prob(torch.tensor([t.action for t in buffer.data]))
        error=float((lp-torch.tensor([t.log_prob for t in buffer.data])).abs().max())
    assert error<1e-5
    return buffer.data,dict(difficulty=difficulty,seed=seed,success=int(planner.best is not None),
                           seconds=elapsed,attempts=len(planner.history),ratio_log_error=error)

def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    out=Path(__file__).resolve().parents[1]/'results'/'learning_probe_v1'
    out.mkdir(exist_ok=False)
    agent=PPOUpdater(9)
    reference=PPOUpdater(9)
    reference.net.load_state_dict(agent.net.state_dict())
    logs=[]
    for update in range(30):
        transitions=[]
        for difficulty in ['simple','medium','hard']:
            batch,_=episode(agent,difficulty,1000+update)
            transitions.extend(batch)
        stats=agent.update(transitions,epochs=4)
        logs.append(dict(stats,update=update,transitions=len(transitions)))
        print(update,stats,flush=True)
    agent.save(str(out/'trained.pt'))
    loaded=PPOUpdater(9)
    loaded.load(str(out/'trained.pt'))
    rows=[]
    for difficulty in ['simple','medium','hard']:
        for seed in range(2000,2010):
            for name,model in [('untrained',reference),('trained',loaded)]:
                _,row=episode(model,difficulty,seed)
                rows.append(dict(row,algorithm=name))
    write_csv(logs,str(out/'training.csv'))
    write_csv(rows,str(out/'evaluation.csv'))
    summary={name:sum(r['success'] for r in rows if r['algorithm']==name)/30 for name in ['untrained','trained']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(summary)

if __name__=='__main__':
    main()

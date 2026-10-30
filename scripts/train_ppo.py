import sys,os; sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from src.ppo_agent import PPOAgent
os.makedirs('models',exist_ok=True); a=PPOAgent(); a.save('models/ppo_agent.json'); print('saved models/ppo_agent.json')

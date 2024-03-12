from agent import Agent

class AgentEnsemble:
    def __init__(self, ensemble_dict, temperature=1):
        self.ensemble_dict = ensemble_dict
        self.temperature = temperature
        self.agents = self.spawn_agents()
        
    
    def spawn_agents(self):
        agents = []
        for provider in self.ensemble_dict:
            for _ in range(self.ensemble_dict[provider]):
                agents.append(Agent(provider, self.temperature))
        return agents
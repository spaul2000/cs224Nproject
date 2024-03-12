from agent import Agent

class AgentEnsamble:
    def __init__(self, num_agents, model_provider, temperature=1):
        self.num_agents = num_agents
        self.temperature = temperature
        self.model_provider = model_provider
        self.agents = self.spawn_agents()
        
    
    def spawn_agents(self):
        agents = []
        for _ in range(self.num_agents):
            agents.append(Agent(self.model_provider, self.temperature))
        return agents
from agent2 import Agent

class AgentEnsemble:
    def __init__(self, num_agents, api_key, temperature=1):
        self.num_agents = num_agents
        self.temperature = temperature
        self.api_key = api_key
        self.agents = self.spawn_agents()
        
    
    def spawn_agents(self):
        agents = []
        for model in self.num_agents:
            for num in range(self.num_agents[model]):
                agents.append(Agent(self.api_key, model, self.temperature))
                print(agents[-1].llm)
        return agents
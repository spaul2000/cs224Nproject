from agent import Agent

class AgentEnsamble:
    def __init__(self, num_agents, model_provider, model_type, api_key, temperature=1):
        self.model_type = model_type
        self.num_agents = num_agents
        self.temperature = temperature
        self.api_key = api_key
        self.model_provider = model_provider
        self.agents = self.spawn_agents()
        
    
    def spawn_agents(self):
        agents = []
        for _ in range(self.num_agents):
            agents.append(Agent(self.api_key, 'Llama', self.model_type, self.temperature))
        return agents
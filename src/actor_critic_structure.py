

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):

    # shared network observations are local not centralized.

    def __init__(self, obs_dim, n_agents, action_dim, hidden=128):
        super().__init__()

        self.n_agents = n_agents

        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_agents, hidden),
            nn.Tanh(),

            nn.Linear(hidden, hidden),
            nn.Tanh(),

            nn.Linear(hidden, action_dim)
        )

    def forward(self, obs, agent_id_onehot):
        # Combine local observation with the identity of the agent
        x = torch.cat([obs, agent_id_onehot], dim=-1)

        # Network outputs one logit for each possible action
        logits = self.net(x)

        # Convert into a categorical probability
        return Categorical(logits=logits)


class CentralizedCritic(nn.Module):
    #centralized critic, share4d reward
    def __init__(self, global_state_dim, n_agents, hidden=128):
        super().__init__()

        self.n_agents = n_agents

        self.net = nn.Sequential(
            nn.Linear(global_state_dim + n_agents, hidden),
            nn.Tanh(),

            nn.Linear(hidden, hidden),
            nn.Tanh(),

            nn.Linear(hidden, 1)
        )

    def forward(self, global_state, agent_id_onehot):
        # Combine global state with the agent's identity
        x = torch.cat([global_state, agent_id_onehot], dim=-1)

        # Output a single value estimate
        return self.net(x).squeeze(-1)
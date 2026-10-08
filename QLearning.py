import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DECLINE, APPROVE = 0, 1


class OfflineBuffer:
    """Offline data containing BOTH actions for every applicant.

    The raw loan file only contains loans that were funded (action = approve).
    To teach the model what "decline" is worth, each applicant is stored twice:
      - APPROVE : reward = -1 if the loan defaulted, else net interest margin
      - DECLINE : reward = 0 (no loan, no profit, no loss) - same as the environment
    Applicants are independent of each other, so each row is a one-step
    (contextual-bandit) transition: done = 1 and gamma = 0 when training.
    """

    def __init__(self, capacity, state_dim):
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.ones(capacity, dtype=np.float32)
        self.size = 0
        self.capacity = capacity

    def populate(self, states, defaults, repo_rates, assumed_apr=0.18, decline_reward=0.0):
        n = len(states)
        if 2 * n > self.capacity:
            raise ValueError(f"Buffer capacity {self.capacity} is too small; need {2 * n}.")
        defaults = np.asarray(defaults)
        repo_rates = np.asarray(repo_rates, dtype=np.float32)

        approve_reward = np.where(defaults == 1, -1.0, (assumed_apr - repo_rates) * 3.5)

        # first half: APPROVE, second half: DECLINE
        self.states[:n] = states
        self.actions[:n] = APPROVE
        self.rewards[:n] = approve_reward
        self.states[n:2 * n] = states
        self.actions[n:2 * n] = DECLINE
        self.rewards[n:2 * n] = decline_reward
        self.next_states[: 2 * n] = np.concatenate([states, states])  # unused (done = 1)
        self.size = 2 * n
        print(
            f"Populated buffer with {self.size} transitions "
            f"({n} approve + {n} decline) | default rate {defaults.mean():.2%} | "
            f"mean approve reward {approve_reward.mean():+.4f} vs decline {decline_reward:+.4f}"
        )

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.tensor(self.states[idx]),
            torch.tensor(self.actions[idx], dtype=torch.long),
            torch.tensor(self.rewards[idx]).unsqueeze(1),
            torch.tensor(self.next_states[idx]),
            torch.tensor(self.dones[idx]).unsqueeze(1),
        )


class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, state):
        return self.net(state)


def train_cql(q_net, target_net, buffer, steps=3000, batch_size=256,
              alpha_cql=0.1, lr=3e-4, gamma=0.0, tau=0.05, log_every=500):
    """Offline CQL. (Moved out of QNetwork: the old `train` method shadowed nn.Module.train.)"""
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    print(">>[phase 2] CQL training on approve+decline data")
    for step in range(steps):
        s, a, r, s_, d = buffer.sample(batch_size)
        with torch.no_grad():
            y = r + gamma * (1.0 - d) * target_net(s_).max(1, keepdim=True)[0]
        all_q = q_net(s)
        q_pred = all_q.gather(1, a.unsqueeze(1))
        td_loss = loss_fn(q_pred, y)
        cql_penalty = (torch.logsumexp(all_q, dim=1).mean() - q_pred.mean()) * alpha_cql
        loss = td_loss + cql_penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for p, tq in zip(q_net.parameters(), target_net.parameters()):
            tq.data.copy_(tau * p.data + (1 - tau) * tq.data)
        if (step + 1) % log_every == 0:
            print(f"Step [{step + 1}/{steps}] | TD loss: {td_loss.item():.4f} | CQL reg: {cql_penalty.item():.4f}")
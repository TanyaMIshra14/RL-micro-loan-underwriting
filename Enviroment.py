import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    def __init__(self,states:np.array,default:np.array,contexts:np.array):
        super().__init__()
        self.states=states
        self.default=default
        self.contexts=contexts
        self.num_states=len(states)
        self.state_space=states.shape[1]+2
        self.observation_space=spaces.Box(
            low=-np.inf,high=np.inf,shape=(self.state_space,),dtype=np.float32
        )
        self.action_space=spaces.Tuple((
            spaces.Discrete(2),
            spaces.Box(low=0.08,high=0.36,shape=(1,),dtype=np.float32)
        ))
        self.window_size=100
        self.trailing_defaults=deque(maxlen=self.window_size)
        self.group_approvals={0:0,1:0,2:0,3:0}
        self.current_idx=0
        self.max_steps=2000

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.current_idx=0
        start_range = self.num_states - self.max_steps
        if start_range <= 0:
            raise ValueError(f"Environment needs more than {self.max_steps} rows of data.")
        self.currentstep=np.random.randint(0, start_range)
        self.trailing_defaults.clear()
        self.group_approvals={i:0 for i in range(4)}
        observation=self._get_observation(self.currentstep)
        return observation, {}

    def _get_observation(self,idx):
        base_state=self.states[idx]
        gnp=np.mean(self.trailing_defaults) if self.trailing_defaults else 0.02
        total_approvals=max(1,sum(self.group_approvals.values()))
        psl_ratio=(self.group_approvals[0]+self.group_approvals[1])/total_approvals
        return np.concatenate([base_state,[gnp,psl_ratio]]).astype(np.float32)
    
    
    def step(self, action):
        state_idx = self.currentstep
        self.current_idx += 1
        decision = action[0]
        quoted_approval = np.asarray(np.clip(action[1], 0.08, 0.36)).item()
        actual_default = self.default[state_idx]
        current_context = self.contexts[state_idx]
        repo_rate = self.states[state_idx, 14]
        reward = 0.0
        info = {}

        if decision == 0:
            reward = 0.0
        else:
            self.group_approvals[current_context] += 1
            p_accept = 1.0 / (1.0 + np.exp(14.0 * (quoted_approval - 0.18)))
            accepted = np.random.rand() < p_accept

            if not accepted:
                reward = -0.02
            else:
                adverse_selection = 0.5 * max(0.0, quoted_approval - 0.22)
                effective_default = np.clip(float(actual_default) + adverse_selection, 0.0, 1.0)
                loan_defaulted = np.random.rand() < effective_default
                self.trailing_defaults.append(1.0 if loan_defaulted else 0.0)

                if loan_defaulted:
                    reward = -1.0
                else:
                    net_interest_margin = quoted_approval - repo_rate
                    reward = net_interest_margin * 3.5

        trailing_gnpa = np.mean(self.trailing_defaults) if self.trailing_defaults else 0.0
        if trailing_gnpa > 0.04:
            reward -= 5.0 * (trailing_gnpa - 0.04)

        tot_approvals = sum(self.group_approvals.values())
        if tot_approvals > 20:
            psl_share = (self.group_approvals[0] + self.group_approvals[1]) / tot_approvals
            if psl_share < 0.40:
                reward -= 0.1 * (0.40 - psl_share)

        terminated = self.current_idx >= self.max_steps
        truncated = False
        self.currentstep += 1
        next_observation = self._get_observation(self.currentstep)
        info["gnpa"] = trailing_gnpa

        return next_observation, float(reward), terminated, truncated, info
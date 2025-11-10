# Monte Carlo (First-Visit) Control on Two_D_GridEnv
# The code trains a policy using First-Visit Monte Carlo Control with epsilon-greedy policy.
# It is self-contained (includes the environment definition you provided) and will:
# 1. Train for `n_episodes`
# 2. Print progress and final policy/Q summary
# 3. Evaluate the learned greedy policy for a few episodes

import random
import numpy as np
from collections import defaultdict

# --- Environment (user-provided) ---
import gymnasium as gym
from gymnasium import spaces

class Two_D_GridEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(Two_D_GridEnv, self).__init__()
        self.grid_size = 5
        self.position = [0,0]
        self.top_left = [0,0]
        self.bottom_right = [self.grid_size - 1, self.grid_size - 1]
        self.observation_space = spaces.Box(low=np.array([0, 0]),
                                            high=np.array([self.grid_size - 1, self.grid_size - 1]),
                                            dtype=np.int32)
        self.action_space = spaces.Discrete(4)

    def reset(self):
        self.position = [0, 0]
        return np.array(self.position, dtype=np.int32)

    def step(self, action):
        if action == 0 and self.position[1] > 0:  # up
            self.position[1] -= 1
        elif action == 1 and self.position[1] < self.grid_size - 1:  # down
            self.position[1] += 1
        elif action == 2 and self.position[0] > 0:  # left
            self.position[0] -= 1
        elif action == 3 and self.position[0] < self.grid_size - 1:  # right
            self.position[0] += 1
        
        done = False
        reward = -1  # step penalty
        if self.position == self.bottom_right:
            done = True
            reward = 10
        info = {}
        return np.array(self.position, dtype=np.int32), reward, done, info

    def render(self, mode='human'):
        print(f"Agent's current position: {self.position}")
    def close(self):
        pass

# --- Monte Carlo Control (First-Visit) ---
def state_to_key(state):
    # Convert numpy array state to tuple key for dict
    return (int(state[0]), int(state[1]))

def epsilon_greedy_action(Q, state_key, n_actions, epsilon):
    # If state unseen, treat Q-values as zero
    if random.random() < epsilon:
        return random.randrange(n_actions)
    else:
        qvals = [Q[(state_key, a)] for a in range(n_actions)]
        maxv = max(qvals)
        # break ties randomly
        max_actions = [a for a, v in enumerate(qvals) if v == maxv]
        return random.choice(max_actions)

def train_mc_control(env, n_episodes=5000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_episodes=4000):
    n_actions = env.action_space.n
    Q = defaultdict(float)  # Q[(state_key, action)] = value
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    for ep in range(1, n_episodes + 1):
        # linearly anneal epsilon
        frac = min(1.0, ep / epsilon_decay_episodes)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        episode = []  # list of (state_key, action, reward)
        state = env.reset()
        done = False
        steps = 0
        max_steps = 100  # safety cap for episode length
        while not done and steps < max_steps:
            s_key = state_to_key(state)
            a = epsilon_greedy_action(Q, s_key, n_actions, epsilon)
            next_state, r, done, info = env.step(a)
            episode.append((s_key, a, r))
            state = next_state
            steps += 1

        # Compute returns G and first-visit updates
        G = 0.0
        visited = set()  # to track first visits for (state,action)
        for t in reversed(range(len(episode))):
            s_key, a, r = episode[t]
            G = gamma * G + r
            sa = (s_key, a)
            if sa not in visited:
                visited.add(sa)
                returns_sum[sa] += G
                returns_count[sa] += 1.0
                Q[sa] = returns_sum[sa] / returns_count[sa]

        # Optionally print progress
        if ep % (n_episodes // 10) == 0 or ep == 1:
            print(f"Episode {ep}/{n_episodes} | Episode length: {len(episode)} | Epsilon: {epsilon:.3f}")

    # Derive final greedy policy from Q
    policy = {}
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            s_key = (x, y)
            qvals = [Q[(s_key, a)] for a in range(n_actions)]
            maxv = max(qvals)
            best_actions = [a for a, v in enumerate(qvals) if v == maxv]
            policy[s_key] = random.choice(best_actions)
    return Q, policy

def evaluate_policy(env, policy, n_episodes=50, render=False):
    total_returns = []
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        G = 0.0
        steps = 0
        while not done and steps < 100:
            s_key = state_to_key(state)
            action = policy.get(s_key, env.action_space.sample())
            state, r, done, info = env.step(action)
            G += r
            steps += 1
        total_returns.append(G)
        if render:
            print("Episode return:", G)
    avg_return = np.mean(total_returns)
    std_return = np.std(total_returns)
    return avg_return, std_return, total_returns

# --- Run training ---
env = Two_D_GridEnv()
Q, learned_policy = train_mc_control(env, n_episodes=3000, gamma=0.99,
                                     epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=2500)

# --- Evaluate learned greedy policy ---
avg_ret, std_ret, rets = evaluate_policy(env, learned_policy, n_episodes=100)
print("\nEvaluation of learned greedy policy:")
print(f"Average return over 100 episodes: {avg_ret:.2f} ± {std_ret:.2f}")

# Show policy grid (action codes)
action_map = {0: '^', 1: 'v', 2: '<', 3: '>'}
print("\nLearned greedy policy (as arrows, grid coordinates x across, y down):")
for y in range(env.grid_size):
    row = ""
    for x in range(env.grid_size):
        if (x, y) == tuple(env.bottom_right):
            row += " G  "  # goal
        else:
            row += f" {action_map[learned_policy[(x,y)]]}  "
    print(row)

# Show some example Q-values for a few states
print("\nSample Q-values (state -> [a0,a1,a2,a3]):")
for sx in [0, 1, 2, 3, 4]:
    sk = (sx, sx)  # diagonal states for sampling
    qvals = [round(Q[(sk, a)], 2) for a in range(env.action_space.n)]
    print(f"{sk}: {qvals}")

print("\nDone training and evaluation.")

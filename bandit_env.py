import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BanditEnv(gym.Env):
    """
    A multi-armed bandit environment:
    - There are N arms (slot machines) each with different reward probabilities.
    - Each arm gives a reward based on a Bernoulli distribution.
    - The agent must learn which arm gives the highest expected reward.
    - Each action corresponds to pulling one of the arms.
    - Episode length is fixed (no termination condition except max steps).
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, n_arms=5, max_steps=100):
        super(BanditEnv, self).__init__()
        
        self.n_arms = n_arms
        self.max_steps = max_steps
        self.current_step = 0
        
        # Each arm has a different probability of success (hidden from agent)
        # Generate random probabilities for each arm
        self.arm_probabilities = np.random.uniform(0.1, 0.9, size=n_arms)
        self.optimal_arm = np.argmax(self.arm_probabilities)
        
        # Track statistics
        self.arm_counts = np.zeros(n_arms)  # How many times each arm was pulled
        self.arm_rewards = np.zeros(n_arms)  # Total rewards from each arm
        
        # Observation space: just the current step number (minimal information)
        self.observation_space = spaces.Box(
            low=0, high=max_steps, shape=(1,), dtype=np.float32
        )
        
        # Action space: choose which arm to pull (0 to n_arms-1)
        self.action_space = spaces.Discrete(n_arms)
        
        self.reset()

    def reset(self):
        """Resets the environment to the initial state."""
        self.current_step = 0
        self.arm_counts = np.zeros(self.n_arms)
        self.arm_rewards = np.zeros(self.n_arms)
        return np.array([self.current_step], dtype=np.float32)

    def step(self, action):
        """Applies an action and returns the new state, reward, done, info."""
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        self.current_step += 1
        
        # Pull the selected arm and get reward based on its probability
        reward = float(np.random.binomial(1, self.arm_probabilities[action]))
        
        # Update statistics
        self.arm_counts[action] += 1
        self.arm_rewards[action] += reward
        
        # Episode ends after max_steps
        done = self.current_step >= self.max_steps
        
        # Information about the environment (for analysis)
        info = {
            'optimal_arm': self.optimal_arm,
            'arm_probabilities': self.arm_probabilities.copy(),
            'arm_counts': self.arm_counts.copy(),
            'arm_rewards': self.arm_rewards.copy(),
            'regret': self.arm_probabilities[self.optimal_arm] - self.arm_probabilities[action]
        }
        
        return np.array([self.current_step], dtype=np.float32), reward, done, info

    def render(self, mode='human'):
        """Optional rendering function for visualization."""
        print(f"Step: {self.current_step}/{self.max_steps}")
        print("Arm Statistics:")
        print("Arm | Probability | Times Pulled | Total Rewards | Avg Reward")
        print("-" * 65)
        
        for i in range(self.n_arms):
            prob = self.arm_probabilities[i]
            count = int(self.arm_counts[i])
            total_reward = self.arm_rewards[i]
            avg_reward = total_reward / count if count > 0 else 0.0
            optimal_marker = " ⭐" if i == self.optimal_arm else ""
            
            print(f" {i:2d} |    {prob:.3f}    |      {count:3d}      |      {total_reward:3.0f}     |   {avg_reward:.3f}{optimal_marker}")
        
        print(f"\nOptimal arm: {self.optimal_arm} (probability: {self.arm_probabilities[self.optimal_arm]:.3f})")

    def close(self):
        """Optional cleanup code."""
        pass

    def get_regret(self):
        """Calculate the cumulative regret so far."""
        total_possible_reward = self.current_step * self.arm_probabilities[self.optimal_arm]
        total_actual_reward = np.sum(self.arm_rewards)
        return total_possible_reward - total_actual_reward


if __name__ == "__main__":
    env = BanditEnv(n_arms=4, max_steps=50)
    obs = env.reset()
    
    print("Multi-Armed Bandit Environment")
    print("=" * 50)
    print(f"Number of arms: {env.n_arms}")
    print(f"Max steps: {env.max_steps}")
    print("Hidden arm probabilities:", [f"{p:.3f}" for p in env.arm_probabilities])
    print("=" * 50)
    
    env.render()
    done = False
    count = 0
    cumulative_reward = 0
    
    # Simple strategy: explore each arm a few times, then exploit the best one
    exploration_steps = 12  # Explore for first 12 steps
    
    while not done:
        if count < exploration_steps:
            # Exploration phase: try each arm roughly equally
            action = count % env.n_arms
            strategy = "Explore"
        else:
            # Exploitation phase: choose arm with highest average reward
            avg_rewards = []
            for i in range(env.n_arms):
                if env.arm_counts[i] > 0:
                    avg_rewards.append(env.arm_rewards[i] / env.arm_counts[i])
                else:
                    avg_rewards.append(0)
            action = np.argmax(avg_rewards)
            strategy = "Exploit"
        
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        
        print(f"\nStep {count + 1}: {strategy} - Pulled arm {action}, Reward: {reward}")
        
        if count % 10 == 9 or done:  # Show stats every 10 steps or at end
            env.render()
            print(f"Cumulative regret: {env.get_regret():.2f}")
            print("-" * 50)
        
        count += 1
        
        if done:
            print("Episode finished!")
            break
    
    print(f"\nFinal Results:")
    print(f"Total steps taken: {count}")
    print(f"Cumulative reward: {cumulative_reward}")
    print(f"Cumulative regret: {env.get_regret():.2f}")
    print(f"Average reward per step: {cumulative_reward/count:.3f}")
    print(f"Optimal average reward: {env.arm_probabilities[env.optimal_arm]:.3f}")
    
    # Show final arm selection statistics
    print(f"\nArm Selection Summary:")
    for i in range(env.n_arms):
        percentage = (env.arm_counts[i] / count) * 100 if count > 0 else 0
        print(f"Arm {i}: {int(env.arm_counts[i])} times ({percentage:.1f}%)")
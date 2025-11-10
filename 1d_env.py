import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleLineEnv(gym.Env):
    """
    A simple 1D line environment:
    - The agent starts at position 0 on a line of length 10.
    - It can move left or right (actions 0 or 1).
    - Reaches the left end (position -5): reward +10, episode done.
    - Reaches the right end (position +5): reward +10, episode done.
    - Each step has a small penalty (-1) to encourage faster solutions.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SimpleLineEnv, self).__init__()
        self.position = 0
        self.left_end = -5
        self.right_end = 5

        # Observation space is a single number: current position
        self.observation_space = spaces.Box(low=np.array([self.left_end]),
                                            high=np.array([self.right_end]),
                                            dtype=np.float32)

        # Action space: 0 = move left, 1 = move right
        self.action_space = spaces.Discrete(2)

    def reset(self):
        """Resets the environment to the initial state."""
        self.position = 0
        return np.array([self.position], dtype=np.float32)

    def step(self, action):
        """Applies an action and returns the new state, reward, done, info."""
        # Action: 0 = left, 1 = right
        if action == 0:
            self.position -= 1
        else:
            self.position += 1
        
        # Reward logic
        done = False
        reward = -1  # small step punishment to encourage faster completion

        if self.position <= self.left_end:
            done = True
            reward = 10  # reward for reaching left end
        elif self.position >= self.right_end:
            done = True
            reward = 10  # reward for reaching right end
        
        info = {}
        return np.array([self.position], dtype=np.float32), reward, done, info

    def render(self, mode='human'):
        """Optional rendering function for visualization."""
        print(f"Agent's current position: {self.position}")
    
    def close(self):
        """Optional cleanup code."""
        pass


if __name__ == "__main__":
    env = SimpleLineEnv()
    obs = env.reset()
    env.render()
    done = False
    count = 0
    cumulative_reward = 0
    while not done:
        action = env.action_space.sample()  # choose random action
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        env.render()
        count += 1
        if done:
            print("Episode finished!")
            break
    print(f"Total steps taken: {count}")
    print(f"Cumulative reward: {cumulative_reward}")
    print("Final observation:", obs)
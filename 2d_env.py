import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Two_D_GridEnv(gym.Env):
    """
    A simple 2D grid environment:
    - The agent starts at position (0, 0) on a 5x5 grid.
    - It can move up, down, left, or right (actions 0, 1, 2, 3).
    - Reaches the top-left corner (position (0, 0)): reward +10, episode done.
    - Reaches the bottom-right corner (position (4, 4)): reward +10, episode done.
    - Each step has a small penalty (-1) to encourage faster solutions.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Two_D_GridEnv, self).__init__()
        self.grid_size = 5
        self.position = [0,0]
        self.top_left = [0,0]
        self.bottom_right = [self.grid_size - 1, self.grid_size - 1]

        # Observation space is a 2D coordinate: current position
        self.observation_space = spaces.Box(low=np.array([0, 0]),
                                            high=np.array([self.grid_size - 1, self.grid_size - 1]),
                                            dtype=np.int32)

        # Action space: 0 = up, 1 = down, 2 = left, 3 = right
        self.action_space = spaces.Discrete(4)

    def reset(self):
        """Resets the environment to the initial state."""
        self.position = [0, 0]
        return np.array(self.position, dtype=np.int32)

    def step(self, action):
        """Applies an action and returns the new state, reward, done, info."""
        if action == 0 and self.position[1] > 0:  # up
            self.position[1] -= 1
        elif action == 1 and self.position[1] < self.grid_size - 1:  # down
            self.position[1] += 1
        elif action == 2 and self.position[0] > 0:  # left
            self.position[0] -= 1
        elif action == 3 and self.position[0] < self.grid_size - 1:  # right
            self.position[0] += 1
        
        # Reward logic
        done = False
        reward = -1  # small step punishment to encourage faster completion

    
        if self.position == self.bottom_right:
            done = True
            reward = 10  # reward for reaching bottom-right corner

        info = {}
        return np.array(self.position, dtype=np.int32), reward, done, info


    def render(self, mode='human'):
        """Optional rendering function for visualization."""
        print(f"Agent's current position: {self.position}")

    def close(self):
        """Optional cleanup code."""
        pass

if __name__ == "__main__":
    env = Two_D_GridEnv()
    obs = env.reset()
    env.render()
    done = False
    count = 0
    cumulative_reward = 0
    while not done:
        action = env.action_space.sample()  
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
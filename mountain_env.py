import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MountainClimbEnv(gym.Env):
    """
    A simplified Mountain Car environment:
    - A car is stuck in a valley and needs to build momentum to reach the goal.
    - The car starts at the bottom of the valley (position 0).
    - Goal is to reach the top of the mountain (position >= 0.5).
    - Actions: 0 = push left, 1 = do nothing, 2 = push right.
    - Reward: -1 for each step until goal is reached, +100 for reaching goal.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MountainClimbEnv, self).__init__()
        
        # Environment parameters
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = 0
        
        # Physics parameters
        self.force = 0.001
        self.gravity = 0.0025
        
        # State: [position, velocity]
        self.observation_space = spaces.Box(
            low=np.array([self.min_position, -self.max_speed], dtype=np.float32),
            high=np.array([self.max_position, self.max_speed], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: 0 = push left, 1 = no action, 2 = push right
        self.action_space = spaces.Discrete(3)
        
        self.reset()

    def reset(self):
        """Resets the environment to the initial state."""
        # Start at random position in the valley with zero velocity
        self.state = np.array([
            np.random.uniform(low=-0.6, high=-0.4),
            0
        ], dtype=np.float32)
        return self.state.copy()

    def step(self, action):
        """Applies an action and returns the new state, reward, done, info."""
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        position, velocity = self.state
        
        # Apply action force
        if action == 0:  # Push left
            velocity += -self.force
        elif action == 2:  # Push right
            velocity += self.force
        # action == 1 means no force applied
        
        # Apply gravity (always pulls towards the valley)
        velocity += np.cos(3 * position) * (-self.gravity)
        
        # Clip velocity to max speed
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        
        # Update position
        position += velocity
        
        # Clip position to boundaries
        if position < self.min_position:
            position = self.min_position
            velocity = 0  # Stop at the left boundary
        elif position > self.max_position:
            position = self.max_position
        
        self.state = np.array([position, velocity], dtype=np.float32)
        
        # Check if goal is reached
        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        
        # Reward structure
        if done:
            reward = 100.0  # Large reward for reaching the goal
        else:
            reward = -1.0  # Small penalty for each step to encourage efficiency
        
        info = {}
        return self.state.copy(), reward, done, info

    def render(self, mode='human'):
        """Optional rendering function for visualization."""
        position, velocity = self.state
        # Simple text-based visualization
        valley_width = int((self.max_position - self.min_position) * 20)
        car_pos = int((position - self.min_position) * 20)
        goal_pos = int((self.goal_position - self.min_position) * 20)
        
        valley_line = ['.'] * valley_width
        if 0 <= car_pos < valley_width:
            valley_line[car_pos] = 'C'  # Car position
        if 0 <= goal_pos < valley_width:
            valley_line[goal_pos] = 'G'  # Goal position
        
        print(f"Position: {position:.3f}, Velocity: {velocity:.3f}")
        print(''.join(valley_line))
        print(f"Goal at position {self.goal_position}")

    def close(self):
        """Optional cleanup code."""
        pass


if __name__ == "__main__":
    env = MountainClimbEnv()
    obs = env.reset()
    env.render()
    done = False
    count = 0
    cumulative_reward = 0
    
    print("Starting Mountain Climb simulation...")
    print("=" * 60)
    
    while not done and count < 200:  # Max 200 steps
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        
        action_names = ['Push Left', 'No Action', 'Push Right']
        print(f"Step {count + 1}: Action: {action_names[action]}, Reward: {reward}")
        env.render()
        print("-" * 40)
        count += 1
        
        if done:
            print("🎉 Goal reached! Episode finished!")
            break
    
    if not done:
        print("❌ Episode ended without reaching goal (max steps reached)")
    
    print(f"Total steps taken: {count}")
    print(f"Cumulative reward: {cumulative_reward}")
    print("Final observation:", obs)
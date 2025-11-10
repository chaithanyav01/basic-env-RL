import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math


class SimpleCartPoleEnv(gym.Env):
    """
    A simplified CartPole environment:
    - A pole is balanced on a cart that can move left or right.
    - The goal is to keep the pole upright by moving the cart.
    - Episode ends if pole angle exceeds ±15 degrees or cart moves too far.
    - Reward +1 for each timestep the pole stays upright.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SimpleCartPoleEnv, self).__init__()
        
        # Physical constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # half-pole length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        
        # Thresholds
        self.theta_threshold_radians = 15 * 2 * math.pi / 360  # ±15 degrees
        self.x_threshold = 2.4
        
        # State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # Action space: 0 = push left, 1 = push right
        self.action_space = spaces.Discrete(2)
        
        self.reset()

    def reset(self):
        """Resets the environment to the initial state."""
        # Start with small random values around equilibrium
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        return self.state.copy()

    def step(self, action):
        """Applies an action and returns the new state, reward, done, info."""
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        # Physics simulation
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # Update state using Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        
        # Check if episode is done
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        # Reward is 1 for each timestep the pole stays upright
        reward = 1.0 if not done else 0.0
        
        info = {}
        return self.state.copy(), reward, done, info

    def render(self, mode='human'):
        """Optional rendering function for visualization."""
        x, x_dot, theta, theta_dot = self.state
        print(f"Cart Position: {x:.2f}, Cart Velocity: {x_dot:.2f}")
        print(f"Pole Angle: {theta:.3f} rad ({math.degrees(theta):.1f}°), Pole Angular Velocity: {theta_dot:.3f}")

    def close(self):
        """Optional cleanup code."""
        pass


if __name__ == "__main__":
    env = SimpleCartPoleEnv()
    obs = env.reset()
    env.render()
    done = False
    count = 0
    cumulative_reward = 0
    
    while not done and count < 200:  # Max 200 steps
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        print(f"Step {count + 1}: Action: {action}, Reward: {reward}, Done: {done}")
        env.render()
        print("-" * 50)
        count += 1
        
        if done:
            print("Episode finished!")
            break
    
    print(f"Total steps taken: {count}")
    print(f"Cumulative reward: {cumulative_reward}")
    print("Final observation:", obs)
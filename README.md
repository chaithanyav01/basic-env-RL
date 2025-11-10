# Reinforcement Learning Environments

This readme provides detailed explanations of the step function logic for all five custom RL environments.

## Table of Contents
1. [1D Line Environment](#1d-line-environment)
2. [2D Grid Environment](#2d-grid-environment)
3. [CartPole Environment](#cartpole-environment)
4. [Mountain Car Environment](#mountain-car-environment)
5. [Multi-Armed Bandit Environment](#multi-armed-bandit-environment)

---

## 1D Line Environment

### Overview
A simple linear world where an agent moves left or right on a line to reach either end.

### Step Function Logic

```python
def step(self, action):
```

**Input**: `action` (0 = move left, 1 = move right)

**Process Flow**:

1. **Action Processing**:
   ```python
   if action == 0:
       self.position -= 1  # Move left
   else:
       self.position += 1  # Move right
   ```

2. **Reward Calculation**:
   ```python
   done = False
   reward = -1  # Step penalty to encourage efficiency
   
   if self.position <= self.left_end:    # Reached left boundary (-5)
       done = True
       reward = 10
   elif self.position >= self.right_end: # Reached right boundary (+5)
       done = True
       reward = 10
   ```

3. **Return Values**:
   - **State**: Current position as numpy array `[position]`
   - **Reward**: -1 (step penalty) or +10 (goal reached)
   - **Done**: True if reached either boundary, False otherwise
   - **Info**: Empty dictionary

**Key Design Decisions**:
- Step penalty (-1) encourages the agent to reach goals quickly
- Both ends are equally rewarded to allow exploration of different strategies
- No bounds checking needed as agent can move beyond boundaries (triggers termination)

---

## 2D Grid Environment

### Overview
A 2D grid world where an agent navigates from start position (0,0) to the bottom-right corner (4,4).

### Step Function Logic

```python
def step(self, action):
```

**Input**: `action` (0 = up, 1 = down, 2 = left, 3 = right)

**Process Flow**:

1. **Action Processing with Boundary Checking**:
   ```python
   if action == 0 and self.position[1] > 0:          # Move up (decrease y)
       self.position[1] -= 1
   elif action == 1 and self.position[1] < self.grid_size - 1:  # Move down (increase y)
       self.position[1] += 1
   elif action == 2 and self.position[0] > 0:        # Move left (decrease x)
       self.position[0] -= 1
   elif action == 3 and self.position[0] < self.grid_size - 1:  # Move right (increase x)
       self.position[0] += 1
   ```

2. **Reward Calculation**:
   ```python
   done = False
   reward = -1  # Step penalty
   
   if self.position == self.bottom_right:  # Reached goal [4,4]
       done = True
       reward = 10
   ```

3. **Return Values**:
   - **State**: Current position as numpy array `[x, y]`
   - **Reward**: -1 (step penalty) or +10 (goal reached)
   - **Done**: True only when reaching bottom-right corner
   - **Info**: Empty dictionary

**Key Design Decisions**:
- Boundary enforcement prevents invalid moves (agent stays in place if trying to move out of bounds)
- Only one goal location (bottom-right) creates a clear objective
- Grid coordinates: (0,0) is top-left, (4,4) is bottom-right

---

## CartPole Environment

### Overview
A physics simulation where an agent balances a pole on a movable cart by applying forces.

### Step Function Logic

```python
def step(self, action):
```

**Input**: `action` (0 = push left, 1 = push right)

**Process Flow**:

1. **Force Application**:
   ```python
   force = self.force_mag if action == 1 else -self.force_mag  # ±10.0 N
   ```

2. **Physics Simulation**:
   ```python
   x, x_dot, theta, theta_dot = self.state
   costheta = math.cos(theta)
   sintheta = math.sin(theta)
   
   # Calculate accelerations using physics equations
   temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
   thetaacc = (self.gravity * sintheta - costheta * temp) / (
       self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
   )
   xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
   ```

3. **State Integration** (Euler method):
   ```python
   x = x + self.tau * x_dot          # Update position
   x_dot = x_dot + self.tau * xacc   # Update velocity
   theta = theta + self.tau * theta_dot      # Update angle
   theta_dot = theta_dot + self.tau * thetaacc  # Update angular velocity
   ```

4. **Termination Conditions**:
   ```python
   done = bool(
       x < -self.x_threshold          # Cart too far left (-2.4)
       or x > self.x_threshold        # Cart too far right (+2.4)
       or theta < -self.theta_threshold_radians  # Pole fell left (-15°)
       or theta > self.theta_threshold_radians   # Pole fell right (+15°)
   )
   ```

5. **Reward Calculation**:
   ```python
   reward = 1.0 if not done else 0.0  # +1 for each step balanced, 0 when failed
   ```

**Key Design Decisions**:
- Continuous physics simulation using Euler integration
- Multiple failure conditions (cart position and pole angle)
- Reward structure encourages longer episodes (survival time)
- Small time step (τ = 0.02s) for stable simulation

---

## Mountain Car Environment

### Overview
An underpowered car in a valley must build momentum to reach a goal on top of a hill.

### Step Function Logic

```python
def step(self, action):
```

**Input**: `action` (0 = push left, 1 = no action, 2 = push right)

**Process Flow**:

1. **Force Application**:
   ```python
   if action == 0:      # Push left
       velocity += -self.force    # -0.001
   elif action == 2:    # Push right
       velocity += self.force     # +0.001
   # action == 1: no force applied
   ```

2. **Gravity Effect**:
   ```python
   velocity += np.cos(3 * position) * (-self.gravity)  # -0.0025 * cos(3x)
   ```
   - The `cos(3 * position)` creates a hill shape
   - Gravity always pulls the car toward the valley bottom

3. **Velocity Constraints**:
   ```python
   velocity = np.clip(velocity, -self.max_speed, self.max_speed)  # [-0.07, +0.07]
   ```

4. **Position Update**:
   ```python
   position += velocity
   ```

5. **Boundary Handling**:
   ```python
   if position < self.min_position:     # Hit left wall (-1.2)
       position = self.min_position
       velocity = 0                     # Stop at boundary
   elif position > self.max_position:  # Beyond right boundary (0.6)
       position = self.max_position
   ```

6. **Goal Check**:
   ```python
   done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
   # Must reach position ≥ 0.5 with non-negative velocity
   ```

7. **Reward Calculation**:
   ```python
   if done:
       reward = 100.0   # Large reward for success
   else:
       reward = -1.0    # Step penalty encourages efficiency
   ```

**Key Design Decisions**:
- Weak engine force (0.001) requires momentum building
- Cosine-based gravity creates realistic hill physics
- Velocity requirement at goal prevents "lucky" slow arrivals
- Left boundary acts as a wall (stops movement)

---

## Multi-Armed Bandit Environment

### Overview
A classic explore-vs-exploit problem with multiple slot machines, each having different hidden reward probabilities.

### Step Function Logic

```python
def step(self, action):
```

**Input**: `action` (0 to n_arms-1, selecting which arm to pull)

**Process Flow**:

1. **Step Counter Update**:
   ```python
   self.current_step += 1
   ```

2. **Reward Generation**:
   ```python
   reward = float(np.random.binomial(1, self.arm_probabilities[action]))
   ```
   - Uses Bernoulli distribution (binomial with n=1)
   - Each arm has a hidden success probability
   - Reward is either 0 or 1

3. **Statistics Tracking**:
   ```python
   self.arm_counts[action] += 1      # Increment pull count for this arm
   self.arm_rewards[action] += reward # Add reward to arm's total
   ```

4. **Episode Termination**:
   ```python
   done = self.current_step >= self.max_steps  # Fixed episode length
   ```

5. **Information Dictionary**:
   ```python
   info = {
       'optimal_arm': self.optimal_arm,                    # Index of best arm
       'arm_probabilities': self.arm_probabilities.copy(), # True probabilities (hidden)
       'arm_counts': self.arm_counts.copy(),              # Pull counts per arm
       'arm_rewards': self.arm_rewards.copy(),            # Total rewards per arm
       'regret': self.arm_probabilities[self.optimal_arm] - self.arm_probabilities[action]
   }
   ```

6. **Return Values**:
   - **State**: Current step number `[current_step]`
   - **Reward**: 0 or 1 based on stochastic outcome
   - **Done**: True after max_steps reached
   - **Info**: Rich information for analysis

**Key Design Decisions**:
- Fixed episode length (no early termination)
- Minimal state information (just step number) forces learning from rewards
- Bernoulli rewards create clear probability learning problem
- Extensive info tracking enables regret analysis
- Hidden probabilities create exploration challenge

---

## Common Patterns Across Environments

### 1. **Action Validation**
All environments validate actions using:
```python
assert self.action_space.contains(action), f"Invalid action {action}"
```

### 2. **State Representation**
- States returned as numpy arrays with appropriate dtypes
- Consistent shape matching observation_space definition

### 3. **Reward Design**
- **Sparse rewards**: Large positive rewards for goals
- **Step penalties**: Small negative rewards encouraging efficiency
- **Survival rewards**: Positive rewards for maintaining good states

### 4. **Termination Conditions**
- **Goal-based**: Episode ends when objective achieved
- **Failure-based**: Episode ends when constraints violated
- **Time-based**: Episode ends after maximum steps

### 5. **Information Dictionary**
- Most environments return empty `info = {}`
- Bandit environment provides rich analytics
- Can be extended for debugging and analysis

This documentation provides the foundation for understanding how each environment processes actions and generates the learning signal for RL algorithms.
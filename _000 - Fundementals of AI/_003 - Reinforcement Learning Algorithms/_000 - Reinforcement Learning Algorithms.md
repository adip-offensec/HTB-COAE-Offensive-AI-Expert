
# Reinforcement Learning Algorithms

---

`Reinforcement learning` (`RL`) introduces a unique paradigm in `machine learning` where an agent learns by interacting with an environment. Unlike `supervised learning`, which relies on labeled data, or `unsupervised learning`, which explores unlabeled data, `RL` focuses on learning through trial and error, guided by feedback in the form of rewards or penalties. This approach mimics how humans learn through experience, making `RL` particularly suitable for tasks that involve sequential decision-making in dynamic environments.

Think of it like training a dog. You don't give the dog explicit instructions on sitting, staying, or fetching. Instead, you reward it with treats and praise when it performs the desired actions and correct it when it doesn't. The dog learns to associate specific actions with positive outcomes through trial, error, and feedback.

## How Reinforcement Learning Works

In `RL`, an agent interacts with an environment by acting and observing the consequences. The environment provides feedback through rewards or penalties, guiding the agent toward learning an optimal policy. A `policy` is a strategy for selecting actions that maximize cumulative rewards over time.

`Reinforcement learning` algorithms can be broadly categorized into:

1. `Model-Based RL:` The agent learns a model of the environment, which it uses to predict future states and plan its actions. This approach is analogous to having a map of a maze before navigating it. The agent can use this map to plan the most efficient path to the goal, reducing the need for trial and error.
2. `Model-Free RL:` The agent learns directly from experience without explicitly modeling the environment. This is like navigating a maze without a map, where the agent relies solely on trial and error and feedback from the environment to learn the best actions. The agent gradually improves its policy by exploring different paths and learning from the rewards or penalties it receives.

## Core Concepts in Reinforcement Learning

Understanding `Reinforcement Learning` (`RL`) requires a grasp of its core concepts. These concepts provide the foundation for understanding how agents learn and interact with their environment to achieve their goals.

### Agent

The `agent` is the learner and decision-maker in an `RL` system. It interacts with the environment, taking action and observing the consequences. The agent aims to learn an optimal policy that maximizes cumulative rewards over time.

Think of the `agent` as a robot navigating a maze, a program playing a game, or a self-driving car navigating through traffic. In each case, the `agent` makes decisions and learns from its experiences.

### Environment

The `environment` is the external system or context in which the agent operates. It encompasses everything outside the agent, including the physical world, a simulated world, or even a game board. The `environment` responds to the agent's actions and provides feedback through rewards or penalties.

In the maze navigation example, the `environment` is the maze itself, with its walls, paths, and goal location. In a game-play scenario, the `environment` is the game with its rules and opponent moves.

### State

The `state` represents the current situation or condition of the environment. It provides a snapshot of the relevant information that the agent needs to make informed decisions. The `state` can include various aspects of the environment, such as the agent's position, the positions of other objects, and any other relevant variables.

The state of a robot navigating a maze might include its current location and the surrounding walls. In a chess game, the `state` is the current configuration of the chessboard.

### Action

An `action` is an agent's move or decision that affects the environment. The agent selects actions based on its current state and its policy. The `environment` responds to the `action` and transitions to a new state.

In the maze example, the robot's actions might be to move forward, turn left, or turn right. In a game, the actions might be to move a piece or make a specific play.

### Reward

The `reward` is feedback from the environment indicating the desirability of the agent's action. It is a scalar value that can be positive, negative, or zero. Positive rewards encourage the agent to repeat the action, while negative rewards (penalties) discourage it. The agent's goal is to maximize cumulative rewards over time.

In the maze example, the robot might receive a positive reward for getting closer to the goal and a penalty for hitting a wall. In a game, the reward might be positive for winning and negative for losing.

### Policy

A `policy` is a strategy or mapping from states to actions the agent follows. It determines which action the agent should take in a given state. The agent aims to learn an optimal policy that maximizes cumulative rewards.

The policy can be deterministic, where it always selects the same action in a given state, or stochastic, where it selects actions with certain probabilities.

### Value Function

The `value function` estimates the long-term value of being in a particular state or taking a specific action. It predicts the expected cumulative reward that the agent can obtain from that state or action onwards. The value function is a crucial component in many `RL` algorithms, as it guides the agent towards choosing actions that lead to higher long-term rewards.

There are two main types of value functions:

- `State-value function:` Estimates the expected cumulative reward from starting in a given state and following a particular policy.
- `Action-value function:` Estimates the expected cumulative reward from taking a specific action in a given state and then following a particular policy.

### Discount Factor

The `discount factor` (`γ`) is a `RL` parameter that determines future rewards' present value. It ranges between 0 and 1, with values closer to 1 giving more weight to long-term rewards and values closer to 0 emphasizing short-term rewards.

- `γ=0` means the agent only considers immediate rewards.
- `γ=1` means the agent values all future rewards equally.

### Episodic vs. Continuous Tasks

`Episodic tasks` involve the agent interacting with the environment in episodes, each ending at a terminal state (e.g., reaching the goal in a maze). In contrast, `continuous tasks` have no explicit end and continue indefinitely (e.g., controlling a robot arm).


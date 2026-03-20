"""
agents.py — Tabular RL Agents (Q-Learning & SARSA)
===================================================
Implements two classical RL algorithms for the EV charging problem:
  1. Q-Learning (off-policy, optimistic)
  2. SARSA      (on-policy, conservative)

Both use epsilon-greedy exploration with configurable decay.

Authors: Sydney Nzunguli & Utkarsh Singh
Course:  Reinforcement Learning – CS, Université Paris-Saclay (2025–2026)
"""

import numpy as np


class BaseAgent:
    """
    Base class for tabular RL agents.

    Handles Q-table initialization, epsilon-greedy action selection,
    and common bookkeeping. Subclasses implement the specific update rule.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int = None,
    ):
        """
        Parameters
        ----------
        n_states : int
            Size of the state space.
        n_actions : int
            Size of the action space.
        alpha : float
            Learning rate (step size for Q-value updates).
        gamma : float
            Discount factor for future rewards.
        epsilon : float
            Initial exploration rate (probability of random action).
        epsilon_min : float
            Minimum exploration rate after decay.
        epsilon_decay : float
            Multiplicative decay applied to epsilon after each episode.
        seed : int or None
            Random seed for reproducibility.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # ── Q-table: initialized to zeros ────────────────────────
        # Shape: (n_states, n_actions)
        self.Q = np.zeros((n_states, n_actions))

        # ── Random generator ─────────────────────────────────────
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state: int) -> int:
        """
        Epsilon-greedy action selection.

        With probability epsilon, pick a random action (exploration).
        Otherwise, pick the action with the highest Q-value (exploitation).
        Ties are broken randomly for fairness.

        Parameters
        ----------
        state : int
            Current state index.

        Returns
        -------
        action : int
            Selected action.
        """
        if self.rng.random() < self.epsilon:
            # Explore: random action
            return self.rng.integers(0, self.n_actions)
        else:
            # Exploit: greedy w.r.t. Q-values (random tie-breaking)
            q_values = self.Q[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return self.rng.choice(best_actions)

    def decay_epsilon(self):
        """Decay epsilon after each episode (multiplicative decay with floor)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self) -> np.ndarray:
        """
        Extract the greedy policy from the current Q-table.

        Returns
        -------
        policy : np.ndarray of shape (n_states,)
            Greedy action for each state.
        """
        return np.argmax(self.Q, axis=1)

    def get_value_function(self) -> np.ndarray:
        """
        Extract the value function V(s) = max_a Q(s, a).

        Returns
        -------
        V : np.ndarray of shape (n_states,)
        """
        return np.max(self.Q, axis=1)


class QLearningAgent(BaseAgent):
    """
    Q-Learning: Off-policy TD control.

    Update rule:
        Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]

    Uses the maximum Q-value of the next state regardless of the action
    actually taken → "optimistic" about future returns.
    """

    def __init__(self, n_states, n_actions, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.name = "Q-Learning"

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int = None,  # unused, kept for interface compatibility
        done: bool = False,
    ):
        """
        Perform Q-Learning update.

        Parameters
        ----------
        state : int
            Current state.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : int
            State transitioned to.
        next_action : int
            Ignored (Q-learning is off-policy).
        done : bool
            Whether the episode has terminated.
        """
        # Target: r + γ max_a' Q(s', a')  (0 if terminal)
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        # TD error
        td_error = target - self.Q[state, action]

        # Update Q-value
        self.Q[state, action] += self.alpha * td_error


class SARSAAgent(BaseAgent):
    """
    SARSA: On-policy TD control.

    Update rule:
        Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]

    Uses the Q-value of the action actually taken in the next state
    → "realistic" about future returns under the current policy.
    This makes SARSA more conservative / risk-averse than Q-learning.
    """

    def __init__(self, n_states, n_actions, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.name = "SARSA"

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool = False,
    ):
        """
        Perform SARSA update.

        Parameters
        ----------
        state : int
            Current state.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : int
            State transitioned to.
        next_action : int
            Action chosen (by current policy) in next_state.
        done : bool
            Whether the episode has terminated.
        """
        # Target: r + γ Q(s', a')  (0 if terminal)
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, next_action]

        # TD error
        td_error = target - self.Q[state, action]

        # Update Q-value
        self.Q[state, action] += self.alpha * td_error


# ══════════════════════════════════════════════════════════════════
#  Quick test
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Verify agents can be instantiated and basic operations work
    q_agent = QLearningAgent(n_states=504, n_actions=3, seed=42)
    sarsa_agent = SARSAAgent(n_states=504, n_actions=3, seed=42)

    print(f"Q-Learning agent: Q-table shape = {q_agent.Q.shape}")
    print(f"SARSA agent:      Q-table shape = {sarsa_agent.Q.shape}")

    # Test action selection
    action = q_agent.choose_action(state=0)
    print(f"\nQ-Learning chose action {action} in state 0 (epsilon={q_agent.epsilon:.2f})")

    # Test update
    q_agent.update(state=0, action=1, reward=-0.5, next_state=25, done=False)
    print(f"After update: Q(0, 1) = {q_agent.Q[0, 1]:.4f}")

    print("\nAgents initialized successfully!")
"""
train.py — Training Loop for Q-Learning & SARSA
================================================
Trains both agents on the EV Charging environment and saves
all results (learning curves, Q-tables, episode logs) to disk.

Usage:
    python train.py

Output:
    results/training_results.npz  — compressed NumPy archive with all data

Authors: Sydney Nzunguli & Utkarsh Singh
Course:  Reinforcement Learning – CS, Université Paris-Saclay (2025–2026)
"""

import os
import numpy as np
from environment import EVChargingEnv
from agents import QLearningAgent, SARSAAgent


def train_qlearning(env: EVChargingEnv, agent: QLearningAgent, n_episodes: int) -> dict:
    """
    Train a Q-Learning agent.

    The Q-learning loop:
        1. Observe state s
        2. Choose action a (epsilon-greedy)
        3. Take action, observe r, s'
        4. Update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        5. s ← s'

    Parameters
    ----------
    env : EVChargingEnv
        The environment instance.
    agent : QLearningAgent
        The Q-learning agent.
    n_episodes : int
        Number of training episodes.

    Returns
    -------
    logs : dict
        Training logs including rewards, costs, epsilon history.
    """
    # ── Logging arrays ───────────────────────────────────────────
    episode_rewards = np.zeros(n_episodes)
    episode_electricity = np.zeros(n_episodes)
    episode_degradation = np.zeros(n_episodes)
    episode_lowbat = np.zeros(n_episodes)
    epsilon_history = np.zeros(n_episodes)

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        total_elec = 0.0
        total_deg = 0.0
        total_low = 0.0

        while not env.done:
            # Choose action using epsilon-greedy
            action = agent.choose_action(state)

            # Take step in environment
            next_state, reward, done, info = env.step(action)

            # Q-Learning update (off-policy: uses max over next actions)
            agent.update(state, action, reward, next_state, done=done)

            # Accumulate logs
            total_reward += reward
            total_elec += info["electricity_cost"]
            total_deg += info["degradation_cost"]
            total_low += info["low_battery_penalty"]

            state = next_state

        # End of episode: decay exploration
        agent.decay_epsilon()

        # Store logs
        episode_rewards[ep] = total_reward
        episode_electricity[ep] = total_elec
        episode_degradation[ep] = total_deg
        episode_lowbat[ep] = total_low
        epsilon_history[ep] = agent.epsilon

        # Progress print every 500 episodes
        if (ep + 1) % 500 == 0:
            avg_reward = np.mean(episode_rewards[max(0, ep - 99):ep + 1])
            print(
                f"  [Q-Learning] Episode {ep + 1:5d}/{n_episodes} | "
                f"Avg Reward (last 100): {avg_reward:8.3f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    return {
        "rewards": episode_rewards,
        "electricity_costs": episode_electricity,
        "degradation_costs": episode_degradation,
        "low_battery_penalties": episode_lowbat,
        "epsilon_history": epsilon_history,
    }


def train_sarsa(env: EVChargingEnv, agent: SARSAAgent, n_episodes: int) -> dict:
    """
    Train a SARSA agent.

    The SARSA loop:
        1. Observe state s, choose action a
        2. Take action a, observe r, s'
        3. Choose next action a' (epsilon-greedy)
        4. Update: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
        5. s ← s', a ← a'

    Key difference from Q-learning: the update uses Q(s', a') where a'
    is the action actually chosen by the policy (not the max).

    Parameters
    ----------
    env : EVChargingEnv
        The environment instance.
    agent : SARSAAgent
        The SARSA agent.
    n_episodes : int
        Number of training episodes.

    Returns
    -------
    logs : dict
        Training logs including rewards, costs, epsilon history.
    """
    # ── Logging arrays ───────────────────────────────────────────
    episode_rewards = np.zeros(n_episodes)
    episode_electricity = np.zeros(n_episodes)
    episode_degradation = np.zeros(n_episodes)
    episode_lowbat = np.zeros(n_episodes)
    epsilon_history = np.zeros(n_episodes)

    for ep in range(n_episodes):
        state = env.reset()
        action = agent.choose_action(state)  # SARSA: choose initial action
        total_reward = 0.0
        total_elec = 0.0
        total_deg = 0.0
        total_low = 0.0

        while not env.done:
            # Take step in environment
            next_state, reward, done, info = env.step(action)

            # Choose next action (SARSA needs this BEFORE the update)
            next_action = agent.choose_action(next_state)

            # SARSA update (on-policy: uses the actual next action)
            agent.update(state, action, reward, next_state, next_action, done=done)

            # Accumulate logs
            total_reward += reward
            total_elec += info["electricity_cost"]
            total_deg += info["degradation_cost"]
            total_low += info["low_battery_penalty"]

            # Transition
            state = next_state
            action = next_action

        # End of episode: decay exploration
        agent.decay_epsilon()

        # Store logs
        episode_rewards[ep] = total_reward
        episode_electricity[ep] = total_elec
        episode_degradation[ep] = total_deg
        episode_lowbat[ep] = total_low
        epsilon_history[ep] = agent.epsilon

        # Progress print every 500 episodes
        if (ep + 1) % 500 == 0:
            avg_reward = np.mean(episode_rewards[max(0, ep - 99):ep + 1])
            print(
                f"  [SARSA]      Episode {ep + 1:5d}/{n_episodes} | "
                f"Avg Reward (last 100): {avg_reward:8.3f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    return {
        "rewards": episode_rewards,
        "electricity_costs": episode_electricity,
        "degradation_costs": episode_degradation,
        "low_battery_penalties": episode_lowbat,
        "epsilon_history": epsilon_history,
    }


def main():
    """
    Main training routine.

    Trains Q-Learning and SARSA agents with identical environment seeds,
    then saves all results to a compressed .npz file.
    """
    # ── Hyperparameters ──────────────────────────────────────────
    N_EPISODES = 5000
    ALPHA = 0.1          # learning rate
    GAMMA = 0.99         # discount factor
    EPSILON = 1.0        # initial exploration
    EPSILON_MIN = 0.01   # minimum exploration
    EPSILON_DECAY = 0.998  # decay per episode
    ENV_SEED = 42

    print("=" * 65)
    print("  EV Charging RL — Training Q-Learning & SARSA")
    print("=" * 65)
    print(f"  Episodes:      {N_EPISODES}")
    print(f"  Alpha (lr):    {ALPHA}")
    print(f"  Gamma:         {GAMMA}")
    print(f"  Epsilon:       {EPSILON} → {EPSILON_MIN} (decay={EPSILON_DECAY})")
    print(f"  Env seed:      {ENV_SEED}")
    print("=" * 65)

    # ── Create output directory ──────────────────────────────────
    os.makedirs("results", exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    #  Train Q-Learning
    # ══════════════════════════════════════════════════════════════
    print("\n▶ Training Q-Learning agent...")
    env_q = EVChargingEnv(seed=ENV_SEED)
    q_agent = QLearningAgent(
        n_states=env_q.n_states,
        n_actions=env_q.n_actions,
        alpha=ALPHA, gamma=GAMMA,
        epsilon=EPSILON, epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        seed=100,
    )
    q_logs = train_qlearning(env_q, q_agent, N_EPISODES)

    # ══════════════════════════════════════════════════════════════
    #  Train SARSA
    # ══════════════════════════════════════════════════════════════
    print("\n▶ Training SARSA agent...")
    env_s = EVChargingEnv(seed=ENV_SEED)
    sarsa_agent = SARSAAgent(
        n_states=env_s.n_states,
        n_actions=env_s.n_actions,
        alpha=ALPHA, gamma=GAMMA,
        epsilon=EPSILON, epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        seed=200,
    )
    s_logs = train_sarsa(env_s, sarsa_agent, N_EPISODES)

    # ══════════════════════════════════════════════════════════════
    #  Save all results
    # ══════════════════════════════════════════════════════════════
    print("\n▶ Saving results to results/training_results.npz ...")

    np.savez_compressed(
        "results/training_results.npz",
        # Q-Learning data
        q_rewards=q_logs["rewards"],
        q_electricity=q_logs["electricity_costs"],
        q_degradation=q_logs["degradation_costs"],
        q_lowbat=q_logs["low_battery_penalties"],
        q_epsilon=q_logs["epsilon_history"],
        q_table=q_agent.Q,
        q_policy=q_agent.get_policy(),
        # SARSA data
        s_rewards=s_logs["rewards"],
        s_electricity=s_logs["electricity_costs"],
        s_degradation=s_logs["degradation_costs"],
        s_lowbat=s_logs["low_battery_penalties"],
        s_epsilon=s_logs["epsilon_history"],
        s_table=sarsa_agent.Q,
        s_policy=sarsa_agent.get_policy(),
        # Environment info
        battery_grid=env_q.battery_grid,
        n_hours=np.array(env_q.n_hours),
        n_battery_levels=np.array(env_q.n_battery_levels),
    )

    print("\n✓ Training complete!")
    print(f"  Q-Learning final avg reward (last 100): "
          f"{np.mean(q_logs['rewards'][-100:]):.3f}")
    print(f"  SARSA      final avg reward (last 100): "
          f"{np.mean(s_logs['rewards'][-100:]):.3f}")


if __name__ == "__main__":
    main()
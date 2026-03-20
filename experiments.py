"""
experiments.py — Sensitivity Analysis Experiments
=================================================
Runs controlled experiments varying key hyperparameters:
  1. Battery degradation penalty weight
  2. Discount factor (gamma)

For each configuration, trains both Q-Learning and SARSA agents
and records final performance + learned policies.

Usage:
    python experiments.py

Output:
    results/sensitivity_degradation.npz
    results/sensitivity_gamma.npz

Authors: Sydney Nzunguli & Utkarsh Singh
Course:  Reinforcement Learning – CS, Université Paris-Saclay (2025–2026)
"""

import os
import numpy as np
from environment import EVChargingEnv
from agents import QLearningAgent, SARSAAgent
from train import train_qlearning, train_sarsa


def run_sensitivity_degradation():
    """
    Experiment 1: Vary battery degradation penalty.

    We test a range of degradation_fast values to see how agents
    shift from aggressive fast-charging to conservative slow-charging
    as the degradation cost increases.

    The slow degradation is kept proportional (always 1/4 of fast).
    """
    print("\n" + "=" * 65)
    print("  Experiment 1: Sensitivity to Degradation Penalty")
    print("=" * 65)

    # ── Parameter sweep ──────────────────────────────────────────
    degradation_values = [0.5, 1.0, 2.0, 4.0, 8.0]
    N_EPISODES = 3000  # fewer episodes per config for efficiency
    ALPHA = 0.1
    GAMMA = 0.99
    EPSILON_DECAY = 0.998

    results = {
        "degradation_values": np.array(degradation_values),
        "q_final_rewards": [],
        "s_final_rewards": [],
        "q_fast_charge_pcts": [],  # % of time fast charge is chosen
        "s_fast_charge_pcts": [],
        "q_policies": [],
        "s_policies": [],
    }

    for deg_fast in degradation_values:
        deg_slow = deg_fast / 4.0  # keep ratio consistent
        print(f"\n  ▶ degradation_fast = {deg_fast:.1f}, degradation_slow = {deg_slow:.2f}")

        # ── Train Q-Learning ─────────────────────────────────────
        env = EVChargingEnv(
            degradation_fast=deg_fast,
            degradation_slow=deg_slow,
            seed=42,
        )
        q_agent = QLearningAgent(
            n_states=env.n_states, n_actions=env.n_actions,
            alpha=ALPHA, gamma=GAMMA,
            epsilon=1.0, epsilon_min=0.01, epsilon_decay=EPSILON_DECAY,
            seed=100,
        )
        q_logs = train_qlearning(env, q_agent, N_EPISODES)

        # ── Train SARSA ──────────────────────────────────────────
        env2 = EVChargingEnv(
            degradation_fast=deg_fast,
            degradation_slow=deg_slow,
            seed=42,
        )
        s_agent = SARSAAgent(
            n_states=env2.n_states, n_actions=env2.n_actions,
            alpha=ALPHA, gamma=GAMMA,
            epsilon=1.0, epsilon_min=0.01, epsilon_decay=EPSILON_DECAY,
            seed=200,
        )
        s_logs = train_sarsa(env2, s_agent, N_EPISODES)

        # ── Collect results ──────────────────────────────────────
        q_final = np.mean(q_logs["rewards"][-200:])
        s_final = np.mean(s_logs["rewards"][-200:])

        # Count fast-charge frequency in learned policy
        q_policy = q_agent.get_policy()
        s_policy = s_agent.get_policy()
        q_fast_pct = np.mean(q_policy == EVChargingEnv.FAST_CHARGE) * 100
        s_fast_pct = np.mean(s_policy == EVChargingEnv.FAST_CHARGE) * 100

        results["q_final_rewards"].append(q_final)
        results["s_final_rewards"].append(s_final)
        results["q_fast_charge_pcts"].append(q_fast_pct)
        results["s_fast_charge_pcts"].append(s_fast_pct)
        results["q_policies"].append(q_policy)
        results["s_policies"].append(s_policy)

        print(f"    Q-Learning: avg reward = {q_final:.3f}, fast charge = {q_fast_pct:.1f}%")
        print(f"    SARSA:      avg reward = {s_final:.3f}, fast charge = {s_fast_pct:.1f}%")

    # ── Save results ─────────────────────────────────────────────
    np.savez_compressed(
        "results/sensitivity_degradation.npz",
        degradation_values=np.array(degradation_values),
        q_final_rewards=np.array(results["q_final_rewards"]),
        s_final_rewards=np.array(results["s_final_rewards"]),
        q_fast_charge_pcts=np.array(results["q_fast_charge_pcts"]),
        s_fast_charge_pcts=np.array(results["s_fast_charge_pcts"]),
        q_policies=np.array(results["q_policies"]),
        s_policies=np.array(results["s_policies"]),
    )
    print("\n  ✓ Saved to results/sensitivity_degradation.npz")


def run_sensitivity_gamma():
    """
    Experiment 2: Vary discount factor (gamma).

    Lower gamma → myopic agent (only cares about immediate cost).
    Higher gamma → far-sighted agent (anticipatory charging).

    We expect higher gamma to produce more proactive charging behavior
    (charging before the battery gets critically low).
    """
    print("\n" + "=" * 65)
    print("  Experiment 2: Sensitivity to Discount Factor (Gamma)")
    print("=" * 65)

    # ── Parameter sweep ──────────────────────────────────────────
    gamma_values = [0.5, 0.8, 0.9, 0.95, 0.99]
    N_EPISODES = 3000
    ALPHA = 0.1
    EPSILON_DECAY = 0.998

    results = {
        "gamma_values": np.array(gamma_values),
        "q_final_rewards": [],
        "s_final_rewards": [],
        "q_avg_battery": [],    # average battery level in last 200 episodes
        "s_avg_battery": [],
        "q_policies": [],
        "s_policies": [],
    }

    for gamma in gamma_values:
        print(f"\n  ▶ gamma = {gamma:.2f}")

        # ── Train Q-Learning ─────────────────────────────────────
        env = EVChargingEnv(seed=42)
        q_agent = QLearningAgent(
            n_states=env.n_states, n_actions=env.n_actions,
            alpha=ALPHA, gamma=gamma,
            epsilon=1.0, epsilon_min=0.01, epsilon_decay=EPSILON_DECAY,
            seed=100,
        )
        q_logs = train_qlearning(env, q_agent, N_EPISODES)

        # ── Train SARSA ──────────────────────────────────────────
        env2 = EVChargingEnv(seed=42)
        s_agent = SARSAAgent(
            n_states=env2.n_states, n_actions=env2.n_actions,
            alpha=ALPHA, gamma=gamma,
            epsilon=1.0, epsilon_min=0.01, epsilon_decay=EPSILON_DECAY,
            seed=200,
        )
        s_logs = train_sarsa(env2, s_agent, N_EPISODES)

        # ── Collect results ──────────────────────────────────────
        q_final = np.mean(q_logs["rewards"][-200:])
        s_final = np.mean(s_logs["rewards"][-200:])

        results["q_final_rewards"].append(q_final)
        results["s_final_rewards"].append(s_final)
        results["q_policies"].append(q_agent.get_policy())
        results["s_policies"].append(s_agent.get_policy())

        # Evaluate average battery level under learned policy
        for agent, label, avg_list in [
            (q_agent, "Q-Learning", results["q_avg_battery"]),
            (s_agent, "SARSA", results["s_avg_battery"]),
        ]:
            eval_env = EVChargingEnv(seed=999)
            battery_levels = []
            for _ in range(200):
                s = eval_env.reset()
                while not eval_env.done:
                    a = np.argmax(agent.Q[s])  # greedy evaluation
                    s, _, _, info = eval_env.step(a)
                    battery_levels.append(info["battery_pct"])
            avg_list.append(np.mean(battery_levels))

        print(f"    Q-Learning: avg reward = {q_final:.3f}, "
              f"avg battery = {results['q_avg_battery'][-1]:.1f}%")
        print(f"    SARSA:      avg reward = {s_final:.3f}, "
              f"avg battery = {results['s_avg_battery'][-1]:.1f}%")

    # ── Save results ─────────────────────────────────────────────
    np.savez_compressed(
        "results/sensitivity_gamma.npz",
        gamma_values=np.array(gamma_values),
        q_final_rewards=np.array(results["q_final_rewards"]),
        s_final_rewards=np.array(results["s_final_rewards"]),
        q_avg_battery=np.array(results["q_avg_battery"]),
        s_avg_battery=np.array(results["s_avg_battery"]),
        q_policies=np.array(results["q_policies"]),
        s_policies=np.array(results["s_policies"]),
    )
    print("\n  ✓ Saved to results/sensitivity_gamma.npz")


def main():
    os.makedirs("results", exist_ok=True)
    run_sensitivity_degradation()
    run_sensitivity_gamma()
    print("\n" + "=" * 65)
    print("  All experiments complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
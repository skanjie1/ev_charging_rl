"""
visualizations.py — Generate All Project Plots
===============================================
Loads training and experiment results, then produces publication-quality
figures for the project report:

  1. Learning curves (Q-Learning vs SARSA)
  2. Policy heatmaps (action chosen per state)
  3. Reward decomposition (cost breakdown)
  4. Electricity pricing profile
  5. Sensitivity analysis: degradation penalty
  6. Sensitivity analysis: discount factor (gamma)

Usage:
    python visualizations.py

Output:
    results/figures/*.png

Authors: Sydney Nzunguli & Utkarsh Singh
Course:  Reinforcement Learning – CS, Université Paris-Saclay (2025–2026)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from environment import EVChargingEnv


# ── Global plot style ────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

# Color palette
C_QLEARN = "#2563EB"   # blue
C_SARSA = "#DC2626"     # red
C_ELEC = "#F59E0B"      # amber
C_DEG = "#8B5CF6"       # purple
C_LOW = "#EF4444"       # red
ACTION_COLORS = ["#E5E7EB", "#3B82F6", "#EF4444"]  # no charge, slow, fast
ACTION_LABELS = ["No Charge", "Slow Charge", "Fast Charge"]


def smooth(data, window=100):
    """Simple moving average for smoothing learning curves."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def load_training_results():
    """Load the main training results from disk."""
    return np.load("results/training_results.npz")


def plot_learning_curves(data, figdir):
    """
    Figure 1: Learning curves — cumulative reward over episodes.

    Shows how Q-Learning (off-policy) and SARSA (on-policy) converge,
    with a smoothed line and faint raw data in the background.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Smoothed learning curves (both agents)
    ax = axes[0]
    window = 100
    q_smooth = smooth(data["q_rewards"], window)
    s_smooth = smooth(data["s_rewards"], window)
    episodes = np.arange(window, len(data["q_rewards"]) + 1)

    ax.plot(episodes, q_smooth, color=C_QLEARN, linewidth=2, label="Q-Learning")
    ax.plot(episodes, s_smooth, color=C_SARSA, linewidth=2, label="SARSA")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward (smoothed)")
    ax.set_title("Learning Curves: Q-Learning vs SARSA")
    ax.legend(frameon=True, fancybox=True)

    # Right: Epsilon decay
    ax = axes[1]
    ax.plot(data["q_epsilon"], color=C_QLEARN, linewidth=1.5, label="Q-Learning ε")
    ax.plot(data["s_epsilon"], color=C_SARSA, linewidth=1.5, label="SARSA ε")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon (exploration rate)")
    ax.set_title("Exploration Decay")
    ax.legend(frameon=True, fancybox=True)

    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "01_learning_curves.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 01_learning_curves.png")


def plot_policy_heatmaps(data, figdir):
    """
    Figure 2: Policy heatmaps — what action the agent takes in each state.

    X-axis: hour of day (0–23)
    Y-axis: battery level (0%–100%)
    Color:  No Charge (gray), Slow Charge (blue), Fast Charge (red)

    Expected pattern:
    - Charging concentrated in off-peak hours (night / early morning)
    - Fast charge only at low battery levels
    - No charge at high battery + peak price hours
    """
    n_battery = int(data["n_battery_levels"])
    n_hours = int(data["n_hours"])
    battery_grid = data["battery_grid"]

    # Custom colormap: gray → blue → red
    cmap = mcolors.ListedColormap(ACTION_COLORS)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, policy_key, title in [
        (axes[0], "q_policy", "Q-Learning Policy"),
        (axes[1], "s_policy", "SARSA Policy"),
    ]:
        policy = data[policy_key].reshape(n_battery, n_hours)
        im = ax.imshow(
            policy, aspect="auto", cmap=cmap, norm=norm,
            origin="lower", extent=[-0.5, n_hours - 0.5, -0.5, n_battery - 0.5],
        )
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Battery Level")
        ax.set_title(title)

        # Y-axis: show battery percentages
        tick_positions = np.arange(0, n_battery, 4)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([f"{battery_grid[i]:.0f}%" for i in tick_positions])

        # X-axis: hours
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 3)], rotation=45)

    # Shared legend
    legend_elements = [
        Patch(facecolor=ACTION_COLORS[i], label=ACTION_LABELS[i])
        for i in range(3)
    ]
    fig.legend(
        handles=legend_elements, loc="upper center",
        ncol=3, frameon=True, fontsize=11,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "02_policy_heatmaps.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 02_policy_heatmaps.png")


def plot_reward_decomposition(data, figdir):
    """
    Figure 3: Reward decomposition — breakdown of costs over training.

    Stacked area plot showing how electricity cost, degradation cost,
    and low-battery penalty evolve as the agent learns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    window = 200

    for ax, prefix, title in [
        (axes[0], "q", "Q-Learning"),
        (axes[1], "s", "SARSA"),
    ]:
        elec = smooth(data[f"{prefix}_electricity"], window)
        deg = smooth(data[f"{prefix}_degradation"], window)
        low = smooth(data[f"{prefix}_lowbat"], window)
        eps = np.arange(window, len(data[f"{prefix}_electricity"]) + 1)

        ax.stackplot(
            eps, elec, deg, low,
            colors=[C_ELEC, C_DEG, C_LOW],
            labels=["Electricity Cost", "Degradation", "Low Battery"],
            alpha=0.8,
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cost Component (smoothed)")
        ax.set_title(f"Reward Decomposition — {title}")
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "03_reward_decomposition.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 03_reward_decomposition.png")


def plot_pricing_profile(figdir):
    """
    Figure 4: Dynamic electricity pricing over 24 hours.

    Shows the sinusoidal price model to help interpret policy decisions.
    """
    env = EVChargingEnv()
    hours = np.arange(24)
    prices = [env.get_price(h) for h in hours]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(hours, prices, alpha=0.3, color=C_ELEC)
    ax.plot(hours, prices, color=C_ELEC, linewidth=2.5, marker="o", markersize=5)

    # Annotate peak and off-peak
    peak_h = hours[np.argmax(prices)]
    off_h = hours[np.argmin(prices)]
    ax.annotate(
        f"Peak: €{max(prices):.3f}",
        xy=(peak_h, max(prices)),
        xytext=(peak_h + 2, max(prices) + 0.005),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=10, color="#B45309",
    )
    ax.annotate(
        f"Off-peak: €{min(prices):.3f}",
        xy=(off_h, min(prices)),
        xytext=(off_h + 3, min(prices) - 0.01),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=10, color="#047857",
    )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Electricity Price (€/kWh)")
    ax.set_title("Dynamic Electricity Pricing (Sinusoidal Model)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 2)], rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "04_pricing_profile.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 04_pricing_profile.png")


def plot_sensitivity_degradation(figdir):
    """
    Figure 5: Sensitivity to degradation penalty.

    Shows how increasing degradation cost reduces fast-charging frequency
    and affects total reward for both agents.
    """
    data = np.load("results/sensitivity_degradation.npz")
    deg_vals = data["degradation_values"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Final reward vs degradation
    ax = axes[0]
    ax.plot(deg_vals, data["q_final_rewards"], "o-",
            color=C_QLEARN, linewidth=2, markersize=8, label="Q-Learning")
    ax.plot(deg_vals, data["s_final_rewards"], "s-",
            color=C_SARSA, linewidth=2, markersize=8, label="SARSA")
    ax.set_xlabel("Fast Charging Degradation Penalty")
    ax.set_ylabel("Average Reward (last 200 episodes)")
    ax.set_title("Reward vs Degradation Penalty")
    ax.legend(frameon=True)

    # Right: Fast-charge frequency vs degradation
    ax = axes[1]
    ax.plot(deg_vals, data["q_fast_charge_pcts"], "o-",
            color=C_QLEARN, linewidth=2, markersize=8, label="Q-Learning")
    ax.plot(deg_vals, data["s_fast_charge_pcts"], "s-",
            color=C_SARSA, linewidth=2, markersize=8, label="SARSA")
    ax.set_xlabel("Fast Charging Degradation Penalty")
    ax.set_ylabel("Fast Charge in Policy (%)")
    ax.set_title("Fast Charge Frequency vs Degradation Penalty")
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "05_sensitivity_degradation.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 05_sensitivity_degradation.png")


def plot_sensitivity_gamma(figdir):
    """
    Figure 6: Sensitivity to discount factor (gamma).

    Shows how higher gamma leads to higher average battery levels
    (anticipatory charging) and affects total reward.
    """
    data = np.load("results/sensitivity_gamma.npz")
    gamma_vals = data["gamma_values"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Final reward vs gamma
    ax = axes[0]
    ax.plot(gamma_vals, data["q_final_rewards"], "o-",
            color=C_QLEARN, linewidth=2, markersize=8, label="Q-Learning")
    ax.plot(gamma_vals, data["s_final_rewards"], "s-",
            color=C_SARSA, linewidth=2, markersize=8, label="SARSA")
    ax.set_xlabel("Discount Factor (γ)")
    ax.set_ylabel("Average Reward (last 200 episodes)")
    ax.set_title("Reward vs Discount Factor")
    ax.legend(frameon=True)

    # Right: Average battery level vs gamma
    ax = axes[1]
    ax.plot(gamma_vals, data["q_avg_battery"], "o-",
            color=C_QLEARN, linewidth=2, markersize=8, label="Q-Learning")
    ax.plot(gamma_vals, data["s_avg_battery"], "s-",
            color=C_SARSA, linewidth=2, markersize=8, label="SARSA")
    ax.set_xlabel("Discount Factor (γ)")
    ax.set_ylabel("Average Battery Level (%)")
    ax.set_title("Battery Maintenance vs Discount Factor")
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "06_sensitivity_gamma.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 06_sensitivity_gamma.png")


def plot_action_distribution(data, figdir):
    """
    Figure 7: Action distribution comparison.

    Bar chart showing how often each agent chooses each action
    in its learned policy.
    """
    q_policy = data["q_policy"]
    s_policy = data["s_policy"]

    q_counts = [np.mean(q_policy == a) * 100 for a in range(3)]
    s_counts = [np.mean(s_policy == a) * 100 for a in range(3)]

    x = np.arange(3)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, q_counts, width, color=C_QLEARN, label="Q-Learning", alpha=0.85)
    bars2 = ax.bar(x + width / 2, s_counts, width, color=C_SARSA, label="SARSA", alpha=0.85)

    ax.set_xlabel("Action")
    ax.set_ylabel("Frequency in Policy (%)")
    ax.set_title("Learned Action Distribution: Q-Learning vs SARSA")
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_LABELS)
    ax.legend(frameon=True)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "07_action_distribution.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 07_action_distribution.png")


def plot_value_function(data, figdir):
    """
    Figure 8: State value function V(s) = max_a Q(s,a).

    Heatmap showing how "valuable" each state is under the learned policy.
    High value = agent expects low future costs from this state.
    """
    n_battery = int(data["n_battery_levels"])
    n_hours = int(data["n_hours"])
    battery_grid = data["battery_grid"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, table_key, title in [
        (axes[0], "q_table", "Q-Learning: V(s)"),
        (axes[1], "s_table", "SARSA: V(s)"),
    ]:
        Q_table = data[table_key].reshape(n_battery, n_hours, 3)
        V = np.max(Q_table, axis=2)  # V(s) = max_a Q(s, a)

        im = ax.imshow(
            V, aspect="auto", cmap="RdYlGn",
            origin="lower", extent=[-0.5, n_hours - 0.5, -0.5, n_battery - 0.5],
        )
        plt.colorbar(im, ax=ax, label="State Value V(s)")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Battery Level")
        ax.set_title(title)

        tick_positions = np.arange(0, n_battery, 4)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([f"{battery_grid[i]:.0f}%" for i in tick_positions])
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 3)], rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "08_value_function.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 08_value_function.png")


def main():
    """Generate all figures."""
    figdir = "results/figures"
    os.makedirs(figdir, exist_ok=True)

    print("=" * 55)
    print("  Generating all figures...")
    print("=" * 55)

    # Load main training results
    data = load_training_results()

    # Generate each figure
    plot_learning_curves(data, figdir)
    plot_policy_heatmaps(data, figdir)
    plot_reward_decomposition(data, figdir)
    plot_pricing_profile(figdir)
    plot_action_distribution(data, figdir)
    plot_value_function(data, figdir)

    # These require sensitivity experiment results
    if os.path.exists("results/sensitivity_degradation.npz"):
        plot_sensitivity_degradation(figdir)
    else:
        print("  ⚠ Skipping degradation sensitivity (run experiments.py first)")

    if os.path.exists("results/sensitivity_gamma.npz"):
        plot_sensitivity_gamma(figdir)
    else:
        print("  ⚠ Skipping gamma sensitivity (run experiments.py first)")

    print("\n" + "=" * 55)
    print(f"  All figures saved to {figdir}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
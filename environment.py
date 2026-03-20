"""
environment.py — EV Charging MDP Environment
=============================================
Simulates an Electric Vehicle charging scenario with:
- Dynamic (sinusoidal) electricity pricing
- Battery degradation modeling
- Stochastic energy consumption from driving

State:  (battery_level, hour_of_day)
Action: 0 = no charge, 1 = slow charge, 2 = fast charge

Authors: Sydney Nzunguli & Utkarsh Singh
Course:  Reinforcement Learning – CS, Université Paris-Saclay (2025–2026)
"""

import numpy as np


class EVChargingEnv:
    """
    Finite MDP environment for EV charging under dynamic pricing.

    The agent decides each hour whether to charge (and how fast),
    balancing electricity cost, battery degradation, and the risk
    of running out of charge.
    """

    # ── Action constants ──────────────────────────────────────────
    NO_CHARGE = 0
    SLOW_CHARGE = 1
    FAST_CHARGE = 2
    ACTIONS = [NO_CHARGE, SLOW_CHARGE, FAST_CHARGE]
    ACTION_NAMES = ["No Charge", "Slow Charge", "Fast Charge"]

    def __init__(
        self,
        battery_levels: int = 21,          # 0%, 5%, 10%, ..., 100%
        hours: int = 24,                   # one full day cycle
        slow_charge_rate: float = 10.0,    # % gained per hour (slow)
        fast_charge_rate: float = 25.0,    # % gained per hour (fast)
        consumption_mean: float = 8.0,     # mean % consumed per hour
        consumption_std: float = 3.0,      # std of consumption noise
        price_base: float = 0.10,          # base electricity price (€/kWh)
        price_amplitude: float = 0.08,     # amplitude of sinusoidal variation
        degradation_slow: float = 0.5,     # degradation penalty for slow charge
        degradation_fast: float = 2.0,     # degradation penalty for fast charge
        low_battery_threshold: float = 20.0,  # % below which penalty applies
        low_battery_penalty: float = 5.0,  # penalty for being below threshold
        gamma: float = 0.99,              # discount factor (used by agents)
        seed: int = None,
    ):
        """
        Initialize the EV charging environment.

        Parameters
        ----------
        battery_levels : int
            Number of discrete battery levels (e.g., 21 → 0%, 5%, ..., 100%).
        hours : int
            Number of hours in one episode (one day = 24).
        slow_charge_rate : float
            Percentage points of battery gained per hour with slow charging.
        fast_charge_rate : float
            Percentage points of battery gained per hour with fast charging.
        consumption_mean : float
            Average battery percentage consumed per hour (driving).
        consumption_std : float
            Standard deviation of hourly consumption noise.
        price_base : float
            Base electricity price in €/kWh.
        price_amplitude : float
            Amplitude of the sinusoidal price variation.
        degradation_slow : float
            Penalty factor for slow charging (battery wear).
        degradation_fast : float
            Penalty factor for fast charging (battery wear).
        low_battery_threshold : float
            Battery % below which the low-battery penalty kicks in.
        low_battery_penalty : float
            Magnitude of penalty when battery is below threshold.
        gamma : float
            Discount factor (stored here for convenience, used by agents).
        seed : int or None
            Random seed for reproducibility.
        """
        # ── Store parameters ─────────────────────────────────────
        self.n_battery_levels = battery_levels
        self.n_hours = hours
        self.n_states = battery_levels * hours
        self.n_actions = len(self.ACTIONS)

        self.slow_charge_rate = slow_charge_rate
        self.fast_charge_rate = fast_charge_rate
        self.consumption_mean = consumption_mean
        self.consumption_std = consumption_std

        self.price_base = price_base
        self.price_amplitude = price_amplitude

        self.degradation_slow = degradation_slow
        self.degradation_fast = degradation_fast

        self.low_battery_threshold = low_battery_threshold
        self.low_battery_penalty = low_battery_penalty

        self.gamma = gamma

        # ── Battery level grid: [0, 5, 10, ..., 100] ────────────
        self.battery_grid = np.linspace(0, 100, battery_levels)
        self.battery_step = self.battery_grid[1] - self.battery_grid[0]

        # ── Random number generator ──────────────────────────────
        self.rng = np.random.default_rng(seed)

        # ── Episode state ────────────────────────────────────────
        self.battery_idx = None   # index into battery_grid
        self.hour = None          # current hour (0–23)
        self.done = False

    # ══════════════════════════════════════════════════════════════
    #  State encoding / decoding
    # ══════════════════════════════════════════════════════════════

    def state_to_idx(self, battery_idx: int, hour: int) -> int:
        """Convert (battery_idx, hour) pair to a flat state index."""
        return battery_idx * self.n_hours + hour

    def idx_to_state(self, idx: int) -> tuple:
        """Convert flat state index back to (battery_idx, hour)."""
        battery_idx = idx // self.n_hours
        hour = idx % self.n_hours
        return battery_idx, hour

    # ══════════════════════════════════════════════════════════════
    #  Electricity pricing model
    # ══════════════════════════════════════════════════════════════

    def get_price(self, hour: int) -> float:
        """
        Sinusoidal electricity price: peaks at noon (hour=12),
        cheapest at midnight (hour=0).

        price(t) = base + amplitude * sin(π * t / 12)

        Returns a value in [base - amplitude, base + amplitude].
        """
        return self.price_base + self.price_amplitude * np.sin(np.pi * hour / 12.0)

    # ══════════════════════════════════════════════════════════════
    #  Core environment interface
    # ══════════════════════════════════════════════════════════════

    def reset(self, initial_battery_pct: float = None) -> int:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        initial_battery_pct : float or None
            Starting battery percentage. If None, random between 40–80%.

        Returns
        -------
        state : int
            Flat state index for the initial state.
        """
        # Random starting battery if not specified
        if initial_battery_pct is None:
            initial_battery_pct = self.rng.uniform(40, 80)

        # Snap to nearest grid level
        self.battery_idx = int(np.clip(
            np.round(initial_battery_pct / self.battery_step),
            0, self.n_battery_levels - 1
        ))
        self.hour = 0
        self.done = False

        return self.state_to_idx(self.battery_idx, self.hour)

    def step(self, action: int) -> tuple:
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int
            0 = no charge, 1 = slow charge, 2 = fast charge.

        Returns
        -------
        next_state : int
            Flat state index after the transition.
        reward : float
            Immediate reward (negative cost).
        done : bool
            Whether the episode has ended (24 hours elapsed).
        info : dict
            Breakdown of reward components for analysis.
        """
        assert not self.done, "Episode is over. Call reset()."
        assert action in self.ACTIONS, f"Invalid action: {action}"

        current_battery = self.battery_grid[self.battery_idx]
        price = self.get_price(self.hour)

        # ── 1. Apply charging ────────────────────────────────────
        charge_gained = 0.0
        electricity_cost = 0.0
        degradation_cost = 0.0

        if action == self.SLOW_CHARGE:
            charge_gained = self.slow_charge_rate
            electricity_cost = price * self.slow_charge_rate / 100.0
            degradation_cost = self.degradation_slow
        elif action == self.FAST_CHARGE:
            charge_gained = self.fast_charge_rate
            electricity_cost = price * self.fast_charge_rate / 100.0
            degradation_cost = self.degradation_fast

        # ── 2. Apply stochastic consumption ──────────────────────
        consumption = max(0, self.rng.normal(self.consumption_mean, self.consumption_std))

        # ── 3. Update battery level ──────────────────────────────
        new_battery = np.clip(current_battery + charge_gained - consumption, 0, 100)

        # Snap to nearest grid level
        self.battery_idx = int(np.clip(
            np.round(new_battery / self.battery_step),
            0, self.n_battery_levels - 1
        ))

        # ── 4. Compute low-battery penalty ───────────────────────
        actual_battery = self.battery_grid[self.battery_idx]
        low_penalty = 0.0
        if actual_battery < self.low_battery_threshold:
            # Penalty proportional to how far below threshold
            low_penalty = self.low_battery_penalty * (
                (self.low_battery_threshold - actual_battery) / self.low_battery_threshold
            )

        # ── 5. Total reward (negative = cost) ────────────────────
        reward = -(electricity_cost + degradation_cost + low_penalty)

        # ── 6. Advance time ──────────────────────────────────────
        self.hour += 1
        if self.hour >= self.n_hours:
            self.done = True

        next_state = self.state_to_idx(self.battery_idx, min(self.hour, self.n_hours - 1))

        info = {
            "electricity_cost": electricity_cost,
            "degradation_cost": degradation_cost,
            "low_battery_penalty": low_penalty,
            "battery_pct": actual_battery,
            "price": price,
            "hour": self.hour - 1,
        }

        return next_state, reward, self.done, info

    # ══════════════════════════════════════════════════════════════
    #  Utility
    # ══════════════════════════════════════════════════════════════

    def get_state_description(self, state_idx: int) -> str:
        """Human-readable description of a state."""
        b_idx, hour = self.idx_to_state(state_idx)
        return f"Battery: {self.battery_grid[b_idx]:.0f}%, Hour: {hour}:00"


# ══════════════════════════════════════════════════════════════════
#  Quick sanity check
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    env = EVChargingEnv(seed=42)
    state = env.reset(initial_battery_pct=60)
    print(f"Initial state: {env.get_state_description(state)}")
    print(f"State space size: {env.n_states}")
    print(f"Action space size: {env.n_actions}")
    print()

    # Run one episode with random actions
    total_reward = 0
    while not env.done:
        action = np.random.choice(env.ACTIONS)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        print(
            f"  Hour {info['hour']:2d} | Action: {env.ACTION_NAMES[action]:11s} | "
            f"Battery: {info['battery_pct']:5.1f}% | Price: €{info['price']:.3f} | "
            f"Reward: {reward:.3f}"
        )

    print(f"\nTotal episode reward: {total_reward:.3f}")
"""
Stochastic renewable generation models.

Wind:  Ornstein-Uhlenbeck process for wind speed -> nonlinear turbine power curve.
Solar: deterministic diurnal clear-sky curve x Beta-distributed clearness index.
"""

import numpy as np


class WindModel:
    """
    OU process: dX_t = theta*(mu - X_t)*dt + sigma*dW_t, discretized (Euler-Maruyama).
    Converts wind speed (m/s) to normalized power output in [0, 1] via a turbine power curve.
    """

    def __init__(self, mu=8.0, theta=0.15, sigma=1.5, dt=1.0,
                 v_cutin=3.0, v_rated=12.0, v_cutout=25.0, rng=None):
        self.mu, self.theta, self.sigma, self.dt = mu, theta, sigma, dt
        self.v_cutin, self.v_rated, self.v_cutout = v_cutin, v_rated, v_cutout
        self.rng = rng if rng is not None else np.random.default_rng()
        self.speed = mu

    def reset(self, speed=None):
        self.speed = speed if speed is not None else self.mu
        return self.speed

    def step(self):
        z = self.rng.standard_normal()
        self.speed = self.speed + self.theta * (self.mu - self.speed) * self.dt \
            + self.sigma * np.sqrt(self.dt) * z
        self.speed = max(0.0, self.speed)  # wind speed can't be negative
        return self.speed

    def power_output(self, speed=None):
        """Normalized power in [0, 1] (multiply by rated_capacity_mw for actual power)."""
        v = self.speed if speed is None else speed
        if v < self.v_cutin or v >= self.v_cutout:
            return 0.0
        if v < self.v_rated:
            # cubic ramp: power ~ v^3 between cut-in and rated speed
            return ((v - self.v_cutin) / (self.v_rated - self.v_cutin)) ** 3
        return 1.0  # rated output between v_rated and v_cutout


class SolarModel:
    """
    Deterministic diurnal clear-sky curve (half-sine between sunrise/sunset)
    modulated by a stochastic Beta-distributed clearness index per step.
    """

    def __init__(self, sunrise_hour=6.0, sunset_hour=18.0, alpha=6.0, beta=2.0, rng=None):
        self.sunrise, self.sunset = sunrise_hour, sunset_hour
        self.alpha, self.beta = alpha, beta
        self.rng = rng if rng is not None else np.random.default_rng()

    def clear_sky_index(self, hour_of_day: float) -> float:
        """Deterministic clear-sky irradiance fraction in [0, 1], 0 outside daylight hours."""
        if hour_of_day <= self.sunrise or hour_of_day >= self.sunset:
            return 0.0
        day_len = self.sunset - self.sunrise
        frac = (hour_of_day - self.sunrise) / day_len  # in (0, 1)
        return np.sin(np.pi * frac)  # half-sine: 0 at sunrise/sunset, 1 at solar noon

    def step(self, hour_of_day: float) -> float:
        """Normalized power output in [0, 1] for the given hour of day."""
        clear_sky = self.clear_sky_index(hour_of_day)
        if clear_sky <= 0.0:
            return 0.0
        clearness = self.rng.beta(self.alpha, self.beta)  # cloud attenuation factor in [0, 1]
        return clear_sky * clearness


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- Wind: check OU stationary statistics match theory ---
    wind = WindModel(mu=8.0, theta=0.15, sigma=1.5, dt=1.0, rng=rng)
    speeds = [wind.step() for _ in range(20000)]
    speeds = np.array(speeds[5000:])  # discard burn-in
    theoretical_var = wind.sigma ** 2 / (2 * wind.theta)
    print("=== Wind (OU process) ===")
    print(f"Empirical mean speed:    {speeds.mean():.3f} m/s (target mu = {wind.mu})")
    print(f"Empirical variance:      {speeds.var():.3f} (theoretical sigma^2/(2*theta) = {theoretical_var:.3f})")
    print(f"Min speed: {speeds.min():.3f}, Max speed: {speeds.max():.3f}")

    powers = [wind.power_output(v) for v in speeds]
    print(f"Power output range: [{min(powers):.3f}, {max(powers):.3f}] (normalized)")
    print(f"Fraction of time at zero power (below cut-in or above cut-out): {np.mean(np.array(powers) == 0.0):.3f}")
    print(f"Fraction of time at rated power: {np.mean(np.array(powers) == 1.0):.3f}")

    # --- Solar: check diurnal pattern is physically sane ---
    solar = SolarModel(rng=rng)
    print("\n=== Solar (clear-sky x Beta clearness) ===")
    for hour in [0, 3, 6, 9, 12, 15, 18, 21]:
        samples = [solar.step(hour) for _ in range(5000)]
        print(f"Hour {hour:>2.0f}:00  mean output = {np.mean(samples):.3f}  "
              f"(clear-sky factor = {solar.clear_sky_index(hour):.3f})")

    assert solar.clear_sky_index(0) == 0.0 and solar.clear_sky_index(23) == 0.0, "Should be zero at night"
    assert solar.clear_sky_index(12) == 1.0, "Should peak at solar noon"
    print("\nSanity checks passed: zero at night, peak at solar noon.")
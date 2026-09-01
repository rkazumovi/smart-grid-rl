"""
Price-elastic demand response model.

D(p) = D_baseline * (p / p_ref)^epsilon,  epsilon < 0 (demand falls as price rises)
Price ratio is clipped before the power law to avoid unphysical blow-up as p -> 0.
"""

import numpy as np

class DemandResponseModel:
    def __init__(self, elasticity=-0.15, reference_price=50.0,
                 min_price_ratio=0.2, max_price_ratio=4.0):
        assert elasticity < 0, "Electricity demand elasticity should be negative (price up -> demand down)"
        self.elasticity = elasticity
        self.reference_price = reference_price
        self.min_price_ratio = min_price_ratio
        self.max_price_ratio = max_price_ratio

    def response(self, baseline_demand_mw: float, price: float) -> float:
        """Actual demand (MW) given a baseline load and a price signal ($/MWh)."""
        price_ratio = np.clip(price / self.reference_price, self.min_price_ratio, self.max_price_ratio)
        return baseline_demand_mw * price_ratio ** self.elasticity


def diurnal_baseline_load(hour_of_day: float, base=0.6, morning_peak=0.25, evening_peak=0.35) -> float:
    """Normalized load shape in roughly [0.4, 1.0]: a base load plus morning + evening peaks."""
    morning = morning_peak * np.exp(-((hour_of_day - 8.0) ** 2) / (2 * 1.5 ** 2))
    evening = evening_peak * np.exp(-((hour_of_day - 19.0) ** 2) / (2 * 2.0 ** 2))
    return base + morning + evening


if __name__ == "__main__":
    dr = DemandResponseModel(elasticity=-0.15, reference_price=50.0)
    baseline = 100.0  # MW

    print("=== Price elasticity sanity checks ===")
    at_ref = dr.response(baseline, 50.0)
    print(f"At reference price ($50/MWh): demand = {at_ref:.3f} MW (should equal baseline exactly)")
    assert abs(at_ref - baseline) < 1e-9

    doubled = dr.response(baseline, 100.0)
    print(f"Price doubled to $100/MWh:    demand = {doubled:.3f} MW ({(doubled/baseline-1)*100:+.1f}%, should decrease)")
    assert doubled < baseline

    halved = dr.response(baseline, 25.0)
    print(f"Price halved to $25/MWh:      demand = {halved:.3f} MW ({(halved/baseline-1)*100:+.1f}%, should increase)")
    assert halved > baseline

    near_zero = dr.response(baseline, 0.01)
    print(f"Price near zero ($0.01/MWh):  demand = {near_zero:.3f} MW (clipped, should NOT blow up to infinity)")
    assert near_zero < baseline * 5  # sane bound, not exploding

    print("\n=== Diurnal baseline load shape (normalized, x system peak MW) ===")
    for hour in [0, 6, 8, 12, 17, 19, 22]:
        print(f"Hour {hour:>2}:00  load factor = {diurnal_baseline_load(hour):.3f}")

    print("\nAll sanity checks passed.")
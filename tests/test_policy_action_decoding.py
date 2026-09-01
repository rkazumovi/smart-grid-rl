"""Tests decode_action's rescaling math (policy_inference.py) against hand-computed
boundary cases -- full-capacity dispatch, zero dispatch, full charge/discharge -- using a
lightweight stand-in for GridEnv rather than a real one, so this test doesn't need a
trained RL checkpoint, stable_baselines3, or torch to run. decode_action only reads three
attributes off its `env` argument (GEN_BUSES, GEN_CAPACITY_MW, BATTERY_MAX_POWER_MW), so a
plain object exposing those three is a faithful stand-in for the one real formula under
test -- it is not attempting to fake GridEnv's actual physics. Note this file still
transitively requires gymnasium and pandapower to be installed, since
policy_inference.py's own module-level `from environment.grid_env import GridEnv` import
pulls those in regardless of what this test itself uses (see requirements-api.txt, which
CI installs before running this suite).
"""
import numpy as np
import pytest

from grid_intelligence.policy_inference import decode_action


class _FakeEnv:
    GEN_BUSES = [1, 2, 5, 7]
    GEN_CAPACITY_MW = {1: 80.0, 2: 60.0, 5: 40.0, 7: 40.0}
    BATTERY_MAX_POWER_MW = 2.5


@pytest.fixture
def env():
    return _FakeEnv()


def test_full_positive_action_dispatches_full_capacity(env):
    # action = [gen_P(4), gen_V(4), battery(1), price(1)], all entries = +1.
    action = np.ones(10)
    decoded = decode_action(env, action)

    assert decoded["gen_dispatch_mw"] == {1: 80.0, 2: 60.0, 5: 40.0, 7: 40.0}
    assert decoded["total_gen_mw"] == pytest.approx(220.0)
    assert decoded["battery_power_mw"] == pytest.approx(2.5)  # +1 -> full discharge
    assert decoded["price_signal_usd_per_mwh"] == pytest.approx(100.0)  # 50*(1+1)


def test_full_negative_action_dispatches_zero(env):
    action = -np.ones(10)
    decoded = decode_action(env, action)

    assert decoded["gen_dispatch_mw"] == {1: 0.0, 2: 0.0, 5: 0.0, 7: 0.0}
    assert decoded["total_gen_mw"] == pytest.approx(0.0)
    assert decoded["battery_power_mw"] == pytest.approx(-2.5)  # -1 -> full charge
    assert decoded["price_signal_usd_per_mwh"] == pytest.approx(0.0)  # 50*(1-1)


def test_zero_action_dispatches_half_capacity(env):
    # frac = (0 + 1) / 2 = 0.5 for each generator.
    action = np.zeros(10)
    decoded = decode_action(env, action)

    assert decoded["gen_dispatch_mw"] == {1: 40.0, 2: 30.0, 5: 20.0, 7: 20.0}
    assert decoded["total_gen_mw"] == pytest.approx(110.0)
    assert decoded["battery_power_mw"] == pytest.approx(0.0)
    assert decoded["price_signal_usd_per_mwh"] == pytest.approx(50.0)


def test_action_outside_valid_range_is_clipped(env):
    # decode_action clips to [-1, 1] before rescaling -- an action of +5 must behave
    # identically to an action of exactly +1, not extrapolate past rated capacity.
    action = np.full(10, 5.0)
    decoded = decode_action(env, action)
    assert decoded["gen_dispatch_mw"] == {1: 80.0, 2: 60.0, 5: 40.0, 7: 40.0}
    assert decoded["battery_power_mw"] == pytest.approx(2.5)
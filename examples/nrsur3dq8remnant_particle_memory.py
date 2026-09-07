from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memorie import (  # noqa: E402
    nrsur3dq8_particle_displacement_memory_mode,
    nrsur3dq8_remnant_state,
)


class _MockRemnantFit:
    def all(self, q: float, chi_a: np.ndarray, chi_b: np.ndarray):
        del q, chi_a, chi_b
        return (
            0.95,
            np.array([0.0, 0.0, 0.7]),
            np.array([3.0e-3, -4.0e-3, 0.0]),
            1.0e-4,
            np.array([0.0, 0.0, 2.0e-4]),
            np.array([1.0e-5, 1.0e-5, 0.0]),
        )


def _self_test() -> None:
    memory, state = nrsur3dq8_particle_displacement_memory_mode(
        2,
        0,
        2.0,
        fit=_MockRemnantFit(),
    )
    np.testing.assert_allclose(state.mass, 0.95, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(state.particle.speed, 5.0e-3, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(state.particle.theta, np.pi / 2.0, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(memory.imag, 0.0, rtol=0.0, atol=1.0e-15)
    state_only = nrsur3dq8_remnant_state(2.0, fit=_MockRemnantFit())
    np.testing.assert_allclose(state_only.kick_velocity, state.kick_velocity)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, default=2.0)
    parser.add_argument("--chi-a-z", type=float, default=0.0)
    parser.add_argument("--chi-b-z", type=float, default=0.0)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--total-mass", type=float, default=1.0)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--gravitational-constant", type=float, default=1.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print("NRSur3dq8Remnant particle-memory self-test passed")
        return 0

    chi_a = (0.0, 0.0, args.chi_a_z)
    chi_b = (0.0, 0.0, args.chi_b_z)
    try:
        memory, state = nrsur3dq8_particle_displacement_memory_mode(
            args.ell,
            args.m,
            args.q,
            chi_a,
            chi_b,
            total_mass=args.total_mass,
            distance=args.distance,
            gravitational_constant=args.gravitational_constant,
        )
    except (ImportError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Mf / M = {state.mass / args.total_mass:.16e}")
    print("chif = [" + ", ".join(f"{value:.16e}" for value in state.final_spin) + "]")
    print("vf = [" + ", ".join(f"{value:.16e}" for value in state.kick_velocity) + "]")
    print(f"|vf| = {state.particle.speed:.16e}")
    print(f"theta_v = {state.particle.theta:.16e}")
    print(f"phi_v = {state.particle.phi:.16e}")
    print(f"Delta h_{{{args.ell},{args.m}}} = {memory.real:.16e}{memory.imag:+.16e}j")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

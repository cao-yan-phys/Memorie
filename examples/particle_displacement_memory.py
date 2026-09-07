from __future__ import annotations

import argparse
import json
from math import pi
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import eval_legendre

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memorie.particle_displacement import (
    NullParticle,
    TimelikeParticle,
    particle_displacement_memory_mode,
    outflow_speed_transfer,
)


def _load_state(path: Path | None) -> tuple[TimelikeParticle | NullParticle, ...]:
    if path is None:
        return ()
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain a JSON array")
    particles: list[TimelikeParticle | NullParticle] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"entry {index} in {path} must be a JSON object")
        try:
            if float(entry.get("speed", 1.0)) == 1.0:
                particles.append(NullParticle(**entry))
            else:
                particles.append(TimelikeParticle(**entry))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid particle {index} in {path}: {exc}") from exc
    return tuple(particles)


def _integral_transfer(ell: int, speed: float) -> float:
    if speed == 0.0:
        return 0.0
    if speed == 1.0:
        return 1.0
    integral, _error = quad(
        lambda z: eval_legendre(ell, z) / (1.0 - speed * z) ** 3,
        -1.0,
        1.0,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
    )
    return (1.0 - speed**2) ** 2 * integral / (2.0 * speed)


def _self_test() -> None:
    for ell in range(2, 7):
        if outflow_speed_transfer(ell, 0.0) != 0.0:
            raise AssertionError("A_l(0) must vanish")
        if outflow_speed_transfer(ell, 1.0) != 1.0:
            raise AssertionError("A_l(1) must equal one")
        for speed in (0.05, 0.4, 0.9):
            np.testing.assert_allclose(
                outflow_speed_transfer(ell, speed),
                _integral_transfer(ell, speed),
                rtol=2.0e-11,
                atol=2.0e-13,
            )

    particle = TimelikeParticle(rest_mass=1.7, speed=0.8, theta=0.9, phi=1.1)
    outgoing = particle_displacement_memory_mode(3, 2, final_particles=(particle,))
    np.testing.assert_allclose(
        particle_displacement_memory_mode(
            3,
            2,
            initial_particles=(particle,),
            final_particles=(particle,),
        ),
        0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        particle_displacement_memory_mode(3, 2, initial_particles=(particle,)),
        -outgoing,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    negative_m = particle_displacement_memory_mode(3, -2, final_particles=(particle,))
    np.testing.assert_allclose(negative_m, np.conjugate(outgoing), rtol=1.0e-14, atol=1.0e-14)

    null_particle = NullParticle(energy=2.3, theta=0.6, phi=-0.4)
    null_outgoing = particle_displacement_memory_mode(3, 2, final_particles=(null_particle,))
    doubled_null = NullParticle(energy=4.6, theta=0.6, phi=-0.4)
    np.testing.assert_allclose(
        particle_displacement_memory_mode(3, 2, final_particles=(doubled_null,)),
        2.0 * null_outgoing,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        particle_displacement_memory_mode(
            3,
            2,
            initial_particles=(null_particle,),
            final_particles=(null_particle,),
        ),
        0.0,
        atol=1.0e-14,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ell", type=int)
    parser.add_argument("--m", type=int)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--final-state", type=Path)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--gravitational-constant", type=float, default=1.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        print("particle displacement-memory self-test passed")
        return 0
    if args.ell is None or args.m is None:
        parser.error("--ell and --m are required unless --self-test is used")
    if args.initial_state is None and args.final_state is None:
        parser.error("provide --initial-state, --final-state, or both")

    try:
        value = particle_displacement_memory_mode(
            args.ell,
            args.m,
            initial_particles=_load_state(args.initial_state),
            final_particles=_load_state(args.final_state),
            distance=args.distance,
            gravitational_constant=args.gravitational_constant,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Delta h_{{{args.ell},{args.m}}} = {value.real:.16e}{value.imag:+.16e}j")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

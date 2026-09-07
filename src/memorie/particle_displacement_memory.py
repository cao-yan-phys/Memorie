from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from math import factorial, pi, sqrt
from pathlib import Path

import numpy as np
from scipy.special import hyp2f1

try:
    from scipy.special import sph_harm_y
except ImportError:
    from scipy.special import sph_harm

    def _spherical_harmonic(ell: int, emm: int, theta: float, phi: float) -> complex:
        return sph_harm(emm, ell, phi, theta)

else:

    def _spherical_harmonic(ell: int, emm: int, theta: float, phi: float) -> complex:
        return sph_harm_y(ell, emm, theta, phi)


@dataclass(frozen=True)
class TimelikeParticle:

    rest_mass: float
    speed: float
    theta: float
    phi: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.rest_mass) or self.rest_mass < 0.0:
            raise ValueError("rest_mass must be finite and nonnegative")
        if not np.isfinite(self.speed) or not 0.0 <= self.speed < 1.0:
            raise ValueError("speed must satisfy 0 <= speed < 1")
        if not np.isfinite(self.theta) or not 0.0 <= self.theta <= pi:
            raise ValueError("theta must satisfy 0 <= theta <= pi")
        if not np.isfinite(self.phi):
            raise ValueError("phi must be finite")

    @property
    def radial_momentum(self) -> float:

        return self.rest_mass * self.speed / sqrt(1.0 - self.speed**2)


@dataclass(frozen=True)
class NullParticle:

    energy: float
    theta: float
    phi: float
    speed: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.energy) or self.energy < 0.0:
            raise ValueError("energy must be finite and nonnegative")
        if self.speed != 1.0:
            raise ValueError("a NullParticle must have speed equal to 1")
        if not np.isfinite(self.theta) or not 0.0 <= self.theta <= pi:
            raise ValueError("theta must satisfy 0 <= theta <= pi")
        if not np.isfinite(self.phi):
            raise ValueError("phi must be finite")

    @property
    def radial_momentum(self) -> float:

        return self.energy


ParticleState = TimelikeParticle | NullParticle


def outflow_speed_transfer(ell: int, speed: float) -> float:

    if ell < 2:
        raise ValueError("ell must be at least 2")
    if not np.isfinite(speed) or not 0.0 <= speed <= 1.0:
        raise ValueError("speed must satisfy 0 <= speed <= 1")
    if speed == 0.0:
        return 0.0
    if speed == 1.0:
        return 1.0
    coefficient = (
        2.0 ** (ell - 1)
        * (ell + 1)
        * (ell + 2)
        * factorial(ell) ** 2
        / factorial(2 * ell + 1)
    )
    return float(
        coefficient
        * speed ** (ell - 1)
        * hyp2f1((ell - 1) / 2.0, ell / 2.0, ell + 1.5, speed**2)
    )


def _mode_weight(ell: int, emm: int, particle: ParticleState) -> complex:
    return (
        particle.radial_momentum
        * outflow_speed_transfer(ell, particle.speed)
        * np.conjugate(_spherical_harmonic(ell, emm, particle.theta, particle.phi))
    )


def particle_displacement_memory_mode(
    ell: int,
    emm: int,
    *,
    final_particles: Iterable[ParticleState] | None = None,
    initial_particles: Iterable[ParticleState] | None = None,
    gravitational_constant: float = 1.0,
    distance: float = 1.0,
) -> complex:

    if ell < 2 or abs(emm) > ell:
        raise ValueError("require ell >= 2 and abs(emm) <= ell")
    if not np.isfinite(gravitational_constant) or gravitational_constant <= 0.0:
        raise ValueError("gravitational_constant must be finite and positive")
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance must be finite and positive")

    final = tuple(final_particles or ())
    initial = tuple(initial_particles or ())
    if any(not isinstance(particle, (TimelikeParticle, NullParticle)) for particle in final + initial):
        raise TypeError("particle states must contain TimelikeParticle or NullParticle instances")

    prefactor = 16.0 * pi * gravitational_constant / distance
    prefactor *= sqrt(factorial(ell - 2) / factorial(ell + 2))
    final_sum = sum((_mode_weight(ell, emm, particle) for particle in final), 0.0j)
    initial_sum = sum((_mode_weight(ell, emm, particle) for particle in initial), 0.0j)
    return complex(prefactor * (final_sum - initial_sum))


def _load_state(path: Path | None) -> tuple[ParticleState, ...]:
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
    particles: list[ParticleState] = []
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ell", type=int)
    parser.add_argument("--m", type=int)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--final-state", type=Path)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--gravitational-constant", type=float, default=1.0)
    args = parser.parse_args()

    if args.ell is None or args.m is None:
        parser.error("--ell and --m are required")
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

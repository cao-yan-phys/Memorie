from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2
from typing import Any

import numpy as np

from .particle_displacement import TimelikeParticle, particle_displacement_memory_mode


@dataclass(frozen=True)
class NRSur3dq8RemnantState:

    mass: float
    mass_error: float
    final_spin: np.ndarray
    final_spin_error: np.ndarray
    kick_velocity: np.ndarray
    kick_velocity_error: np.ndarray
    particle: TimelikeParticle


def _spin_vector(value: Any, name: str) -> np.ndarray:
    spin = np.asarray(value, dtype=float)
    if spin.shape != (3,) or not np.all(np.isfinite(spin)):
        raise ValueError(f"{name} must be a finite length-three vector")
    if not np.allclose(spin[:2], 0.0, atol=1.0e-14, rtol=0.0):
        raise ValueError("NRSur3dq8Remnant accepts only nonprecessing spins")
    return spin


def _load_fit() -> Any:
    try:
        import surfinBH
    except ImportError as exc:
        raise ImportError(
            "NRSur3dq8Remnant requires the optional surfinBH dependency"
        ) from exc
    return surfinBH.LoadFits("NRSur3dq8Remnant")


def nrsur3dq8_remnant_state(
    q: float,
    chi_a: Any = (0.0, 0.0, 0.0),
    chi_b: Any = (0.0, 0.0, 0.0),
    *,
    total_mass: float = 1.0,
    fit: Any | None = None,
) -> NRSur3dq8RemnantState:

    if not np.isfinite(q) or q < 1.0:
        raise ValueError("q must be finite and at least 1")
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("total_mass must be finite and positive")
    spin_a = _spin_vector(chi_a, "chi_a")
    spin_b = _spin_vector(chi_b, "chi_b")
    loaded_fit = _load_fit() if fit is None else fit
    mf, chif, vf, mf_error, chif_error, vf_error = loaded_fit.all(q, spin_a, spin_b)

    mass = float(mf) * total_mass
    kick_velocity = np.asarray(vf, dtype=float)
    speed = float(np.linalg.norm(kick_velocity))
    if kick_velocity.shape != (3,) or not np.all(np.isfinite(kick_velocity)) or speed >= 1.0:
        raise ValueError("NRSur3dq8Remnant returned an invalid kick velocity")
    if speed == 0.0:
        theta, phi = 0.0, 0.0
    else:
        theta = acos(float(np.clip(kick_velocity[2] / speed, -1.0, 1.0)))
        phi = atan2(float(kick_velocity[1]), float(kick_velocity[0]))

    return NRSur3dq8RemnantState(
        mass=mass,
        mass_error=float(mf_error) * total_mass,
        final_spin=np.asarray(chif, dtype=float),
        final_spin_error=np.asarray(chif_error, dtype=float),
        kick_velocity=kick_velocity,
        kick_velocity_error=np.asarray(vf_error, dtype=float),
        particle=TimelikeParticle(mass, speed, theta, phi),
    )


def nrsur3dq8_particle_displacement_memory_mode(
    ell: int,
    emm: int,
    q: float,
    chi_a: Any = (0.0, 0.0, 0.0),
    chi_b: Any = (0.0, 0.0, 0.0),
    *,
    total_mass: float = 1.0,
    gravitational_constant: float = 1.0,
    distance: float = 1.0,
    fit: Any | None = None,
) -> tuple[complex, NRSur3dq8RemnantState]:

    state = nrsur3dq8_remnant_state(
        q,
        chi_a,
        chi_b,
        total_mass=total_mass,
        fit=fit,
    )
    memory = particle_displacement_memory_mode(
        ell,
        emm,
        final_particles=(state.particle,),
        gravitational_constant=gravitational_constant,
        distance=distance,
    )
    return memory, state

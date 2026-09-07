from __future__ import annotations

from collections.abc import Mapping
from math import pi, sqrt
from typing import Any

import numpy as np

from .core import (
    ModeDict,
    cumulative_integral,
    differentiate_modes,
    normalize_mode_dict,
    validate_mode_lengths,
    validate_time_grid,
)


def _validate_modes(h: ModeDict) -> None:
    for ell, emm in h:
        if ell < 2 or abs(emm) > ell:
            raise ValueError(f"invalid spin-weighted strain mode {(ell, emm)}")


def _vector_integral(t: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.column_stack([cumulative_integral(t, values[:, i]) for i in range(3)])


def _a_coefficient(ell: int, emm: int) -> float:
    return sqrt((ell - emm) * (ell + emm + 1)) / (ell * (ell + 1))


def _b_coefficient(ell: int, emm: int) -> float:
    numerator = (ell - 2) * (ell + 2) * (ell + emm) * (ell + emm - 1)
    denominator = (2 * ell - 1) * (2 * ell + 1)
    return sqrt(numerator / denominator) / (2 * ell)


def _c_coefficient(ell: int, emm: int) -> float:
    return 2.0 * emm / (ell * (ell + 1))


def _d_coefficient(ell: int, emm: int) -> float:
    numerator = (ell - 2) * (ell + 2) * (ell - emm) * (ell + emm)
    denominator = (2 * ell - 1) * (2 * ell + 1)
    return sqrt(numerator / denominator) / ell


def compute_poincare_fluxes(
    t: Any,
    h: Mapping[Any, Any],
    hdot: Mapping[Any, Any] | None = None,
) -> dict[str, np.ndarray]:
    t_arr = validate_time_grid(t)
    h_norm = normalize_mode_dict(h)
    validate_mode_lengths(t_arr, h_norm)
    _validate_modes(h_norm)
    hdot_norm = differentiate_modes(t_arr, h_norm) if hdot is None else normalize_mode_dict(hdot)
    validate_mode_lengths(t_arr, hdot_norm)
    if set(hdot_norm) != set(h_norm):
        raise ValueError("hdot must contain exactly the modes supplied in h")

    energy_flux = np.zeros_like(t_arr, dtype=float)
    p_plus = np.zeros_like(t_arr, dtype=complex)
    p_z = np.zeros_like(t_arr, dtype=complex)
    j_plus = np.zeros_like(t_arr, dtype=complex)
    j_minus = np.zeros_like(t_arr, dtype=complex)
    j_z = np.zeros_like(t_arr, dtype=complex)

    for (ell, emm), h_lm in h_norm.items():
        dh_lm = hdot_norm[(ell, emm)]
        energy_flux += np.abs(dh_lm) ** 2
        p_z += _c_coefficient(ell, emm) * dh_lm * np.conjugate(dh_lm)
        j_z += 1.0j * emm * np.conjugate(dh_lm) * h_lm

        next_mode = (ell, emm + 1)
        if next_mode in h_norm:
            dh_next = hdot_norm[next_mode]
            p_plus += _a_coefficient(ell, emm) * dh_lm * np.conjugate(dh_next)
            j_plus += (
                1.0j
                * sqrt((ell - emm) * (ell + emm + 1))
                * np.conjugate(dh_next)
                * h_lm
            )

        previous_ell_mode = (ell - 1, emm + 1)
        if ell > 2 and previous_ell_mode in h_norm:
            p_plus += _b_coefficient(ell, -emm) * dh_lm * np.conjugate(
                hdot_norm[previous_ell_mode]
            )

        next_ell_mode = (ell + 1, emm + 1)
        if next_ell_mode in h_norm:
            p_plus -= _b_coefficient(ell + 1, emm + 1) * dh_lm * np.conjugate(
                hdot_norm[next_ell_mode]
            )

        previous_ell_same_m = (ell - 1, emm)
        if ell > 2 and previous_ell_same_m in h_norm:
            p_z += _d_coefficient(ell, emm) * dh_lm * np.conjugate(
                hdot_norm[previous_ell_same_m]
            )

        next_ell_same_m = (ell + 1, emm)
        if next_ell_same_m in h_norm:
            p_z += _d_coefficient(ell + 1, emm) * dh_lm * np.conjugate(
                hdot_norm[next_ell_same_m]
            )

        previous_mode = (ell, emm - 1)
        if previous_mode in h_norm:
            j_minus += (
                1.0j
                * sqrt((ell + emm) * (ell - emm + 1))
                * np.conjugate(hdot_norm[previous_mode])
                * h_lm
            )

    energy_flux /= 16.0 * pi
    p_plus /= 8.0 * pi
    p_z /= 16.0 * pi
    linear_momentum_flux = np.column_stack(
        (p_plus.real, p_plus.imag, p_z.real)
    )
    angular_momentum_flux = -np.column_stack(
        (
            0.5 * (j_plus.real + j_minus.real),
            0.5 * (j_plus.imag - j_minus.imag),
            j_z.real,
        )
    ) / (16.0 * pi)

    return {
        "energy_flux": energy_flux,
        "linear_momentum_flux": linear_momentum_flux,
        "angular_momentum_flux": angular_momentum_flux,
        "energy_radiated": cumulative_integral(t_arr, energy_flux),
        "linear_momentum_radiated": _vector_integral(t_arr, linear_momentum_flux),
        "angular_momentum_radiated": _vector_integral(t_arr, angular_momentum_flux),
    }

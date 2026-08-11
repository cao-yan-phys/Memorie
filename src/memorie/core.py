"""Core vacuum nonlinear-null memory routines."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from importlib import resources
from math import factorial, pi, sqrt
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid
from sympy import N as sympy_N
from sympy.physics.wigner import wigner_3j

Mode = tuple[int, int]
ModeDict = dict[Mode, np.ndarray]
UnifiedCoeff = tuple[int, int, int, int, int, int, float, float, float]
_DEFAULT_CACHE_LMAX = 10


def parity_sign(n: int) -> float:
    """Return ``(-1)**n`` as a float."""

    return -1.0 if int(n) % 2 else 1.0


def _parse_mode_key(key: Any) -> Mode:
    if isinstance(key, str):
        parts = [part.strip() for part in key.strip().strip("()[]").split(",")]
        if len(parts) != 2:
            raise ValueError(f"cannot parse mode key {key!r}")
        return int(parts[0]), int(parts[1])
    if len(key) != 2:
        raise ValueError(f"cannot parse mode key {key!r}")
    return int(key[0]), int(key[1])


def normalize_mode_dict(h: Mapping[Any, Any]) -> ModeDict:
    """Normalize mode keys and convert arrays to NumPy arrays."""

    out: ModeDict = {}
    for key, value in h.items():
        if hasattr(value, "get"):
            value = value.get()
        out[_parse_mode_key(key)] = np.asarray(value)
    return out


def complete_nonprecessing_modes(h: Mapping[Any, Any]) -> ModeDict:
    """Fill missing negative-m modes using ``h_{l,-m}=(-1)^l conj(h_lm)``."""

    out = dict(normalize_mode_dict(h))
    for (ell, emm), hlm in list(out.items()):
        if emm != 0 and (ell, -emm) not in out:
            out[(ell, -emm)] = parity_sign(ell) * np.conjugate(hlm)
    return out


def validate_time_grid(t: Any) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    if t_arr.ndim != 1 or len(t_arr) < 2:
        raise ValueError("t must be a one-dimensional array with at least two points")
    if np.any(np.diff(t_arr) <= 0.0):
        raise ValueError("t must be strictly increasing")
    return t_arr


def validate_mode_lengths(t: np.ndarray, h: Mapping[Mode, np.ndarray]) -> None:
    for mode, series in h.items():
        if len(series) != len(t):
            raise ValueError(f"mode {mode} has length {len(series)}, but len(t)={len(t)}")


def differentiate_modes(t: Any, h: Mapping[Any, Any], edge_order: int = 2) -> ModeDict:
    """Differentiate all modes on a strictly increasing grid."""

    t_arr = validate_time_grid(t)
    h_norm = normalize_mode_dict(h)
    validate_mode_lengths(t_arr, h_norm)
    actual_edge_order = 2 if edge_order == 2 and len(t_arr) >= 3 else 1
    return {
        mode: np.gradient(series, t_arr, edge_order=actual_edge_order)
        for mode, series in h_norm.items()
    }


def cumulative_integral(t: Any, y: Any) -> np.ndarray:
    """Cumulative trapezoidal integral with initial value zero."""

    t_arr = validate_time_grid(t)
    y_arr = np.asarray(y)
    if len(y_arr) != len(t_arr):
        raise ValueError(f"y has length {len(y_arr)}, but len(t)={len(t_arr)}")
    return cumulative_trapezoid(y_arr, t_arr, initial=0.0)


def _valid_mode(ell: int, emm: int) -> bool:
    return ell >= 0 and abs(emm) <= ell


@lru_cache(maxsize=None)
def gamma_displacement(L: int, M: int, l1: int, m1: int, l2: int, m2: int) -> float:
    """Angular coefficient for vacuum nonlinear-null displacement memory."""

    if L < 2 or abs(M) > L or l1 < 2 or l2 < 2:
        return 0.0
    if not (_valid_mode(l1, m1) and _valid_mode(l2, m2)):
        return 0.0
    if m1 - m2 != M or not (abs(l1 - l2) <= L <= l1 + l2):
        return 0.0

    prefactor = parity_sign(M + m2)
    prefactor *= sqrt(factorial(L - 2) / factorial(L + 2))
    prefactor *= sqrt((2 * l2 + 1) * (2 * L + 1) * (2 * l1 + 1) / (4 * pi))
    return float(
        sympy_N(
            prefactor
            * wigner_3j(l2, L, l1, -2, 0, 2)
            * wigner_3j(l2, L, l1, -m2, -M, m1)
        )
    )


@lru_cache(maxsize=None)
def B_coefficient(
    L: int,
    M: int,
    s1: int,
    l1: int,
    m1: int,
    s2: int,
    l2: int,
    m2: int,
) -> float:
    """Integral of ``_s1Y_l1m1 _s2Y_l2m2 conjugate(_{s1+s2}Y_LM)``."""

    if M != m1 + m2 or abs(M) > L:
        return 0.0
    if not (_valid_mode(l1, m1) and _valid_mode(l2, m2)):
        return 0.0
    if abs(s1) > l1 or abs(s2) > l2 or abs(s1 + s2) > L:
        return 0.0
    if not (abs(l1 - l2) <= L <= l1 + l2):
        return 0.0

    prefactor = parity_sign(M + s1 + s2)
    prefactor *= sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * L + 1) / (4 * pi))
    return float(
        sympy_N(
            prefactor
            * wigner_3j(l1, l2, L, -s1, -s2, s1 + s2)
            * wigner_3j(l1, l2, L, m1, m2, -M)
        )
    )


@lru_cache(maxsize=None)
def C_coefficient(L: int, M: int, l1: int, m1: int, l2: int, m2: int) -> float:
    """Angular coefficient shared by spin and CM memory."""

    term1 = 3.0 * sqrt((l1 - 1) * (l1 + 2)) * B_coefficient(
        L, M, -1, l1, m1, 2, l2, m2
    )
    term2 = 0.0
    if l2 >= 3:
        term2 = sqrt((l2 - 2) * (l2 + 3)) * B_coefficient(
            L, M, -2, l1, m1, 3, l2, m2
        )
    return term1 + term2


def _spin_cm_prefactor(L: int, m2: int) -> float:
    return (
        parity_sign(m2)
        * sqrt(factorial(L - 2) / factorial(L + 2))
        / (4.0 * sqrt(L * (L + 1)))
    )


@lru_cache(maxsize=None)
def gamma_spin(L: int, M: int, l1: int, m1: int, l2: int, m2: int) -> float:
    """Direct strain-mode coefficient for spin memory."""

    if L < 2 or abs(M) > L:
        return 0.0
    if not (_valid_mode(l1, m1) and _valid_mode(l2, m2)):
        return 0.0
    if m1 - m2 != M or not (abs(l1 - l2) <= L <= l1 + l2):
        return 0.0
    epsilon = parity_sign(L + l1 + l2)
    c12 = C_coefficient(L, M, l1, m1, l2, -m2)
    c21 = C_coefficient(L, M, l2, -m2, l1, m1)
    return _spin_cm_prefactor(L, m2) * (c12 + epsilon * c21)


@lru_cache(maxsize=None)
def gamma_cm(L: int, M: int, l1: int, m1: int, l2: int, m2: int) -> float:
    """Direct strain-mode coefficient for vacuum-null CM memory."""

    if L < 2 or abs(M) > L:
        return 0.0
    if not (_valid_mode(l1, m1) and _valid_mode(l2, m2)):
        return 0.0
    if m1 - m2 != M or not (abs(l1 - l2) <= L <= l1 + l2):
        return 0.0
    epsilon = parity_sign(L + l1 + l2)
    c12 = C_coefficient(L, M, l1, m1, l2, -m2)
    c21 = C_coefficient(L, M, l2, -m2, l1, m1)
    return _spin_cm_prefactor(L, m2) * (c12 - epsilon * c21)


def precompute_memory_coeffs(
    L: int,
    M: int,
    l1_max: int = _DEFAULT_CACHE_LMAX,
    l2_max: int | None = _DEFAULT_CACHE_LMAX,
    l1_min: int = 2,
    tol: float = 1e-15,
    use_cache: bool = True,
) -> list[UnifiedCoeff]:
    """Build coefficient table entries ``(..., Gamma_D, Gamma_S, Gamma_CM)``.

    All valid azimuthal modes are tabulated, including ``|m|=1``.
    """

    if use_cache:
        cached = load_precomputed_memory_coeffs(
            L=L,
            M=M,
            lmax=l1_max,
            l2_max=l2_max,
            l1_min=l1_min,
            tol=tol,
        )
        if cached is not None:
            return cached

    coeffs: list[UnifiedCoeff] = []
    for l1 in range(l1_min, l1_max + 1):
        for m1 in range(-l1, l1 + 1):
            m2 = m1 - M
            l2_min = max(abs(L - l1), 2, abs(m2))
            l2_upper = L + l1 if l2_max is None else min(L + l1, l2_max)
            for l2 in range(l2_min, l2_upper + 1):
                gD = gamma_displacement(L, M, l1, m1, l2, m2)
                gS = gamma_spin(L, M, l1, m1, l2, m2)
                gCM = gamma_cm(L, M, l1, m1, l2, m2)
                if max(abs(gD), abs(gS), abs(gCM)) > tol:
                    coeffs.append((L, M, l1, m1, l2, m2, gD, gS, gCM))
    return coeffs


def _coeffs_from_array(array: np.ndarray) -> list[UnifiedCoeff]:
    coeffs: list[UnifiedCoeff] = []
    for row in np.asarray(array, dtype=float):
        coeffs.append(
            (
                int(row[0]),
                int(row[1]),
                int(row[2]),
                int(row[3]),
                int(row[4]),
                int(row[5]),
                float(row[6]),
                float(row[7]),
                float(row[8]),
            )
        )
    return coeffs


def load_precomputed_memory_coeffs(
    L: int,
    M: int,
    lmax: int = _DEFAULT_CACHE_LMAX,
    l2_max: int | None = _DEFAULT_CACHE_LMAX,
    l1_min: int = 2,
    tol: float = 1e-15,
) -> list[UnifiedCoeff] | None:
    """Load bundled coefficient tables when they match the requested setup."""

    if lmax != _DEFAULT_CACHE_LMAX or l2_max != _DEFAULT_CACHE_LMAX:
        return None
    if l1_min != 2 or tol != 1e-15:
        return None

    key = f"L{int(L)}_M{int(M)}".replace("-", "m")
    try:
        table_path = resources.files(__package__).joinpath(
            "data", f"gamma_coeffs_lmax{_DEFAULT_CACHE_LMAX}.npz"
        )
        with resources.as_file(table_path) as path:
            with np.load(path) as data:
                if key not in data:
                    return None
                return _coeffs_from_array(data[key])
    except (FileNotFoundError, ModuleNotFoundError):
        return None


def compute_vacuum_null_memory_mode(
    t: Any,
    h: Mapping[Any, Any],
    coeffs: Iterable[tuple],
    hdot: Mapping[Any, Any] | None = None,
    edge_order: int = 2,
) -> dict[str, Any]:
    """Compute displacement, spin, and CM observables for one target mode."""

    coeff_list = list(coeffs)
    if not coeff_list:
        raise ValueError("coefficient list is empty")
    t_arr = validate_time_grid(t)
    h_norm = normalize_mode_dict(h)
    validate_mode_lengths(t_arr, h_norm)
    hdot_norm = differentiate_modes(t_arr, h_norm, edge_order=edge_order) if hdot is None else normalize_mode_dict(hdot)
    validate_mode_lengths(t_arr, hdot_norm)

    L, M = int(coeff_list[0][0]), int(coeff_list[0][1])
    displacement_source = np.zeros_like(t_arr, dtype=complex)
    h_spin_mode = np.zeros_like(t_arr, dtype=complex)
    h_cm_mode = np.zeros_like(t_arr, dtype=complex)
    used_terms = 0
    skipped_terms = 0

    for row in coeff_list:
        target_L, target_M, l1, m1, l2, m2, gD, gS, gCM = row
        if int(target_L) != L or int(target_M) != M:
            raise ValueError("all coefficient entries must share one target")
        mode1 = (int(l1), int(m1))
        mode2 = (int(l2), int(m2))
        if (
            mode1 not in h_norm
            or mode2 not in h_norm
            or mode1 not in hdot_norm
            or mode2 not in hdot_norm
        ):
            skipped_terms += 1
            continue

        h1 = h_norm[mode1]
        h2 = h_norm[mode2]
        dh1 = hdot_norm[mode1]
        dh2 = hdot_norm[mode2]
        displacement_source += float(gD) * dh1 * np.conjugate(dh2)
        h_spin_mode += float(gS) * (h1 * np.conjugate(dh2) - dh1 * np.conjugate(h2))
        h_cm_mode += float(gCM) * (h1 * np.conjugate(dh2) - dh1 * np.conjugate(h2))
        used_terms += 1

    return {
        "target_mode": (L, M),
        "dh_displacement_dt": displacement_source,
        "h_displacement": cumulative_integral(t_arr, displacement_source),
        "h_spin_mode": h_spin_mode,
        "spin_memory_integral": cumulative_integral(t_arr, h_spin_mode),
        "h_cm_mode": h_cm_mode,
        "cm_memory_integral": cumulative_integral(t_arr, h_cm_mode),
        "used_terms": used_terms,
        "skipped_terms": skipped_terms,
    }


def compute_memory_modes(
    t: Any,
    h: Mapping[Any, Any],
    targets: Iterable[tuple[int, int]],
    lmax: int = _DEFAULT_CACHE_LMAX,
    hdot: Mapping[Any, Any] | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Compute several target modes using one differentiated mode dictionary."""

    t_arr = validate_time_grid(t)
    h_norm = normalize_mode_dict(h)
    hdot_norm = differentiate_modes(t_arr, h_norm) if hdot is None else normalize_mode_dict(hdot)
    return {
        target: compute_vacuum_null_memory_mode(
            t_arr,
            h_norm,
            precompute_memory_coeffs(*target, l1_max=lmax, l2_max=lmax),
            hdot=hdot_norm,
        )
        for target in targets
    }

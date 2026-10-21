from __future__ import annotations

from collections.abc import Mapping
from math import factorial, pi, sqrt

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import eval_jacobi, gammaln, roots_legendre

from .core import normalize_mode_dict, validate_time_grid


def bms_rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("axis must be a finite three-vector")
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("axis must be nonzero")
    x, y, z = axis / norm
    half = angle / 2
    return np.cos(half) * np.eye(2, dtype=complex) + 1j * np.sin(half) * np.array(
        [[z, x + 1j * y], [x - 1j * y, -z]], dtype=complex
    )


def bms_boost_matrix(velocity):
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity must be a finite three-vector")
    speed = np.linalg.norm(velocity)
    if speed >= 1:
        raise ValueError("boost speed must be less than one")
    if speed == 0:
        return np.eye(2, dtype=complex)
    x, y, z = velocity / speed
    half = np.arctanh(speed) / 2
    return np.cosh(half) * np.eye(2, dtype=complex) + np.sinh(half) * np.array(
        [[z, x + 1j * y], [x - 1j * y, -z]], dtype=complex
    )


def _spin_harmonic(ell, emm, spin, theta, phi):
    if ell < abs(spin) or abs(emm) > ell:
        return np.zeros_like(theta, dtype=complex)
    row, column = emm, -spin
    sign = 1
    if abs(row) < abs(column):
        sign *= (-1) ** (row - column)
        row, column = column, row
    if row < 0:
        sign *= (-1) ** (row - column)
        row, column = -row, -column
    prefactor = np.exp(0.5 * (
        gammaln(ell + row + 1) + gammaln(ell - row + 1)
        - gammaln(ell + column + 1) - gammaln(ell - column + 1)
    ))
    small_d = (
        sign * (-1) ** (row - column) * prefactor
        * np.sin(theta / 2) ** (row - column)
        * np.cos(theta / 2) ** (row + column)
        * eval_jacobi(ell - row, row - column, row + column, np.cos(theta))
    )
    return (
        (-1) ** spin
        * sqrt((2 * ell + 1) / (4 * pi))
        * small_d
        * np.exp(1j * emm * phi)
    )


def _harmonic_matrix(modes, spin, theta, phi):
    return np.column_stack(
        [_spin_harmonic(ell, emm, spin, theta, phi) for ell, emm in modes]
    )


def _transform_bms_modes_fixed(
    t,
    h: Mapping,
    *,
    supertranslation: Mapping | None = None,
    lorentz=None,
    output_t=None,
    output_lmax=None,
    n_theta=None,
    n_phi=None,
    angular_batch=64,
):
    t = validate_time_grid(t)
    if not np.all(np.isfinite(t)):
        raise ValueError("t must be finite")
    h = normalize_mode_dict(h)
    if not h:
        raise ValueError("h must contain at least one strain mode")
    for (ell, emm), values in h.items():
        if ell < 2 or abs(emm) > ell or np.asarray(values).shape != t.shape:
            raise ValueError(f"invalid strain mode {(ell, emm)}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"nonfinite strain mode {(ell, emm)}")
    alpha = normalize_mode_dict(supertranslation or {})
    for (ell, emm), value in alpha.items():
        if ell < 0 or abs(emm) > ell or np.asarray(value).ndim != 0:
            raise ValueError(f"invalid supertranslation mode {(ell, emm)}")
        if not np.isfinite(value):
            raise ValueError(f"nonfinite supertranslation mode {(ell, emm)}")
        partner = alpha.get((ell, -emm), 0.0)
        if not np.isclose(
            value, (-1) ** emm * np.conjugate(partner), atol=1e-12, rtol=1e-12
        ):
            raise ValueError("supertranslation must be real on the sphere")
    if lorentz is None:
        lorentz = np.eye(2, dtype=complex)
    lorentz = np.asarray(lorentz, dtype=complex)
    if lorentz.shape != (2, 2) or not np.all(np.isfinite(lorentz)):
        raise ValueError("lorentz must be a finite 2x2 complex matrix")
    determinant = np.linalg.det(lorentz)
    if abs(determinant) < 1e-14:
        raise ValueError("lorentz must be nonsingular")
    lorentz = lorentz / np.sqrt(determinant)
    ell_in = max(ell for ell, _ in h)
    ell_alpha = max((ell for ell, _ in alpha), default=0)
    if output_lmax is None:
        output_lmax = max(ell_in, ell_alpha)
    if not isinstance(output_lmax, int) or output_lmax < 2:
        raise ValueError("output_lmax must be an integer at least two")
    resolution = 2 * (ell_in + ell_alpha + output_lmax) + 1
    if n_theta is None:
        n_theta = resolution
    if n_phi is None:
        n_phi = resolution
    if (
        not isinstance(n_theta, int)
        or not isinstance(n_phi, int)
        or min(n_theta, n_phi) < 2 * max(ell_in, ell_alpha, output_lmax) + 1
    ):
        raise ValueError("angular grid is too small for the requested mode range")
    if not isinstance(angular_batch, int) or angular_batch < 1:
        raise ValueError("angular_batch must be positive")
    x, w = roots_legendre(n_theta)
    theta_new = np.repeat(np.arccos(x), n_phi)
    phi_new = np.tile(2 * pi * np.arange(n_phi) / n_phi, n_theta)
    weights = np.repeat(w, n_phi) * (2 * pi / n_phi)
    z_new = np.exp(1j * phi_new) / np.tan(theta_new / 2)
    a, b = lorentz[0]
    c, d = lorentz[1]
    denominator = a - c * z_new
    numerator = d * z_new - b
    pole = (
        (np.abs(denominator) <= 1e-12 * (np.abs(a) + np.abs(c * z_new)))
        | (np.abs(numerator) <= 1e-12 * (np.abs(d * z_new) + np.abs(b)))
    )
    if np.any(pole):
        theta_new[pole] += np.where(theta_new[pole] < pi / 2, 1e-10, -1e-10)
        phi_new[pole] += 1e-10
        z_new[pole] = np.exp(1j * phi_new[pole]) / np.tan(theta_new[pole] / 2)
        denominator[pole] = a - c * z_new[pole]
        numerator[pole] = d * z_new[pole] - b
    z_old = numerator / denominator
    theta_old = 2 * np.arctan2(np.ones_like(z_old.real), np.abs(z_old))
    phi_old = np.angle(z_old)
    k = (np.abs(denominator) ** 2 + np.abs(numerator) ** 2) / (1 + np.abs(z_new) ** 2)
    phase = (
        np.exp(2j * (phi_old - phi_new))
        * (denominator / np.conjugate(denominator)) ** 2
        / k
    )
    alpha_modes = list(alpha)
    if alpha_modes:
        alpha_values = _harmonic_matrix(alpha_modes, 0, theta_old, phi_old) @ np.array(
            [alpha[mode] for mode in alpha_modes], dtype=complex
        )
        if np.max(np.abs(alpha_values.imag)) > 1e-10 * max(
            1, np.max(np.abs(alpha_values.real))
        ):
            raise ValueError("supertranslation is not real on the angular grid")
        alpha_values = alpha_values.real
    else:
        alpha_values = np.zeros_like(k)
    alpha_shift_coefficients = np.array(
        [
            sqrt(factorial(ell + 2) / factorial(ell - 2)) * alpha[(ell, emm)]
            if ell >= 2
            else 0
            for ell, emm in alpha_modes
        ],
        dtype=complex,
    )
    monopole = np.real(alpha.get((0, 0), 0.0)) / sqrt(4 * pi)
    radius = sum(
        sqrt((2 * ell + 1) / (4 * pi)) * np.linalg.norm(
            [value for (degree, _), value in alpha.items() if degree == ell]
        )
        for ell in range(1, ell_alpha + 1)
    )
    inverse = np.linalg.inv(lorentz)
    k_bounds = np.linalg.eigvalsh(inverse.conj().T @ inverse)
    lower = np.max(k_bounds * (t[0] - monopole + radius))
    upper = np.min(k_bounds * (t[-1] - monopole - radius))
    support_tolerance = 32 * np.finfo(float).eps * max(
        1, abs(lower), abs(upper),
        k_bounds[-1] * (max(abs(t[0]), abs(t[-1])) + abs(monopole) + radius),
    )
    if output_t is None:
        monopole = np.real(alpha.get((0, 0), 0.0)) / sqrt(4 * pi)
        candidate = t - monopole
        output_t = candidate[
            (candidate >= lower - support_tolerance)
            & (candidate <= upper + support_tolerance)
        ]
        if not len(output_t):
            raise ValueError("transformed full-sphere time interval has no input samples")
    else:
        output_t = np.asarray(output_t, dtype=float)
        if (
            output_t.ndim != 1
            or not len(output_t)
            or not np.all(np.isfinite(output_t))
            or np.any(np.diff(output_t) <= 0)
        ):
            raise ValueError("output_t must be a nonempty increasing finite array")
        if output_t[0] < lower - support_tolerance or output_t[-1] > upper + support_tolerance:
            raise ValueError("output_t requires extrapolation at some angles")
    input_modes = list(h)
    output_modes = [
        (ell, emm)
        for ell in range(2, output_lmax + 1)
        for emm in range(-ell, ell + 1)
    ]
    input_data = np.column_stack([h[mode] for mode in input_modes])
    input_spline = CubicSpline(t, input_data, extrapolate=False)
    output_data = np.zeros((len(output_t), len(output_modes)), dtype=complex)
    for start in range(0, len(k), angular_batch):
        stop = min(start + angular_batch, len(k))
        old_harmonics = _harmonic_matrix(
            input_modes, -2, theta_old[start:stop], phi_old[start:stop]
        )
        if alpha_modes:
            old_shift = _harmonic_matrix(
                alpha_modes, -2, theta_old[start:stop], phi_old[start:stop]
            ) @ alpha_shift_coefficients
        else:
            old_shift = np.zeros(stop - start, dtype=complex)
        new_field = np.empty((len(output_t), stop - start), dtype=complex)
        for j in range(stop - start):
            index = start + j
            old_times = output_t / k[index] + alpha_values[index]
            tolerance = 32 * np.finfo(float).eps * max(1, abs(t[0]), abs(t[-1]))
            if np.any(old_times < t[0] - tolerance) or np.any(old_times > t[-1] + tolerance):
                raise ValueError("output_t requires extrapolation at some angles")
            values = input_spline(np.clip(old_times, t[0], t[-1])) @ old_harmonics[j]
            new_field[:, j] = (values - old_shift[j]) * phase[index]
        new_harmonics = _harmonic_matrix(
            output_modes, -2, theta_new[start:stop], phi_new[start:stop]
        )
        output_data += (new_field * weights[start:stop]) @ np.conjugate(new_harmonics)
    return output_t, {mode: output_data[:, i] for i, mode in enumerate(output_modes)}


def transform_bms_modes(
    t, h: Mapping, *, supertranslation: Mapping | None = None,
    lorentz=None, output_t=None, output_lmax=None,
    n_theta=None, n_phi=None, angular_batch=64,
    angular_rtol=1e-7, angular_atol=1e-10, max_refinements=5,
):
    if not np.isfinite(angular_atol) or angular_atol < 0:
        raise ValueError("angular_atol must be finite and nonnegative")
    if angular_rtol is not None and (not np.isfinite(angular_rtol) or angular_rtol < 0):
        raise ValueError("angular_rtol must be finite and nonnegative")
    if not isinstance(max_refinements, int) or max_refinements < 1:
        raise ValueError("max_refinements must be a positive integer")
    if angular_rtol is None and (n_theta is None or n_phi is None):
        raise ValueError("fixed quadrature requires explicit n_theta and n_phi")
    options = dict(
        supertranslation=supertranslation, lorentz=lorentz, output_t=output_t,
        output_lmax=output_lmax, angular_batch=angular_batch,
    )
    times, previous = _transform_bms_modes_fixed(
        t, h, n_theta=n_theta, n_phi=n_phi, **options,
    )
    if angular_rtol is None:
        return times, previous
    ell_in = max(ell for ell, _ in normalize_mode_dict(h))
    ell_alpha = max((ell for ell, _ in normalize_mode_dict(supertranslation or {})), default=0)
    ell_out = max(ell for ell, _ in previous)
    resolution = 2 * (ell_in + ell_alpha + ell_out) + 1
    nt = resolution if n_theta is None else n_theta
    np_ = resolution if n_phi is None else n_phi
    options["output_t"] = times
    for _ in range(max_refinements):
        nt, np_ = 2 * nt + 1, 2 * np_ + 1
        _, current = _transform_bms_modes_fixed(t, h, n_theta=nt, n_phi=np_, **options)
        if all(np.all(np.abs(current[mode] - previous[mode]) <= (
            angular_atol + angular_rtol * np.abs(current[mode])
        )) for mode in current):
            return times, current
        previous = current
    raise RuntimeError("BMS angular quadrature did not converge; increase resolution or max_refinements")

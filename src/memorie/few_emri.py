"""FastEMRIWaveforms helpers for perturbative EMRI memory modes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from .core import parity_sign, precompute_memory_coeffs
from .pn import k20_lo

MTSUN_SI = 4.925490947641266978197229498498379006e-6


@dataclass(frozen=True)
class FewEmriConfig:
    """Configuration for an eccentric-equatorial FEW EMRI memory run."""

    primary_mass_msun: float = 1.0e6
    secondary_mass_msun: float = 50.0
    spin: float = 0.0
    p0: float = 100.0
    e0: float = 0.0
    x0_inclination: float = 1.0
    t_years: float = 10000.0
    endpoint_factor: float = 1.01
    n_dense: int = 20000
    trajectory_err: float = 1e-11
    buffer_length: int = 20000
    frequency_source: str = "geodesic"


def _as_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "get"):
        values = values.get()
    return np.asarray(values)


def _load_few_objects() -> tuple[Any, Any]:
    try:
        from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
        from few.trajectory.inspiral import EMRIInspiral
        from few.trajectory.ode.flux import KerrEccEqFlux
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "The FEW EMRI helper requires FastEMRIWaveforms, imported as 'few'."
        ) from exc

    return AmpInterpKerrEccEq(), EMRIInspiral(func=KerrEccEqFlux)


def _build_n_map(special_index_map: dict[tuple[int, int, int], int]) -> dict[tuple[int, int], list[int]]:
    n_map: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ell, emm, enn in special_index_map:
        n_map[(int(ell), int(emm))].append(int(enn))
    return n_map


def _mode_from_compressed_few_amplitudes(
    amplitudes: np.ndarray,
    special_index_map: dict[tuple[int, int, int], int],
    ell: int,
    emm: int,
    enn: int,
) -> np.ndarray | None:
    """Return one FEW amplitude, reconstructing a stored negative-m partner."""

    column = special_index_map.get((int(ell), int(emm), int(enn)))
    if column is None:
        return None
    value = np.asarray(amplitudes[:, column])
    if emm < 0:
        # FEW maps (ell, -m, n) to the stored (ell, m, -n) column.
        value = parity_sign(ell) * np.conjugate(value)
    return value


def _endpoint_refined_series(
    t: np.ndarray,
    y: np.ndarray,
    endpoint_index: int,
    n_dense: int,
) -> tuple[np.ndarray, np.ndarray]:
    if endpoint_index < 2:
        raise ValueError("endpoint_index must leave at least two samples before the endpoint")
    if n_dense < 16:
        raise ValueError("n_dense must be at least 16")

    tt = np.asarray(t[:endpoint_index], dtype=float)
    yy = np.asarray(y[:endpoint_index])
    tau = float(t[endpoint_index]) - tt
    mask = tau > 0.0
    tau = tau[mask]
    yy = yy[mask]
    if len(tau) < 2:
        raise ValueError("not enough positive endpoint distances for log interpolation")

    order = np.argsort(tau)
    tau = tau[order]
    yy = yy[order]
    log_tau = np.log(tau)
    dense_log_tau = np.linspace(float(log_tau[0]), float(log_tau[-1]), int(n_dense))
    dense_tau = np.exp(dense_log_tau)
    dense_values = PchipInterpolator(log_tau, yy.real)(dense_log_tau) + 1j * PchipInterpolator(
        log_tau, yy.imag
    )(dense_log_tau)

    t_dense = float(t[endpoint_index]) - dense_tau
    time_order = np.argsort(t_dense)
    return t_dense[time_order], dense_values[time_order]


def _compute_few_memory_sources(
    *,
    all_amplitudes: Any,
    special_index_map: dict[tuple[int, int, int], int],
    coeffs: Iterable[tuple],
    omega_phi: np.ndarray,
    omega_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute FEW orbit-averaged displacement and spin-memory sources."""

    amplitudes = _as_numpy(all_amplitudes)
    n_map = _build_n_map(special_index_map)
    displacement_source = np.zeros(amplitudes.shape[0], dtype=complex)
    spin_mode = np.zeros(amplitudes.shape[0], dtype=complex)
    h_cache: dict[tuple[int, int, int], np.ndarray | None] = {}
    hdot_cache: dict[tuple[int, int, int], np.ndarray | None] = {}
    active_pairs = 0
    active_lmn_contributions = 0

    def get_mode(ell: int, emm: int, enn: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        key = (int(ell), int(emm), int(enn))
        if key not in h_cache:
            h_value = _mode_from_compressed_few_amplitudes(
                amplitudes,
                special_index_map,
                ell,
                emm,
                enn,
            )
            if h_value is None:
                h_cache[key] = None
                hdot_cache[key] = None
            else:
                omega_mn = int(emm) * omega_phi + int(enn) * omega_r
                h_cache[key] = h_value
                hdot_cache[key] = -1j * h_value * omega_mn
        return h_cache[key], hdot_cache[key]

    for target_L, target_M, ell1, emm1, ell2, emm2, gamma_d, gamma_s, _gamma_cm in coeffs:
        if int(target_M) != int(emm1) - int(emm2):
            continue

        n_values = set(n_map.get((int(ell1), int(emm1)), [])) & set(
            n_map.get((int(ell2), int(emm2)), [])
        )
        if not n_values:
            continue

        pair_used = False
        for enn in n_values:
            h1, hdot1 = get_mode(int(ell1), int(emm1), int(enn))
            h2, hdot2 = get_mode(int(ell2), int(emm2), int(enn))
            if h1 is None or h2 is None or hdot1 is None or hdot2 is None:
                continue
            displacement_source += float(gamma_d) * hdot1 * np.conjugate(hdot2)
            spin_mode += float(gamma_s) * (h1 * np.conjugate(hdot2) - hdot1 * np.conjugate(h2))
            active_lmn_contributions += 1
            pair_used = True
        if pair_used:
            active_pairs += 1

    diagnostics = {
        "active_pairs": int(active_pairs),
        "active_lmn_contributions": int(active_lmn_contributions),
        "cached_modes": int(sum(value is not None for value in h_cache.values())),
        "amplitude_shape": tuple(int(value) for value in amplitudes.shape),
    }
    return displacement_source, spin_mode, diagnostics


def compute_few_emri_memory_modes(
    config: FewEmriConfig | None = None,
    lmax: int = 10,
) -> dict[str, Any]:
    """Generate FEW EMRI perturbative $h_{20}$ and $h_{30}$ memory modes.

    The returned modes are distance-rescaled dimensionless amplitudes.  The
    FEW amplitudes are converted with the same ``nu**2 * Mtot`` normalization
    used in the internal FEW h20-grid generator.
    """

    cfg = FewEmriConfig() if config is None else config
    if cfg.frequency_source not in {"geodesic", "phase-gradient"}:
        raise ValueError("frequency_source must be 'geodesic' or 'phase-gradient'")

    try:
        from few.utils.geodesic import get_fundamental_frequencies, get_separatrix
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "The FEW EMRI helper requires FastEMRIWaveforms, imported as 'few'."
        ) from exc

    q = float(cfg.primary_mass_msun / cfg.secondary_mass_msun)
    nu = q / (1.0 + q) ** 2
    total_mass_msun = float(cfg.primary_mass_msun + cfg.secondary_mass_msun)
    total_mass_seconds = total_mass_msun * MTSUN_SI
    primary_mass_seconds = float(cfg.primary_mass_msun) * MTSUN_SI

    amp, trajectory = _load_few_objects()
    start = perf_counter()
    t, p, e, x, phi_phi, _phi_theta, phi_r = trajectory(
        cfg.primary_mass_msun,
        cfg.secondary_mass_msun,
        cfg.spin,
        cfg.p0,
        cfg.e0,
        cfg.x0_inclination,
        T=cfg.t_years,
        err=cfg.trajectory_err,
        buffer_length=cfg.buffer_length,
    )
    trajectory_runtime_s = perf_counter() - start
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    e = np.asarray(e, dtype=float)
    x = np.asarray(x, dtype=float)
    phase_gradient_omega_phi = np.gradient(np.asarray(phi_phi, dtype=float), t)
    phase_gradient_omega_r = np.gradient(np.asarray(phi_r, dtype=float), t)

    if cfg.frequency_source == "phase-gradient":
        omega_phi = phase_gradient_omega_phi
        omega_r = phase_gradient_omega_r
        omega_scale_seconds = total_mass_seconds
    else:
        omega_phi_dimless, _omega_theta_dimless, omega_r_dimless = get_fundamental_frequencies(
            cfg.spin, p, e, x
        )
        omega_phi = np.asarray(omega_phi_dimless, dtype=float) / primary_mass_seconds
        omega_r = np.asarray(omega_r_dimless, dtype=float) / primary_mass_seconds
        omega_scale_seconds = primary_mass_seconds

    all_amplitudes = amp(cfg.spin, p, e, x)
    p_sep = float(get_separatrix(cfg.spin, float(e[-1]), float(x[-1])))
    crossing = np.where(p < cfg.endpoint_factor * p_sep)[0]
    endpoint_index = int(crossing[0]) if len(crossing) else int(len(p) - 1)
    endpoint_index = min(max(endpoint_index, 2), len(p) - 1)

    coeffs_20 = precompute_memory_coeffs(2, 0, l1_max=lmax, l2_max=lmax)
    coeffs_30 = precompute_memory_coeffs(3, 0, l1_max=lmax, l2_max=lmax)
    h20_source, _unused_h20_spin, h20_diagnostics = _compute_few_memory_sources(
        all_amplitudes=all_amplitudes,
        special_index_map=amp.special_index_map,
        coeffs=coeffs_20,
        omega_phi=omega_phi,
        omega_r=omega_r,
    )
    _unused_h30_source, h30_spin_seconds, h30_diagnostics = _compute_few_memory_sources(
        all_amplitudes=all_amplitudes,
        special_index_map=amp.special_index_map,
        coeffs=coeffs_30,
        omega_phi=omega_phi,
        omega_r=omega_r,
    )

    t_dense_seconds, h20_source_dense = _endpoint_refined_series(
        t,
        h20_source,
        endpoint_index=endpoint_index,
        n_dense=cfg.n_dense,
    )
    t_h30_seconds, h30_spin_dense_seconds = _endpoint_refined_series(
        t,
        h30_spin_seconds,
        endpoint_index=endpoint_index,
        n_dense=cfg.n_dense,
    )
    if not np.allclose(t_dense_seconds, t_h30_seconds, rtol=0.0, atol=1e-9):
        raise RuntimeError("internal FEW dense grids for h20 and h30 do not match")

    scale = nu**2 * total_mass_seconds
    h20_dimensionless = scale * cumulative_trapezoid(h20_source_dense, t_dense_seconds, initial=0.0)
    h30_dimensionless = scale * h30_spin_dense_seconds
    t_dense_dimensionless = t_dense_seconds / total_mass_seconds
    sample_for_slope = min(10, len(t_dense_dimensionless) - 1)
    dhdt0 = np.real(
        (h20_dimensionless[sample_for_slope] - h20_dimensionless[0])
        / (t_dense_dimensionless[sample_for_slope] - t_dense_dimensionless[0])
    )
    x_eff = float((dhdt0 / ((64.0 / 5.0) * nu * k20_lo(q))) ** 0.2)
    x_orb0 = float((total_mass_seconds * omega_phi[0]) ** (2.0 / 3.0))

    return {
        "config": asdict(cfg),
        "model": "FastEMRIWaveforms",
        "q": q,
        "nu": float(nu),
        "total_mass_msun": total_mass_msun,
        "total_mass_seconds": total_mass_seconds,
        "primary_mass_seconds": primary_mass_seconds,
        "t_dense_dimensionless": t_dense_dimensionless,
        "h20_dimensionless": h20_dimensionless,
        "h30_dimensionless": h30_dimensionless,
        "delta_h20_dimensionless": complex(h20_dimensionless[-1] - h20_dimensionless[0]),
        "delta_h30_dimensionless": complex(h30_dimensionless[-1] - h30_dimensionless[0]),
        "delta_h20_over_nu": complex((h20_dimensionless[-1] - h20_dimensionless[0]) / nu),
        "delta_h30_over_nu": complex((h30_dimensionless[-1] - h30_dimensionless[0]) / nu),
        "prehistory_0pn_dimensionless": complex(k20_lo(q) * x_eff),
        "x_eff_0pn": x_eff,
        "x_orb0": x_orb0,
        "initial_dh20_dt_dimensionless": float(dhdt0),
        "omega_phi_start_dimensionless": float(total_mass_seconds * omega_phi[0]),
        "omega_phi_stop_dimensionless": float(total_mass_seconds * omega_phi[endpoint_index - 1]),
        "phase_gradient_omega_phi_start_dimensionless": float(
            total_mass_seconds * phase_gradient_omega_phi[0]
        ),
        "phase_gradient_omega_phi_stop_dimensionless": float(
            total_mass_seconds * phase_gradient_omega_phi[endpoint_index - 1]
        ),
        "omega_scale_seconds": float(omega_scale_seconds),
        "trajectory_sample_count": int(len(t)),
        "trajectory_runtime_s": float(trajectory_runtime_s),
        "trajectory_t_start_s": float(t[0]),
        "trajectory_t_stop_s": float(t[-1]),
        "trajectory_p_start": float(p[0]),
        "trajectory_p_stop": float(p[-1]),
        "p_sep": p_sep,
        "endpoint_index": int(endpoint_index),
        "integration_stop_index": int(endpoint_index - 1),
        "endpoint_p_crossing": float(p[endpoint_index]),
        "integration_stop_p": float(p[endpoint_index - 1]),
        "h20_diagnostics": h20_diagnostics,
        "h30_diagnostics": h30_diagnostics,
    }

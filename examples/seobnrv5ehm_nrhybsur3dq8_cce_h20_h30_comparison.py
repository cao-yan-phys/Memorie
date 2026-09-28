from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MTSUN_SI = 4.925490947641266978197229498498379006e-6
DEFAULT_X_START = 0.015
DEFAULT_OMEGA_TARGET = DEFAULT_X_START**1.5

from memorie import (
    complete_nonprecessing_modes,
    compute_memory_modes,
    compute_poincare_fluxes,
    cumulative_integral,
    differentiate_modes,
    infer_x_eff_from_dh20,
    symmetric_mass_ratio,
)


def _spin_vector(z_spin: float = 0.0) -> list[float]:
    return [0.0, 0.0, float(z_spin)]


def _normalize_surrogate_modes(raw_modes: dict) -> dict[tuple[int, int], np.ndarray]:
    return {tuple(map(int, mode)): np.asarray(series) for mode, series in raw_modes.items()}


def _orbital_frequency_from_h22(t: np.ndarray, h22: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(h22))
    return 0.5 * np.abs(np.gradient(phase, t, edge_order=2))


def _find_cce_start_time(
    surrogate,
    q: float,
    omega_target: float,
    search_start: float,
    search_stop: float,
    search_dt: float,
) -> float:
    times = np.arange(search_start, search_stop + 0.5 * search_dt, search_dt)
    t, raw_modes, _dyn = surrogate(q, _spin_vector(), _spin_vector(), f_low=0, times=times)
    modes = _normalize_surrogate_modes(raw_modes)
    omega = _orbital_frequency_from_h22(t, modes[(2, 2)])
    hits = np.flatnonzero(omega >= omega_target)
    if not len(hits):
        raise ValueError(
            f"target omega={omega_target} is outside the CCE search range "
            f"[{float(np.nanmin(omega))}, {float(np.nanmax(omega))}]"
        )
    i1 = int(hits[0])
    if i1 == 0:
        return float(t[0])
    i0 = i1 - 1
    return float(np.interp(omega_target, [omega[i0], omega[i1]], [t[i0], t[i1]]))


def _load_cce_segment(
    surrogate,
    q: float,
    t_start: float,
    t_stop: float,
    delta_t: float,
    refinement_start: float,
    refinement_delta_t: float,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    fine_start = min(max(float(refinement_start), t_start), t_stop)
    coarse_times = np.arange(t_start, fine_start, delta_t)
    fine_times = np.arange(
        fine_start,
        t_stop + 0.5 * refinement_delta_t,
        refinement_delta_t,
    )
    times = np.unique(np.concatenate([coarse_times, fine_times]))
    t, raw_modes, _dyn = surrogate(q, _spin_vector(), _spin_vector(), f_low=0, times=times)
    return t, _normalize_surrogate_modes(raw_modes)


def _generate_pyseobnr_modes(
    q: float,
    omega_start: float,
    pyseobnr_delta_t: float,
    total_mass_solar: float,
    approximant: str,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    from pyseobnr.generate_waveform import generate_modes_opt

    t, raw_modes = generate_modes_opt(
        q,
        0.0,
        0.0,
        omega_start,
        eccentricity=0.0,
        approximant=approximant,
        settings={
            "EccIC": 0,
            "M": total_mass_solar,
            "dt": pyseobnr_delta_t * total_mass_solar * MTSUN_SI,
            "lmax_nyquist": 1,
        },
    )
    positive_modes = {tuple(map(int, key.split(","))): value for key, value in raw_modes.items()}
    return t, complete_nonprecessing_modes(positive_modes)


def _interp_complex_with_plateau(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:

    return np.interp(x_new, x, np.real(y)) + 1j * np.interp(x_new, x, np.imag(y))


def _endpoint_refined_indices(
    t: np.ndarray,
    max_points: int,
    refine_duration: float = 2500.0,
) -> np.ndarray:
    n = len(t)
    if n <= max_points:
        return np.arange(n, dtype=int)
    split = int(np.searchsorted(t, t[-1] - refine_duration, side="left"))
    tail_size = n - split
    tail_budget = min(tail_size, max_points // 2)
    head_budget = min(split, max_points - tail_budget)
    head = np.linspace(0, split - 1, head_budget, dtype=int) if head_budget else np.array([], dtype=int)
    tail = np.linspace(split, n - 1, tail_budget, dtype=int) if tail_budget else np.array([], dtype=int)
    return np.unique(np.concatenate([head, tail]))


def _flux_csv_indices(t: np.ndarray, max_points: int) -> np.ndarray:
    if len(t) <= max_points:
        return np.arange(len(t), dtype=int)
    early_end = int(np.searchsorted(t, t[0] + 50000.0, side="right"))
    early_budget = min(early_end, max_points // 2)
    early = np.linspace(0, early_end - 1, early_budget, dtype=int) if early_budget else np.array([], dtype=int)
    remainder = _endpoint_refined_indices(t, max_points - early_budget)
    return np.unique(np.concatenate([early, remainder]))


def _include_extrema(indices: np.ndarray, *series: np.ndarray) -> np.ndarray:
    critical = [0, len(series[0]) - 1]
    for values in series:
        values = np.asarray(values, dtype=float)
        finite = np.flatnonzero(np.isfinite(values))
        if len(finite):
            critical.extend(
                [
                    int(finite[np.argmin(values[finite])]),
                    int(finite[np.argmax(values[finite])]),
                ]
            )
    return np.unique(np.concatenate([indices, np.asarray(critical, dtype=int)]))


def _positive_for_log(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[out <= 0.0] = np.nan
    return out


def _positive_log_limits(
    *series: np.ndarray,
    upper_series: np.ndarray | None = None,
) -> tuple[float, float]:
    values = np.concatenate([np.asarray(item, dtype=float).ravel() for item in series])
    values = values[np.isfinite(values) & (values > 0.0)]
    if not len(values):
        return 1e-15, 1.0
    upper_values = values if upper_series is None else np.asarray(upper_series, dtype=float)
    upper_values = upper_values[np.isfinite(upper_values) & (upper_values > 0.0)]
    ymin = max(float(np.min(values)) * 0.5, 1e-300)
    ymax = max(float(np.max(upper_values)) * 1.05, ymin * 10.0)
    return ymin, ymax


def _format_complex(value: complex) -> str:
    value = complex(value)
    return f"{value.real:+.6e}{value.imag:+.6e}j"


def _plot_linear_h20_comparison(
    cce_time: np.ndarray,
    cce_h20: np.ndarray,
    cce_perturbative_h20: np.ndarray,
    pyseobnr_time: np.ndarray,
    pyseobnr_h20: np.ndarray,
    png_path: Path,
    pyseobnr_approximant: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["axes.formatter.use_mathtext"] = True
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    figure, axis = plt.subplots(figsize=(9, 3.7), constrained_layout=True)
    axis.plot(
        cce_time,
        np.real(cce_h20),
        color="black",
        linewidth=1.4,
        label=r"$\mathtt{NRHybSur3dq8\_CCE}$",
    )
    axis.plot(
        cce_time,
        np.real(cce_perturbative_h20),
        color="blue",
        linestyle="-",
        linewidth=1.3,
        label=r"$\mathtt{NRHybSur3dq8\_CCE}$ perturbative",
    )
    axis.plot(
        pyseobnr_time,
        np.real(pyseobnr_h20),
        color="red",
        linestyle="--",
        linewidth=1.3,
        label=rf"$\mathtt{{{pyseobnr_approximant}}}$ perturbative",
    )
    axis.set_ylabel(r"$\mathrm{Re}\,\Delta h_{2,0}/(\nu M/R)$")
    axis.set_xlabel(r"$(t-t_0)/M$")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", frameon=False)

    end_time = float(np.nanmax(cce_time))
    inset_start = max(float(np.nanmin(cce_time)), end_time - 2500.0)
    cce_inset_mask = cce_time >= inset_start
    pyseobnr_inset_mask = pyseobnr_time >= inset_start
    inset = axis.inset_axes([0.53, 0.32, 0.37, 0.42])
    inset.plot(cce_time[cce_inset_mask], np.real(cce_h20[cce_inset_mask]), color="black", linewidth=1.2)
    inset.plot(
        cce_time[cce_inset_mask],
        np.real(cce_perturbative_h20[cce_inset_mask]),
        color="blue",
        linewidth=1.1,
    )
    inset.plot(
        pyseobnr_time[pyseobnr_inset_mask],
        np.real(pyseobnr_h20[pyseobnr_inset_mask]),
        color="red",
        linestyle="--",
        linewidth=1.1,
    )
    inset.set_xlim(inset_start, end_time)
    inset_values = np.concatenate(
        [
            np.real(cce_h20[cce_inset_mask]),
            np.real(cce_perturbative_h20[cce_inset_mask]),
            np.real(pyseobnr_h20[pyseobnr_inset_mask]),
        ]
    )
    inset_values = inset_values[np.isfinite(inset_values)]
    inset_span = float(np.ptp(inset_values))
    inset_padding = 0.08 * inset_span if inset_span else max(abs(float(inset_values[0])) * 0.08, 1e-16)
    inset.set_ylim(float(np.min(inset_values)) - inset_padding, float(np.max(inset_values)) + inset_padding)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.22, linewidth=0.5)
    mark_inset(axis, inset, loc1=1, loc2=4, fc="none", ec="0.35", linewidth=0.75)

    cce_peak_time = float(cce_time[np.nanargmax(np.real(cce_h20))])
    detail_start = max(inset_start, cce_peak_time - 18.0)
    previous_detail_end = min(end_time, cce_peak_time + 28.0)
    detail_end = min(end_time, detail_start + 2.0 * (previous_detail_end - detail_start))
    cce_detail_mask = (cce_time >= detail_start) & (cce_time <= detail_end)
    pyseobnr_detail_mask = (pyseobnr_time >= detail_start) & (pyseobnr_time <= detail_end)
    detail = inset.inset_axes([0.07, 0.15, 0.35, 0.80])
    detail.plot(cce_time[cce_detail_mask], np.real(cce_h20[cce_detail_mask]), color="black", linewidth=1.0)
    detail.plot(
        cce_time[cce_detail_mask],
        np.real(cce_perturbative_h20[cce_detail_mask]),
        color="blue",
        linewidth=0.95,
    )
    detail.plot(
        pyseobnr_time[pyseobnr_detail_mask],
        np.real(pyseobnr_h20[pyseobnr_detail_mask]),
        color="red",
        linestyle="--",
        linewidth=0.95,
    )
    detail.set_xlim(detail_start, detail_end)
    detail_values = np.concatenate(
        [
            np.real(cce_h20[cce_detail_mask]),
            np.real(cce_perturbative_h20[cce_detail_mask]),
            np.real(pyseobnr_h20[pyseobnr_detail_mask]),
        ]
    )
    detail_values = detail_values[np.isfinite(detail_values)]
    detail_span = float(np.ptp(detail_values))
    detail_padding = 0.08 * detail_span if detail_span else max(abs(float(detail_values[0])) * 0.08, 1e-16)
    detail.set_ylim(float(np.min(detail_values)) - detail_padding, float(np.max(detail_values)) + detail_padding)
    detail.tick_params(labelbottom=False, labelleft=False)
    detail.grid(alpha=0.22, linewidth=0.4)
    mark_inset(inset, detail, loc1=1, loc2=4, fc="none", ec="0.35", linewidth=0.55)
    figure.savefig(png_path, dpi=180)
    plt.close(figure)


def _replot_linear_h20_from_csv(
    csv_path: Path,
    png_path: Path,
    pyseobnr_approximant: str,
) -> None:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    time = np.asarray(data["t_minus_t0_M"], dtype=float)
    cce_h20 = np.asarray(data["NRHybSur3dq8_CCE_delta_h20_real_over_nu"], dtype=float)
    cce_perturbative_h20 = np.asarray(
        data["NRHybSur3dq8_CCE_perturbative_delta_h20_real_over_nu"], dtype=float
    )
    pyseobnr_h20 = np.asarray(data["SEOBNRv5EHM_delta_h20_real_over_nu"], dtype=float)
    _plot_linear_h20_comparison(
        time,
        cce_h20,
        cce_perturbative_h20,
        time,
        pyseobnr_h20,
        png_path,
        pyseobnr_approximant,
    )


def _interp_real_with_nan(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full(len(x_new), np.nan, dtype=float)
    mask = (x_new >= x[0]) & (x_new <= x[-1])
    out[mask] = np.interp(x_new[mask], x, y)
    return out


def _final_kerr_momentum(final_kerr_mass: float, final_kerr_velocity: np.ndarray) -> float:
    speed = float(np.linalg.norm(final_kerr_velocity))
    if not np.isfinite(final_kerr_mass) or final_kerr_mass <= 0.0 or not np.isfinite(speed) or speed >= 1.0:
        raise ValueError("final Kerr mass and velocity must be finite and physical")
    return float(final_kerr_mass * speed / np.sqrt(1.0 - speed**2))


def _pn_bms_remnant(q: float) -> tuple[float, np.ndarray, float]:
    import surfinBH

    chi = np.zeros(3)
    ordinary = surfinBH.LoadFits("NRSur3dq8Remnant")
    mass, _spin, _velocity, _mass_error, _spin_error, _velocity_error = ordinary.all(q, chi, chi)
    bms = surfinBH.LoadFits("NRSur3dq8BMSRemnant")
    _alpha, boost, _alpha_error, _boost_error = bms.all(q, chi, chi)
    boost = np.asarray(boost, dtype=float)
    momentum = _final_kerr_momentum(float(mass), boost)
    return float(mass), boost, momentum


def _bms_aligned_planar_momentum(
    time: np.ndarray,
    modes: dict[tuple[int, int], np.ndarray],
    momentum: np.ndarray,
) -> tuple[np.ndarray, float]:
    if (2, 1) not in modes:
        raise ValueError("the PN-BMS phase alignment requires the supplied (2, 1) mode")
    amplitude = np.zeros_like(time, dtype=float)
    for values in modes.values():
        amplitude += np.abs(values) ** 2
    peak_time = float(time[np.argmax(amplitude)])
    reference_time = peak_time - 100.0
    if reference_time < time[0] or reference_time > time[-1]:
        raise ValueError("the PN-BMS phase reference lies outside the waveform")
    h21_phase = np.unwrap(np.angle(-np.asarray(modes[(2, 1)])))
    rotation = -float(np.interp(reference_time, time, h21_phase)) + 0.5 * np.pi
    planar = np.asarray(momentum[:, 0], dtype=float) + 1j * np.asarray(momentum[:, 1], dtype=float)
    aligned = planar * np.exp(-1j * rotation)
    return np.column_stack((aligned.real, aligned.imag)), reference_time


def _effective_0pn_fluxes(
    time: np.ndarray,
    q: float,
    fluxes: dict[str, np.ndarray],
    x_momentum_phase: float,
) -> dict[str, np.ndarray | float]:
    nu = symmetric_mass_ratio(q)
    delta = (float(q) - 1.0) / (float(q) + 1.0)
    energy_flux0 = float(fluxes["energy_flux"][0])
    jz_flux0 = float(fluxes["angular_momentum_flux"][0, 2])
    pflux0 = np.asarray(fluxes["linear_momentum_flux"][0, :2], dtype=float)
    pflux0_complex = complex(pflux0[0], pflux0[1])
    pflux0_magnitude = abs(pflux0_complex)
    energy_coefficient = (32.0 / 5.0) * nu**2
    momentum_coefficient = (464.0 / 105.0) * nu**2 * delta
    if energy_flux0 <= 0.0 or jz_flux0 <= 0.0 or pflux0_magnitude <= 0.0 or momentum_coefficient <= 0.0:
        raise ValueError("effective 0PN flux matching requires positive initial fluxes")

    x_energy = (energy_flux0 / energy_coefficient) ** 0.2
    x_jz = (jz_flux0 / energy_coefficient) ** (2.0 / 7.0)
    x_momentum = (pflux0_magnitude / momentum_coefficient) ** (2.0 / 11.0)

    def x_series(x_initial: float) -> np.ndarray:
        denominator = 1.0 - (256.0 / 5.0) * nu * x_initial**4 * time
        values = np.full_like(time, np.nan, dtype=float)
        valid = denominator > 0.0
        values[valid] = x_initial * denominator[valid] ** (-0.25)
        values[values > 0.3] = np.nan
        return values

    x_energy_series = x_series(x_energy)
    x_jz_series = x_series(x_jz)
    x_momentum_series = x_series(x_momentum_phase)
    energy_radiated = 0.5 * nu * (x_energy_series - x_energy)
    angular_momentum_radiated = nu * (x_jz ** (-0.5) - x_jz_series ** (-0.5))
    phase = (x_momentum_phase ** (-2.5) - x_momentum_series ** (-2.5)) / (32.0 * nu)
    momentum_flux = (
        pflux0_complex
        * (x_momentum_series / x_momentum_phase) ** 5.5
        * np.exp(1j * phase)
    )
    momentum = np.full_like(momentum_flux, np.nan, dtype=complex)
    momentum_valid = np.isfinite(momentum_flux)
    momentum[momentum_valid] = cumulative_integral(time[momentum_valid], momentum_flux[momentum_valid])
    return {
        "energy_radiated": energy_radiated,
        "angular_momentum_radiated": angular_momentum_radiated,
        "planar_momentum": np.abs(momentum),
        "x_energy": x_energy,
        "x_jz": x_jz,
        "x_momentum": x_momentum,
        "x_momentum_phase": x_momentum_phase,
        "prehistory_energy": 0.5 * nu * x_energy,
        "prehistory_planar_momentum": pflux0_magnitude / x_momentum_phase**1.5,
    }


def _plot_flux_comparison(
    cce_time: np.ndarray,
    cce_fluxes: dict[str, np.ndarray],
    pyseobnr_time: np.ndarray,
    pyseobnr_fluxes: dict[str, np.ndarray],
    effective_0pn_fluxes: dict[str, np.ndarray | float],
    png_path: Path,
    pyseobnr_approximant: str,
    bms_remnant_momentum: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    series = [
        (
            cce_fluxes["energy_radiated"],
            pyseobnr_fluxes["energy_radiated"],
            np.asarray(effective_0pn_fluxes["energy_radiated"], dtype=float),
            r"$\Delta E^{\mathrm{gw}}/M$",
        ),
        (
            cce_fluxes["angular_momentum_radiated"][:, 2],
            pyseobnr_fluxes["angular_momentum_radiated"][:, 2],
            np.asarray(effective_0pn_fluxes["angular_momentum_radiated"], dtype=float),
            r"$\Delta J_z^{\mathrm{gw}}/M^2$",
        ),
        (
            cce_fluxes["planar_momentum"],
            pyseobnr_fluxes["planar_momentum"],
            np.asarray(effective_0pn_fluxes["planar_momentum"], dtype=float),
            r"$|\Delta\mathbf{P}^{\mathrm{gw}}|/M$",
        ),
    ]
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for axis, (cce_values, pyseobnr_values, effective_values, ylabel) in zip(axes, series, strict=True):
        axis.plot(
            cce_time,
            cce_values,
            color="black",
            linewidth=1.4,
            label=r"$\mathtt{NRHybSur3dq8\_CCE}$",
        )
        axis.plot(
            pyseobnr_time,
            pyseobnr_values,
            color="0.45",
            linestyle="--",
            linewidth=1.3,
            label=rf"$\mathtt{{{pyseobnr_approximant}}}$",
        )
        effective_plot_values = effective_values
        if ylabel == r"$|\Delta\mathbf{P}^{\mathrm{gw}}|/M$":
            effective_plot_values = effective_values.copy()
            effective_plot_values[pyseobnr_time > 50000.0] = np.nan
        axis.plot(
            pyseobnr_time,
            effective_plot_values,
            color="red",
            linestyle="--",
            linewidth=1.3,
            label="effective 0PN",
        )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="best", frameon=False)
    axes[-1].set_xlabel(r"$(t-t_0)/M$")
    momentum_upper = max(
        float(np.nanmax(cce_fluxes["planar_momentum"])),
        float(np.nanmax(pyseobnr_fluxes["planar_momentum"])),
    )
    axes[-1].set_ylim(0.0, 1.05 * momentum_upper)
    detail_start = max(
        float(min(cce_time[0], pyseobnr_time[0])),
        float(max(cce_time[-1], pyseobnr_time[-1]) - 2500.0),
    )
    detail_end = float(max(cce_time[-1], pyseobnr_time[-1]))
    cce_mask = cce_time >= detail_start
    pyseobnr_mask = pyseobnr_time >= detail_start
    inset = axes[-1].inset_axes([0.55, 0.54, 0.37, 0.38])
    inset.plot(
        cce_time[cce_mask],
        cce_fluxes["planar_momentum"][cce_mask],
        color="black",
        linewidth=1.2,
    )
    inset.plot(
        pyseobnr_time[pyseobnr_mask],
        pyseobnr_fluxes["planar_momentum"][pyseobnr_mask],
        color="0.45",
        linestyle="--",
        linewidth=1.1,
    )
    inset.plot(
        pyseobnr_time[pyseobnr_mask],
        np.asarray(effective_0pn_fluxes["planar_momentum"], dtype=float)[pyseobnr_mask],
        color="red",
        linestyle="--",
        linewidth=1.1,
    )
    inset.axhline(
        bms_remnant_momentum,
        color="blue",
        linestyle="--",
        linewidth=1.1,
        label=r"$\gamma_{\rm f}M_{\rm f}|\mathbf{v}_{\rm f}|/M$",
    )
    inset.set_xlim(detail_start, detail_end)
    detail_values = np.concatenate(
        [
            cce_fluxes["planar_momentum"][cce_mask],
            pyseobnr_fluxes["planar_momentum"][pyseobnr_mask],
            np.asarray(effective_0pn_fluxes["planar_momentum"], dtype=float)[pyseobnr_mask],
            np.asarray([bms_remnant_momentum]),
        ]
    )
    detail_values = detail_values[np.isfinite(detail_values)]
    inset.set_ylim(0.0, 1.05 * float(np.max(detail_values)))
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.22, linewidth=0.5)
    inset.legend(loc="best", frameon=False, fontsize=7)
    mark_inset(axes[-1], inset, loc1=2, loc2=3, fc="none", ec="0.35", linewidth=0.75)
    early_end = min(float(min(cce_time[-1], pyseobnr_time[-1])), 50000.0)
    cce_early_mask = cce_time <= early_end
    pyseobnr_early_mask = pyseobnr_time <= early_end
    early_inset = axes[-1].inset_axes([0.05, 0.54, 0.37, 0.38])
    early_inset.plot(
        cce_time[cce_early_mask],
        cce_fluxes["planar_momentum"][cce_early_mask],
        color="black",
        linewidth=1.2,
    )
    early_inset.plot(
        pyseobnr_time[pyseobnr_early_mask],
        pyseobnr_fluxes["planar_momentum"][pyseobnr_early_mask],
        color="0.45",
        linestyle="--",
        linewidth=1.1,
    )
    early_inset.plot(
        pyseobnr_time[pyseobnr_early_mask],
        np.asarray(effective_0pn_fluxes["planar_momentum"], dtype=float)[pyseobnr_early_mask],
        color="red",
        linestyle="--",
        linewidth=1.1,
    )
    early_inset.set_xlim(0.0, early_end)
    early_values = np.concatenate(
        [
            cce_fluxes["planar_momentum"][cce_early_mask],
            pyseobnr_fluxes["planar_momentum"][pyseobnr_early_mask],
            np.asarray(effective_0pn_fluxes["planar_momentum"], dtype=float)[pyseobnr_early_mask],
        ]
    )
    early_values = early_values[np.isfinite(early_values)]
    early_inset.set_ylim(0.0, 1.05 * float(np.max(early_values)))
    early_formatter = ScalarFormatter(useMathText=True)
    early_formatter.set_scientific(True)
    early_formatter.set_powerlimits((0, 0))
    early_formatter.set_useOffset(False)
    early_inset.yaxis.set_major_formatter(early_formatter)
    early_inset.tick_params(labelsize=7)
    early_inset.grid(alpha=0.22, linewidth=0.5)
    mark_inset(axes[-1], early_inset, loc1=1, loc2=4, fc="none", ec="0.35", linewidth=0.75)
    figure.savefig(png_path, dpi=180)
    plt.close(figure)


def _write_flux_csv(
    csv_path: Path,
    cce_time: np.ndarray,
    cce_fluxes: dict[str, np.ndarray],
    pyseobnr_time: np.ndarray,
    pyseobnr_fluxes: dict[str, np.ndarray],
    effective_0pn_fluxes: dict[str, np.ndarray | float],
    bms_radiated_momentum: np.ndarray,
    max_plot_points: int,
) -> None:
    points_per_model = max(16, max_plot_points // 2)
    cce_indices = _flux_csv_indices(cce_time, points_per_model)
    pyseobnr_indices = _flux_csv_indices(pyseobnr_time, points_per_model)
    time = np.unique(np.concatenate([cce_time[cce_indices], pyseobnr_time[pyseobnr_indices]]))
    cce_values = [
        cce_fluxes["energy_radiated"],
        cce_fluxes["angular_momentum_radiated"][:, 2],
        cce_fluxes["planar_momentum"],
    ]
    pyseobnr_values = [
        pyseobnr_fluxes["energy_radiated"],
        pyseobnr_fluxes["angular_momentum_radiated"][:, 2],
        pyseobnr_fluxes["planar_momentum"],
    ]
    effective_values = [
        np.asarray(effective_0pn_fluxes["energy_radiated"], dtype=float),
        np.asarray(effective_0pn_fluxes["angular_momentum_radiated"], dtype=float),
        np.asarray(effective_0pn_fluxes["planar_momentum"], dtype=float),
    ]
    interpolated = [
        _interp_real_with_nan(time, source_time, values)
        for source_time, source_values in (
            (cce_time, cce_values),
            (pyseobnr_time, pyseobnr_values),
            (pyseobnr_time, effective_values),
        )
        for values in source_values
    ]
    cce_bms_planar = np.asarray(cce_fluxes["bms_aligned_planar_momentum"], dtype=float)
    pyseobnr_bms_planar = np.asarray(pyseobnr_fluxes["bms_aligned_planar_momentum"], dtype=float)
    cce_bms_x = _interp_real_with_nan(time, cce_time, cce_bms_planar[:, 0])
    cce_bms_y = _interp_real_with_nan(time, cce_time, cce_bms_planar[:, 1])
    pyseobnr_bms_x = _interp_real_with_nan(time, pyseobnr_time, pyseobnr_bms_planar[:, 0])
    pyseobnr_bms_y = _interp_real_with_nan(time, pyseobnr_time, pyseobnr_bms_planar[:, 1])
    bms_target = np.tile(np.asarray(bms_radiated_momentum, dtype=float), (len(time), 1))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "t_minus_t0_M",
                "NRHybSur3dq8_CCE_energy_radiated_over_M",
                "SEOBNRv5EHM_energy_radiated_over_M",
                "effective_0PN_energy_radiated_over_M",
                "NRHybSur3dq8_CCE_angular_momentum_z_radiated_over_M2",
                "SEOBNRv5EHM_angular_momentum_z_radiated_over_M2",
                "effective_0PN_angular_momentum_z_radiated_over_M2",
                "NRHybSur3dq8_CCE_planar_momentum_over_M",
                "SEOBNRv5EHM_planar_momentum_over_M",
                "effective_0PN_planar_momentum_over_M",
                "NRHybSur3dq8_CCE_BMS_aligned_planar_momentum_x_over_M",
                "NRHybSur3dq8_CCE_BMS_aligned_planar_momentum_y_over_M",
                "SEOBNRv5EHM_BMS_aligned_planar_momentum_x_over_M",
                "SEOBNRv5EHM_BMS_aligned_planar_momentum_y_over_M",
                "BMS_remnant_radiated_momentum_x_over_M",
                "BMS_remnant_radiated_momentum_y_over_M",
                "BMS_remnant_radiated_momentum_z_over_M",
            ]
        )
        for values in zip(
            time,
            *interpolated,
            cce_bms_x,
            cce_bms_y,
            pyseobnr_bms_x,
            pyseobnr_bms_y,
            bms_target[:, 0],
            bms_target[:, 1],
            bms_target[:, 2],
            strict=True,
        ):
            (
                time_value,
                cce_energy,
                cce_jz,
                cce_pperp,
                pyseobnr_energy,
                pyseobnr_jz,
                pyseobnr_pperp,
                effective_energy,
                effective_jz,
                effective_pperp,
                cce_bms_p_x,
                cce_bms_p_y,
                pyseobnr_bms_p_x,
                pyseobnr_bms_p_y,
                bms_target_p_x,
                bms_target_p_y,
                bms_target_p_z,
            ) = values
            writer.writerow(
                [
                    time_value,
                    cce_energy,
                    pyseobnr_energy,
                    effective_energy,
                    cce_jz,
                    pyseobnr_jz,
                    effective_jz,
                    cce_pperp,
                    pyseobnr_pperp,
                    effective_pperp,
                    cce_bms_p_x,
                    cce_bms_p_y,
                    pyseobnr_bms_p_x,
                    pyseobnr_bms_p_y,
                    bms_target_p_x,
                    bms_target_p_y,
                    bms_target_p_z,
                ]
            )


def _replot_flux_comparison_from_csv(
    csv_path: Path,
    png_path: Path,
    pyseobnr_approximant: str,
    q: float,
) -> None:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    time = np.asarray(data["t_minus_t0_M"], dtype=float)
    cce_fluxes = {
        "energy_radiated": np.asarray(data["NRHybSur3dq8_CCE_energy_radiated_over_M"], dtype=float),
        "angular_momentum_radiated": np.column_stack(
            [
                np.zeros_like(time),
                np.zeros_like(time),
                np.asarray(data["NRHybSur3dq8_CCE_angular_momentum_z_radiated_over_M2"], dtype=float),
            ]
        ),
        "planar_momentum": np.asarray(
            data["NRHybSur3dq8_CCE_planar_momentum_over_M"], dtype=float
        ),
    }
    pyseobnr_fluxes = {
        "energy_radiated": np.asarray(data["SEOBNRv5EHM_energy_radiated_over_M"], dtype=float),
        "angular_momentum_radiated": np.column_stack(
            [
                np.zeros_like(time),
                np.zeros_like(time),
                np.asarray(data["SEOBNRv5EHM_angular_momentum_z_radiated_over_M2"], dtype=float),
            ]
        ),
        "planar_momentum": np.asarray(
            data["SEOBNRv5EHM_planar_momentum_over_M"], dtype=float
        ),
    }
    effective_0pn_fluxes = {
        "energy_radiated": np.asarray(data["effective_0PN_energy_radiated_over_M"], dtype=float),
        "angular_momentum_radiated": np.asarray(
            data["effective_0PN_angular_momentum_z_radiated_over_M2"], dtype=float
        ),
        "planar_momentum": np.asarray(data["effective_0PN_planar_momentum_over_M"], dtype=float),
    }
    _bms_remnant_mass, _bms_boost, bms_remnant_momentum = _pn_bms_remnant(q)
    _plot_flux_comparison(
        time,
        cce_fluxes,
        time,
        pyseobnr_fluxes,
        effective_0pn_fluxes,
        png_path,
        pyseobnr_approximant,
        bms_remnant_momentum,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, default=2.0)
    parser.add_argument("--x-start", type=float, default=DEFAULT_X_START)
    parser.add_argument("--cce-stop", type=float, default=100.0)
    parser.add_argument("--delta-t", type=float, default=20.0)
    parser.add_argument("--cce-refinement-start", type=float, default=-3000.0)
    parser.add_argument("--cce-refinement-delta-t", type=float, default=0.5)
    parser.add_argument("--pyseobnr-delta-t", type=float, default=1.0)
    parser.add_argument("--pyseobnr-approximant", default="SEOBNRv5EHM")
    parser.add_argument("--search-start", type=float, default=-2_500_000.0)
    parser.add_argument("--search-stop", type=float, default=-1_000_000.0)
    parser.add_argument("--search-dt", type=float, default=200.0)
    parser.add_argument("--lmax", type=int, default=10)
    parser.add_argument("--total-mass-solar", type=float, default=50.0)
    parser.add_argument("--max-plot-points", type=int, default=8000)
    parser.add_argument("--output-dir", default=str(ROOT / "examples" / "output"))
    parser.add_argument("--plot-from-csv", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"seobnrv5ehm_nrhybsur3dq8_cce_h20_h30_q{args.q:g}_x{args.x_start:g}"
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"
    linear_h20_png_path = output_dir / f"{stem}_h20_linear.png"
    flux_csv_path = output_dir / f"{stem}_fluxes.csv"
    flux_png_path = output_dir / f"{stem}_fluxes.png"
    if args.plot_from_csv:
        if not csv_path.exists():
            raise FileNotFoundError(f"cannot replot missing CSV: {csv_path}")
        _replot_linear_h20_from_csv(
            csv_path,
            linear_h20_png_path,
            args.pyseobnr_approximant,
        )
        if flux_csv_path.exists():
            _replot_flux_comparison_from_csv(
                flux_csv_path,
                flux_png_path,
                args.pyseobnr_approximant,
                args.q,
            )
        print(f"Redrew linear h20 comparison: {linear_h20_png_path}")
        if flux_csv_path.exists():
            print(f"Redrew Poincaré flux comparison: {flux_png_path}")
        return 0

    try:
        import gwsurrogate
    except ImportError as exc:
        raise SystemExit("This example requires gwsurrogate.") from exc

    omega_target = float(args.x_start) ** 1.5
    surrogate = gwsurrogate.LoadSurrogate("NRHybSur3dq8_CCE")
    cce_start = _find_cce_start_time(
        surrogate,
        args.q,
        omega_target,
        args.search_start,
        args.search_stop,
        args.search_dt,
    )
    t_cce, h_cce = _load_cce_segment(
        surrogate,
        args.q,
        cce_start,
        args.cce_stop,
        args.delta_t,
        args.cce_refinement_start,
        args.cce_refinement_delta_t,
    )
    cce_modes = complete_nonprecessing_modes(h_cce)
    cce_oscillatory_modes = {
        mode: values
        for mode, values in cce_modes.items()
        if mode[1] != 0
    }
    cce_hdot = differentiate_modes(t_cce, cce_oscillatory_modes)
    cce_flux_hdot = differentiate_modes(t_cce, cce_modes)
    cce_fluxes = compute_poincare_fluxes(t_cce, cce_modes, hdot=cce_flux_hdot)
    cce_memory = compute_memory_modes(
        t_cce,
        cce_oscillatory_modes,
        [(2, 0)],
        lmax=args.lmax,
        hdot=cce_hdot,
    )
    h20_cce_perturbative = cce_memory[(2, 0)]["h_displacement"]
    omega_pyseobnr_start = _orbital_frequency_from_h22(t_cce, h_cce[(2, 2)])[0]
    t_pyseobnr, pyseobnr_modes = _generate_pyseobnr_modes(
        args.q,
        omega_pyseobnr_start,
        args.pyseobnr_delta_t,
        args.total_mass_solar,
        args.pyseobnr_approximant,
    )
    omega_pyseobnr_actual_start = _orbital_frequency_from_h22(
        t_pyseobnr, pyseobnr_modes[(2, 2)]
    )[0]
    pyseobnr_hdot = differentiate_modes(t_pyseobnr, pyseobnr_modes)
    pyseobnr_fluxes = compute_poincare_fluxes(
        t_pyseobnr,
        pyseobnr_modes,
        hdot=pyseobnr_hdot,
    )
    bms_remnant_mass, bms_boost, bms_remnant_momentum = _pn_bms_remnant(args.q)
    bms_gamma = 1.0 / np.sqrt(1.0 - float(np.dot(bms_boost, bms_boost)))
    bms_radiated_momentum = -bms_gamma * bms_remnant_mass * bms_boost
    cce_fluxes["planar_momentum"] = np.linalg.norm(
        cce_fluxes["linear_momentum_radiated"][:, :2],
        axis=1,
    )
    pyseobnr_fluxes["planar_momentum"] = np.linalg.norm(
        pyseobnr_fluxes["linear_momentum_radiated"][:, :2],
        axis=1,
    )
    cce_fluxes["bms_aligned_planar_momentum"], cce_bms_reference_time = _bms_aligned_planar_momentum(
        t_cce,
        cce_modes,
        cce_fluxes["linear_momentum_radiated"],
    )
    pyseobnr_fluxes["bms_aligned_planar_momentum"], pyseobnr_bms_reference_time = _bms_aligned_planar_momentum(
        t_pyseobnr,
        pyseobnr_modes,
        pyseobnr_fluxes["linear_momentum_radiated"],
    )
    cce_bms_terminal_momentum = np.array(
        [
            cce_fluxes["bms_aligned_planar_momentum"][-1, 0],
            cce_fluxes["bms_aligned_planar_momentum"][-1, 1],
            cce_fluxes["linear_momentum_radiated"][-1, 2],
        ]
    )
    pyseobnr_bms_terminal_momentum = np.array(
        [
            pyseobnr_fluxes["bms_aligned_planar_momentum"][-1, 0],
            pyseobnr_fluxes["bms_aligned_planar_momentum"][-1, 1],
            pyseobnr_fluxes["linear_momentum_radiated"][-1, 2],
        ]
    )
    pyseobnr_memory = compute_memory_modes(
        t_pyseobnr,
        pyseobnr_modes,
        [(2, 0), (3, 0)],
        lmax=args.lmax,
        hdot=pyseobnr_hdot,
    )
    h20_pyseobnr = pyseobnr_memory[(2, 0)]["h_displacement"]
    dh20_dt_pyseobnr = pyseobnr_memory[(2, 0)]["dh_displacement_dt"]
    h30_pyseobnr = pyseobnr_memory[(3, 0)]["h_spin_mode"]

    rel_cce = t_cce - t_cce[0]
    rel_pyseobnr = t_pyseobnr - t_pyseobnr[0]
    cce_x_momentum_phase = omega_pyseobnr_start ** (2.0 / 3.0)
    pyseobnr_x_momentum_phase = omega_pyseobnr_actual_start ** (2.0 / 3.0)
    cce_effective_0pn_fluxes = _effective_0pn_fluxes(
        rel_cce,
        args.q,
        cce_fluxes,
        cce_x_momentum_phase,
    )
    pyseobnr_effective_0pn_fluxes = _effective_0pn_fluxes(
        rel_pyseobnr,
        args.q,
        pyseobnr_fluxes,
        pyseobnr_x_momentum_phase,
    )
    pyseobnr_plateau_duration = max(0.0, float(rel_cce[-1] - rel_pyseobnr[-1]))
    dh20_cce = h_cce[(2, 0)] - h_cce[(2, 0)][0]
    dh20_cce_perturbative = h20_cce_perturbative - h20_cce_perturbative[0]
    dh30_cce = h_cce[(3, 0)] - h_cce[(3, 0)][0]
    dh20_pyseobnr = h20_pyseobnr - h20_pyseobnr[0]
    dh30_pyseobnr = h30_pyseobnr - h30_pyseobnr[0]
    nu = symmetric_mass_ratio(args.q)
    x0 = omega_pyseobnr_start ** (2.0 / 3.0)
    x_eff = infer_x_eff_from_dh20(dh20_dt_pyseobnr[0], args.q)

    dh20_cce_norm = dh20_cce / nu
    dh20_cce_perturbative_norm = dh20_cce_perturbative / nu
    dh20_pyseobnr_norm = dh20_pyseobnr / nu
    dh30_cce_norm = dh30_cce / nu
    dh30_pyseobnr_norm = dh30_pyseobnr / nu

    points_per_model = max(16, args.max_plot_points // 2)
    cce_plot_idx = _endpoint_refined_indices(rel_cce, points_per_model)
    pyseobnr_plot_idx = _endpoint_refined_indices(rel_pyseobnr, points_per_model)
    cce_plot_idx = _include_extrema(
        cce_plot_idx,
        np.real(dh20_cce_norm),
        np.real(dh20_cce_perturbative_norm),
        np.imag(dh30_cce_norm),
    )
    pyseobnr_plot_idx = _include_extrema(
        pyseobnr_plot_idx,
        np.real(dh20_pyseobnr_norm),
        np.imag(dh30_pyseobnr_norm),
    )
    csv_t = np.unique(
        np.concatenate(
            [
                rel_cce[cce_plot_idx],
                rel_pyseobnr[pyseobnr_plot_idx],
                np.asarray([rel_cce[-1]]),
            ]
        )
    )
    dh20_cce_csv = _interp_complex_with_plateau(csv_t, rel_cce, dh20_cce_norm)
    dh20_cce_perturbative_csv = _interp_complex_with_plateau(
        csv_t, rel_cce, dh20_cce_perturbative_norm
    )
    dh30_cce_csv = _interp_complex_with_plateau(csv_t, rel_cce, dh30_cce_norm)
    dh20_pyseobnr_csv = _interp_complex_with_plateau(
        csv_t, rel_pyseobnr, dh20_pyseobnr_norm
    )
    dh30_pyseobnr_csv = _interp_complex_with_plateau(
        csv_t, rel_pyseobnr, dh30_pyseobnr_norm
    )

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "t_minus_t0_M",
                "NRHybSur3dq8_CCE_delta_h20_real_over_nu",
                "NRHybSur3dq8_CCE_delta_h20_imag_over_nu",
                "NRHybSur3dq8_CCE_perturbative_delta_h20_real_over_nu",
                "NRHybSur3dq8_CCE_perturbative_delta_h20_imag_over_nu",
                "SEOBNRv5EHM_delta_h20_real_over_nu",
                "SEOBNRv5EHM_delta_h20_imag_over_nu",
                "NRHybSur3dq8_CCE_delta_h30_real_over_nu",
                "NRHybSur3dq8_CCE_delta_h30_imag_over_nu",
                "SEOBNRv5EHM_delta_h30_real_over_nu",
                "SEOBNRv5EHM_delta_h30_imag_over_nu",
            ]
        )
        for values in zip(
            csv_t,
            dh20_cce_csv,
            dh20_cce_perturbative_csv,
            dh20_pyseobnr_csv,
            dh30_cce_csv,
            dh30_pyseobnr_csv,
            strict=True,
        ):
            time_value, c20, c20_perturbative, e20, c30, e30 = values
            writer.writerow(
                [
                    time_value,
                    c20.real,
                    c20.imag,
                    c20_perturbative.real,
                    c20_perturbative.imag,
                    e20.real,
                    e20.imag,
                    c30.real,
                    c30.imag,
                    e30.real,
                    e30.imag,
                ]
            )

    _write_flux_csv(
        flux_csv_path,
        rel_cce,
        cce_fluxes,
        rel_pyseobnr,
        pyseobnr_fluxes,
        pyseobnr_effective_0pn_fluxes,
        bms_radiated_momentum,
        args.max_plot_points,
    )

    import matplotlib.pyplot as plt

    cce_label = r"$\mathtt{NRHybSur3dq8\_CCE}$ $h_{l,m}$"
    pyseobnr_label_hD20 = (
        rf"$\mathtt{{{args.pyseobnr_approximant}}}$ $h^{{\mathrm{{D}}}}_{{2,0}}$"
    )
    pyseobnr_label_hS30 = (
        rf"$\mathtt{{{args.pyseobnr_approximant}}}$ $h^{{\mathrm{{S}}}}_{{3,0}}$"
    )
    y20_cce = _positive_for_log(np.real(dh20_cce_norm))
    y20_pyseobnr = _positive_for_log(np.real(dh20_pyseobnr_norm))
    y30_cce = _positive_for_log(np.imag(dh30_cce_norm))
    y30_pyseobnr = _positive_for_log(np.imag(dh30_pyseobnr_norm))
    y20_lim = _positive_log_limits(y20_cce, y20_pyseobnr)
    y30_lim = _positive_log_limits(y30_cce, y30_pyseobnr, upper_series=y30_pyseobnr)

    pyseobnr_plot_t = rel_pyseobnr[pyseobnr_plot_idx]
    y20_pyseobnr_plot = y20_pyseobnr[pyseobnr_plot_idx]
    y30_pyseobnr_plot = y30_pyseobnr[pyseobnr_plot_idx]
    h20_pyseobnr_plot = np.real(dh20_pyseobnr_norm[pyseobnr_plot_idx])
    if pyseobnr_plateau_duration:
        pyseobnr_plot_t = np.append(pyseobnr_plot_t, rel_cce[-1])
        y20_pyseobnr_plot = np.append(y20_pyseobnr_plot, y20_pyseobnr[-1])
        y30_pyseobnr_plot = np.append(y30_pyseobnr_plot, y30_pyseobnr[-1])
        h20_pyseobnr_plot = np.append(h20_pyseobnr_plot, h20_pyseobnr_plot[-1])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(
        rel_cce[cce_plot_idx],
        y20_cce[cce_plot_idx],
        color="black",
        linewidth=1.4,
        label=cce_label,
    )
    axes[0].plot(
        pyseobnr_plot_t,
        y20_pyseobnr_plot,
        color="red",
        linestyle="--",
        linewidth=1.3,
        label=pyseobnr_label_hD20,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylim(*y20_lim)
    axes[0].set_ylabel(r"$\mathrm{Re}\,\Delta h_{2,0}/(\nu M/R)$")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(loc="best", frameon=False)

    axes[1].plot(
        rel_cce[cce_plot_idx],
        y30_cce[cce_plot_idx],
        color="black",
        linewidth=1.4,
        label=cce_label,
    )
    axes[1].plot(
        pyseobnr_plot_t,
        y30_pyseobnr_plot,
        color="red",
        linestyle="--",
        linewidth=1.3,
        label=pyseobnr_label_hS30,
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(*y30_lim)
    axes[1].set_ylabel(r"$\mathrm{Im}\,\Delta h_{3,0}/(\nu M/R)$")
    axes[1].set_xlabel(r"$(t-t_0)/M$")
    axes[1].grid(True, which="both", alpha=0.25)
    fig.suptitle(
        rf"$\mathtt{{{args.pyseobnr_approximant}}}$ vs $\mathtt{{NRHybSur3dq8\_CCE}}$, "
        rf"$q={args.q:g}$, $x_0={x0:.6g}$, "
        rf"$x_{{\rm eff}}={x_eff:.6g}$"
    )
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    _plot_linear_h20_comparison(
        rel_cce[cce_plot_idx],
        dh20_cce_norm[cce_plot_idx],
        dh20_cce_perturbative_norm[cce_plot_idx],
        pyseobnr_plot_t,
        h20_pyseobnr_plot,
        linear_h20_png_path,
        args.pyseobnr_approximant,
    )
    _plot_flux_comparison(
        rel_cce,
        cce_fluxes,
        rel_pyseobnr,
        pyseobnr_fluxes,
        pyseobnr_effective_0pn_fluxes,
        flux_png_path,
        args.pyseobnr_approximant,
        bms_remnant_momentum,
    )

    print(f"{args.pyseobnr_approximant} vs NRHybSur3dq8_CCE h20/h30 comparison")
    print(f"q = {args.q:g}")
    print(f"target x_start = {args.x_start:.12e}")
    print(f"target Omega = {omega_target:.12e}")
    print(f"NRHybSur3dq8_CCE t0 = {t_cce[0]:.3f} M, final time = {t_cce[-1]:.3f} M")
    print(
        f"NRHybSur3dq8_CCE initial Omega used for {args.pyseobnr_approximant} = "
        f"{omega_pyseobnr_start:.12e}"
    )
    print(
        f"{args.pyseobnr_approximant} initial Omega from h22 = "
        f"{omega_pyseobnr_actual_start:.12e}"
    )
    print(f"x0 = {x0:.12e}")
    print(f"x_eff = {x_eff:.12e}")
    print(f"effective 0PN x_energy = {pyseobnr_effective_0pn_fluxes['x_energy']:.12e}")
    print(f"effective 0PN x_jz = {pyseobnr_effective_0pn_fluxes['x_jz']:.12e}")
    print(f"effective 0PN x_momentum = {pyseobnr_effective_0pn_fluxes['x_momentum']:.12e}")
    print(f"effective 0PN x_momentum_phase = {pyseobnr_effective_0pn_fluxes['x_momentum_phase']:.12e}")
    print(f"CCE effective 0PN x_energy = {cce_effective_0pn_fluxes['x_energy']:.12e}")
    print(f"CCE effective 0PN x_jz = {cce_effective_0pn_fluxes['x_jz']:.12e}")
    print(f"CCE effective 0PN x_momentum = {cce_effective_0pn_fluxes['x_momentum']:.12e}")
    print(f"CCE effective 0PN x_momentum_phase = {cce_effective_0pn_fluxes['x_momentum_phase']:.12e}")
    print(
        "effective 0PN pre-t0 Delta E^gw / M = "
        f"{pyseobnr_effective_0pn_fluxes['prehistory_energy']:.12e}"
    )
    print(
        "effective 0PN pre-t0 |Delta P^gw| / M = "
        f"{pyseobnr_effective_0pn_fluxes['prehistory_planar_momentum']:.12e}"
    )
    print(f"nu = {nu:.12e}")
    print(f"NRHybSur3dq8_CCE early delta_t = {args.delta_t:g} M")
    print(
        "NRHybSur3dq8_CCE merger refinement = "
        f"{args.cce_refinement_delta_t:g} M from t={args.cce_refinement_start:g} M"
    )
    print(f"{args.pyseobnr_approximant} delta_t = {args.pyseobnr_delta_t:g} M")
    if pyseobnr_plateau_duration:
        print(f"{args.pyseobnr_approximant} curve held at final value for the last {pyseobnr_plateau_duration:.1f} M")
    print(f"{args.pyseobnr_approximant} oscillatory modes = {sorted(pyseobnr_modes)}")
    print(f"final NRHybSur3dq8_CCE Delta h20 = {_format_complex(dh20_cce[-1])}")
    print(
        "final NRHybSur3dq8_CCE perturbative Delta h20 = "
        f"{_format_complex(dh20_cce_perturbative[-1])}"
    )
    print(f"final {args.pyseobnr_approximant} Delta h20 = {_format_complex(dh20_pyseobnr[-1])}")
    print(f"final NRHybSur3dq8_CCE Delta h30 = {_format_complex(dh30_cce[-1])}")
    print(f"final {args.pyseobnr_approximant} Delta h30 = {_format_complex(dh30_pyseobnr[-1])}")
    print(f"final NRHybSur3dq8_CCE Delta h20 / nu = {_format_complex(dh20_cce_norm[-1])}")
    print(
        "final NRHybSur3dq8_CCE perturbative Delta h20 / nu = "
        f"{_format_complex(dh20_cce_perturbative_norm[-1])}"
    )
    print(f"final {args.pyseobnr_approximant} Delta h20 / nu = {_format_complex(dh20_pyseobnr_norm[-1])}")
    print(f"final NRHybSur3dq8_CCE Delta h30 / nu = {_format_complex(dh30_cce_norm[-1])}")
    print(f"final {args.pyseobnr_approximant} Delta h30 / nu = {_format_complex(dh30_pyseobnr_norm[-1])}")
    print(f"PN-BMS remnant boost = {bms_boost.tolist()}")
    print(f"PN-BMS remnant radiated-momentum target = {bms_radiated_momentum.tolist()}")
    print(f"PN-BMS remnant momentum magnitude = {bms_remnant_momentum:.12e}")
    print(
        "NRHybSur3dq8_CCE PN-BMS phase-reference time since t0 = "
        f"{cce_bms_reference_time - t_cce[0]:.12e}"
    )
    print(
        f"{args.pyseobnr_approximant} PN-BMS phase-reference time since t0 = "
        f"{pyseobnr_bms_reference_time - t_pyseobnr[0]:.12e}"
    )
    print(f"final NRHybSur3dq8_CCE Delta E^gw / M = {cce_fluxes['energy_radiated'][-1]:.12e}")
    print(f"final {args.pyseobnr_approximant} Delta E^gw / M = {pyseobnr_fluxes['energy_radiated'][-1]:.12e}")
    print(
        "final NRHybSur3dq8_CCE Delta J_z^gw / M^2 = "
        f"{cce_fluxes['angular_momentum_radiated'][-1, 2]:.12e}"
    )
    print(
        f"final {args.pyseobnr_approximant} Delta J_z^gw / M^2 = "
        f"{pyseobnr_fluxes['angular_momentum_radiated'][-1, 2]:.12e}"
    )
    print(
        "final NRHybSur3dq8_CCE |Delta P^gw| / M = "
        f"{cce_fluxes['planar_momentum'][-1]:.12e}"
    )
    print(
        f"final {args.pyseobnr_approximant} |Delta P^gw| / M = "
        f"{pyseobnr_fluxes['planar_momentum'][-1]:.12e}"
    )
    print(
        "final NRHybSur3dq8_CCE Delta P_z^gw / M = "
        f"{cce_fluxes['linear_momentum_radiated'][-1, 2]:.12e}"
    )
    print(
        f"final {args.pyseobnr_approximant} Delta P_z^gw / M = "
        f"{pyseobnr_fluxes['linear_momentum_radiated'][-1, 2]:.12e}"
    )
    print(
        "final NRHybSur3dq8_CCE PN-BMS-aligned Delta P^gw / M = "
        f"{cce_bms_terminal_momentum.tolist()}"
    )
    print(
        f"final {args.pyseobnr_approximant} PN-BMS-aligned Delta P^gw / M = "
        f"{pyseobnr_bms_terminal_momentum.tolist()}"
    )
    print(
        "NRHybSur3dq8_CCE PN-BMS vector residual / M = "
        f"{(cce_bms_terminal_momentum - bms_radiated_momentum).tolist()}"
    )
    print(
        f"{args.pyseobnr_approximant} PN-BMS vector residual / M = "
        f"{(pyseobnr_bms_terminal_momentum - bms_radiated_momentum).tolist()}"
    )
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")
    print(f"Saved linear h20 plot: {linear_h20_png_path}")
    print(f"Saved Poincaré flux CSV: {flux_csv_path}")
    print(f"Saved Poincaré flux plot: {flux_png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

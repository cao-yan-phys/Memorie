"""Compare SEOBNRv5EHM perturbative h20/h30 with NRHybSur3dq8_CCE h20/h30.

The SEOBNRv5EHM starting orbital frequency is measured from the initial
NRHybSur3dq8_CCE h22 mode. Both comparisons plot h(t)-h(t0).
"""

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

from vacuum_memory_modes import (  # noqa: E402
    complete_nonprecessing_modes,
    compute_memory_modes,
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


def _fit_initial_orbital_frequency(
    t: np.ndarray,
    h22: np.ndarray,
    fit_duration: float,
) -> float:
    n_fit = int(np.searchsorted(t, t[0] + fit_duration, side="right"))
    n_fit = max(8, min(n_fit, len(t)))
    phase = np.unwrap(np.angle(h22[:n_fit]))
    slope, _intercept = np.polyfit(t[:n_fit] - t[0], phase, deg=1)
    return float(abs(slope) / 2.0)


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
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    times = np.arange(t_start, t_stop + 0.5 * delta_t, delta_t)
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
    """Interpolate a complex series and hold its final value beyond the end."""

    return np.interp(x_new, x, np.real(y)) + 1j * np.interp(x_new, x, np.imag(y))


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    step = max(1, int(np.ceil(n / max_points)))
    return np.arange(0, n, step, dtype=int)


def _positive_for_log(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[out <= 0.0] = np.nan
    return out


def _positive_log_limits(*series: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(item, dtype=float).ravel() for item in series])
    values = values[np.isfinite(values) & (values > 0.0)]
    if not len(values):
        return 1e-12, 1.0
    ymin = max(float(np.min(values)) * 0.5, 1e-300)
    ymax = max(float(np.max(values)) * 1.5, ymin * 10.0)
    return ymin, ymax


def _format_complex(value: complex) -> str:
    value = complex(value)
    return f"{value.real:+.6e}{value.imag:+.6e}j"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, default=2.0)
    parser.add_argument("--x-start", type=float, default=DEFAULT_X_START)
    parser.add_argument("--cce-stop", type=float, default=100.0)
    parser.add_argument("--delta-t", type=float, default=20.0, help="NRHybSur3dq8_CCE/output spacing in units of M")
    parser.add_argument("--pyseobnr-delta-t", type=float, default=1.0, help="pyseobnr time step in units of M")
    parser.add_argument("--pyseobnr-approximant", default="SEOBNRv5EHM")
    parser.add_argument("--fit-duration", type=float, default=4000.0)
    parser.add_argument("--search-start", type=float, default=-2_500_000.0)
    parser.add_argument("--search-stop", type=float, default=-1_000_000.0)
    parser.add_argument("--search-dt", type=float, default=200.0)
    parser.add_argument("--lmax", type=int, default=10)
    parser.add_argument("--total-mass-solar", type=float, default=50.0)
    parser.add_argument("--max-plot-points", type=int, default=8000)
    parser.add_argument("--output-dir", default=str(ROOT / "examples" / "output"))
    args = parser.parse_args()

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
    )
    omega_pyseobnr_start = _fit_initial_orbital_frequency(t_cce, h_cce[(2, 2)], args.fit_duration)
    t_pyseobnr, pyseobnr_modes = _generate_pyseobnr_modes(
        args.q,
        omega_pyseobnr_start,
        args.pyseobnr_delta_t,
        args.total_mass_solar,
        args.pyseobnr_approximant,
    )
    pyseobnr_hdot = differentiate_modes(t_pyseobnr, pyseobnr_modes)
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
    pyseobnr_plateau_duration = max(0.0, float(rel_cce[-1] - rel_pyseobnr[-1]))
    h20_pyseobnr_on_cce = _interp_complex_with_plateau(rel_cce, rel_pyseobnr, h20_pyseobnr)
    h30_pyseobnr_on_cce = _interp_complex_with_plateau(rel_cce, rel_pyseobnr, h30_pyseobnr)

    dh20_cce = h_cce[(2, 0)] - h_cce[(2, 0)][0]
    dh30_cce = h_cce[(3, 0)] - h_cce[(3, 0)][0]
    dh20_pyseobnr = h20_pyseobnr_on_cce - h20_pyseobnr_on_cce[0]
    dh30_pyseobnr = h30_pyseobnr_on_cce - h30_pyseobnr_on_cce[0]
    nu = symmetric_mass_ratio(args.q)
    x0 = omega_pyseobnr_start ** (2.0 / 3.0)
    x_eff = infer_x_eff_from_dh20(dh20_dt_pyseobnr[0], args.q)

    dh20_cce_norm = dh20_cce / nu
    dh20_pyseobnr_norm = dh20_pyseobnr / nu
    dh30_cce_norm = dh30_cce / nu
    dh30_pyseobnr_norm = dh30_pyseobnr / nu

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"seobnrv5ehm_nrhybsur3dq8_cce_h20_h30_q{args.q:g}_x{args.x_start:g}"
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"
    plot_idx = _downsample_indices(len(rel_cce), args.max_plot_points)

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "t_minus_t0_M",
                "NRHybSur3dq8_CCE_delta_h20_real_over_nu",
                "NRHybSur3dq8_CCE_delta_h20_imag_over_nu",
                "SEOBNRv5EHM_delta_h20_real_over_nu",
                "SEOBNRv5EHM_delta_h20_imag_over_nu",
                "NRHybSur3dq8_CCE_delta_h30_real_over_nu",
                "NRHybSur3dq8_CCE_delta_h30_imag_over_nu",
                "SEOBNRv5EHM_delta_h30_real_over_nu",
                "SEOBNRv5EHM_delta_h30_imag_over_nu",
            ]
        )
        for values in zip(
            rel_cce[plot_idx],
            dh20_cce_norm[plot_idx],
            dh20_pyseobnr_norm[plot_idx],
            dh30_cce_norm[plot_idx],
            dh30_pyseobnr_norm[plot_idx],
        ):
            time_value, c20, e20, c30, e30 = values
            writer.writerow(
                [
                    time_value,
                    c20.real,
                    c20.imag,
                    e20.real,
                    e20.imag,
                    c30.real,
                    c30.imag,
                    e30.real,
                    e30.imag,
                ]
            )

    import matplotlib.pyplot as plt

    cce_label = r"$\mathtt{NRHybSur3dq8\_CCE}$"
    pyseobnr_label = rf"$\mathtt{{{args.pyseobnr_approximant}}}$ perturbative"
    y20_cce = _positive_for_log(np.real(dh20_cce_norm))
    y20_pyseobnr = _positive_for_log(np.real(dh20_pyseobnr_norm))
    y30_cce = _positive_for_log(np.imag(dh30_cce_norm))
    y30_pyseobnr = _positive_for_log(np.imag(dh30_pyseobnr_norm))
    y20_lim = _positive_log_limits(y20_cce, y20_pyseobnr)
    y30_lim = _positive_log_limits(y30_cce, y30_pyseobnr)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(rel_cce[plot_idx], y20_cce[plot_idx], color="black", linewidth=1.4, label=cce_label)
    axes[0].plot(
        rel_cce[plot_idx],
        y20_pyseobnr[plot_idx],
        color="red",
        linestyle="--",
        linewidth=1.3,
        label=pyseobnr_label,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylim(*y20_lim)
    axes[0].set_ylabel(r"$\mathrm{Re}\,\Delta h_{20}/(\nu M/R)$")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(loc="best", frameon=False)

    axes[1].plot(rel_cce[plot_idx], y30_cce[plot_idx], color="black", linewidth=1.4, label=cce_label)
    axes[1].plot(
        rel_cce[plot_idx],
        y30_pyseobnr[plot_idx],
        color="red",
        linestyle="--",
        linewidth=1.3,
        label=pyseobnr_label,
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(*y30_lim)
    axes[1].set_ylabel(r"$\mathrm{Im}\,\Delta h_{30}/(\nu M/R)$")
    axes[1].set_xlabel(r"$t-t_0$ [$M$]")
    axes[1].grid(True, which="both", alpha=0.25)
    fig.suptitle(
        rf"$\mathtt{{{args.pyseobnr_approximant}}}$ vs $\mathtt{{NRHybSur3dq8\_CCE}}$, "
        rf"$q={args.q:g}$, $x_0={x0:.6g}$, "
        rf"$x_{{\rm eff}}={x_eff:.6g}$"
    )
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"{args.pyseobnr_approximant} vs NRHybSur3dq8_CCE h20/h30 comparison")
    print(f"q = {args.q:g}")
    print(f"target x_start = {args.x_start:.12e}")
    print(f"target Omega = {omega_target:.12e}")
    print(f"NRHybSur3dq8_CCE t0 = {t_cce[0]:.3f} M, final time = {t_cce[-1]:.3f} M")
    print(f"NRHybSur3dq8_CCE-fit Omega_start used for {args.pyseobnr_approximant} = {omega_pyseobnr_start:.12e}")
    print(f"x0 = {x0:.12e}")
    print(f"x_eff = {x_eff:.12e}")
    print(f"nu = {nu:.12e}")
    print(f"NRHybSur3dq8_CCE/output delta_t = {args.delta_t:g} M")
    print(f"{args.pyseobnr_approximant} delta_t = {args.pyseobnr_delta_t:g} M")
    if pyseobnr_plateau_duration:
        print(f"{args.pyseobnr_approximant} curve held at final value for the last {pyseobnr_plateau_duration:.1f} M")
    print(f"{args.pyseobnr_approximant} oscillatory modes = {sorted(pyseobnr_modes)}")
    print(f"final NRHybSur3dq8_CCE Delta h20 = {_format_complex(dh20_cce[-1])}")
    print(f"final {args.pyseobnr_approximant} Delta h20 = {_format_complex(dh20_pyseobnr[-1])}")
    print(f"final NRHybSur3dq8_CCE Delta h30 = {_format_complex(dh30_cce[-1])}")
    print(f"final {args.pyseobnr_approximant} Delta h30 = {_format_complex(dh30_pyseobnr[-1])}")
    print(f"final NRHybSur3dq8_CCE Delta h20 / nu = {_format_complex(dh20_cce_norm[-1])}")
    print(f"final {args.pyseobnr_approximant} Delta h20 / nu = {_format_complex(dh20_pyseobnr_norm[-1])}")
    print(f"final NRHybSur3dq8_CCE Delta h30 / nu = {_format_complex(dh30_cce_norm[-1])}")
    print(f"final {args.pyseobnr_approximant} Delta h30 / nu = {_format_complex(dh30_pyseobnr_norm[-1])}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

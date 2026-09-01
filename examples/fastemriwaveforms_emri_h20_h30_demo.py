"""Generate FEW EMRI perturbative h20/h30 memory waveforms."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memorie import (  # noqa: E402
    FewEmriConfig,
    compute_few_emri_memory_modes,
    h20_lo,
    h30_spin_lo,
)


def _format_complex(value: complex) -> str:
    value = complex(value)
    return f"{value.real:+.6e}{value.imag:+.6e}j"


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
        return 1e-18, 1.0
    ymin = max(float(np.min(values)) * 0.5, 1e-300)
    ymax = max(float(np.max(values)) * 1.5, ymin * 10.0)
    return ymin, ymax


def _token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _interp_complex_with_nan(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    real = np.interp(x_new, x, np.real(y), left=np.nan, right=np.nan)
    imag = np.interp(x_new, x, np.imag(y), left=np.nan, right=np.nan)
    return real + 1j * imag


def _x_0pn_series(t: np.ndarray, x0: float, q: float) -> np.ndarray:
    nu = float(q) / (1.0 + float(q)) ** 2
    denominator = float(x0) ** -4 - (256.0 / 5.0) * nu * (t - t[0])
    x = np.full_like(denominator, np.nan, dtype=float)
    valid = denominator > 0.0
    x[valid] = denominator[valid] ** -0.25
    return x


def _eccentric_effective_0pn_h20_delta(
    t: np.ndarray,
    h20: np.ndarray,
    q: float,
    e0: float,
) -> np.ndarray:
    """Return the exact-in-e Newtonian DC effective 0PN h20 increment."""

    t_arr = np.asarray(t, dtype=float)
    h20_arr = np.asarray(h20)
    e0 = float(e0)
    if len(t_arr) < 2 or len(h20_arr) != len(t_arr):
        raise ValueError("t and h20 must have matching lengths of at least two")
    if not (0.0 < e0 < 1.0):
        raise ValueError("the eccentric effective 0PN model requires 0 < e0 < 1")

    nu = float(q) / (1.0 + float(q)) ** 2
    slope_index = min(10, len(t_arr) - 1)
    s_dc = float(
        np.real(
            (h20_arr[slope_index] - h20_arr[0])
            / (t_arr[slope_index] - t_arr[0])
        )
    )
    if s_dc <= 0.0:
        raise ValueError(f"expected positive DC h20 slope, got {s_dc}")

    c20 = (256.0 / 7.0) * np.sqrt(np.pi / 30.0)
    eccentric_factor = (1.0 - e0**2) ** 1.5 * (
        1.0 + (145.0 / 48.0) * e0**2 + (73.0 / 192.0) * e0**4
    )
    rho_eff = (s_dc / (c20 * nu**2 * eccentric_factor)) ** 0.2
    a = 121.0 / 304.0
    g0 = e0 ** (12.0 / 19.0) * (1.0 + a * e0**2) ** (870.0 / 2299.0)
    lambda_grid = nu * (t_arr - t_arr[0])

    def rhs(_lambda: float, values: np.ndarray) -> np.ndarray:
        eccentricity = values[0]
        g = eccentricity ** (12.0 / 19.0) * (1.0 + a * eccentricity**2) ** (
            870.0 / 2299.0
        )
        rho = rho_eff * g0 / g
        one_minus_e2 = 1.0 - eccentricity**2
        de_dlambda = -(
            (304.0 / 15.0)
            * rho**4
            * eccentricity
            * one_minus_e2**1.5
            * (1.0 + a * eccentricity**2)
        )
        dh20_dlambda = (
            c20
            * nu
            * rho**5
            * one_minus_e2**1.5
            * (1.0 + (145.0 / 48.0) * eccentricity**2 + (73.0 / 192.0) * eccentricity**4)
        )
        return np.array([de_dlambda, dh20_dlambda])

    solution = solve_ivp(
        rhs,
        (float(lambda_grid[0]), float(lambda_grid[-1])),
        np.array([e0, 0.0]),
        t_eval=lambda_grid,
        rtol=1e-10,
        atol=(1e-12, 1e-16),
    )
    if not solution.success or len(solution.t) != len(lambda_grid):
        raise RuntimeError("eccentric effective 0PN evolution did not cover the requested grid")
    return solution.y[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-mass-msun", type=float, default=1.0e6)
    parser.add_argument("--secondary-mass-msun", type=float, default=10.0)
    parser.add_argument("--spin", type=float, default=0.8)
    parser.add_argument("--p0", type=float, default=100.0)
    parser.add_argument("--e0", type=float, default=0.0)
    parser.add_argument("--comparison-e0", type=float, default=0.8)
    parser.add_argument("--x0-inclination", type=float, default=1.0)
    parser.add_argument("--t-years", type=float, default=60000.0)
    parser.add_argument("--endpoint-factor", type=float, default=1.01)
    parser.add_argument("--n-dense", type=int, default=20000)
    parser.add_argument("--trajectory-err", type=float, default=1e-11)
    parser.add_argument("--buffer-length", type=int, default=20000)
    parser.add_argument("--frequency-source", choices=["geodesic", "phase-gradient"], default="geodesic")
    parser.add_argument("--lmax", type=int, default=10)
    parser.add_argument("--max-plot-points", type=int, default=8000)
    parser.add_argument("--output-dir", default=str(ROOT / "examples" / "output"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    config = FewEmriConfig(
        primary_mass_msun=args.primary_mass_msun,
        secondary_mass_msun=args.secondary_mass_msun,
        spin=args.spin,
        p0=args.p0,
        e0=args.e0,
        x0_inclination=args.x0_inclination,
        t_years=args.t_years,
        endpoint_factor=args.endpoint_factor,
        n_dense=args.n_dense,
        trajectory_err=args.trajectory_err,
        buffer_length=args.buffer_length,
        frequency_source=args.frequency_source,
    )
    result = compute_few_emri_memory_modes(config, lmax=args.lmax)
    comparison_result = compute_few_emri_memory_modes(
        replace(config, e0=args.comparison_e0),
        lmax=args.lmax,
    )

    t = result["t_dense_dimensionless"]
    rel_t = t - t[0]
    nu = result["nu"]
    dh20 = result["h20_dimensionless"] - result["h20_dimensionless"][0]
    dh30 = result["h30_dimensionless"] - result["h30_dimensionless"][0]
    dh20_norm = dh20 / nu
    dh30_norm = dh30 / nu
    x_0pn = _x_0pn_series(t, result["x_eff_0pn"], result["q"])
    h20_0pn = np.array([h20_lo(result["q"], x_value) for x_value in x_0pn])
    h30_0pn = np.array([h30_spin_lo(result["q"], x_value) for x_value in x_0pn])
    dh20_0pn_norm = (h20_0pn - h20_0pn[0]) / nu
    dh30_0pn_norm = (h30_0pn - h30_0pn[0]) / nu

    comparison_t = comparison_result["t_dense_dimensionless"]
    comparison_rel_t = comparison_t - comparison_t[0]
    comparison_nu = comparison_result["nu"]
    comparison_dh20_norm = (
        comparison_result["h20_dimensionless"] - comparison_result["h20_dimensionless"][0]
    ) / comparison_nu
    comparison_dh30_norm = (
        comparison_result["h30_dimensionless"] - comparison_result["h30_dimensionless"][0]
    ) / comparison_nu
    comparison_dh20_eccentric_0pn_norm = (
        _eccentric_effective_0pn_h20_delta(
            comparison_t,
            comparison_result["h20_dimensionless"],
            comparison_result["q"],
            args.comparison_e0,
        )
        / comparison_nu
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        "fastemriwaveforms_emri_h20_h30"
        f"_q{_token(result['q'])}_p0_{_token(args.p0)}_chi{_token(args.spin)}"
    )
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"
    plot_idx = _downsample_indices(len(rel_t), args.max_plot_points)
    comparison_plot_idx = _downsample_indices(len(comparison_rel_t), args.max_plot_points)
    csv_t = np.unique(
        np.concatenate(
            [
                rel_t[plot_idx],
                comparison_rel_t[comparison_plot_idx],
            ]
        )
    )
    h20_on_csv_grid = _interp_complex_with_nan(csv_t, rel_t, dh20_norm)
    h30_on_csv_grid = _interp_complex_with_nan(csv_t, rel_t, dh30_norm)
    h20_0pn_on_csv_grid = _interp_complex_with_nan(csv_t, rel_t, dh20_0pn_norm)
    h30_0pn_on_csv_grid = _interp_complex_with_nan(csv_t, rel_t, dh30_0pn_norm)
    comparison_h20_on_csv_grid = _interp_complex_with_nan(
        csv_t, comparison_rel_t, comparison_dh20_norm
    )
    comparison_h30_on_csv_grid = _interp_complex_with_nan(
        csv_t, comparison_rel_t, comparison_dh30_norm
    )
    comparison_h20_eccentric_0pn_on_csv_grid = _interp_complex_with_nan(
        csv_t,
        comparison_rel_t,
        comparison_dh20_eccentric_0pn_norm,
    )

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "t_minus_t0_M",
                "FastEMRIWaveforms_delta_h20_real_over_nu",
                "FastEMRIWaveforms_delta_h20_imag_over_nu",
                "effective_0PN_delta_h20_real_over_nu",
                "effective_0PN_delta_h20_imag_over_nu",
                "FastEMRIWaveforms_delta_h30_real_over_nu",
                "FastEMRIWaveforms_delta_h30_imag_over_nu",
                "effective_0PN_delta_h30_real_over_nu",
                "effective_0PN_delta_h30_imag_over_nu",
                f"FastEMRIWaveforms_e0_{_token(args.comparison_e0)}_delta_h20_real_over_nu",
                f"FastEMRIWaveforms_e0_{_token(args.comparison_e0)}_delta_h20_imag_over_nu",
                f"FastEMRIWaveforms_e0_{_token(args.comparison_e0)}_delta_h30_real_over_nu",
                f"FastEMRIWaveforms_e0_{_token(args.comparison_e0)}_delta_h30_imag_over_nu",
                f"eccentric_effective_0PN_e0_{_token(args.comparison_e0)}_delta_h20_real_over_nu",
                f"eccentric_effective_0PN_e0_{_token(args.comparison_e0)}_delta_h20_imag_over_nu",
            ]
        )
        for (
            time_value,
            h20_value,
            h20_0pn_value,
            h30_value,
            h30_0pn_value,
            comparison_h20_value,
            comparison_h30_value,
            comparison_h20_eccentric_0pn_value,
        ) in zip(
            csv_t,
            h20_on_csv_grid,
            h20_0pn_on_csv_grid,
            h30_on_csv_grid,
            h30_0pn_on_csv_grid,
            comparison_h20_on_csv_grid,
            comparison_h30_on_csv_grid,
            comparison_h20_eccentric_0pn_on_csv_grid,
        ):
            writer.writerow(
                [
                    time_value,
                    h20_value.real,
                    h20_value.imag,
                    h20_0pn_value.real,
                    h20_0pn_value.imag,
                    h30_value.real,
                    h30_value.imag,
                    h30_0pn_value.real,
                    h30_0pn_value.imag,
                    comparison_h20_value.real,
                    comparison_h20_value.imag,
                    comparison_h30_value.real,
                    comparison_h30_value.imag,
                    comparison_h20_eccentric_0pn_value.real,
                    comparison_h20_eccentric_0pn_value.imag,
                ]
            )

    if not args.no_plot:
        import matplotlib.pyplot as plt

        y20 = _positive_for_log(np.real(dh20_norm))
        y30 = _positive_for_log(np.imag(dh30_norm))
        y20_0pn = _positive_for_log(np.real(dh20_0pn_norm))
        y30_0pn = _positive_for_log(np.imag(dh30_0pn_norm))
        comparison_y20 = _positive_for_log(np.real(comparison_dh20_norm))
        comparison_y30 = _positive_for_log(np.imag(comparison_dh30_norm))
        comparison_y20_eccentric_0pn = _positive_for_log(
            np.real(comparison_dh20_eccentric_0pn_norm)
        )
        y20_lim = _positive_log_limits(
            y20,
            y20_0pn,
            comparison_y20,
            comparison_y20_eccentric_0pn,
        )
        y30_lim = _positive_log_limits(y30, y30_0pn, comparison_y30)
        few_label = rf"$\mathtt{{FastEMRIWaveforms}}$ ($e_0={args.e0:g}$)"
        comparison_label = rf"$\mathtt{{FastEMRIWaveforms}}$ ($e_0={args.comparison_e0:g}$)"
        effective_label = rf"effective 0PN ($e_0={args.e0:g}$)"
        comparison_effective_label = rf"effective 0PN ($e_0={args.comparison_e0:g}$)"

        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
        axes[0].plot(rel_t[plot_idx], y20[plot_idx], color="black", linewidth=1.4, label=few_label)
        axes[0].plot(
            comparison_rel_t[comparison_plot_idx],
            comparison_y20[comparison_plot_idx],
            color="#0072B2",
            linewidth=1.4,
            label=comparison_label,
        )
        axes[0].plot(
            rel_t[plot_idx],
            y20_0pn[plot_idx],
            color="red",
            linestyle="--",
            linewidth=1.3,
            label=effective_label,
        )
        axes[0].plot(
            comparison_rel_t[comparison_plot_idx],
            comparison_y20_eccentric_0pn[comparison_plot_idx],
            color="red",
            linestyle="--",
            linewidth=1.3,
            label=comparison_effective_label,
        )
        axes[0].set_yscale("log")
        axes[0].set_ylim(*y20_lim)
        axes[0].set_ylabel(r"$\mathrm{Re}\,\Delta h^{\mathrm{D}}_{2,0}/(\nu M/R)$")
        axes[0].grid(True, which="both", alpha=0.25)
        axes[0].legend(loc="best", frameon=False, ncol=2)

        axes[1].plot(rel_t[plot_idx], y30[plot_idx], color="black", linewidth=1.4, label=few_label)
        axes[1].plot(
            comparison_rel_t[comparison_plot_idx],
            comparison_y30[comparison_plot_idx],
            color="#0072B2",
            linewidth=1.4,
            label=comparison_label,
        )
        axes[1].plot(
            rel_t[plot_idx],
            y30_0pn[plot_idx],
            color="red",
            linestyle="--",
            linewidth=1.3,
            label=effective_label,
        )
        axes[1].set_yscale("log")
        axes[1].set_ylim(*y30_lim)
        axes[1].set_ylabel(r"$\mathrm{Im}\,\Delta h^{\mathrm{S}}_{3,0}/(\nu M/R)$")
        axes[1].set_xlabel(r"$t-t_0$ [$M$]")
        axes[1].grid(True, which="both", alpha=0.25)
        axes[1].legend(loc="best", frameon=False)
        fig.suptitle(
            rf"$\mathtt{{FastEMRIWaveforms}}$, "
            rf"$q={result['q']:.6g}$, $x_0={result['x_orb0']:.6g}$, "
            rf"$x_{{\rm eff}}={result['x_eff_0pn']:.6g}$, "
            rf"$\chi={args.spin:.6g}$"
        )
        fig.savefig(png_path, dpi=180)
        plt.close(fig)

    print("FastEMRIWaveforms EMRI h20/h30 memory demo")
    print(f"q = {result['q']:.12e}")
    print(f"nu = {nu:.12e}")
    print(f"primary_mass_msun = {args.primary_mass_msun:.12e}")
    print(f"secondary_mass_msun = {args.secondary_mass_msun:.12e}")
    print(f"spin = {args.spin:.12e}")
    print(f"p0 = {args.p0:.12e}")
    print(f"trajectory samples = {result['trajectory_sample_count']}")
    print(f"dense samples = {len(t)}")
    print(f"time range = [{t[0]:.3f}, {t[-1]:.3f}] M")
    print(f"x0 = {result['x_orb0']:.12e}")
    print(f"x_eff = {result['x_eff_0pn']:.12e}")
    print(f"initial Re(dot h20) = {result['initial_dh20_dt_dimensionless']:.12e}")
    print(f"final FastEMRIWaveforms Delta h20 = {_format_complex(dh20[-1])}")
    print(f"final FastEMRIWaveforms Delta h30 = {_format_complex(dh30[-1])}")
    print(f"final FastEMRIWaveforms Delta h20 / nu = {_format_complex(dh20_norm[-1])}")
    print(f"final FastEMRIWaveforms Delta h30 / nu = {_format_complex(dh30_norm[-1])}")
    print(f"comparison e0 = {args.comparison_e0:.12e}")
    print(f"comparison trajectory samples = {comparison_result['trajectory_sample_count']}")
    print(f"comparison time range = [{comparison_t[0]:.3f}, {comparison_t[-1]:.3f}] M")
    print(
        "comparison final FastEMRIWaveforms Delta h20 / nu = "
        f"{_format_complex(comparison_dh20_norm[-1])}"
    )
    print(
        "comparison final FastEMRIWaveforms Delta h30 / nu = "
        f"{_format_complex(comparison_dh30_norm[-1])}"
    )
    print(f"Saved CSV: {csv_path}")
    if not args.no_plot:
        print(f"Saved plot: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

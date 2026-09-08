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

from memorie import (
    FewEmriConfig,
    compute_few_emri_memory_modes,
    h20_lo,
)


REFERENCE_PATH = ROOT / "examples" / "data" / "arxiv_2407_19017_fig3_h20.csv"
X_MIN = -1.0e6
X_MAX = 2.5e4


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    step = max(1, int(np.ceil(n / max_points)))
    indices = np.arange(0, n, step, dtype=int)
    if indices[-1] != n - 1:
        indices = np.append(indices, n - 1)
    return indices


def _load_reference_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return (
        np.asarray(data["nu_t_over_M"], dtype=float),
        np.asarray(data["R_h20_over_nu_M"], dtype=float),
    )


def _token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _latex_number(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / 10.0**exponent
    if np.isclose(coefficient, 1.0):
        return rf"10^{{{exponent}}}"
    return rf"{coefficient:g}\times10^{{{exponent}}}"


def _x_0pn_series(t_over_m: np.ndarray, x0: float, q: float) -> np.ndarray:
    nu = float(q) / (1.0 + float(q)) ** 2
    denominator = float(x0) ** -4 - (256.0 / 5.0) * nu * (t_over_m - t_over_m[0])
    x = np.full_like(denominator, np.nan, dtype=float)
    valid = denominator > 0.0
    x[valid] = denominator[valid] ** -0.25
    return x


def _oscillatory_mode_segments(
    x: np.ndarray,
    mode: np.ndarray,
    phase: np.ndarray,
    x_min: float,
    x_max: float,
    n_segments: int = 2400,
) -> np.ndarray:
    left = max(float(x_min), float(x[0]))
    right = min(float(x_max), float(x[-1]))
    edges = np.linspace(left, right, n_segments + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    amplitude = np.interp(centers, x, np.abs(mode))
    phase_edges = np.interp(edges, x, phase)
    real_edges = np.interp(edges, x, np.real(mode))
    segments = np.empty((n_segments, 2, 2), dtype=float)
    for index, (x_left, x_right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if abs(phase_edges[index + 1] - phase_edges[index]) >= 2.0 * np.pi:
            segments[index] = ((centers[index], -amplitude[index]), (centers[index], amplitude[index]))
        else:
            segments[index] = ((x_left, real_edges[index]), (x_right, real_edges[index + 1]))
    return segments


def _plot(
    *,
    png_path: Path,
    mode_png_path: Path,
    q: float,
    spin: float,
    nu_t_over_m: np.ndarray,
    r_h20_over_nu_m: np.ndarray,
    effective_0pn_over_nu_m: np.ndarray,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    h2_minus2_nu_t_over_m: np.ndarray,
    h2_minus2_over_nu_m: np.ndarray,
    h2_minus2_phase: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    mask = (nu_t_over_m >= X_MIN) & (nu_t_over_m <= X_MAX)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(
        nu_t_over_m[mask],
        r_h20_over_nu_m[mask],
        color="black",
        linewidth=1.8,
        label=r"$\mathtt{FastEMRIWaveforms}$",
    )
    ax.plot(
        nu_t_over_m[mask],
        effective_0pn_over_nu_m[mask],
        color="red",
        linestyle="--",
        linewidth=1.6,
        label="effective 0PN",
    )
    ax.plot(
        reference_x,
        reference_y,
        color="#0072B2",
        linestyle="--",
        linewidth=1.6,
        label=r"arXiv:2407.19017",
    )
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0.0, 0.11)
    ax.set_xlabel(r"$(\nu/M)t$")
    ax.set_ylabel(r"$R h^{\mathrm{D}}_{2,0}/(\nu M)$")
    ax.set_title(rf"$q={_latex_number(q)}$, $\chi={spin:g}$, $e_0=0$")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6), useMathText=True)
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    h2_minus2_segments = _oscillatory_mode_segments(
        h2_minus2_nu_t_over_m,
        h2_minus2_over_nu_m,
        h2_minus2_phase,
        X_MIN,
        X_MAX,
    )
    mode_figure, mode_axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    mode_axis.add_collection(
        LineCollection(
            h2_minus2_segments,
            colors="#0072B2",
            alpha=0.35,
            linewidths=0.65,
            label=r"$(2,-2)$",
            zorder=1,
        )
    )
    mode_axis.plot(
        nu_t_over_m[mask],
        r_h20_over_nu_m[mask],
        color="black",
        linewidth=1.5,
        label=r"$(2,0)$ null displacement memory",
        zorder=2,
    )
    mode_amplitude = float(np.max(np.abs(h2_minus2_segments[:, :, 1])))
    mode_axis.set_xlim(X_MIN, X_MAX)
    mode_axis.set_ylim(-1.1 * mode_amplitude, 1.1 * mode_amplitude)
    mode_axis.set_xlabel(r"$(\nu/M)t$")
    mode_axis.set_ylabel(r"$\operatorname{Re} h_{l,m}/(\nu M/R)$")
    mode_axis.ticklabel_format(axis="x", style="sci", scilimits=(6, 6), useMathText=True)
    mode_axis.grid(alpha=0.22, linewidth=0.6)
    mode_axis.legend(loc="upper left", frameon=False)
    mode_figure.savefig(mode_png_path, dpi=220)
    plt.close(mode_figure)


def _load_output_data(
    csv_path: Path,
    mode_csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    mode_data = np.genfromtxt(mode_csv_path, delimiter=",", names=True)
    return (
        np.asarray(data["nu_t_over_M"], dtype=float),
        np.asarray(data["FastEMRIWaveforms_R_h20_over_nu_M"], dtype=float),
        np.asarray(data["effective_0PN_R_h20_over_nu_M"], dtype=float),
        np.asarray(data["arXiv_2407_19017_Fig3_R_h20_over_nu_M"], dtype=float),
        np.asarray(mode_data["nu_t_over_M"], dtype=float),
        np.asarray(mode_data["FastEMRIWaveforms_Re_h2_minus2_over_nu_M"], dtype=float)
        + 1j * np.asarray(mode_data["FastEMRIWaveforms_Im_h2_minus2_over_nu_M"], dtype=float),
        np.asarray(mode_data["minus_two_phi_phi"], dtype=float),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-mass-msun", type=float, default=1.0e6)
    parser.add_argument("--secondary-mass-msun", type=float, default=10.0)
    parser.add_argument("--spin", type=float, default=0.0)
    parser.add_argument("--p0", type=float, default=100.0)
    parser.add_argument("--t-years", type=float, default=50000.0)
    parser.add_argument("--endpoint-factor", type=float, default=1.01)
    parser.add_argument("--n-dense", type=int, default=200000)
    parser.add_argument("--trajectory-err", type=float, default=1e-11)
    parser.add_argument("--buffer-length", type=int, default=20000)
    parser.add_argument("--lmax", type=int, default=10)
    parser.add_argument("--max-output-points", type=int, default=8000)
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples" / "output")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-from-csv", action="store_true")
    args = parser.parse_args()

    q = float(args.primary_mass_msun / args.secondary_mass_msun)
    stem = (
        "fastemriwaveforms_arxiv_2407_19017_h20_comparison"
        f"_q{_token(q)}_chi{_token(args.spin)}"
    )
    csv_path = args.output_dir / f"{stem}.csv"
    png_path = args.output_dir / f"{stem}.png"
    mode_csv_path = args.output_dir / f"{stem}_h2_minus2_h20.csv"
    mode_png_path = args.output_dir / f"{stem}_h2_minus2_h20.png"
    if args.plot_from_csv:
        (
            output_x,
            output_h20,
            output_0pn,
            output_reference,
            output_h2_minus2_x,
            output_h2_minus2,
            output_h2_minus2_phase,
        ) = _load_output_data(csv_path, mode_csv_path)
        _plot(
            png_path=png_path,
            mode_png_path=mode_png_path,
            q=q,
            spin=args.spin,
            nu_t_over_m=output_x,
            r_h20_over_nu_m=output_h20,
            effective_0pn_over_nu_m=output_0pn,
            reference_x=output_x,
            reference_y=output_reference,
            h2_minus2_nu_t_over_m=output_h2_minus2_x,
            h2_minus2_over_nu_m=output_h2_minus2,
            h2_minus2_phase=output_h2_minus2_phase,
        )
        print(f"wrote {png_path}")
        print(f"wrote {mode_png_path}")
        return 0

    config = FewEmriConfig(
        primary_mass_msun=args.primary_mass_msun,
        secondary_mass_msun=args.secondary_mass_msun,
        spin=args.spin,
        p0=args.p0,
        e0=0.0,
        x0_inclination=1.0,
        t_years=args.t_years,
        endpoint_factor=args.endpoint_factor,
        n_dense=args.n_dense,
        trajectory_err=args.trajectory_err,
        buffer_length=args.buffer_length,
        frequency_source="geodesic",
    )
    result = compute_few_emri_memory_modes(config, lmax=args.lmax)

    nu = float(result["nu"])
    t_over_m = np.asarray(result["t_dense_dimensionless"], dtype=float)
    h20 = np.asarray(result["h20_dimensionless"], dtype=complex)
    h20_total = h20 + complex(result["prehistory_0pn_dimensionless"])
    nu_t_over_m = nu * (t_over_m - t_over_m[-1])
    r_h20_over_nu_m = np.real(h20_total / nu)
    h2_minus2_t_over_m = np.asarray(result["t_trajectory_dimensionless"], dtype=float)
    h2_minus2_nu_t_over_m = nu * (h2_minus2_t_over_m - t_over_m[-1])
    h2_minus2_over_nu_m = np.asarray(result["h2_minus2_dimensionless"], dtype=complex) / nu
    h2_minus2_phase = -2.0 * np.asarray(result["phi_phi_trajectory"], dtype=float)
    x_0pn = _x_0pn_series(t_over_m, result["x_eff_0pn"], result["q"])
    effective_0pn_over_nu_m = np.asarray(
        [h20_lo(result["q"], x_value) for x_value in x_0pn],
        dtype=float,
    ) / nu

    order = np.argsort(nu_t_over_m)
    nu_t_over_m = nu_t_over_m[order]
    r_h20_over_nu_m = r_h20_over_nu_m[order]
    effective_0pn_over_nu_m = effective_0pn_over_nu_m[order]
    h2_minus2_order = np.argsort(h2_minus2_nu_t_over_m)
    h2_minus2_nu_t_over_m = h2_minus2_nu_t_over_m[h2_minus2_order]
    h2_minus2_over_nu_m = h2_minus2_over_nu_m[h2_minus2_order]
    h2_minus2_phase = h2_minus2_phase[h2_minus2_order]
    reference_x, reference_y = _load_reference_curve(args.reference)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_csv_path = output_dir / f"{stem}_h2_minus2_h20.csv"

    output_indices = _downsample_indices(len(nu_t_over_m), args.max_output_points)
    output_x = nu_t_over_m[output_indices]
    reference_on_output_grid = np.interp(
        output_x,
        reference_x,
        reference_y,
        left=np.nan,
        right=np.nan,
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "nu_t_over_M",
                "FastEMRIWaveforms_R_h20_over_nu_M",
                "effective_0PN_R_h20_over_nu_M",
                "arXiv_2407_19017_Fig3_R_h20_over_nu_M",
            ]
        )
        writer.writerows(
            zip(
                output_x,
                r_h20_over_nu_m[output_indices],
                effective_0pn_over_nu_m[output_indices],
                reference_on_output_grid,
                strict=True,
            )
        )
    h20_on_h2_minus2_grid = np.interp(
        h2_minus2_nu_t_over_m,
        nu_t_over_m,
        r_h20_over_nu_m,
    )
    with mode_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "nu_t_over_M",
                "FastEMRIWaveforms_Re_h2_minus2_over_nu_M",
                "FastEMRIWaveforms_Im_h2_minus2_over_nu_M",
                "minus_two_phi_phi",
                "FastEMRIWaveforms_R_h20_over_nu_M",
            ]
        )
        writer.writerows(
            zip(
                h2_minus2_nu_t_over_m,
                np.real(h2_minus2_over_nu_m),
                np.imag(h2_minus2_over_nu_m),
                h2_minus2_phase,
                h20_on_h2_minus2_grid,
                strict=True,
            )
        )

    relative_difference_end = (r_h20_over_nu_m[-1] - reference_y[-1]) / reference_y[-1]

    if not args.no_plot:
        _plot(
            png_path=png_path,
            mode_png_path=mode_png_path,
            q=result["q"],
            spin=args.spin,
            nu_t_over_m=nu_t_over_m,
            r_h20_over_nu_m=r_h20_over_nu_m,
            effective_0pn_over_nu_m=effective_0pn_over_nu_m,
            reference_x=reference_x,
            reference_y=reference_y,
            h2_minus2_nu_t_over_m=h2_minus2_nu_t_over_m,
            h2_minus2_over_nu_m=h2_minus2_over_nu_m,
            h2_minus2_phase=h2_minus2_phase,
        )

    print(f"model={result['model']}")
    print(f"q={result['q']:.12g}, nu={nu:.12g}, spin={args.spin:g}, e0=0")
    print(f"x_eff_0pn={result['x_eff_0pn']:.12g}")
    print(f"prehistory_0pn_over_nu={np.real(result['prehistory_0pn_dimensionless'] / nu):.12g}")
    print(f"effective_0PN_end={effective_0pn_over_nu_m[-1]:.12g}")
    print(f"R_h20_over_nu_M_end={r_h20_over_nu_m[-1]:.12g}")
    print(f"reference_end={reference_y[-1]:.12g}")
    print(f"relative_difference_end={relative_difference_end:+.6%}")
    print(f"wrote {csv_path}")
    print(f"wrote {mode_csv_path}")
    if not args.no_plot:
        print(f"wrote {png_path}")
        print(f"wrote {mode_png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

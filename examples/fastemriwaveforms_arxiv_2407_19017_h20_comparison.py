"""Compare FastEMRIWaveforms h20 memory with arXiv:2407.19017, Fig. 3.

The bundled reference CSV was digitized from the vector source of the main
panel in https://arxiv.org/abs/2407.19017.
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

from vacuum_memory_modes import FewEmriConfig, compute_few_emri_memory_modes  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FastEMRIWaveforms perturbative h20 memory with the main panel "
            "of arXiv:2407.19017, Fig. 3."
        )
    )
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
    args = parser.parse_args()

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

    order = np.argsort(nu_t_over_m)
    nu_t_over_m = nu_t_over_m[order]
    r_h20_over_nu_m = r_h20_over_nu_m[order]
    reference_x, reference_y = _load_reference_curve(args.reference)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        "fastemriwaveforms_arxiv_2407_19017_h20_comparison"
        f"_q{_token(result['q'])}_chi{_token(args.spin)}"
    )
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"

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
                "arXiv_2407_19017_Fig3_R_h20_over_nu_M",
            ]
        )
        writer.writerows(
            zip(
                output_x,
                r_h20_over_nu_m[output_indices],
                reference_on_output_grid,
                strict=True,
            )
        )

    relative_difference_end = (r_h20_over_nu_m[-1] - reference_y[-1]) / reference_y[-1]

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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
            reference_x,
            reference_y,
            color="red",
            linestyle="--",
            linewidth=1.6,
            label=r"arXiv:2407.19017",
        )
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(0.0, 0.11)
        ax.set_xlabel(r"$(\nu/M)t$")
        ax.set_ylabel(r"$R h_{2,0}/(\nu M)$")
        ax.set_title(rf"$q={_latex_number(result['q'])}$, $\chi={args.spin:g}$, $e_0=0$")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6), useMathText=True)
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.legend(loc="upper left", frameon=False)
        fig.savefig(png_path, dpi=220)
        plt.close(fig)

    print(f"model={result['model']}")
    print(f"q={result['q']:.12g}, nu={nu:.12g}, spin={args.spin:g}, e0=0")
    print(f"x_eff_0pn={result['x_eff_0pn']:.12g}")
    print(f"prehistory_0pn_over_nu={np.real(result['prehistory_0pn_dimensionless'] / nu):.12g}")
    print(f"R_h20_over_nu_M_end={r_h20_over_nu_m[-1]:.12g}")
    print(f"reference_end={reference_y[-1]:.12g}")
    print(f"relative_difference_end={relative_difference_end:+.6%}")
    print(f"wrote {csv_path}")
    if not args.no_plot:
        print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

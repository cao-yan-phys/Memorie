"""Compare numerical and analytic memory from the fundamental Kerr 220 QNM."""

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

from memorie import (  # noqa: E402
    KerrQNMExcitation,
    KerrRingdownConfig,
    analytic_single_exponential_memory_modes,
    compute_kerr_ringdown_memory_modes,
)


def _token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _max_relative_error(numerical: np.ndarray, analytic: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(analytic))), np.finfo(float).tiny)
    return float(np.max(np.abs(numerical - analytic)) / scale)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate h20 and h30 memory from the fundamental Kerr 220 QNM."
    )
    parser.add_argument("--final-spin", type=float, default=0.7)
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--start-time-M", type=float, default=0.0)
    parser.add_argument("--end-time-M", type=float, default=160.0)
    parser.add_argument("--n-samples", type=int, default=4097)
    parser.add_argument("--lmax", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples" / "output")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    config = KerrRingdownConfig(
        final_spin=args.final_spin,
        start_time_M=args.start_time_M,
        end_time_M=args.end_time_M,
        n_samples=args.n_samples,
    )
    excitation = KerrQNMExcitation(
        spheroidal_ell=2,
        m=2,
        overtone=0,
        amplitude=complex(args.amplitude),
    )
    targets = ((2, 0), (3, 0))
    result = compute_kerr_ringdown_memory_modes(
        config,
        (excitation,),
        targets=targets,
        lmax=args.lmax,
        include_cm=False,
    )
    component = result["qnm_components"][0]
    analytic = analytic_single_exponential_memory_modes(
        result["t_dimensionless"],
        component["spherical_amplitudes_dimensionless"],
        component["omega_dimensionless"],
        targets=targets,
        lmax=args.lmax,
    )

    time = np.asarray(result["t_dimensionless"], dtype=float)
    elapsed = time - time[0]
    numerical_h20 = np.asarray(result["memory"][(2, 0)]["h_displacement"], dtype=complex)
    analytic_h20 = np.asarray(analytic[(2, 0)]["h_displacement"], dtype=complex)
    numerical_h30 = np.asarray(result["memory"][(3, 0)]["h_spin_mode"], dtype=complex)
    analytic_h30 = np.asarray(analytic[(3, 0)]["h_spin_mode"], dtype=complex)
    delta_h30_numerical = numerical_h30 - numerical_h30[0]
    delta_h30_analytic = analytic_h30 - analytic_h30[0]
    h20_error = _max_relative_error(numerical_h20, analytic_h20)
    h30_error = _max_relative_error(delta_h30_numerical, delta_h30_analytic)
    if h20_error > 1.0e-5 or h30_error > 1.0e-12:
        raise RuntimeError(
            f"analytic comparison failed: h20={h20_error:.3e}, h30={h30_error:.3e}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"qnm_kerr_ringdown_220_h20_h30_chi{_token(args.final_spin)}"
    csv_path = args.output_dir / f"{stem}.csv"
    png_path = args.output_dir / f"{stem}.png"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "t_minus_t0_M",
                "qnm_delta_h20_real_over_Mf_over_R",
                "analytic_delta_h20_real_over_Mf_over_R",
                "qnm_delta_h30_imag_over_Mf_over_R",
                "analytic_delta_h30_imag_over_Mf_over_R",
            ]
        )
        writer.writerows(
            zip(
                elapsed,
                np.real(numerical_h20),
                np.real(analytic_h20),
                np.imag(delta_h30_numerical),
                np.imag(delta_h30_analytic),
                strict=True,
            )
        )

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_slice = slice(1, None)
        fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.6), sharex=True, constrained_layout=True)
        axes[0].plot(
            elapsed[plot_slice],
            np.real(numerical_h20[plot_slice]),
            color="black",
            linewidth=1.5,
            label=r"$\mathtt{qnm}$",
        )
        axes[0].plot(
            elapsed[plot_slice],
            np.real(analytic_h20[plot_slice]),
            color="red",
            linestyle="--",
            linewidth=1.3,
            label="analytic",
        )
        axes[0].set_ylabel(r"$\mathrm{Re}\,\Delta h_{2,0}/(M_f/R)$")
        axes[0].grid(alpha=0.22, linewidth=0.6)
        axes[0].legend(frameon=False)

        axes[1].plot(
            elapsed[plot_slice],
            np.imag(delta_h30_numerical[plot_slice]),
            color="black",
            linewidth=1.5,
            label=r"$\mathtt{qnm}$",
        )
        axes[1].plot(
            elapsed[plot_slice],
            np.imag(delta_h30_analytic[plot_slice]),
            color="red",
            linestyle="--",
            linewidth=1.3,
            label="analytic",
        )
        axes[1].set_xlabel(r"$t-t_0$ [$M_f$]")
        axes[1].set_ylabel(r"$\mathrm{Im}\,\Delta h_{3,0}/(M_f/R)$")
        axes[1].grid(alpha=0.22, linewidth=0.6)
        axes[1].legend(frameon=False)
        fig.suptitle(
            rf"$(\ell_{{\rm s}},m,n)=(2,2,0)$, "
            rf"$\chi_f={args.final_spin:g}$, $A_{{220}}={args.amplitude:g}$"
        )
        fig.savefig(png_path, dpi=220)
        plt.close(fig)

    omega = complex(component["omega_dimensionless"])
    print(f"model={result['model']} {result['qnm_version']}")
    print(f"omega_220_M={omega.real:.12g}{omega.imag:+.12g}j")
    print(f"retained_spherical_modes={sorted(component['spherical_amplitudes_dimensionless'])}")
    print(f"max_relative_error_h20={h20_error:.3e}")
    print(f"max_relative_error_h30={h30_error:.3e}")
    print(f"wrote {csv_path}")
    if not args.no_plot:
        print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

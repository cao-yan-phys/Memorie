"""Compare numerical and analytic memory from a physical Kerr 220 QNM pair."""

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
    complete_nonprecessing_modes,
    compute_memory_modes,
    generate_kerr_ringdown_modes,
)


def _token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _max_relative_error(numerical: np.ndarray, analytic: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(analytic))), np.finfo(float).tiny)
    return float(np.max(np.abs(numerical - analytic)) / scale)


def _delta(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    return values - values[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate h20 and h30 memory from the fundamental Kerr 220 QNM."
    )
    parser.add_argument("--spin", type=float, default=0.7)
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--start-time-M", type=float, default=0.0)
    parser.add_argument("--end-time-M", type=float, default=160.0)
    parser.add_argument("--n-samples", type=int, default=4097)
    parser.add_argument("--lmax", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples" / "output")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.amplitude == 0.0:
        parser.error("--amplitude must be nonzero")
    amplitude_abs_squared = abs(complex(args.amplitude)) ** 2

    config = KerrRingdownConfig(
        spin=args.spin,
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
    result = generate_kerr_ringdown_modes(
        config,
        (excitation,),
        lmax=args.lmax,
    )
    # A physical, reflection-symmetric ringdown contains the mirror of the
    # explicitly supplied m=2 QNM.  Complete both h_lm and dot{h}_lm before
    # forming the memory source.
    oscillatory_modes = complete_nonprecessing_modes(result["oscillatory_modes_dimensionless"])
    oscillatory_hdot = complete_nonprecessing_modes(result["oscillatory_hdot_dimensionless"])
    memory = compute_memory_modes(
        result["t_dimensionless"],
        oscillatory_modes,
        targets=targets,
        lmax=args.lmax,
        hdot=oscillatory_hdot,
        include_cm=False,
    )
    component = result["qnm_components"][0]
    positive_m_analytic = analytic_single_exponential_memory_modes(
        result["t_dimensionless"],
        component["spherical_amplitudes_dimensionless"],
        component["omega_dimensionless"],
        targets=targets,
        lmax=args.lmax,
    )

    time = np.asarray(result["t_dimensionless"], dtype=float)
    elapsed = time - time[0]
    delta_components = {
        (target, component_name): _delta(memory[target][component_name])
        for target in targets
        for component_name in ("h_displacement", "h_spin_mode")
    }
    delta_hD20 = delta_components[((2, 0), "h_displacement")]
    delta_hS20 = delta_components[((2, 0), "h_spin_mode")]
    delta_hD30 = delta_components[((3, 0), "h_displacement")]
    delta_hS30 = delta_components[((3, 0), "h_spin_mode")]

    # The analytic helper starts from the explicitly supplied m=2 QNM.  The
    # reflection partner makes an equal contribution to the parity-selected
    # h^D_20 and h^S_30 components.
    analytic_hD20 = 2.0 * _delta(positive_m_analytic[(2, 0)]["h_displacement"])
    analytic_hS30 = 2.0 * _delta(positive_m_analytic[(3, 0)]["h_spin_mode"])
    h20_error = _max_relative_error(delta_hD20, analytic_hD20)
    h30_error = _max_relative_error(delta_hS30, analytic_hS30)
    cross_scale = max(float(np.max(np.abs(delta_hD20))), float(np.max(np.abs(delta_hS30))))
    cross_terms = (delta_hS20, delta_hD30)
    if h20_error > 1.0e-5 or h30_error > 1.0e-12 or any(
        float(np.max(np.abs(values))) > 1.0e-12 * cross_scale for values in cross_terms
    ):
        raise RuntimeError(
            "reflection/analytic comparison failed: "
            f"hD20={h20_error:.3e}, hS30={h30_error:.3e}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"qnm_kerr_ringdown_220_h20_h30_chi{_token(args.spin)}"
    csv_path = args.output_dir / f"{stem}.csv"
    png_path = args.output_dir / f"{stem}.png"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "t_minus_t0_M",
                "qnm_delta_hD20_real_over_abs_A220_squared_M_over_R",
                "qnm_delta_hD20_imag_over_abs_A220_squared_M_over_R",
                "qnm_delta_hS20_real_over_abs_A220_squared_M_over_R",
                "qnm_delta_hS20_imag_over_abs_A220_squared_M_over_R",
                "qnm_delta_hD30_real_over_abs_A220_squared_M_over_R",
                "qnm_delta_hD30_imag_over_abs_A220_squared_M_over_R",
                "qnm_delta_hS30_real_over_abs_A220_squared_M_over_R",
                "qnm_delta_hS30_imag_over_abs_A220_squared_M_over_R",
                "analytic_delta_hD20_real_over_abs_A220_squared_M_over_R",
                "analytic_delta_hD20_imag_over_abs_A220_squared_M_over_R",
                "analytic_delta_hS30_real_over_abs_A220_squared_M_over_R",
                "analytic_delta_hS30_imag_over_abs_A220_squared_M_over_R",
            ]
        )
        writer.writerows(
            zip(
                elapsed,
                delta_hD20.real / amplitude_abs_squared,
                delta_hD20.imag / amplitude_abs_squared,
                delta_hS20.real / amplitude_abs_squared,
                delta_hS20.imag / amplitude_abs_squared,
                delta_hD30.real / amplitude_abs_squared,
                delta_hD30.imag / amplitude_abs_squared,
                delta_hS30.real / amplitude_abs_squared,
                delta_hS30.imag / amplitude_abs_squared,
                analytic_hD20.real / amplitude_abs_squared,
                analytic_hD20.imag / amplitude_abs_squared,
                analytic_hS30.real / amplitude_abs_squared,
                analytic_hS30.imag / amplitude_abs_squared,
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
            np.real(delta_hD20[plot_slice]) / amplitude_abs_squared,
            color="black",
            linewidth=1.5,
            label=r"$\mathtt{qnm}$",
        )
        axes[0].plot(
            elapsed[plot_slice],
            np.real(analytic_hD20[plot_slice]) / amplitude_abs_squared,
            color="red",
            linestyle="--",
            linewidth=1.3,
            label="analytic",
        )
        axes[0].set_ylabel(r"$\mathrm{Re}\,\Delta h^{\mathrm{D}}_{2,0}/(|A_{220}|^2 M/R)$")
        axes[0].grid(alpha=0.22, linewidth=0.6)
        axes[0].legend(frameon=False)

        axes[1].plot(
            elapsed[plot_slice],
            np.imag(delta_hS30[plot_slice]) / amplitude_abs_squared,
            color="black",
            linewidth=1.5,
            label=r"$\mathtt{qnm}$",
        )
        axes[1].plot(
            elapsed[plot_slice],
            np.imag(analytic_hS30[plot_slice]) / amplitude_abs_squared,
            color="red",
            linestyle="--",
            linewidth=1.3,
            label="analytic",
        )
        axes[1].set_xlabel(r"$t-t_0$ [$M$]")
        axes[1].set_ylabel(r"$\mathrm{Im}\,\Delta h^{\mathrm{S}}_{3,0}/(|A_{220}|^2 M/R)$")
        axes[1].grid(alpha=0.22, linewidth=0.6)
        axes[1].legend(frameon=False)
        fig.suptitle(
            rf"$(l_{{\mathrm{{s}}}},m,n)=(2,2,0)$, "
            rf"$\chi={args.spin:g}$"
        )
        fig.savefig(png_path, dpi=220)
        plt.close(fig)

    omega = complex(component["omega_dimensionless"])
    print(f"model={result['model']} {result['qnm_version']}")
    print(f"omega_220_M={omega.real:.12g}{omega.imag:+.12g}j")
    print(f"retained_spherical_modes={sorted(component['spherical_amplitudes_dimensionless'])}")
    print(f"max_relative_error_hD20={h20_error:.3e}")
    print(f"max_relative_error_hS30={h30_error:.3e}")
    for target, short_name in (((2, 0), "20"), ((3, 0), "30")):
        for component_name, symbol in (
            ("h_displacement", "D"),
            ("h_spin_mode", "S"),
        ):
            value = delta_components[(target, component_name)][-1] / amplitude_abs_squared
            print(
                f"delta_h{symbol}{short_name}_final_over_abs_A220_squared="
                f"{value.real:+.12e}{value.imag:+.12e}j"
            )
    print(f"wrote {csv_path}")
    if not args.no_plot:
        print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

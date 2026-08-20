"""Compute nonlinear-null memory from the E=1.01 axial Kerr-plunge waveform."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memorie import compute_memory_modes  # noqa: E402


def _token(value: float) -> str:
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def _latex_scientific(value: float) -> str:
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / 10.0**exponent
    if np.isclose(coefficient, 1.0):
        return rf"10^{{{exponent}}}"
    return rf"{coefficient:g}\times10^{{{exponent}}}"


def _read_total_axisymmetric_modes(path: Path) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """Read the total ``h=h_regular+h_memory`` time-domain modes from CSV."""

    required_columns = {
        "spherical_l",
        "m",
        "time",
        "re_h",
        "im_h",
    }
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required_columns <= set(reader.fieldnames):
            missing = sorted(required_columns - set(reader.fieldnames or ()))
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in reader:
            mode = (int(row["spherical_l"]), int(row["m"]))
            grouped[mode].append(row)

    if not grouped:
        raise ValueError(f"{path} contains no modes")
    if any(emm != 0 for _ell, emm in grouped):
        raise ValueError("the axial-plunge example requires only m=0 input modes")

    first_mode = min(grouped)
    time = np.asarray([float(row["time"]) for row in grouped[first_mode]], dtype=float)
    if len(time) < 3 or np.any(np.diff(time) <= 0.0):
        raise ValueError("the input time grid must be strictly increasing with at least three samples")

    modes: dict[tuple[int, int], np.ndarray] = {}
    for mode, rows in sorted(grouped.items()):
        mode_time = np.asarray([float(row["time"]) for row in rows], dtype=float)
        if not np.allclose(mode_time, time, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"time grid for mode {mode} differs from the first mode")
        modes[mode] = np.asarray(
            [float(row["re_h"]) + 1j * float(row["im_h"]) for row in rows],
            dtype=complex,
        )
    return time, modes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute displacement and spin memory components for an axial Kerr plunge."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--nu", type=float, default=1.0e-4)
    parser.add_argument("--lmax", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples" / "output")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.nu < 1.0:
        raise ValueError("nu must lie between zero and one")
    if not args.input_csv.is_file():
        raise FileNotFoundError(f"input CSV not found: {args.input_csv}")

    time, all_linear_modes = _read_total_axisymmetric_modes(args.input_csv)
    available_lmax = max(ell for ell, _emm in all_linear_modes)
    if not 2 <= args.lmax <= available_lmax:
        raise ValueError(f"lmax must lie between 2 and {available_lmax}")
    linear_modes = {
        mode: waveform
        for mode, waveform in all_linear_modes.items()
        if mode[0] <= args.lmax
    }
    if (2, 0) not in linear_modes or (3, 0) not in linear_modes:
        raise ValueError("the input CSV must include (2, 0) and (3, 0) modes")

    # The CSV is normalized by nu M/R.  The quadratic results are normalized
    # by nu**2 M/R and need one factor of nu before they are combined with the
    # linear waveform in the plotted normalization.
    memory = compute_memory_modes(
        time,
        linear_modes,
        targets=((2, 0), (3, 0)),
        lmax=args.lmax,
        include_cm=False,
    )
    delta_components = {
        (target, component_name): args.nu
        * (
            np.asarray(memory[target][component_name], dtype=complex)
            - memory[target][component_name][0]
        )
        for target in ((2, 0), (3, 0))
        for component_name in ("h_displacement", "h_spin_mode")
    }
    nonlinear_h20_D = delta_components[((2, 0), "h_displacement")]
    nonlinear_h20_S = delta_components[((2, 0), "h_spin_mode")]
    nonlinear_h30_D = delta_components[((3, 0), "h_displacement")]
    nonlinear_h30_S = delta_components[((3, 0), "h_spin_mode")]
    nonlinear_h20 = nonlinear_h20_D + nonlinear_h20_S
    nonlinear_h30 = nonlinear_h30_D + nonlinear_h30_S

    linear_h20 = linear_modes[(2, 0)]
    linear_h30 = linear_modes[(3, 0)]
    total_h20 = linear_h20 + nonlinear_h20
    total_h30 = linear_h30 + nonlinear_h30
    delta_linear_h20 = linear_h20 - linear_h20[0]
    delta_linear_h30 = linear_h30 - linear_h30[0]
    delta_nonlinear_h20 = nonlinear_h20
    delta_nonlinear_h30 = nonlinear_h30

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"kerr_axial_plunge_e1p01_h20_h30_nu{_token(args.nu)}"
    csv_path = args.output_dir / f"{stem}.csv"
    png_path = args.output_dir / f"{stem}.png"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "t_over_M",
                "linear_h20_real_over_nu_M_over_R",
                "linear_h20_imag_over_nu_M_over_R",
                "delta_hD20_real_over_nu_M_over_R",
                "delta_hD20_imag_over_nu_M_over_R",
                "delta_hS20_real_over_nu_M_over_R",
                "delta_hS20_imag_over_nu_M_over_R",
                "nonlinear_null_total_h20_real_over_nu_M_over_R",
                "nonlinear_null_total_h20_imag_over_nu_M_over_R",
                "total_h20_real_over_nu_M_over_R",
                "total_h20_imag_over_nu_M_over_R",
                "linear_h30_real_over_nu_M_over_R",
                "linear_h30_imag_over_nu_M_over_R",
                "delta_hD30_real_over_nu_M_over_R",
                "delta_hD30_imag_over_nu_M_over_R",
                "delta_hS30_real_over_nu_M_over_R",
                "delta_hS30_imag_over_nu_M_over_R",
                "nonlinear_null_total_h30_real_over_nu_M_over_R",
                "nonlinear_null_total_h30_imag_over_nu_M_over_R",
                "total_h30_real_over_nu_M_over_R",
                "total_h30_imag_over_nu_M_over_R",
            ]
        )
        writer.writerows(
            zip(
                time,
                linear_h20.real,
                linear_h20.imag,
                nonlinear_h20_D.real,
                nonlinear_h20_D.imag,
                nonlinear_h20_S.real,
                nonlinear_h20_S.imag,
                nonlinear_h20.real,
                nonlinear_h20.imag,
                total_h20.real,
                total_h20.imag,
                linear_h30.real,
                linear_h30.imag,
                nonlinear_h30_D.real,
                nonlinear_h30_D.imag,
                nonlinear_h30_S.real,
                nonlinear_h30_S.imag,
                nonlinear_h30.real,
                nonlinear_h30.imag,
                total_h30.real,
                total_h30.imag,
                strict=True,
            )
        )

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.1), sharex=True, constrained_layout=True)
        panels = (
            (axes[0, 0], delta_nonlinear_h20.real, nonlinear_h20_D.real, delta_linear_h20.real, r"$\mathrm{Re}\,\Delta h_{2,0}/(\nu M/R)$"),
            (axes[0, 1], delta_nonlinear_h20.imag, nonlinear_h20_D.imag, delta_linear_h20.imag, r"$\mathrm{Im}\,\Delta h_{2,0}/(\nu M/R)$"),
            (axes[1, 0], delta_nonlinear_h30.real, nonlinear_h30_D.real, delta_linear_h30.real, r"$\mathrm{Re}\,\Delta h_{3,0}/(\nu M/R)$"),
            (axes[1, 1], delta_nonlinear_h30.imag, nonlinear_h30_D.imag, delta_linear_h30.imag, r"$\mathrm{Im}\,\Delta h_{3,0}/(\nu M/R)$"),
        )
        for axis, memory_mode, displacement_mode, linear, ylabel in panels:
            axis.plot(time, linear, color="blue", linewidth=1.1, label="linear")
            axis.plot(
                time,
                1.0e5 * memory_mode,
                color="black",
                linewidth=1.25,
                label=r"$h^{\mathrm{D}}+h^{\mathrm{S}}$ ($\times10^5$)",
            )
            axis.plot(
                time,
                1.0e5 * displacement_mode,
                color="red",
                linestyle="--",
                linewidth=1.25,
                label=r"$h^{\mathrm{D}}$ ($\times10^5$)",
            )
            axis.set_ylabel(ylabel)
            axis.set_xlim(-500.0, 500.0)
            axis.grid(alpha=0.22, linewidth=0.6)
        axes[0, 0].legend(frameon=False, loc="best")
        axes[1, 0].set_xlabel(r"$t/M$")
        axes[1, 1].set_xlabel(r"$t/M$")
        figure.suptitle(
            rf"$\chi=0.999$, $E=1.01$, $\nu={_latex_scientific(args.nu)}$"
        )
        figure.savefig(png_path, dpi=220)
        plt.close(figure)

    print(f"input={args.input_csv}")
    print(f"nu={args.nu:.12g}, lmax={args.lmax}, samples={len(time)}")
    for target, short_name in (((2, 0), "20"), ((3, 0), "30")):
        for component_name, symbol in (
            ("h_displacement", "D"),
            ("h_spin_mode", "S"),
        ):
            value = delta_components[(target, component_name)][-1]
            print(f"delta_h{symbol}{short_name}_over_nu={value.real:+.12e}{value.imag:+.12e}j")
    print(f"wrote {csv_path}")
    if not args.no_plot:
        print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compute h20, h30, and CM memory from a circular nonprecessing SEOBNRv5EHM event."""

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
DEFAULT_OMEGA_START = DEFAULT_X_START**1.5

from vacuum_memory_modes import (  # noqa: E402
    cm_strain_lo_modes,
    complete_nonprecessing_modes,
    compute_memory_modes,
    cumulative_integral,
    delta_mass_fraction,
    differentiate_modes,
    h30_spin_lo,
    infer_x_eff_from_dh20,
    k20_lo,
    phase_from_h22_lo,
    symmetric_mass_ratio,
)


def _relative_error(value: complex, reference: complex) -> float:
    reference_abs = abs(reference)
    return float(abs(value - reference) / reference_abs) if reference_abs else np.nan


def _format_complex(value: complex) -> str:
    value = complex(value)
    return f"{value.real:+.6e}{value.imag:+.6e}j"


def _x_0pn_series(t: np.ndarray, x0: float, q: float) -> np.ndarray:
    """Evolve ``x`` with ``dx/dt = (64 nu / 5) x^5`` from the first sample."""

    nu = symmetric_mass_ratio(q)
    denominator = float(x0) ** -4 - (256.0 / 5.0) * nu * (t - t[0])
    if np.any(denominator <= 0.0):
        raise ValueError("0PN x evolution reached its formal coalescence before the plot end")
    return denominator ** -0.25


def _plot_indices(t: np.ndarray, duration: float | None) -> np.ndarray:
    if duration is None or duration <= 0.0:
        end = len(t)
    else:
        end = int(np.searchsorted(t, t[0] + duration, side="right"))
        end = max(2, min(end, len(t)))
    return np.arange(0, end, dtype=int)


def _cycle_max_envelope(
    t: np.ndarray,
    orbital_phase: np.ndarray,
    values: np.ndarray,
    emm: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one maximum of ``values`` per azimuthal-mode cycle."""

    t = np.asarray(t, dtype=float)
    phase = np.unwrap(np.asarray(orbital_phase, dtype=float))
    values = np.asarray(values, dtype=float)
    if not (len(t) == len(phase) == len(values)):
        raise ValueError("t, orbital_phase, and values must have equal lengths")
    if not len(t):
        return t, values

    phase_progress = phase - phase[0]
    if phase_progress[-1] < 0.0:
        phase_progress = -phase_progress
    phase_progress = np.maximum.accumulate(phase_progress)
    cycle = np.floor(abs(int(emm)) * phase_progress / (2.0 * np.pi)).astype(np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(cycle)) + 1]
    stops = np.r_[starts[1:], len(t)]

    envelope_t: list[float] = []
    envelope_y: list[float] = []
    complete_starts = starts[:-1] if len(starts) > 1 else starts
    complete_stops = stops[:-1] if len(stops) > 1 else stops
    for start, stop in zip(complete_starts, complete_stops, strict=True):
        segment = values[start:stop]
        finite = np.flatnonzero(np.isfinite(segment))
        if not len(finite):
            continue
        envelope_t.append(float(t[stop - 1]))
        envelope_y.append(float(segment[finite[np.argmax(segment[finite])]]))
    return np.asarray(envelope_t), np.asarray(envelope_y)


def _h31_0pn_from_h22(
    t: np.ndarray,
    h22: np.ndarray,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the missing leading-PN ``h_31`` from the model's ``h_22`` phase."""

    phase = -0.5 * np.unwrap(np.angle(-np.asarray(h22)))
    omega = np.gradient(phase, t, edge_order=2)
    x = np.maximum(omega, np.finfo(float).tiny) ** (2.0 / 3.0)
    nu = symmetric_mass_ratio(q)
    dm = delta_mass_fraction(q)
    h31 = (
        -(2j / (3.0 * np.sqrt(2.0)))
        * np.sqrt(np.pi / 35.0)
        * dm
        * nu
        * x**1.5
        * np.exp(-1j * phase)
    )
    return h31, np.gradient(h31, t, edge_order=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, default=2.0)
    parser.add_argument("--omega-start", type=float, default=DEFAULT_OMEGA_START)
    parser.add_argument("--lmax", type=int, default=10)
    parser.add_argument("--delta-t", type=float, default=20.0, help="output spacing in units of M")
    parser.add_argument("--total-mass-solar", type=float, default=50.0)
    parser.add_argument("--output-dir", default=str(ROOT / "examples" / "output"))
    parser.add_argument("--plot-duration", type=float, default=1_250_000.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    try:
        from pyseobnr.generate_waveform import generate_modes_opt
    except ImportError as exc:
        raise SystemExit("This example requires pyseobnr.") from exc

    t, raw_modes = generate_modes_opt(
        args.q,
        0.0,
        0.0,
        args.omega_start,
        eccentricity=0.0,
        approximant="SEOBNRv5EHM",
        settings={
            "EccIC": 0,
            "M": args.total_mass_solar,
            "dt": args.delta_t * args.total_mass_solar * MTSUN_SI,
            "lmax_nyquist": 1,
        },
    )
    pyseobnr_positive_modes = {tuple(map(int, key.split(","))): value for key, value in raw_modes.items()}
    oscillatory_modes = complete_nonprecessing_modes(pyseobnr_positive_modes)
    oscillatory_hdot = differentiate_modes(t, oscillatory_modes)

    cm_targets = [(3, 1), (3, 3), (5, 1), (5, 3), (5, 5), (7, 1), (7, 3)]
    targets = [(2, 0), (3, 0), (4, 0), *cm_targets]
    primary = compute_memory_modes(t, oscillatory_modes, targets, lmax=args.lmax, hdot=oscillatory_hdot)

    h20 = primary[(2, 0)]["h_displacement"]
    dh20_dt = primary[(2, 0)]["dh_displacement_dt"]
    x_eff = infer_x_eff_from_dh20(dh20_dt[0], args.q)
    h40 = primary[(4, 0)]["h_displacement"]
    dh40_dt = primary[(4, 0)]["dh_displacement_dt"]

    modes_with_h20 = dict(oscillatory_modes)
    modes_with_h20[(2, 0)] = h20
    hdot_with_h20 = dict(oscillatory_hdot)
    hdot_with_h20[(2, 0)] = dh20_dt
    h31_0pn, dh31_0pn_dt = _h31_0pn_from_h22(
        t,
        oscillatory_modes[(2, 2)],
        args.q,
    )
    supplemented_modes = dict(modes_with_h20)
    supplemented_modes[(3, 1)] = h31_0pn
    supplemented_modes[(3, -1)] = -np.conjugate(h31_0pn)
    supplemented_modes[(4, 0)] = h40
    supplemented_hdot = dict(hdot_with_h20)
    supplemented_hdot[(3, 1)] = dh31_0pn_dt
    supplemented_hdot[(3, -1)] = -np.conjugate(dh31_0pn_dt)
    supplemented_hdot[(4, 0)] = dh40_dt
    with_supplemented_modes = compute_memory_modes(
        t,
        supplemented_modes,
        cm_targets,
        lmax=args.lmax,
        hdot=supplemented_hdot,
    )

    lo_input_modes = {
        (2, -2),
        (2, 0),
        (2, 2),
        (3, -3),
        (3, -1),
        (3, 1),
        (3, 3),
        (4, 0),
    }
    truncated_modes = {
        mode: value for mode, value in supplemented_modes.items() if mode in lo_input_modes
    }
    truncated_hdot = {
        mode: value for mode, value in supplemented_hdot.items() if mode in lo_input_modes
    }
    with_truncated_modes = compute_memory_modes(
        t,
        truncated_modes,
        cm_targets,
        lmax=args.lmax,
        hdot=truncated_hdot,
    )

    phase0 = phase_from_h22_lo(oscillatory_modes[(2, 2)][0])
    cm_lo_full = cm_strain_lo_modes(args.q, x_eff, phase0)

    h30_num = primary[(3, 0)]["h_spin_mode"][0]
    h30_lo = h30_spin_lo(args.q, x_eff)

    print("SEOBNRv5EHM circular nonprecessing memory demo")
    print(f"q = {args.q:g}, omega_start = {args.omega_start:g}")
    print(f"samples = {len(t)}, time range = [{t[0]:.3f}, {t[-1]:.3f}] M")
    print(f"pyseobnr positive-m modes = {sorted(pyseobnr_positive_modes)}")
    print()
    print("Effective 0PN initial x from dot h20")
    print(f"Re(dot h20)_0 = {np.real(dh20_dt[0]):.12e}")
    print(f"x_eff = {x_eff:.12e}")
    print(f"h20_numeric(t0) = {h20[0].real:.12e}")
    print(f"h40_numeric(t0) = {h40[0].real:.12e}")
    print()
    print("Initial spin-memory h30")
    print("mode        numeric                 LO PN                 rel.err")
    print(
        f"h30   {_format_complex(h30_num):>24}  {_format_complex(h30_lo):>24}"
        f"  {_relative_error(h30_num, h30_lo):.3e}"
    )
    print()
    print("Initial CM strain modes")
    print("mode        full + 0PN (3,1)        truncated + 0PN         corrected LO PN")
    rows = []
    for target in cm_targets:
        supplemented = with_supplemented_modes[target]["h_cm_mode"][0]
        truncated = with_truncated_modes[target]["h_cm_mode"][0]
        full = cm_lo_full[target]
        rows.append(
            {
                "mode": str(target),
                "supplemented": supplemented,
                "truncated": truncated,
                "lo_full": full,
            }
        )
        print(
            f"{target!s:<6} {_format_complex(supplemented):>24}"
            f"  {_format_complex(truncated):>24}  {_format_complex(full):>24}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"seobnrv5ehm_circular_memory_q{args.q:g}_omega{args.omega_start:g}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "quantity",
                "supplemented_real",
                "supplemented_imag",
                "truncated_real",
                "truncated_imag",
                "lo_full_real",
                "lo_full_imag",
                "relerr_supplemented",
                "relerr_truncated",
            ]
        )
        writer.writerow(
            [
                "h30",
                h30_num.real,
                h30_num.imag,
                h30_num.real,
                h30_num.imag,
                h30_lo.real,
                h30_lo.imag,
                _relative_error(h30_num, h30_lo),
                _relative_error(h30_num, h30_lo),
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["mode"],
                    row["supplemented"].real,
                    row["supplemented"].imag,
                    row["truncated"].real,
                    row["truncated"].imag,
                    row["lo_full"].real,
                    row["lo_full"].imag,
                    _relative_error(row["supplemented"], row["lo_full"]),
                    _relative_error(row["truncated"], row["lo_full"]),
                ]
            )
    print(f"Saved comparison table: {csv_path}")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        x_0pn = _x_0pn_series(t, x_eff, args.q)
        phase_0pn = phase0 + cumulative_integral(t, x_0pn**1.5)
        indices = _plot_indices(t, args.plot_duration)
        t_plot = t[indices] - t[0]
        x_plot = x_0pn[indices]
        phase_plot = phase_0pn[indices]

        h20_0pn = k20_lo(args.q) * x_plot
        h30_0pn = np.array([h30_spin_lo(args.q, x_value) for x_value in x_plot])
        cm_0pn_coefficients = cm_strain_lo_modes(args.q, 1.0, 0.0)
        cm_0pn = {
            target: (
                cm_0pn_coefficients[target]
                * x_plot**4
                * np.exp(-1j * target[1] * phase_plot)
            )
            for target in cm_targets
        }
        nu = symmetric_mass_ratio(args.q)
        numeric_orbital_phase = -0.5 * np.unwrap(
            np.angle(oscillatory_modes[(2, 2)][indices])
        )

        plot_specs = [
            (
                r"Re $\Delta h_{2,0}/(\nu M/R)$",
                t_plot,
                np.real(h20[indices] - h20[0]) / nu,
                t_plot,
                np.real(h20_0pn - h20_0pn[0]) / nu,
            ),
            (
                r"Im $\Delta h_{3,0}/(\nu M/R)$",
                t_plot,
                np.imag(primary[(3, 0)]["h_spin_mode"][indices] - primary[(3, 0)]["h_spin_mode"][0]) / nu,
                t_plot,
                np.imag(h30_0pn - h30_0pn[0]) / nu,
            ),
        ]
        for target in cm_targets:
            supplemented = np.abs(
                with_supplemented_modes[target]["h_cm_mode"][indices]
                - with_supplemented_modes[target]["h_cm_mode"][0]
            ) / nu
            truncated = np.abs(
                with_truncated_modes[target]["h_cm_mode"][indices]
                - with_truncated_modes[target]["h_cm_mode"][0]
            ) / nu
            effective_0pn = np.abs(cm_0pn[target] - cm_0pn[target][0]) / nu
            supplemented_t, supplemented_envelope = _cycle_max_envelope(
                t_plot,
                numeric_orbital_phase,
                supplemented,
                target[1],
            )
            truncated_t, truncated_envelope = _cycle_max_envelope(
                t_plot,
                numeric_orbital_phase,
                truncated,
                target[1],
            )
            effective_t, effective_envelope = _cycle_max_envelope(
                t_plot,
                phase_plot,
                effective_0pn,
                target[1],
            )
            plot_specs.append(
                (
                    rf"$|\Delta h_{{{target[0]},{target[1]}}}^{{\rm CM}}|_{{\rm envelope}}/(\nu M/R)$",
                    supplemented_t,
                    supplemented_envelope,
                    truncated_t,
                    truncated_envelope,
                    effective_t,
                    effective_envelope,
                )
            )

        fig, axes = plt.subplots(5, 2, figsize=(11, 12.5), sharex=True, constrained_layout=True)
        flat_axes = axes.ravel()
        plot_pairs = zip(flat_axes, plot_specs, strict=False)
        for (
            ax,
            spec,
        ) in plot_pairs:
            label, numeric_t, numeric, *remaining = spec
            if len(remaining) == 2:
                supplemented_t = supplemented = None
                effective_t, effective_0pn = remaining
            else:
                supplemented_t, supplemented, effective_t, effective_0pn = remaining
            ax.plot(
                numeric_t,
                numeric,
                color="black",
                linewidth=1.4,
                label="SEOBNRv5EHM + 0PN (3,1)",
            )
            if supplemented is not None:
                ax.plot(
                    supplemented_t,
                    supplemented,
                    color="tab:blue",
                    linestyle="-.",
                    linewidth=1.3,
                    label="SEOBNRv5EHM (truncated) + 0PN (3,1)",
                )
            ax.plot(
                effective_t,
                effective_0pn,
                color="red",
                linestyle="--",
                linewidth=1.3,
                label="effective 0PN",
            )
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
        for ax in flat_axes[len(plot_specs) :]:
            ax.set_visible(False)
        for ax in flat_axes[-2:]:
            if ax.get_visible():
                ax.set_xlabel(r"$t-t_0$ [$M$]")
        flat_axes[0].legend(loc="best", frameon=False)
        flat_axes[2].legend(loc="best", frameon=False)
        fig.suptitle(f"SEOBNRv5EHM memory-mode waveform check, q={args.q:g}")
        png_path = output_dir / f"seobnrv5ehm_circular_memory_q{args.q:g}_omega{args.omega_start:g}.png"
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        print(f"Saved comparison plot: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

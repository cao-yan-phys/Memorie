"""Generate vector-cloud displacement and spin-memory modes with SuperRad."""

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
    SuperRadConfig,
    compute_superrad_memory_modes,
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
    return rf"{coefficient:.3g}\times10^{{{exponent}}}"


def _endpoint_matched_quadrupolar_model(
    result: dict[str, object],
    h20: np.ndarray,
) -> np.ndarray:
    """Return the endpoint-matched saturated-cloud quadrupolar h20 model."""

    time = np.asarray(result["t_dimensionless"], dtype=float)
    saturation_index = int(np.argmin(np.abs(np.asarray(result["t_seconds"], dtype=float))))
    post_saturation_time = time - time[saturation_index]
    mask = post_saturation_time >= 0.0
    tau_gw = float(result["gw_time_dimensionless"])
    growth = 1.0 - tau_gw / (post_saturation_time[mask] + tau_gw)
    if growth[-1] == 0.0:
        raise ValueError("the quadrupolar h20 model has a zero final growth factor")

    delta_h20 = np.real(h20 - h20[0])
    model = np.full_like(delta_h20, np.nan)
    initial_value = float(delta_h20[saturation_index])
    final_value = float(delta_h20[-1])
    model[mask] = initial_value + (final_value - initial_value) * growth / growth[-1]
    return model


def _endpoint_matched_quadrupolar_h30_model(
    result: dict[str, object],
    h30: np.ndarray,
) -> np.ndarray:
    """Return the endpoint-matched saturated-cloud quadrupolar h30 model.

    For a quadrupolar primary waveform, ``h30`` is proportional to
    ``i omega_gw A**2``.  The saturated-cloud amplitude therefore gives a
    ``[1 + (t - t_sat)/tau_gw]**-2`` decay.
    """

    time = np.asarray(result["t_dimensionless"], dtype=float)
    saturation_index = int(np.argmin(np.abs(np.asarray(result["t_seconds"], dtype=float))))
    post_saturation_time = time - time[saturation_index]
    mask = post_saturation_time >= 0.0
    tau_gw = float(result["gw_time_dimensionless"])
    decay = (tau_gw / (post_saturation_time[mask] + tau_gw)) ** 2
    growth = 1.0 - decay
    if growth[-1] == 0.0:
        raise ValueError("the quadrupolar h30 model has a zero final change factor")

    delta_h30 = np.imag(h30 - h30[0])
    model = np.full_like(delta_h30, np.nan)
    initial_value = float(delta_h30[saturation_index])
    final_value = float(delta_h30[-1])
    model[mask] = initial_value + (final_value - initial_value) * growth / growth[-1]
    return model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SuperRad vector-cloud h20 and h30 memory waveforms."
    )
    parser.add_argument("--black-hole-mass-msun", type=float, default=20.8)
    parser.add_argument("--black-hole-spin", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--growth-start-fraction", type=float, default=1.0)
    parser.add_argument("--end-gw-time", type=float, default=10.0)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--lmax", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples" / "output")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    config = SuperRadConfig(
        black_hole_mass_msun=args.black_hole_mass_msun,
        black_hole_spin=args.black_hole_spin,
        alpha=args.alpha,
        boson_spin=1,
        cloud_model="relativistic",
        evolution="full",
        growth_start_fraction=args.growth_start_fraction,
        end_gw_time=args.end_gw_time,
        n_samples=args.n_samples,
    )
    result = compute_superrad_memory_modes(config, targets=((2, 0), (3, 0)), lmax=args.lmax)

    time = np.asarray(result["t_dimensionless"], dtype=float)
    elapsed = time - time[0]
    saturation_index = int(np.argmin(np.abs(np.asarray(result["t_seconds"], dtype=float))))
    efold_time = float(result["efold_time_seconds"]) / float(result["mass_scale_seconds"])
    h20 = np.asarray(result["h20_dimensionless"], dtype=complex)
    h30 = np.asarray(result["h30_dimensionless"], dtype=complex)
    delta_h20 = h20 - h20[0]
    delta_h30 = h30 - h30[0]
    h20_reference = _endpoint_matched_quadrupolar_model(result, h20)
    h30_reference = _endpoint_matched_quadrupolar_h30_model(result, h30)
    beta0 = float(result["cloud_mass_msun"][0]) / float(args.black_hole_mass_msun)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        "superrad_vector_1011_h20_h30"
        f"_alpha{_token(args.alpha)}_chi{_token(args.black_hole_spin)}"
    )
    csv_path = args.output_dir / f"{stem}.csv"
    png_path = args.output_dir / f"{stem}.png"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "t_minus_t0_M",
                "SuperRad_delta_h20_real_over_M_over_R",
                "quadrupolar_model_delta_h20_over_M_over_R",
                "SuperRad_delta_h30_imag_over_M_over_R",
                "quadrupolar_model_delta_h30_imag_over_M_over_R",
            ]
        )
        writer.writerows(
            zip(
                elapsed,
                np.real(delta_h20),
                h20_reference,
                np.imag(delta_h30),
                h30_reference,
                strict=True,
            )
        )

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset

        plot_slice = slice(1, None)
        superrad_label = r"$\mathtt{SuperRad}$"
        reference_label = "quadrupolar model"
        fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.5), sharex=True, constrained_layout=True)
        axes[0].plot(
            elapsed[plot_slice],
            np.real(delta_h20[plot_slice]),
            color="black",
            linewidth=1.5,
            label=superrad_label,
        )
        axes[0].plot(
            elapsed[plot_slice],
            h20_reference[plot_slice],
            color="red",
            linestyle="--",
            linewidth=1.4,
            label=reference_label,
        )
        axes[0].set_ylabel(r"$\mathrm{Re}\,\Delta h_{2,0}/(M/R)$")
        axes[0].grid(alpha=0.22, linewidth=0.6)
        axes[0].legend(loc="upper left", frameon=False)

        axes[1].plot(
            elapsed[plot_slice],
            np.imag(delta_h30[plot_slice]),
            color="black",
            linewidth=1.5,
            label=superrad_label,
        )
        axes[1].plot(
            elapsed[plot_slice],
            h30_reference[plot_slice],
            color="red",
            linestyle="--",
            linewidth=1.4,
            label=reference_label,
        )
        axes[1].set_xlabel(r"$t-t_0$ [$M$]")
        axes[1].set_ylabel(r"$\mathrm{Im}\,\Delta h_{3,0}/(M/R)$")
        axes[1].grid(alpha=0.22, linewidth=0.6)
        axes[1].legend(loc="center", bbox_to_anchor=(0.40, 0.78), frameon=False)

        saturation_elapsed = elapsed[saturation_index]
        inset_half_width = 8.0 * efold_time
        inset_mask = np.abs(elapsed - saturation_elapsed) <= inset_half_width
        if np.count_nonzero(inset_mask) < 2:
            raise RuntimeError("the time grid does not resolve the saturation transition")

        def add_transition_inset(
            axis: object,
            numerical: np.ndarray,
            reference: np.ndarray | None,
            bounds: list[float],
            corners: tuple[int, int],
            zero_floor: bool = False,
        ) -> None:
            inset = axis.inset_axes(bounds)
            inset.plot(
                elapsed[inset_mask],
                numerical[inset_mask],
                color="black",
                linewidth=1.2,
            )
            if reference is not None:
                inset.plot(
                    elapsed[inset_mask],
                    reference[inset_mask],
                    color="red",
                    linestyle="--",
                    linewidth=1.1,
                )
            inset.set_xlim(
                saturation_elapsed - inset_half_width,
                saturation_elapsed + inset_half_width,
            )
            series = [numerical[inset_mask]]
            if reference is not None:
                series.append(reference[inset_mask])
            values = np.concatenate(series)
            values = values[np.isfinite(values)]
            span = float(np.max(values) - np.min(values))
            if zero_floor:
                inset.set_ylim(0.0, 1.03 * float(np.max(values)))
            else:
                padding = 0.12 * span if span > 0.0 else max(abs(float(values[0])) * 0.1, 1e-16)
                inset.set_ylim(float(np.min(values)) - padding, float(np.max(values)) + padding)
            inset.tick_params(labelsize=7)
            inset.grid(alpha=0.22, linewidth=0.5)
            mark_inset(
                axis,
                inset,
                loc1=corners[0],
                loc2=corners[1],
                fc="none",
                ec="0.35",
                linewidth=0.75,
            )

        add_transition_inset(
            axes[0],
            np.real(delta_h20),
            h20_reference,
            [0.57, 0.12, 0.38, 0.30],
            (2, 3),
        )
        add_transition_inset(
            axes[1],
            np.imag(delta_h30),
            h30_reference,
            [0.57, 0.56, 0.38, 0.35],
            (2, 3),
            zero_floor=True,
        )
        fig.suptitle(
            rf"Vector $|1011\rangle$, $\alpha={args.alpha:g}$, "
            rf"$\chi_{{\rm BH}}(t_0)={args.black_hole_spin:g}$, "
            rf"$\beta(t_0)=M_{{\rm c}}(t_0)/M={_latex_number(beta0)}$"
        )
        fig.savefig(png_path, dpi=220)
        plt.close(fig)

    print(f"model={result['model']} {result['superrad_version']}")
    print(f"waveform_class={result['waveform_class']}")
    print(f"alpha={args.alpha:.12g}, chi={args.black_hole_spin:.12g}")
    print(f"time_range_M=[{time[0]:.12g}, {time[-1]:.12g}]")
    print(f"gw_time_M={result['gw_time_dimensionless']:.12g}")
    print(f"final Re(Delta h20)/(M/R)={np.real(delta_h20[-1]):.12e}")
    print(f"final quadrupolar-model h20/(M/R)={h20_reference[-1]:.12e}")
    print(f"final Im(Delta h30)/(M/R)={np.imag(delta_h30[-1]):.12e}")
    print(f"max_power_relative_error={result['max_power_relative_error']:.6%}")
    print(f"wrote {csv_path}")
    if not args.no_plot:
        print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import ScaledTranslation

from memorie import (
    NullParticle,
    TimelikeParticle,
    evaluate_supertranslation,
    memory_to_supertranslation,
)


ROOT = Path(__file__).resolve().parents[1]


def _states(q, momentum):
    if not np.isfinite(q) or q <= 0 or not np.isfinite(momentum) or momentum <= 0:
        raise ValueError("q and p_initial must be finite and positive")
    masses = np.array([q, 1.0]) / (1.0 + q)
    energies = np.hypot(masses, momentum)
    energy = float(energies.sum())
    photon_energy = (energy**2 - 1) / (2 * energy)
    final_speed = photon_energy / (energy - photon_energy)
    initial = tuple(
        TimelikeParticle(float(mass), float(momentum / initial_energy), theta, 0.0)
        for mass, initial_energy, theta in zip(masses, energies, (0.0, np.pi), strict=True)
    )
    final = tuple(TimelikeParticle(float(mass), final_speed, np.pi / 2, np.pi) for mass in masses)
    photon = NullParticle(photon_energy, np.pi / 2, 0.0)
    residual = np.array([
        energy - 1 / np.sqrt(1 - final_speed**2) - photon_energy,
        final_speed / np.sqrt(1 - final_speed**2) - photon_energy,
    ])
    if np.max(np.abs(residual)) > 1e-12 * energy:
        raise ValueError("energy-momentum conservation failed")
    print(f"E_initial/M={energy:.12g}, E_null/M={photon_energy:.12g}, "
          f"v_final={final_speed:.12g}, conservation={np.max(np.abs(residual)):.3e}")
    return initial, final, photon


def _calculate(q, momentum, lmax):
    initial, final, photon = _states(q, momentum)
    ordinary = memory_to_supertranslation({}, initial_particles=initial, final_particles=final, lmax=lmax)
    null = memory_to_supertranslation({}, final_particles=(photon,), lmax=lmax)
    total = memory_to_supertranslation({}, initial_particles=initial, final_particles=(*final, photon), lmax=lmax)
    error = max(abs(total[key] - ordinary[key] - null[key]) for key in total)
    if error > 1e-12:
        raise ValueError("ordinary/null decomposition failed")
    keys = sorted(total)
    data = np.array([
        [q, momentum, ell, emm,
         ordinary[(ell, emm)].real, ordinary[(ell, emm)].imag,
         null[(ell, emm)].real, null[(ell, emm)].imag,
         total[(ell, emm)].real, total[(ell, emm)].imag]
        for ell, emm in keys
    ])
    print(f"alpha_total-alpha_ordinary-alpha_null={error:.3e}")
    return data


def _particle_markers(ax, component):
    if component in ("ordinary", "total"):
        for latitude, color in ((np.pi / 2, "red"), (-np.pi / 2, "limegreen")):
            ax.plot(
                0, latitude, marker="o", markersize=9, linestyle="none",
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.8, zorder=10, clip_on=False,
            )
        for size, color in ((13, "red"), (8, "limegreen")):
            ax.plot(
                np.pi / 2, 0, marker="o", markersize=size, linestyle="none",
                markerfacecolor="black", markeredgecolor=color,
                markeredgewidth=1.8, zorder=10, clip_on=False,
            )
    if component in ("null", "total"):
        ax.plot(
            -np.pi / 2, 0, marker="o", markersize=9, linestyle="none",
            markerfacecolor="black", markeredgecolor="blue",
            markeredgewidth=1.8, zorder=10,
        )


def _plot(data, path, plot_cache=None):
    longitude = np.linspace(-np.pi, np.pi, 362)
    latitude = np.linspace(-np.pi / 2, np.pi / 2, 182)
    lon, lat = np.meshgrid(
        0.5 * (longitude[:-1] + longitude[1:]),
        0.5 * (latitude[:-1] + latitude[1:]),
    )
    if plot_cache is not None and plot_cache.exists():
        with np.load(plot_cache, allow_pickle=False) as cache:
            if not (
                np.array_equal(cache["coefficients"], data)
                and np.array_equal(cache["longitude"], longitude)
                and np.array_equal(cache["latitude"], latitude)
                and cache["phi_center"].item() == np.pi / 2
            ):
                raise ValueError("plot cache does not match the coefficients or angular grid")
            fields = cache["fields"]
    else:
        fields = []
        for column in (4, 6, 8):
            alpha = {
                (int(row[2]), int(row[3])): complex(row[column], row[column + 1])
                for row in data
            }
            fields.append(evaluate_supertranslation(alpha, np.pi / 2 - lat, lon + np.pi / 2))
        if plot_cache is not None:
            plot_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                plot_cache, coefficients=data, longitude=longitude, latitude=latitude,
                phi_center=np.pi / 2, fields=np.asarray(fields),
            )
    bound = max(float(np.max(np.abs(field))) for field in fields)
    norm = Normalize(vmin=-bound, vmax=bound)
    plt.rcParams.update({key: plt.rcParamsDefault[key] for key in ("font.family", "mathtext.fontset")})
    fig, axes = plt.subplots(
        1, 3, figsize=(13.8, 4.2), subplot_kw={"projection": "mollweide"},
        layout="constrained",
    )
    for ax, field, title in zip(axes, fields, ("ordinary", "null", "total"), strict=True):
        image = ax.pcolormesh(longitude, latitude, field, shading="flat", cmap="RdBu_r", norm=norm, rasterized=True)
        ax.set_title(title, fontsize=12, pad=13)
        ax.set_xticks(np.deg2rad([-90, 0, 90]))
        ax.set_xticklabels([r"$0^\circ$", r"$90^\circ$", r"$180^\circ$"])
        ax.set_yticks(np.deg2rad([-60, -30, 0, 30, 60]))
        ax.tick_params(labelsize=8, pad=3)
        for label in ax.get_xticklabels():
            label.set_transform(label.get_transform() + ScaledTranslation(0, 10 / 72, fig.dpi_scale_trans))
        ax.grid(True, linewidth=0.45, color="0.45", alpha=0.5)
        ax.spines["geo"].set_linewidth(0.7)
        _particle_markers(ax, title)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    colorbar = fig.colorbar(
        image, ax=axes, orientation="horizontal", fraction=0.07,
        shrink=0.6, pad=0.07, aspect=40, format=formatter,
    )
    colorbar.set_label(r"$\alpha/M$", fontsize=12)
    colorbar.outline.set_linewidth(0.6)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=face, markeredgecolor=color, markeredgewidth=1.8,
               label=label)
        for color, label, face in (("red", "1", "white"), ("limegreen", "2", "white"), ("blue", "3 (null)", "black"))
    ]
    handles.extend(
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=face, markeredgecolor="0.35", label=label)
        for face, label in (("white", "in"), ("black", "out"))
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False,
               handletextpad=0.45, columnspacing=1.8)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--p-initial", type=float, default=3 / 8)
    parser.add_argument("--lmax", type=int, default=32)
    parser.add_argument("--plot-from-csv", action="store_true")
    parser.add_argument("--plot-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples" / "output")
    args = parser.parse_args()
    stem = f"particle_scattering_supertranslation_q{args.q:g}_p{args.p_initial:g}".replace(".", "p")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / (stem + ".csv")
    if args.plot_from_csv:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    else:
        data = _calculate(args.q, args.p_initial, args.lmax)
        with csv_path.open("w", encoding="utf-8", newline="\n") as stream:
            np.savetxt(
                stream, data, delimiter=",", comments="",
                header="q,p_initial_over_M,l,m,alpha_ordinary_real_over_M,alpha_ordinary_imag_over_M,"
                       "alpha_null_real_over_M,alpha_null_imag_over_M,alpha_total_real_over_M,alpha_total_imag_over_M",
                fmt="%.17e",
            )
    _plot(data, args.output_dir / (stem + ".png"), args.plot_cache)


if __name__ == "__main__":
    main()

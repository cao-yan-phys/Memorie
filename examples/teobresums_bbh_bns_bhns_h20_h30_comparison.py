from __future__ import annotations

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorie import complete_nonprecessing_modes, compute_memory_modes, differentiate_modes


SYSTEMS = ("BBH", "BNS", "BH-NS")
INPUT_MODES = ((2, 1), (2, 2), (3, 2), (3, 3), (4, 4))


def _eos_parameters(name, mass_msun):
    import lal
    import lalsimulation

    eos = lalsimulation.SimNeutronStarEOSByName(name)
    family = lalsimulation.CreateSimNeutronStarFamily(eos)
    mass = mass_msun * lal.MSUN_SI
    radius = lalsimulation.SimNeutronStarRadius(mass, family)
    love = lalsimulation.SimNeutronStarLoveNumberK2(mass, family)
    compactness = lal.G_SI * mass / (radius * lal.C_SI**2)
    return {
        "eos": name,
        "ns_radius_m": radius,
        "ns_love_k2": love,
        "ns_lambda2": 2 * love / (3 * compactness**5),
        "mass_time_seconds": 2 * mass_msun * lal.MTSUN_SI,
    }


def _resample_waveform(time, raw, args, system):
    time = np.asarray(time, dtype=float)
    if len(time) < 5 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise RuntimeError(f"invalid time grid for {system}")
    stop = len(time)
    limiting_modes = []
    for ell, emm in INPUT_MODES:
        index = ell * (ell - 1) // 2 + emm - 2
        amplitude, phase = map(np.asarray, raw[str(index)])
        bad = np.flatnonzero(~np.isfinite(amplitude) | ~np.isfinite(phase))
        if len(bad):
            first_bad = int(bad[0])
            if first_bad < stop:
                stop = first_bad
                limiting_modes = [(ell, emm)]
            elif first_bad == stop:
                limiting_modes.append((ell, emm))
    if stop < 5:
        raise RuntimeError(f"insufficient finite input for {system}")
    sample_count = int(np.ceil((time[stop - 1] - time[0]) / args.delta_t)) + 1
    output_time = np.linspace(time[0], time[stop - 1], sample_count)
    positive = {}
    for ell, emm in INPUT_MODES:
        index = ell * (ell - 1) // 2 + emm - 2
        amplitude, phase = raw[str(index)]
        interpolated_amplitude = CubicSpline(time[:stop], np.asarray(amplitude)[:stop])(output_time)
        interpolated_phase = CubicSpline(time[:stop], np.asarray(phase)[:stop])(output_time)
        positive[(ell, emm)] = -0.25 * interpolated_amplitude * np.exp(-1j * interpolated_phase)
    print(f"{system}: native_end_M={time[-1] - time[0]:.12g}, "
          f"finite_end_M={output_time[-1] - time[0]:.12g}, limiting_modes={limiting_modes}", flush=True)
    return {
        "time": output_time - time[0],
        "modes": positive,
        "native_time": time - time[0],
        "native_amplitude": np.column_stack([raw[str(l * (l - 1) // 2 + m - 2)][0]
                                              for l, m in INPUT_MODES]),
        "native_phase": np.column_stack([raw[str(l * (l - 1) // 2 + m - 2)][1]
                                          for l, m in INPUT_MODES]),
    }


def _generate_waveforms(args):
    import EOBRun_module

    metadata = {
        "mass_msun": args.mass,
        "f22_start_hz": args.f22_start,
        "delta_t_M": args.delta_t,
        "input_modes": [list(mode) for mode in INPUT_MODES],
        "endpoint": "last_finite_model_sample",
        "model": "Giotto",
        "teobresums_version": version("teobresums"),
        **_eos_parameters(args.eos, args.mass),
    }
    indices = [ell * (ell - 1) // 2 + emm - 2 for ell, emm in INPUT_MODES]
    tidal = metadata["ns_lambda2"]
    parameters = {
        "M": 2 * args.mass,
        "q": 1.0,
        "chi1": 0.0,
        "chi2": 0.0,
        "ecc": 0.0,
        "model": "Giotto",
        "domain": 0,
        "initial_frequency": args.f22_start * metadata["mass_time_seconds"],
        "use_geometric_units": "yes",
        "interp_uniform_grid": "no",
        "use_mode_lm": indices,
        "arg_out": "yes",
        "output_hpc": "no",
        "output_multipoles": "no",
        "output_dynamics": "no",
        "output_nqc": "no",
        "output_nqc_coefs": "no",
        "output_ringdown": "no",
    }
    waveforms = {}
    for system, lambdas in zip(SYSTEMS, ((0.0, 0.0), (tidal, tidal), (0.0, tidal)), strict=True):
        values = EOBRun_module.EOBRunPy(
            {**parameters, "LambdaAl2": lambdas[0], "LambdaBl2": lambdas[1]}
        )
        time, _, _, raw, _ = values
        waveforms[system] = _resample_waveform(time, raw, args, system)
        print(f"{system}: samples={len(waveforms[system]['time'])}, "
              f"t_end_M={waveforms[system]['time'][-1]:.9g}", flush=True)
    return metadata, waveforms


def _save_waveforms(path, metadata, waveforms):
    arrays = {"metadata": np.array(json.dumps(metadata))}
    for system in SYSTEMS:
        item = waveforms[system]
        arrays[f"{system}_time"] = item["time"]
        arrays[f"{system}_lm"] = np.array(list(item["modes"]), dtype=int)
        arrays[f"{system}_modes"] = np.column_stack(list(item["modes"].values()))
        for name in ("native_time", "native_amplitude", "native_phase"):
            arrays[f"{system}_{name}"] = item[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_waveforms(path, args):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        expected = {
            "mass_msun": args.mass,
            "eos": args.eos,
            "f22_start_hz": args.f22_start,
            "delta_t_M": args.delta_t,
            "input_modes": [list(mode) for mode in INPUT_MODES],
            "endpoint": "last_finite_model_sample",
            "model": "Giotto",
        }
        if any(metadata.get(name) != value for name, value in expected.items()):
            raise ValueError("waveform cache parameters do not match the requested parameters")
        waveforms = {}
        for system in SYSTEMS:
            modes = {
                tuple(map(int, mode)): data[f"{system}_modes"][:, index]
                for index, mode in enumerate(data[f"{system}_lm"])
            }
            waveforms[system] = {
                "time": data[f"{system}_time"],
                "modes": modes,
                **{name: data[f"{system}_{name}"]
                   for name in ("native_time", "native_amplitude", "native_phase")},
            }
    return metadata, waveforms


def _memory(time, positive_modes, stride=1):
    selected = {mode: values[::stride] for mode, values in positive_modes.items()}
    modes = complete_nonprecessing_modes(selected)
    result = compute_memory_modes(
        time[::stride], modes, ((2, 0), (3, 0)), lmax=10, include_cm=False
    )
    displacement = result[(2, 0)]["h_displacement"]
    spin = result[(3, 0)]["h_spin"]
    return 4 * (displacement - displacement[0]).real, 4 * (spin - spin[0]).imag


def _add_seobnrv5ehm(args, reference_path):
    if args.waveform_cache is None or args.seobnrv5ehm_cache is None:
        raise ValueError("both waveform cache paths are required")
    with np.load(args.waveform_cache, allow_pickle=False) as source:
        source_metadata = json.loads(str(source["metadata"]))
        expected = {"mass_msun": args.mass, "eos": args.eos, "f22_start_hz": args.f22_start}
        if any(source_metadata.get(name) != value for name, value in expected.items()):
            raise ValueError("reference waveform parameters do not match")
        index22 = list(map(tuple, source["BBH_lm"])).index((2, 2))
        source_time = source["BBH_time"][:32]
        source_h22 = source["BBH_modes"][:32, index22].copy()
    source_hdot = differentiate_modes(source_time, {(2, 2): source_h22})[(2, 2)]
    target_omega22 = abs(np.imag(source_hdot[0] / source_h22[0]))
    mass_time = source_metadata["mass_time_seconds"]
    metadata = {
        "model": "SEOBNRv5EHM", "mass_msun": args.mass, "q": 1.0,
        "chi1": 0.0, "chi2": 0.0, "eccentricity": 0.0,
        "target_omega22_M": target_omega22, "delta_t_M": args.delta_t,
        "input_modes": [list(mode) for mode in INPUT_MODES],
    }
    cache_path = args.seobnrv5ehm_cache
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            cached_metadata = json.loads(str(saved["metadata"]))
            if any(cached_metadata.get(name) != value for name, value in metadata.items()):
                raise ValueError("SEOBNRv5EHM cache parameters do not match")
            arrays = {name: saved[name] for name in saved.files if name != "metadata"}
        metadata = cached_metadata
    else:
        from pyseobnr.generate_waveform import generate_modes_opt

        omega_start = 0.99 * target_omega22 / 2
        native_time, raw = generate_modes_opt(
            1.0, 0.0, 0.0, omega_start, eccentricity=0.0,
            approximant="SEOBNRv5EHM",
            settings={"EccIC": 0, "M": 2 * args.mass,
                      "dt": args.delta_t * mass_time, "lmax_nyquist": 1,
                      "return_modes": list(INPUT_MODES)},
        )
        native_modes = {tuple(map(int, key.split(","))): np.asarray(value)
                        for key, value in raw.items()}
        if set(native_modes) != set(INPUT_MODES):
            raise RuntimeError("unexpected SEOBNRv5EHM input modes")
        arrays = {"native_time": np.asarray(native_time),
                  "native_modes": np.column_stack([native_modes[mode] for mode in INPUT_MODES])}
        metadata.update(omega_start_M=omega_start, pyseobnr_version=version("pyseobnr"))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, metadata=np.array(json.dumps(metadata)), **arrays)
    if "displacement" not in arrays:
        native_time, native_modes = arrays["native_time"], arrays["native_modes"]
        if not np.all(np.isfinite(native_modes)) or np.any(np.diff(native_time) <= 0):
            raise RuntimeError("invalid SEOBNRv5EHM waveform")
        phase = CubicSpline(native_time, np.unwrap(np.angle(native_modes[:, 1])))
        frequency = np.abs(phase.derivative()(native_time))
        crossing = np.flatnonzero(frequency >= target_omega22)
        if not len(crossing) or crossing[0] == 0:
            raise RuntimeError("initial-frequency match is not bracketed")
        index = int(crossing[0])
        match_time = brentq(lambda t: abs(phase.derivative()(t)) - target_omega22,
                            native_time[index - 1], native_time[index], xtol=1e-10)
        sample_count = int(np.ceil((native_time[-1] - match_time) / args.delta_t)) + 1
        time = np.linspace(0, native_time[-1] - match_time, sample_count)
        values = CubicSpline(native_time, native_modes, axis=0)(time + match_time)
        modes = {mode: values[:, index] for index, mode in enumerate(INPUT_MODES)}
        displacement, spin = _memory(time, modes)
        hdot22 = differentiate_modes(time[:32], {(2, 2): values[:32, 1]})[(2, 2)]
        matched_omega22 = abs(np.imag(hdot22[0] / values[0, 1]))
        residual = abs(matched_omega22 / target_omega22 - 1)
        if residual > 1e-7:
            raise RuntimeError(f"initial-frequency mismatch: {residual}")
        for stride in (2, 4):
            d_coarse, s_coarse = _memory(time, modes, stride=stride)
            errors = [float(np.max(np.abs(coarse - fine[::stride])) / np.max(np.abs(fine)))
                      for coarse, fine in ((d_coarse, displacement), (s_coarse, spin))]
            metadata[f"stride_{stride}_relative_errors"] = errors
            if max(errors) > 0.01:
                raise RuntimeError(f"SEOBNRv5EHM sampling check failed: {errors}")
        metadata.update(match_time_M=float(match_time),
                        discarded_duration_M=float(match_time - native_time[0]),
                        target_f22_hz=float(target_omega22 / (2 * np.pi * mass_time)),
                        matched_f22_hz=float(matched_omega22 / (2 * np.pi * mass_time)),
                        frequency_relative_error=float(residual))
        arrays.update(time=time, modes=values, displacement=displacement, spin=spin)
        np.savez_compressed(cache_path, metadata=np.array(json.dumps(metadata)), **arrays)
    time = arrays["time"]
    peak_time = time[np.argmax(np.abs(arrays["modes"][:, 1]))]
    output_stride = max(1, int(np.ceil(len(time) / 12000)))
    indices = np.unique(np.r_[np.arange(0, len(time), output_stride),
                              np.flatnonzero((time >= time[-1] - 100)
                                             | (np.abs(time - peak_time) <= 100))])
    with reference_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("t_minus_t0_M", "delta_hD20_real_over_nu_M_over_R",
                         "delta_hS30_imag_over_nu_M_over_R"))
        writer.writerows((time[index], arrays["displacement"][index], arrays["spin"][index])
                         for index in indices)
    print(json.dumps(metadata, sort_keys=True), flush=True)
    print(f"SEOBNRv5EHM: D20_final={arrays['displacement'][-1]:.12g}, "
          f"S30_final={arrays['spin'][-1]:.12g}, S30_peak={np.max(arrays['spin']):.12g}, "
          f"t_end_M={time[-1]:.12g}", flush=True)


def _plot(csv_path, png_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    reference_path = csv_path.with_name(f"{csv_path.stem}_seobnrv5ehm.csv")
    reference = np.genfromtxt(reference_path, delimiter=",", names=True) if reference_path.exists() else None
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True, constrained_layout=True)
    styles = (("black", "-"), ("#175cd3", "-"), ("#d92d20", "--"))
    columns = ("delta_hD20_real_over_nu_M_over_R", "delta_hS30_imag_over_nu_M_over_R")
    for system, (color, linestyle) in zip(SYSTEMS, styles, strict=True):
        selected = data[data["system"] == system]
        for axis, column in zip(axes, columns, strict=True):
            axis.plot(selected["t_minus_t0_M"][1:], selected[column][1:], color=color,
                      linestyle=linestyle, linewidth=1.5, label=system)
    axes[0].set_ylabel(r"$\mathrm{Re}\,\Delta h^{\mathrm{D}}_{2,0}/(\nu M/R)$")
    axes[1].set_ylabel(r"$\mathrm{Im}\,\Delta h^{\mathrm{S}}_{3,0}/(\nu M/R)$")
    axes[1].set_xlabel(r"$(t-t_0)/M$")
    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.6)
        axis.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4), useMathText=True)
        axis.legend(frameon=False)
        axis.margins(x=0.02, y=0.08)
    endpoints = [data["t_minus_t0_M"][data["system"] == system][-1] for system in SYSTEMS]
    left, right = min(endpoints) - 180, max(endpoints) + 12
    if reference is not None:
        right = max(right, reference["t_minus_t0_M"][-1] + 12)
    for axis, column in zip(axes, columns, strict=True):
        inset = axis.inset_axes([0.29, 0.48, 0.41, 0.45])
        inset_values = []
        for system, (color, linestyle) in zip(SYSTEMS, styles, strict=True):
            selected = data[(data["system"] == system) & (data["t_minus_t0_M"] >= left)]
            inset.plot(selected["t_minus_t0_M"], selected[column], color=color,
                       linestyle=linestyle, linewidth=1.2)
            inset_values.extend(selected[column])
        if reference is not None:
            selected = reference[(reference["t_minus_t0_M"] >= left)
                                 & (reference["t_minus_t0_M"] <= right)]
            inset.plot(selected["t_minus_t0_M"], selected[column], color="0.5",
                       linestyle="--", linewidth=1.2, label="SEOBNRv5EHM")
            inset_values.extend(selected[column])
            inset.legend(frameon=False, fontsize=7, loc="upper left")
        inset.set_xlim(left, right)
        lower, upper = min(inset_values), max(inset_values)
        padding = 0.06 * (upper - lower)
        inset.set_ylim(lower - padding, upper + padding)
        inset.grid(alpha=0.2, linewidth=0.5)
        inset.tick_params(labelsize=8)
        inset.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4),
                               useMathText=True, useOffset=False)
        mark_inset(axis, inset, loc1=1, loc2=4, fc="none", ec="0.45", linewidth=0.65)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mass", type=float, default=1.4)
    parser.add_argument("--eos", default="SLY")
    parser.add_argument("--f22-start", type=float, default=200.0)
    parser.add_argument("--delta-t", type=float, default=0.03125)
    parser.add_argument("--waveform-cache", type=Path)
    parser.add_argument("--seobnrv5ehm-cache", type=Path)
    parser.add_argument("--add-seobnrv5ehm", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples" / "output")
    parser.add_argument("--plot-from-csv", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.mass <= 0 or args.f22_start <= 0 or args.delta_t <= 0:
        parser.error("invalid mass, frequency, or time step")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mass_token = f"{args.mass:g}".replace(".", "p")
    frequency_token = f"{args.f22_start:g}".replace(".", "p")
    stem = (f"teobresums_bbh_bns_bhns_h20_h30_m{mass_token}_m{mass_token}_"
            f"{args.eos.lower()}_f{frequency_token}")
    csv_path = args.output_dir / f"{stem}.csv"
    png_path = args.output_dir / f"{stem}.png"
    if args.add_seobnrv5ehm:
        if not csv_path.exists():
            parser.error("the TEOBResumS reference CSV must already exist")
        reference_path = csv_path.with_name(f"{stem}_seobnrv5ehm.csv")
        _add_seobnrv5ehm(args, reference_path)
        _plot(csv_path, png_path)
        return
    if args.plot_from_csv:
        _plot(csv_path, png_path)
        return
    if args.waveform_cache is not None and args.waveform_cache.exists():
        metadata, waveforms = _load_waveforms(args.waveform_cache, args)
    else:
        metadata, waveforms = _generate_waveforms(args)
        if args.waveform_cache is not None:
            _save_waveforms(args.waveform_cache, metadata, waveforms)
    print(json.dumps(metadata, sort_keys=True), flush=True)
    results = {}
    for system in SYSTEMS:
        item = waveforms[system]
        time, positive = item["time"], item["modes"]
        displacement, spin = _memory(time, positive)
        hdot = differentiate_modes(time, {(2, 2): positive[(2, 2)]})[(2, 2)]
        omega0 = abs(np.imag(hdot[0] / positive[(2, 2)][0])) / 2
        f_end = abs(np.imag(hdot[-1] / positive[(2, 2)][-1])) / (2 * np.pi * metadata["mass_time_seconds"])
        ratio = abs(positive[(2, 2)][0]) / (2 * np.sqrt(np.pi / 5) * omega0**(2 / 3))
        if not 0.5 < ratio < 1.5:
            raise RuntimeError(f"unexpected (2,2) normalization for {system}: {ratio}")
        print(f"{system}: f22_0_Hz={omega0 / (np.pi * metadata['mass_time_seconds']):.9g}, "
              f"h22_over_Newtonian={ratio:.9g}, D20_final={displacement[-1]:.12g}, "
              f"S30_final={spin[-1]:.12g}, S30_peak={np.max(spin):.12g}", flush=True)
        if args.validate:
            for stride in (2, 4):
                d_coarse, s_coarse = _memory(time, positive, stride=stride)
                errors = [np.max(np.abs(coarse - fine[::stride])) / np.max(np.abs(fine))
                          for coarse, fine in ((d_coarse, displacement), (s_coarse, spin))]
                print(f"{system}: stride={stride}, D20_error={errors[0]:.6g}, "
                      f"S30_error={errors[1]:.6g}", flush=True)
                if max(errors) > 0.01:
                    raise RuntimeError(f"time-step convergence failed for {system}")
        results[system] = displacement, spin, f_end
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("system", "t_minus_t0_M", "delta_hD20_real_over_nu_M_over_R",
                         "delta_hS30_imag_over_nu_M_over_R", "component_mass_msun", "eos", "ns_lambda2",
                         "f22_start_hz", "f22_end_hz"))
        for system in SYSTEMS:
            item = waveforms[system]
            time = item["time"]
            displacement, spin, f_end = results[system]
            output_stride = max(1, int(np.ceil(len(time) / 12000)))
            peak_time = time[np.argmax(np.abs(item["modes"][(2, 2)]))]
            output_indices = np.unique(np.r_[np.arange(0, len(time), output_stride),
                                              np.flatnonzero((time >= time[-1] - 100)
                                                             | (np.abs(time - peak_time) <= 100))])
            writer.writerows((system, time[index], displacement[index], spin[index],
                              args.mass, args.eos, metadata["ns_lambda2"],
                              args.f22_start, f_end)
                             for index in output_indices)
    _plot(csv_path, png_path)
    print(f"wrote {csv_path.name} and {png_path.name}")


if __name__ == "__main__":
    main()

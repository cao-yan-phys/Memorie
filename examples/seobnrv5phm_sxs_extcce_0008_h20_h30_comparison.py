"""Compare `SEOBNRv5PHM` modes plus perturbative memory with SXS h20/h30.

The public Ext-CCE strain is fetched from the SXS Zenodo record on first use.
The SXS target is its supplied h20/h30. The EOB total combines its supplied
m=0 modes with perturbative memory from complete signed-m inertial-frame modes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memorie import compute_memory_modes, differentiate_modes, symmetric_mass_ratio  # noqa: E402


MTSUN_SI = 4.925490947641266978197229498498379006e-6
SXS_ID = "SXS:BBH_ExtCCE:0008"
SXS_ZENODO_RECORD = "10783582"
SXS_FILE_PREFIX = f"https://zenodo.org/api/records/{SXS_ZENODO_RECORD}/files"
SXS_FILES = {
    "strain": "Lev5/rhOverM_BondiCce_R0305_CoM.h5",
    "sidecar": "Lev5/rhOverM_BondiCce_R0305_CoM.json",
    "metadata": "Lev5/metadata.json",
}
DEFAULT_MATCH_TIME = 500.0
DEFAULT_ALIGNMENT_WINDOW = (700.0, 2200.0)
DEFAULT_ALIGNMENT_MAX_TIME_SHIFT = 200.0
DEFAULT_ALIGNMENT_MAXITER = 32


def _download(url: str, destination: Path) -> None:
    """Download one immutable SXS artifact without replacing an existing file."""

    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("xb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def _ensure_sxs_cce_files(cache_dir: Path) -> dict[str, Path]:
    local_names = {key: Path(remote_name).name for key, remote_name in SXS_FILES.items()}
    for key, remote_name in SXS_FILES.items():
        url = f"{SXS_FILE_PREFIX}/{quote(remote_name, safe='/')}/content"
        _download(url, cache_dir / local_names[key])
    return {key: cache_dir / name for key, name in local_names.items()}


def _require_signed_m_modes(modes: dict[tuple[int, int], np.ndarray], source: str) -> None:
    positive_modes = [mode for mode in modes if mode[1] > 0]
    missing_partners = [mode for mode in positive_modes if (mode[0], -mode[1]) not in modes]
    if missing_partners:
        raise ValueError(f"{source} is missing signed-m partners for {missing_partners}")


def _load_sxs_modes(
    cache_dir: Path,
    source_lmax: int,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray], dict[str, object]]:
    try:
        import scri
    except ImportError as exc:
        raise SystemExit("This example requires scri to read the compressed SXS waveform.") from exc

    files = _ensure_sxs_cce_files(cache_dir)
    waveform = scri.rpxmb.load(str(files["strain"]), transform_to_inertial=True)
    modes = {
        tuple(map(int, mode)): np.asarray(waveform.data[:, index], dtype=complex)
        for index, mode in enumerate(waveform.LM)
        if int(mode[0]) <= source_lmax
    }
    if not modes:
        raise ValueError(f"{SXS_ID} has no modes through l={source_lmax}")
    _require_signed_m_modes(modes, SXS_ID)
    return (
        np.asarray(waveform.t, dtype=float),
        modes,
        json.loads(files["metadata"].read_text(encoding="utf-8")),
    )


def _normalized(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} has zero norm")
    return vector / norm


def _reference_eob_inputs(
    metadata: dict[str, object],
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Express the SXS reference spins in the EOB source frame."""

    omega = np.asarray(metadata["reference_orbital_frequency"], dtype=float)
    separation = np.asarray(metadata["reference_position1"], dtype=float) - np.asarray(
        metadata["reference_position2"], dtype=float
    )
    z_axis = _normalized(omega, "reference_orbital_frequency")
    x_axis = separation - z_axis * np.dot(separation, z_axis)
    x_axis = _normalized(x_axis, "reference separation projected orthogonal to orbital angular momentum")
    y_axis = _normalized(np.cross(z_axis, x_axis), "reference source-frame y axis")
    x_axis = np.cross(y_axis, z_axis)
    source_to_sxs = np.column_stack((x_axis, y_axis, z_axis))

    q = float(metadata["reference_mass1"]) / float(metadata["reference_mass2"])
    if q < 1.0:
        raise ValueError("this example requires SXS metadata ordered with mass1 >= mass2")
    chi1 = source_to_sxs.T @ np.asarray(metadata["reference_dimensionless_spin1"], dtype=float)
    chi2 = source_to_sxs.T @ np.asarray(metadata["reference_dimensionless_spin2"], dtype=float)
    return q, chi1, chi2, float(np.linalg.norm(omega))


def _parse_pyseobnr_modes(raw_modes: dict[str, np.ndarray]) -> dict[tuple[int, int], np.ndarray]:
    modes = {
        tuple(map(int, key.split(","))): np.asarray(series, dtype=complex)
        for key, series in raw_modes.items()
    }
    _require_signed_m_modes(modes, "SEOBNRv5PHM")
    return modes


def _generate_seobnrv5phm_modes(
    q: float,
    chi1: np.ndarray,
    chi2: np.ndarray,
    omega_start: float,
    delta_t_m: float,
    total_mass_solar: float,
    eob_lmax: int,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    try:
        from pyseobnr.generate_waveform import generate_modes_opt
    except ImportError as exc:
        raise SystemExit("This example requires pyseobnr.") from exc

    time, raw_modes = generate_modes_opt(
        q,
        chi1,
        chi2,
        omega_start,
        omega_ref=omega_start,
        approximant="SEOBNRv5PHM",
        settings={
            "M": total_mass_solar,
            "dt": delta_t_m * total_mass_solar * MTSUN_SI,
            "lmax": eob_lmax,
            "lmax_nyquist": 1,
            "enable_antisymmetric_modes": True,
            "antisymmetric_modes": [(2, 2), (3, 3), (4, 4)],
        },
    )
    return np.asarray(time, dtype=float), _parse_pyseobnr_modes(raw_modes)


def _extend_to_reference_end(
    time: np.ndarray,
    values: np.ndarray,
    reference_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict to the CCE interval and hold a shorter EOB waveform at its final value."""

    keep = time <= reference_end
    clipped_time = time[keep]
    clipped_values = values[keep]
    if not len(clipped_time):
        raise ValueError("The EOB waveform does not overlap the CCE interval")
    if clipped_time[-1] < reference_end:
        clipped_time = np.append(clipped_time, reference_end)
        clipped_values = np.append(clipped_values, clipped_values[-1])
    return clipped_time, clipped_values


def _interpolate_with_plateau(
    target_time: np.ndarray,
    source_time: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.interp(target_time, source_time, np.real(values)) + 1j * np.interp(
        target_time, source_time, np.imag(values)
    )


def _start_index(time: np.ndarray, elapsed_time: float) -> int:
    index = int(np.searchsorted(time - time[0], elapsed_time, side="left"))
    if index >= len(time):
        raise ValueError(f"match time {elapsed_time:g} M is outside the waveform interval")
    return index


def _mode_order(lmax: int) -> list[tuple[int, int]]:
    return [(ell, m) for ell in range(2, lmax + 1) for m in range(-ell, ell + 1)]


def _mode_matrix(
    modes: dict[tuple[int, int], np.ndarray],
    lmax: int,
    source: str,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    mode_order = _mode_order(lmax)
    missing = [mode for mode in mode_order if mode not in modes]
    if missing:
        raise ValueError(f"{source} lacks modes required for rotational alignment: {missing}")
    return mode_order, np.column_stack([modes[mode] for mode in mode_order])


def _euler_rotor(alpha: float, beta: float, gamma: float) -> object:
    try:
        import quaternion
        import quaternionic
    except ImportError as exc:
        raise SystemExit("This example requires the quaternion dependencies installed with scri.") from exc

    return quaternion.from_float_array(
        np.asarray(quaternionic.array.from_euler_angles(alpha, beta, gamma))
    )


def _rotate_mode_data(
    time: np.ndarray,
    data: np.ndarray,
    lmax: int,
    euler_angles: np.ndarray,
) -> np.ndarray:
    try:
        import scri
    except ImportError as exc:
        raise SystemExit("This example requires scri for rotational alignment.") from exc

    waveform = scri.WaveformModes(
        t=np.asarray(time, dtype=float).copy(),
        data=np.asarray(data, dtype=complex).copy(),
        ell_min=2,
        ell_max=lmax,
        frameType=scri.Inertial,
        dataType=scri.h,
        r_is_scaled_out=True,
        m_is_scaled_out=True,
    )
    scri.rotations.rotate_physical_system(waveform, _euler_rotor(*euler_angles))
    return waveform.data


def _rotate_modes(
    time: np.ndarray,
    modes: dict[tuple[int, int], np.ndarray],
    lmax: int,
    euler_angles: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    mode_order, data = _mode_matrix(modes, lmax, "SEOBNRv5PHM")
    rotated = _rotate_mode_data(time, data, lmax, euler_angles)
    return {mode: rotated[:, index] for index, mode in enumerate(mode_order)}


def _interpolate_complex(
    target_time: np.ndarray | float,
    source_time: np.ndarray,
    values: np.ndarray,
) -> np.ndarray | complex:
    return np.interp(target_time, source_time, np.real(values)) + 1j * np.interp(
        target_time, source_time, np.imag(values)
    )


def _interpolate_mode_matrix(
    target_time: np.ndarray,
    source_time: np.ndarray,
    data: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [_interpolate_complex(target_time, source_time, values) for values in data.T]
    )


def _fit_rigid_alignment(
    t_sxs: np.ndarray,
    h_sxs: dict[tuple[int, int], np.ndarray],
    t_eob: np.ndarray,
    h_eob: dict[tuple[int, int], np.ndarray],
    lmax: int,
    window_start: float,
    window_end: float,
    max_time_shift: float,
    maxiter: int,
) -> tuple[float, np.ndarray, float, float]:
    """Fit EOB elapsed time = SXS elapsed time + delta_t and a constant rotation."""

    try:
        from scipy.optimize import differential_evolution
    except ImportError as exc:
        raise SystemExit("This example requires SciPy for rigid waveform alignment.") from exc

    if not 0.0 <= window_start < window_end:
        raise ValueError("the alignment window must have 0 <= start < end")
    if max_time_shift <= 0.0:
        raise ValueError("alignment max time shift must be positive")
    if maxiter < 1:
        raise ValueError("alignment maxiter must be positive")

    mode_order, sxs_data = _mode_matrix(h_sxs, lmax, SXS_ID)
    eob_mode_order, eob_data = _mode_matrix(h_eob, lmax, "SEOBNRv5PHM")
    if eob_mode_order != mode_order:
        raise RuntimeError("inconsistent alignment mode ordering")

    sxs_elapsed = t_sxs - t_sxs[0]
    eob_elapsed = t_eob - t_eob[0]
    grid = np.linspace(window_start, window_end, 401)
    if grid[-1] > sxs_elapsed[-1]:
        raise ValueError("the alignment window exceeds the SXS waveform")

    fit_mask = (eob_elapsed >= window_start - max_time_shift) & (
        eob_elapsed <= window_end + max_time_shift
    )
    if not np.any(fit_mask):
        raise ValueError("the EOB waveform does not cover the alignment window")
    fit_time = eob_elapsed[fit_mask]
    if fit_time[0] > window_start - max_time_shift or fit_time[-1] < window_end + max_time_shift:
        raise ValueError("the EOB waveform does not cover the allowed alignment time shifts")
    fit_data = eob_data[fit_mask]
    radiative_columns = np.array([mode[1] != 0 for mode in mode_order])
    sxs_grid = _interpolate_mode_matrix(grid, sxs_elapsed, sxs_data[:, radiative_columns])
    normalization = float(np.vdot(sxs_grid, sxs_grid).real)
    if normalization == 0.0:
        raise ValueError("the SXS alignment window has zero radiative norm")

    def mismatch(parameters: np.ndarray) -> float:
        delta_t, alpha, beta, gamma = parameters
        shifted_grid = grid + delta_t
        if shifted_grid[0] < fit_time[0] or shifted_grid[-1] > fit_time[-1]:
            return 1.0e6
        rotated = _rotate_mode_data(fit_time, fit_data, lmax, np.array([alpha, beta, gamma]))
        eob_grid = _interpolate_mode_matrix(shifted_grid, fit_time, rotated[:, radiative_columns])
        residual = sxs_grid - eob_grid
        return float(np.vdot(residual, residual).real / normalization)

    identity_mismatch = mismatch(np.zeros(4))
    result = differential_evolution(
        mismatch,
        bounds=[
            (-max_time_shift, max_time_shift),
            (-np.pi, np.pi),
            (0.0, np.pi),
            (-np.pi, np.pi),
        ],
        popsize=8,
        maxiter=maxiter,
        tol=5.0e-4,
        polish=True,
        seed=20260903,
        updating="immediate",
        workers=1,
    )
    if result.fun >= identity_mismatch:
        raise RuntimeError("rigid alignment did not improve the radiative-mode mismatch")
    return float(result.x[0]), np.asarray(result.x[1:], dtype=float), identity_mismatch, float(result.fun)


def _aligned_eob_change(
    elapsed_time: np.ndarray,
    values: np.ndarray,
    sxs_reference_elapsed: float,
    delta_t: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    eob_reference_elapsed = sxs_reference_elapsed + delta_t
    if not elapsed_time[0] <= eob_reference_elapsed <= elapsed_time[-1]:
        raise ValueError("the aligned EOB reference time is outside the waveform")
    baseline = _interpolate_complex(eob_reference_elapsed, elapsed_time, values)
    start = int(np.searchsorted(elapsed_time, eob_reference_elapsed, side="left"))
    return (
        elapsed_time[start:] - eob_reference_elapsed,
        values[start:] - baseline,
        eob_reference_elapsed,
    )


def _write_csv(
    path: Path,
    time: np.ndarray,
    h20_sxs: np.ndarray,
    h20_eob_no_memory: np.ndarray,
    h20_eob: np.ndarray,
    h30_sxs: np.ndarray,
    h30_eob_no_memory: np.ndarray,
    h30_eob: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "t_minus_t0_M",
                "SXS_BBH_ExtCCE_0008_delta_h20_real_over_nu",
                "SXS_BBH_ExtCCE_0008_delta_h20_imag_over_nu",
                "SEOBNRv5PHM_delta_h20_real_over_nu",
                "SEOBNRv5PHM_delta_h20_imag_over_nu",
                "SEOBNRv5PHM_plus_memory_delta_h20_real_over_nu",
                "SEOBNRv5PHM_plus_memory_delta_h20_imag_over_nu",
                "SXS_BBH_ExtCCE_0008_delta_h30_real_over_nu",
                "SXS_BBH_ExtCCE_0008_delta_h30_imag_over_nu",
                "SEOBNRv5PHM_delta_h30_real_over_nu",
                "SEOBNRv5PHM_delta_h30_imag_over_nu",
                "SEOBNRv5PHM_plus_memory_delta_h30_real_over_nu",
                "SEOBNRv5PHM_plus_memory_delta_h30_imag_over_nu",
            ]
        )
        for row in zip(
            time,
            h20_sxs,
            h20_eob_no_memory,
            h20_eob,
            h30_sxs,
            h30_eob_no_memory,
            h30_eob,
            strict=True,
        ):
            writer.writerow(
                [
                    row[0],
                    row[1].real,
                    row[1].imag,
                    row[2].real,
                    row[2].imag,
                    row[3].real,
                    row[3].imag,
                    row[4].real,
                    row[4].imag,
                    row[5].real,
                    row[5].imag,
                    row[6].real,
                    row[6].imag,
                ]
            )


def _replot_from_csv(csv_path: Path, png_path: Path) -> None:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    time = data["t_minus_t0_M"]
    h20_sxs = data["SXS_BBH_ExtCCE_0008_delta_h20_real_over_nu"] + 1j * data[
        "SXS_BBH_ExtCCE_0008_delta_h20_imag_over_nu"
    ]
    h20_eob_no_memory = data["SEOBNRv5PHM_delta_h20_real_over_nu"] + 1j * data[
        "SEOBNRv5PHM_delta_h20_imag_over_nu"
    ]
    h20_eob = data["SEOBNRv5PHM_plus_memory_delta_h20_real_over_nu"] + 1j * data[
        "SEOBNRv5PHM_plus_memory_delta_h20_imag_over_nu"
    ]
    h30_sxs = data["SXS_BBH_ExtCCE_0008_delta_h30_real_over_nu"] + 1j * data[
        "SXS_BBH_ExtCCE_0008_delta_h30_imag_over_nu"
    ]
    h30_eob_no_memory = data["SEOBNRv5PHM_delta_h30_real_over_nu"] + 1j * data[
        "SEOBNRv5PHM_delta_h30_imag_over_nu"
    ]
    h30_eob = data["SEOBNRv5PHM_plus_memory_delta_h30_real_over_nu"] + 1j * data[
        "SEOBNRv5PHM_plus_memory_delta_h30_imag_over_nu"
    ]
    _plot(
        png_path,
        time,
        h20_sxs,
        h30_sxs,
        time,
        h20_eob_no_memory,
        h30_eob_no_memory,
        h20_eob,
        h30_eob,
    )


def _plot(
    path: Path,
    time_sxs: np.ndarray,
    h20_sxs: np.ndarray,
    h30_sxs: np.ndarray,
    time_eob: np.ndarray,
    h20_eob_no_memory: np.ndarray,
    h30_eob_no_memory: np.ndarray,
    h20_eob: np.ndarray,
    h30_eob: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    cce_h20_label = r"$\mathtt{SXS\!:\!BBH\_ExtCCE\!:\!0008}$"
    cce_h30_label = r"$\mathtt{SXS\!:\!BBH\_ExtCCE\!:\!0008}$"
    eob_no_memory_label = r"$\mathtt{SEOBNRv5PHM}$"
    eob_h20_label = r"$\mathtt{SEOBNRv5PHM}$ + perturbative null memory"
    eob_h30_label = r"$\mathtt{SEOBNRv5PHM}$ + perturbative null memory"
    panels = (
        (
            r"$\mathrm{Re}\,\Delta h_{2,0}/(\nu M/R)$",
            h20_sxs.real,
            h20_eob_no_memory.real,
            h20_eob.real,
            cce_h20_label,
            eob_h20_label,
        ),
        (
            r"$\mathrm{Im}\,\Delta h_{3,0}/(\nu M/R)$",
            h30_sxs.imag,
            h30_eob_no_memory.imag,
            h30_eob.imag,
            cce_h30_label,
            eob_h30_label,
        ),
    )

    figure, axes = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True)
    for axis, (ylabel, sxs_values, eob_no_memory_values, eob_values, cce_label, eob_label) in zip(
        axes, panels, strict=True
    ):
        axis.plot(time_sxs, sxs_values, color="black", linewidth=1.25, label=cce_label)
        axis.plot(
            time_eob,
            eob_no_memory_values,
            color="blue",
            linestyle="-",
            alpha=0.45,
            linewidth=1.25,
            label=eob_no_memory_label,
        )
        axis.plot(
            time_eob,
            eob_values,
            color="red",
            linestyle="-",
            alpha=0.65,
            linewidth=1.25,
            label=eob_label,
        )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(loc="best", frameon=False)
    axes[1].legend(loc="best", frameon=False)
    axes[1].set_xlabel(r"$t-t_0$ [$M$]")
    figure.suptitle(
        r"$\mathtt{SEOBNRv5PHM}$ vs $\mathtt{SXS\!:\!BBH\_ExtCCE\!:\!0008}$, $q=1$",
        y=0.99,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _format_complex(value: complex) -> str:
    return f"{value.real:+.6e}{value.imag:+.6e}j"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--omega-start",
        type=float,
        default=None,
        help="override the SXS reference orbital frequency used to start SEOBNRv5PHM",
    )
    parser.add_argument("--eob-delta-t", type=float, default=1.0, help="SEOBNRv5PHM time step in units of M")
    parser.add_argument("--eob-lmax", type=int, default=5)
    parser.add_argument("--memory-lmax", type=int, default=10)
    parser.add_argument("--match-time", type=float, default=DEFAULT_MATCH_TIME)
    parser.add_argument("--alignment-window-start", type=float, default=DEFAULT_ALIGNMENT_WINDOW[0])
    parser.add_argument("--alignment-window-end", type=float, default=DEFAULT_ALIGNMENT_WINDOW[1])
    parser.add_argument("--alignment-max-time-shift", type=float, default=DEFAULT_ALIGNMENT_MAX_TIME_SHIFT)
    parser.add_argument("--alignment-maxiter", type=int, default=DEFAULT_ALIGNMENT_MAXITER)
    parser.add_argument("--total-mass-solar", type=float, default=50.0)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "memorie" / "sxs_bbh_extcce_0008_lev5_r0305_com",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples" / "output")
    parser.add_argument("--plot-from-csv", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "seobnrv5phm_sxs_bbh_extcce_0008_h20_h30_q1"
    csv_path = args.output_dir / f"{stem}.csv"
    png_path = args.output_dir / f"{stem}.png"
    if args.plot_from_csv:
        if not csv_path.exists():
            raise FileNotFoundError(f"cannot replot missing CSV: {csv_path}")
        _replot_from_csv(csv_path, png_path)
        print(f"Redrew plot: {png_path}")
        return 0

    t_sxs, h_sxs, metadata = _load_sxs_modes(args.cache_dir, args.eob_lmax)
    q, chi1, chi2, metadata_omega_start = _reference_eob_inputs(metadata)
    omega_start = args.omega_start if args.omega_start is not None else metadata_omega_start
    t_eob, h_eob = _generate_seobnrv5phm_modes(
        q,
        chi1,
        chi2,
        omega_start,
        args.eob_delta_t,
        args.total_mass_solar,
        args.eob_lmax,
    )
    alignment_delta_t, alignment_euler, identity_mismatch, aligned_mismatch = _fit_rigid_alignment(
        t_sxs,
        h_sxs,
        t_eob,
        h_eob,
        args.eob_lmax,
        args.alignment_window_start,
        args.alignment_window_end,
        args.alignment_max_time_shift,
        args.alignment_maxiter,
    )
    h_eob = _rotate_modes(t_eob, h_eob, args.eob_lmax, alignment_euler)
    eob_memory = compute_memory_modes(
        t_eob,
        h_eob,
        [(2, 0), (3, 0)],
        lmax=args.memory_lmax,
        hdot=differentiate_modes(t_eob, h_eob),
        include_cm=False,
    )

    sxs_start = _start_index(t_sxs, args.match_time)
    sxs_elapsed = t_sxs - t_sxs[0]
    eob_elapsed = t_eob - t_eob[0]
    sxs_reference_elapsed = sxs_elapsed[sxs_start]
    relative_sxs_time = t_sxs[sxs_start:] - t_sxs[sxs_start]
    nu = symmetric_mass_ratio(q)
    delta_h20_sxs = (h_sxs[(2, 0)][sxs_start:] - h_sxs[(2, 0)][sxs_start]) / nu
    delta_h30_sxs = (h_sxs[(3, 0)][sxs_start:] - h_sxs[(3, 0)][sxs_start]) / nu
    relative_eob_time, delta_h20_eob_no_memory, eob_reference_elapsed = _aligned_eob_change(
        eob_elapsed, h_eob[(2, 0)], sxs_reference_elapsed, alignment_delta_t
    )
    _relative_eob_time_h30, delta_h30_eob_no_memory, _eob_reference_elapsed_h30 = _aligned_eob_change(
        eob_elapsed, h_eob[(3, 0)], sxs_reference_elapsed, alignment_delta_t
    )
    h20_null_memory = eob_memory[(2, 0)]["h_displacement"]
    h30_null_memory = eob_memory[(3, 0)]["h_spin_mode"]
    _relative_eob_time_h20_memory, delta_h20_memory, _eob_reference_elapsed_h20_memory = _aligned_eob_change(
        eob_elapsed,
        h20_null_memory,
        sxs_reference_elapsed,
        alignment_delta_t,
    )
    _relative_eob_time_h30_memory, delta_h30_memory, _eob_reference_elapsed_h30_memory = _aligned_eob_change(
        eob_elapsed,
        h30_null_memory,
        sxs_reference_elapsed,
        alignment_delta_t,
    )
    if not all(
        np.array_equal(relative_eob_time, time)
        for time in (
            _relative_eob_time_h30,
            _relative_eob_time_h20_memory,
            _relative_eob_time_h30_memory,
        )
    ) or not np.allclose(
        eob_reference_elapsed,
        (
            _eob_reference_elapsed_h30,
            _eob_reference_elapsed_h20_memory,
            _eob_reference_elapsed_h30_memory,
        ),
    ):
        raise RuntimeError("inconsistent aligned SEOBNRv5PHM reference grids")
    delta_h20_eob_no_memory /= nu
    delta_h30_eob_no_memory /= nu
    delta_h20_eob = delta_h20_eob_no_memory + delta_h20_memory / nu
    delta_h30_eob = delta_h30_eob_no_memory + delta_h30_memory / nu
    plot_time_eob, plot_h20_eob = _extend_to_reference_end(
        relative_eob_time, delta_h20_eob, relative_sxs_time[-1]
    )
    _plot_time_eob, plot_h20_eob_no_memory = _extend_to_reference_end(
        relative_eob_time, delta_h20_eob_no_memory, relative_sxs_time[-1]
    )
    _plot_time_eob_h30, plot_h30_eob = _extend_to_reference_end(
        relative_eob_time, delta_h30_eob, relative_sxs_time[-1]
    )
    _plot_time_eob_h30_no_memory, plot_h30_eob_no_memory = _extend_to_reference_end(
        relative_eob_time, delta_h30_eob_no_memory, relative_sxs_time[-1]
    )
    if not all(
        np.array_equal(plot_time_eob, time)
        for time in (_plot_time_eob, _plot_time_eob_h30, _plot_time_eob_h30_no_memory)
    ):
        raise RuntimeError("inconsistent SEOBNRv5PHM plot grids")

    _write_csv(
        csv_path,
        relative_sxs_time,
        delta_h20_sxs,
        _interpolate_with_plateau(relative_sxs_time, relative_eob_time, delta_h20_eob_no_memory),
        _interpolate_with_plateau(relative_sxs_time, relative_eob_time, delta_h20_eob),
        delta_h30_sxs,
        _interpolate_with_plateau(relative_sxs_time, relative_eob_time, delta_h30_eob_no_memory),
        _interpolate_with_plateau(relative_sxs_time, relative_eob_time, delta_h30_eob),
    )
    _plot(
        png_path,
        relative_sxs_time,
        delta_h20_sxs,
        delta_h30_sxs,
        plot_time_eob,
        plot_h20_eob_no_memory,
        plot_h30_eob_no_memory,
        plot_h20_eob,
        plot_h30_eob,
    )

    print(f"{SXS_ID} vs SEOBNRv5PHM h20/h30 comparison")
    print(f"omega_start = {omega_start:.12e}")
    print(f"x0 = {omega_start ** (2.0 / 3.0):.12e}")
    print(f"nu = {nu:.12e}")
    print(f"match time = {args.match_time:.6f} M")
    print(
        "alignment window = "
        f"[{args.alignment_window_start:.6f}, {args.alignment_window_end:.6f}] M"
    )
    print(f"alignment delta_t = {alignment_delta_t:.6f} M")
    print(f"alignment Euler angles = {alignment_euler}")
    print(f"alignment radiative mismatch = {identity_mismatch:.6e} -> {aligned_mismatch:.6e}")
    print(f"SXS comparison range = [{t_sxs[sxs_start]:.6f}, {t_sxs[-1]:.6f}] M")
    print(
        "SEOBNRv5PHM comparison range = "
        f"[{eob_reference_elapsed:.6f}, {eob_elapsed[-1]:.6f}] M after model start"
    )
    print(f"{SXS_ID} signed-m modes = {sorted(h_sxs)}")
    print(f"SEOBNRv5PHM signed-m modes = {sorted(h_eob)}")
    print(f"final SXS Delta h20 / nu = {_format_complex(delta_h20_sxs[-1])}")
    print(f"final SEOBNRv5PHM plus memory Delta h20 / nu = {_format_complex(delta_h20_eob[-1])}")
    print(f"final SXS Delta h30 / nu = {_format_complex(delta_h30_sxs[-1])}")
    print(f"final SEOBNRv5PHM plus memory Delta h30 / nu = {_format_complex(delta_h30_eob[-1])}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

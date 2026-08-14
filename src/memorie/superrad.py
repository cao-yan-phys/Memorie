"""SuperRad helpers for slowly varying axisymmetric memory modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Any, Iterable

import numpy as np

from .core import complete_nonprecessing_modes, compute_memory_modes

MTSUN_SI = 4.925490947641266978197229498498379006e-6
Mode = tuple[int, int]
DEFAULT_TARGETS: tuple[Mode, ...] = ((2, 0), (3, 0))


@dataclass(frozen=True)
class SuperRadConfig:
    """Parameters for a SuperRad boson-cloud memory calculation.

    ``alpha`` is the dimensionless gravitational fine-structure constant.  The
    ``growth_start_fraction`` selects how far back from cloud saturation to
    begin, as a fraction of SuperRad's cloud-growth time.  The endpoint is in
    units of SuperRad's post-saturation GW dissipation time, with ``t=0`` at
    cloud saturation.
    """

    black_hole_mass_msun: float = 20.8
    black_hole_spin: float = 0.7
    alpha: float = 0.2
    boson_spin: int = 1
    cloud_model: str = "relativistic"
    evolution: str = "matched"
    growth_start_fraction: float = 1.0
    end_gw_time: float = 10.0
    n_samples: int = 2048


def _as_1d_real(values: Any, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(size, float(array))
    if array.shape != (size,):
        raise ValueError(f"expected a one-dimensional series of length {size}, got {array.shape}")
    return array


def _full_evolution_coefficients(waveform: Any, t_seconds: np.ndarray) -> np.ndarray:
    """Evaluate SuperRad's instantaneous spherical-harmonic coefficients."""

    t_internal = t_seconds / waveform._tunit
    mass_bh = _as_1d_real(waveform._Mbh(t_internal), len(t_seconds))
    spin_bh = _as_1d_real(waveform._abh(t_internal), len(t_seconds))
    coefficient_function = np.vectorize(
        waveform._cloud_model.strain_sph_harm,
        excluded=[0],
        signature="(),()->(n)",
    )
    return np.asarray(
        coefficient_function(waveform.azimuthal_num(), waveform._mu * mass_bh, spin_bh),
        dtype=complex,
    )


def _spherical_coefficients(waveform: Any, t_seconds: np.ndarray) -> np.ndarray:
    """Return phase-stripped SuperRad coefficients for each supplied time."""

    if callable(getattr(waveform, "_Mbh", None)):
        return _full_evolution_coefficients(waveform, t_seconds)

    coefficients = np.asarray(waveform._hl, dtype=complex)
    return np.broadcast_to(coefficients, (len(t_seconds), len(coefficients))).copy()


def _validate_config(config: SuperRadConfig, targets: tuple[Mode, ...], lmax: int) -> None:
    if config.black_hole_mass_msun <= 0.0:
        raise ValueError("black_hole_mass_msun must be positive")
    if not 0.0 < config.black_hole_spin <= 1.0:
        raise ValueError("black_hole_spin must lie in (0, 1]")
    if config.alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if config.boson_spin not in {0, 1}:
        raise ValueError("boson_spin must be 0 (scalar) or 1 (vector)")
    if config.cloud_model not in {"relativistic", "non-relativistic"}:
        raise ValueError("cloud_model must be 'relativistic' or 'non-relativistic'")
    if config.evolution not in {"matched", "full"}:
        raise ValueError("evolution must be 'matched' or 'full'")
    if config.n_samples < 3:
        raise ValueError("n_samples must be at least 3")
    if not 0.0 <= config.growth_start_fraction <= 1.0:
        raise ValueError("growth_start_fraction must lie in [0, 1]")
    if config.end_gw_time <= 0.0:
        raise ValueError("end_gw_time must be positive")
    if lmax < 2:
        raise ValueError("lmax must be at least 2")
    if not targets:
        raise ValueError("targets must not be empty")
    if any(int(emm) != 0 for _ell, emm in targets):
        raise ValueError("the SuperRad slow-time adapter currently supports only M=0 targets")


def _mode_power(hdot: dict[Mode, np.ndarray]) -> np.ndarray:
    power = np.zeros_like(next(iter(hdot.values())), dtype=float)
    for series in hdot.values():
        power += np.abs(series) ** 2
    return power / (16.0 * np.pi)


def _evolution_time_grid(
    waveform: Any,
    config: SuperRadConfig,
    gw_time_seconds: float,
) -> np.ndarray:
    """Resolve cloud growth, saturation, and the slower dissipative era."""

    growth_start_seconds = -config.growth_start_fraction * float(
        waveform.cloud_growth_time()
    )
    end_seconds = config.end_gw_time * gw_time_seconds
    if growth_start_seconds == 0.0:
        return np.linspace(0.0, end_seconds, config.n_samples)

    transition_half_width = min(10.0 * float(waveform.efold_time()), end_seconds)
    transition_left = max(growth_start_seconds, -transition_half_width)
    transition_right = transition_half_width
    if growth_start_seconds >= transition_left:
        transition_samples = config.n_samples // 2
        return np.concatenate(
            (
                np.linspace(growth_start_seconds, transition_right, transition_samples),
                np.linspace(
                    transition_right,
                    end_seconds,
                    config.n_samples - transition_samples + 1,
                )[1:],
            )
        )

    early_samples = max(3, config.n_samples // 16)
    transition_samples = max(33, config.n_samples // 2)
    late_samples = config.n_samples - early_samples - transition_samples
    early_grid = np.linspace(
        growth_start_seconds,
        transition_left,
        early_samples,
        endpoint=False,
    )
    transition_grid = np.linspace(
        transition_left,
        transition_right,
        transition_samples,
    )
    late_grid = np.linspace(transition_right, end_seconds, late_samples + 1)[1:]
    return np.concatenate((early_grid, transition_grid, late_grid))


def compute_superrad_memory_modes(
    config: SuperRadConfig | None = None,
    *,
    targets: Iterable[Mode] = DEFAULT_TARGETS,
    lmax: int = 10,
) -> dict[str, Any]:
    """Construct SuperRad radiative modes and their slowly varying M=0 memory.

    SuperRad supplies the phase-stripped coefficients of the radiative modes
    ``(ell, 2 m_cloud)``.  This helper restores the common GW phase, constructs
    the negative-m partners, and supplies derivatives with the rapid carrier
    term evaluated analytically to :func:`compute_memory_modes`.  It deliberately
    accepts only ``M=0`` targets: their bilinear sources cancel the rapid
    continuous-wave phase and can be evaluated on the cloud-evolution time grid.

    This adapter computes displacement and spin memory only; it does not
    calculate CM memory modes.  Returned radiative and memory amplitudes are
    distance-rescaled and normalized by the initial black-hole mass.
    Cumulative displacement and spin-memory integrals start from zero at the
    first requested time.
    """

    cfg = SuperRadConfig() if config is None else config
    target_modes = tuple((int(ell), int(emm)) for ell, emm in targets)
    _validate_config(cfg, target_modes, int(lmax))

    try:
        from superrad import ultralight_boson as ub
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("The SuperRad helper requires the optional 'superrad' package.") from exc

    boson = ub.UltralightBoson(spin=cfg.boson_spin, model=cfg.cloud_model)
    waveform = boson.make_waveform(
        cfg.black_hole_mass_msun,
        cfg.black_hole_spin,
        cfg.alpha,
        units="physical+alpha",
        evo_type=cfg.evolution,
    )
    gw_time_seconds = float(waveform.gw_time())
    t_seconds = _evolution_time_grid(waveform, cfg, gw_time_seconds)
    mass_scale_seconds = cfg.black_hole_mass_msun * MTSUN_SI
    t_dimensionless = t_seconds / mass_scale_seconds

    cloud_mass = _as_1d_real(waveform.mass_cloud(t_seconds), len(t_seconds))
    phase = _as_1d_real(waveform.phase_gw(t_seconds), len(t_seconds))
    omega_dimensionless = 2.0 * np.pi * _as_1d_real(
        waveform.freq_gw(t_seconds), len(t_seconds)
    ) * mass_scale_seconds
    coefficients = _spherical_coefficients(waveform, t_seconds)
    if coefficients.ndim != 2 or coefficients.shape[0] != len(t_seconds):
        raise RuntimeError("SuperRad returned spherical-harmonic coefficients with an unexpected shape")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("SuperRad returned non-finite spherical-harmonic coefficients")

    cloud_m = int(waveform.azimuthal_num())
    gw_m = 2 * cloud_m
    ell_values = gw_m + np.arange(coefficients.shape[1], dtype=int)
    retained = ell_values <= int(lmax)
    if not np.any(retained):
        raise ValueError(f"lmax={lmax} excludes every SuperRad radiative mode")

    phase_factor = np.exp(-1j * np.remainder(phase, 2.0 * np.pi))
    amplitudes = coefficients[:, retained] * (cloud_mass[:, None] / cfg.black_hole_mass_msun)
    amplitude_dot = np.gradient(amplitudes, t_dimensionless, axis=0, edge_order=2)
    ell_retained = ell_values[retained]

    positive_modes = {
        (int(ell), gw_m): amplitudes[:, index] * phase_factor
        for index, ell in enumerate(ell_retained)
    }
    positive_hdot = {
        (int(ell), gw_m): (
            amplitude_dot[:, index] - 1j * omega_dimensionless * amplitudes[:, index]
        )
        * phase_factor
        for index, ell in enumerate(ell_retained)
    }
    modes = complete_nonprecessing_modes(positive_modes)
    hdot = complete_nonprecessing_modes(positive_hdot)
    memory = compute_memory_modes(
        t_dimensionless,
        modes,
        target_modes,
        lmax=int(lmax),
        hdot=hdot,
        include_cm=False,
    )

    model_power = _as_1d_real(waveform.power_gw(t_seconds), len(t_seconds)) / waveform._Punit
    mode_power = _mode_power(hdot)
    scale = np.maximum(np.abs(model_power), np.finfo(float).tiny)
    power_relative_error = np.abs(mode_power - model_power) / scale

    result: dict[str, Any] = {
        "config": asdict(cfg),
        "model": "SuperRad",
        "superrad_version": version("superrad"),
        "waveform_class": type(waveform).__name__,
        "t_seconds": t_seconds,
        "t_dimensionless": t_dimensionless,
        "mass_scale_seconds": float(mass_scale_seconds),
        "gw_time_seconds": gw_time_seconds,
        "gw_time_dimensionless": float(gw_time_seconds / mass_scale_seconds),
        "cloud_growth_time_seconds": float(waveform.cloud_growth_time()),
        "efold_time_seconds": float(waveform.efold_time()),
        "cloud_mass_msun": cloud_mass,
        "cloud_azimuthal_number": cloud_m,
        "gw_azimuthal_number": gw_m,
        "ell_values": tuple(int(ell) for ell in ell_retained),
        "dropped_ell_values": tuple(int(ell) for ell in ell_values[~retained]),
        "omega_gw_dimensionless": omega_dimensionless,
        "oscillatory_modes_dimensionless": modes,
        "oscillatory_hdot_dimensionless": hdot,
        "memory": memory,
        "model_power_dimensionless": model_power,
        "mode_power_dimensionless": mode_power,
        "power_relative_error": power_relative_error,
        "max_power_relative_error": float(np.max(power_relative_error)),
    }
    if (2, 0) in memory:
        result["h20_dimensionless"] = memory[(2, 0)]["h_displacement"]
    if (3, 0) in memory:
        result["h30_dimensionless"] = memory[(3, 0)]["h_spin_mode"]
    return result

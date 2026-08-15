"""Kerr quasinormal-mode ringdown waveforms and memory-mode helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Any, Iterable, Mapping

import numpy as np

from .core import (
    Mode,
    ModeDict,
    compute_memory_modes,
    precompute_memory_coeffs,
    validate_time_grid,
)


@dataclass(frozen=True)
class KerrQNMExcitation:
    """One explicitly excited gravitational Kerr QNM.

    ``amplitude`` is the complex coefficient of the spin-weighted spheroidal
    harmonic at ``start_time_M`` in the distance-rescaled strain convention.
    The caller must supply every desired ordinary or mirror contribution.
    """

    spheroidal_ell: int
    m: int
    overtone: int = 0
    amplitude: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        if self.spheroidal_ell < 2:
            raise ValueError("a gravitational QNM needs spheroidal_ell >= 2")
        if abs(self.m) > self.spheroidal_ell:
            raise ValueError("QNM azimuthal number must satisfy abs(m) <= spheroidal_ell")
        if self.overtone < 0:
            raise ValueError("overtone must be non-negative")
        if not np.isfinite(complex(self.amplitude).real) or not np.isfinite(complex(self.amplitude).imag):
            raise ValueError("QNM amplitude must be finite")


@dataclass(frozen=True)
class KerrRingdownConfig:
    """Dimensionless sampling configuration for a perturbation of a Kerr hole."""

    final_spin: float = 0.7
    start_time_M: float = 0.0
    end_time_M: float = 160.0
    n_samples: int = 4097

    def __post_init__(self) -> None:
        if not 0.0 <= self.final_spin < 1.0:
            raise ValueError("final_spin must satisfy 0 <= final_spin < 1")
        if not self.end_time_M > self.start_time_M:
            raise ValueError("end_time_M must be greater than start_time_M")
        if self.n_samples < 3:
            raise ValueError("n_samples must be at least 3")


def _qnm_module() -> Any:
    try:
        import qnm
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("The Kerr-ringdown helper requires the optional 'qnm' package.") from exc
    return qnm


def _spherical_amplitudes(
    qnm: Any,
    excitation: KerrQNMExcitation,
    final_spin: float,
    lmax: int,
) -> tuple[complex, ModeDict]:
    sequence = qnm.modes_cache(
        s=-2,
        l=int(excitation.spheroidal_ell),
        m=int(excitation.m),
        n=int(excitation.overtone),
    )
    omega, _separation_constant, mixing = sequence(a=float(final_spin))
    omega = complex(omega)
    if omega.imag >= 0.0:
        raise RuntimeError(f"qnm returned a non-decaying frequency {omega!r}")

    spherical_ells = qnm.angular.ells(s=-2, m=int(excitation.m), l_max=sequence.l_max)
    amplitudes: ModeDict = {}
    for spherical_ell, coefficient in zip(spherical_ells, mixing, strict=True):
        spherical_ell = int(spherical_ell)
        if 2 <= spherical_ell <= int(lmax):
            amplitudes[(spherical_ell, int(excitation.m))] = (
                complex(excitation.amplitude) * complex(coefficient)
            )
    if not amplitudes:
        raise ValueError(f"lmax={lmax} excludes every spherical mode of {excitation!r}")
    return omega, amplitudes


def generate_kerr_ringdown_modes(
    config: KerrRingdownConfig,
    excitations: Iterable[KerrQNMExcitation],
    lmax: int = 10,
) -> dict[str, Any]:
    """Generate spherical strain modes from explicitly excited Kerr QNMs.

    The returned strain modes are ``H_lm=R*h_lm/M_f`` on a time grid in
    ``t/M_f``.  No reflection or negative-``m`` completion is applied.
    """

    if lmax < 2:
        raise ValueError("lmax must be at least 2")
    excitation_list = tuple(excitations)
    if not excitation_list:
        raise ValueError("at least one QNM excitation is required")

    qnm = _qnm_module()
    time = np.linspace(config.start_time_M, config.end_time_M, config.n_samples, dtype=float)
    elapsed = time - config.start_time_M
    modes: ModeDict = {}
    hdot: ModeDict = {}
    components: list[dict[str, Any]] = []

    for excitation in excitation_list:
        omega, amplitudes = _spherical_amplitudes(qnm, excitation, config.final_spin, lmax)
        phase = np.exp(-1j * omega * elapsed)
        component_modes = {mode: amplitude * phase for mode, amplitude in amplitudes.items()}
        component_hdot = {
            mode: -1j * omega * waveform for mode, waveform in component_modes.items()
        }
        for mode, waveform in component_modes.items():
            modes[mode] = modes.get(mode, np.zeros_like(time, dtype=complex)) + waveform
        for mode, derivative in component_hdot.items():
            hdot[mode] = hdot.get(mode, np.zeros_like(time, dtype=complex)) + derivative
        components.append(
            {
                "excitation": asdict(excitation),
                "omega_dimensionless": omega,
                "spherical_amplitudes_dimensionless": amplitudes,
            }
        )

    return {
        "model": "qnm",
        "qnm_version": version("qnm"),
        "config": asdict(config),
        "lmax": int(lmax),
        "t_dimensionless": time,
        "oscillatory_modes_dimensionless": modes,
        "oscillatory_hdot_dimensionless": hdot,
        "qnm_components": components,
    }


def compute_kerr_ringdown_memory_modes(
    config: KerrRingdownConfig,
    excitations: Iterable[KerrQNMExcitation],
    targets: Iterable[tuple[int, int]],
    lmax: int = 10,
    include_cm: bool = True,
) -> dict[str, Any]:
    """Generate Kerr-ringdown modes and compute their vacuum null memory."""

    result = generate_kerr_ringdown_modes(config, excitations, lmax=lmax)
    result["memory"] = compute_memory_modes(
        result["t_dimensionless"],
        result["oscillatory_modes_dimensionless"],
        targets,
        lmax=lmax,
        hdot=result["oscillatory_hdot_dimensionless"],
        include_cm=include_cm,
    )
    return result


def analytic_single_exponential_memory_modes(
    t_dimensionless: Any,
    spherical_amplitudes: Mapping[Mode, complex],
    omega_dimensionless: complex,
    targets: Iterable[tuple[int, int]],
    lmax: int = 10,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Return exact displacement and spin memory for one shared QNM exponential.

    Every input mode is assumed to equal ``B_lm*exp(-i*omega*(t-t0))``.
    This is useful for validating the numerical memory integration used for a
    single QNM after its spheroidal-to-spherical projection.
    """

    time = validate_time_grid(t_dimensionless)
    omega = complex(omega_dimensionless)
    if omega.imag >= 0.0:
        raise ValueError("omega_dimensionless must have negative imaginary part")
    amplitudes = {mode: complex(value) for mode, value in spherical_amplitudes.items()}
    if not amplitudes:
        raise ValueError("spherical_amplitudes is empty")

    elapsed = time - time[0]
    rate = 2.0 * omega.imag
    decay = np.exp(rate * elapsed)
    integral = np.expm1(rate * elapsed) / rate
    result: dict[tuple[int, int], dict[str, Any]] = {}

    for target in targets:
        coefficient_rows = precompute_memory_coeffs(*target, l1_max=lmax, l2_max=lmax)
        displacement_source_initial = 0.0j
        spin_initial = 0.0j
        used_terms = 0
        skipped_terms = 0
        for _L, _M, l1, m1, l2, m2, gamma_d, gamma_s, _gamma_cm in coefficient_rows:
            amplitude1 = amplitudes.get((int(l1), int(m1)))
            amplitude2 = amplitudes.get((int(l2), int(m2)))
            if amplitude1 is None or amplitude2 is None:
                skipped_terms += 1
                continue
            derivative1 = -1j * omega * amplitude1
            derivative2 = -1j * omega * amplitude2
            displacement_source_initial += float(gamma_d) * derivative1 * np.conjugate(derivative2)
            spin_initial += float(gamma_s) * (
                amplitude1 * np.conjugate(derivative2)
                - derivative1 * np.conjugate(amplitude2)
            )
            used_terms += 1
        result[(int(target[0]), int(target[1]))] = {
            "target_mode": (int(target[0]), int(target[1])),
            "dh_displacement_dt": displacement_source_initial * decay,
            "h_displacement": displacement_source_initial * integral,
            "h_spin_mode": spin_initial * decay,
            "used_terms": used_terms,
            "skipped_terms": skipped_terms,
        }
    return result

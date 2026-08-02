"""Leading-order PN helpers for the SEOBNRv5EHM example."""

from __future__ import annotations

from math import pi, sqrt

import numpy as np

from .core import gamma_cm

Mode = tuple[int, int]


def symmetric_mass_ratio(q: float) -> float:
    """Return ``nu=m1*m2/(m1+m2)^2`` for mass ratio ``q=m1/m2 >= 1``."""

    q = float(q)
    return q / (1.0 + q) ** 2


def delta_mass_fraction(q: float) -> float:
    """Return ``(m1-m2)/(m1+m2)`` for ``q=m1/m2 >= 1``."""

    q = float(q)
    return (q - 1.0) / (q + 1.0)


def k20_lo(q: float) -> float:
    """Leading 0PN coefficient for ``h_20 = K20*x`` in distance-rescaled units."""

    nu = symmetric_mass_ratio(q)
    return nu * 2.0 * sqrt(10.0 * pi / 3.0) / 7.0


def h20_lo(q: float, x: float) -> float:
    """Leading 0PN displacement-memory ``h_20`` mode."""

    return k20_lo(q) * float(x)


def k40_lo(q: float) -> float:
    """Leading 0PN coefficient for the displacement-memory ``h_40`` mode."""

    nu = symmetric_mass_ratio(q)
    return nu * sqrt(pi / 10.0) / 63.0


def h40_lo(q: float, x: float) -> float:
    """Leading 0PN displacement-memory ``h_40`` mode."""

    return k40_lo(q) * float(x)


def infer_x_eff_from_dh20(dh20_dt: complex, q: float) -> float:
    """Infer effective 0PN ``x`` from initial ``dot h_20``.

    Uses ``dot h20 = K20 * (64*nu/5) * x^5`` with dimensionless time ``t/M``.
    """

    nu = symmetric_mass_ratio(q)
    slope = float(np.real(dh20_dt))
    if slope <= 0.0:
        raise ValueError(f"expected positive Re(dot h20), got {slope}")
    return (slope / ((64.0 / 5.0) * nu * k20_lo(q))) ** 0.2


def h30_spin_lo(q: float, x: float) -> complex:
    """Nichols leading PN spin-memory strain mode ``h^S_30``."""

    nu = symmetric_mass_ratio(q)
    return 1j * (16.0 * pi / 25.0) * sqrt(30.0 / (7.0 * pi)) * nu**2 * float(x) ** 3.5


def phase_from_h22_lo(h22: complex) -> float:
    """Infer orbital phase from ``H22 ~= -8 sqrt(pi/5) nu x exp(-2 i phi)``."""

    return float(-0.5 * np.angle(-complex(h22)))


def _with_negative_m(modes: dict[Mode, complex]) -> dict[Mode, complex]:
    out = dict(modes)
    for (ell, emm), value in list(out.items()):
        if emm > 0:
            out[(ell, -emm)] = (-1) ** ell * np.conjugate(value)
    return out


def _filter_modes(modes: dict[Mode, complex], available_modes: set[Mode] | None) -> dict[Mode, complex]:
    if available_modes is None:
        return modes
    return {mode: value for mode, value in modes.items() if mode in available_modes}


def _lo_pn_moments(
    q: float,
    x: float,
    phase: float,
    available_modes: set[Mode] | None = None,
) -> tuple[dict[Mode, complex], dict[Mode, complex]]:
    """Return leading PN mass moments ``U_lm`` and analytic ``dot U_lm``."""

    nu = symmetric_mass_ratio(q)
    dm = delta_mass_fraction(q)
    x = float(x)
    phase = float(phase)
    U = {
        (2, 2): -8.0 * np.sqrt(2.0 * np.pi / 5.0) * nu * x * np.exp(-2j * phase),
        (2, 0): (4.0 / 7.0) * np.sqrt(5.0 * np.pi / 3.0) * nu * x,
        (3, 1): -(2j / 3.0) * np.sqrt(np.pi / 35.0) * dm * nu * x**1.5 * np.exp(-1j * phase),
        (3, 3): 6j * np.sqrt(3.0 * np.pi / 7.0) * dm * nu * x**1.5 * np.exp(-3j * phase),
        (4, 0): np.sqrt(np.pi / 5.0) * nu * x / 63.0,
    }
    Udot = {
        (2, 2): 16j * np.sqrt(2.0 * np.pi / 5.0) * nu * x**2.5 * np.exp(-2j * phase),
        (2, 0): 0.0j,
        (3, 1): -(2.0 / 3.0) * np.sqrt(np.pi / 35.0) * dm * nu * x**3 * np.exp(-1j * phase),
        (3, 3): 18.0 * np.sqrt(3.0 * np.pi / 7.0) * dm * nu * x**3 * np.exp(-3j * phase),
        (4, 0): 0.0j,
    }
    return (
        _filter_modes(_with_negative_m(U), available_modes),
        _filter_modes(_with_negative_m(Udot), available_modes),
    )


def cm_strain_lo_modes(
    q: float,
    x: float,
    phase: float,
    available_modes: set[Mode] | None = None,
) -> dict[Mode, complex]:
    """Corrected leading nonlinear-null CM strain modes.

    If ``available_modes`` is supplied, unavailable PN radiative moments are
    set to zero.  The returned values are strain modes ``h_lm=U_lm/sqrt(2)``.

    The leading displacement-memory ``U_40`` contribution omitted in Nichols
    (2018) is included, producing the additional ``(7,1)`` and ``(7,3)`` modes.
    """

    U, dU = _lo_pn_moments(q, x, phase, available_modes=available_modes)
    h = {mode: value / np.sqrt(2.0) for mode, value in U.items()}
    hdot = {mode: value / np.sqrt(2.0) for mode, value in dU.items()}
    targets = ((3, 1), (3, 3), (5, 1), (5, 3), (5, 5), (7, 1), (7, 3))
    return {
        (L, M): sum(
            gamma_cm(L, M, l1, m1, l2, m2)
            * (
                h1 * np.conjugate(hdot[(l2, m2)])
                - hdot[(l1, m1)] * np.conjugate(h2)
            )
            for (l1, m1), h1 in h.items()
            for (l2, m2), h2 in h.items()
        )
        for L, M in targets
    }

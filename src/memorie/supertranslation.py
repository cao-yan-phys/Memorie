from __future__ import annotations

from collections.abc import Mapping
from math import factorial, pi, sqrt
from operator import index

import numpy as np

from .core import normalize_mode_dict
from .particle_displacement_memory import particle_displacement_memory_mode

try:
    from scipy.special import sph_harm_y
except ImportError:
    from scipy.special import sph_harm

    def _harmonic(ell, emm, theta, phi):
        return sph_harm(emm, ell, phi, theta)

else:

    def _harmonic(ell, emm, theta, phi):
        return sph_harm_y(ell, emm, theta, phi)


def _scalar_coefficients(values, minimum_ell):
    if not isinstance(values, Mapping):
        raise TypeError("modes must be a mapping")
    result = {}
    for (ell, emm), value in normalize_mode_dict(values).items():
        if ell < minimum_ell or abs(emm) > ell:
            raise ValueError(f"invalid mode {(ell, emm)}")
        if value.ndim != 0 or not np.isfinite(value):
            raise ValueError(f"mode {(ell, emm)} must be a finite scalar")
        result[(ell, emm)] = complex(value)
    return result


def memory_to_supertranslation(
    delta_rh,
    *,
    initial_particles=None,
    final_particles=None,
    lmax=None,
    direction="remove",
    project_electric=False,
):
    if direction not in ("remove", "add"):
        raise ValueError("direction must be 'remove' or 'add'")
    modes = _scalar_coefficients(delta_rh, 2)
    initial = tuple(() if initial_particles is None else initial_particles)
    final = tuple(() if final_particles is None else final_particles)
    if lmax is None:
        if not modes:
            raise ValueError("lmax is required when delta_rh is empty")
        lmax = max(ell for ell, _ in modes)
    lmax = index(lmax)
    if lmax < 2 or any(ell > lmax for ell, _ in modes):
        raise ValueError("lmax must be at least 2 and include every supplied mode")
    scale = max((abs(value) for value in modes.values()), default=0.0)
    electric = {}
    for ell in range(2, lmax + 1):
        for emm in range(-ell, ell + 1):
            value = modes.get((ell, emm), 0.0j)
            partner = (-1) ** emm * modes.get((ell, -emm), 0.0j).conjugate()
            if not project_electric and abs(value - partner) > 1e-10 * scale:
                raise ValueError(
                    "delta_rh is not electric parity; supply the full signed-m set "
                    "or explicitly set project_electric=True"
                )
            electric[(ell, emm)] = 0.5 * value + 0.5 * partner
    sign = 1.0 if direction == "remove" else -1.0
    alpha = {(0, 0): 0.0j, (1, -1): 0.0j, (1, 0): 0.0j, (1, 1): 0.0j}
    for (ell, emm), value in electric.items():
        if initial or final:
            value += particle_displacement_memory_mode(
                ell,
                emm,
                initial_particles=initial,
                final_particles=final,
                distance=1.0,
            )
        alpha[(ell, emm)] = sign * sqrt(factorial(ell - 2) / factorial(ell + 2)) * value
        if not np.isfinite(alpha[(ell, emm)]):
            raise ValueError(f"nonfinite supertranslation mode {(ell, emm)}")
    return alpha


def evaluate_supertranslation(alpha, theta, phi):
    modes = _scalar_coefficients(alpha, 0)
    scale = max((abs(value) for value in modes.values()), default=0.0)
    for (ell, emm), value in modes.items():
        partner = (-1) ** emm * modes.get((ell, -emm), 0.0j).conjugate()
        if abs(value - partner) > 1e-10 * scale:
            raise ValueError("alpha must represent a real scalar field on the sphere")
    theta, phi = np.broadcast_arrays(
        np.asarray(theta, dtype=float), np.asarray(phi, dtype=float)
    )
    if not np.all(np.isfinite(theta)) or np.any((theta < 0) | (theta > pi)):
        raise ValueError("theta must be finite and lie in [0, pi]")
    if not np.all(np.isfinite(phi)):
        raise ValueError("phi must be finite")
    values = np.zeros(theta.shape, dtype=complex)
    for (ell, emm), coefficient in modes.items():
        if coefficient != 0:
            values += coefficient * _harmonic(ell, emm, theta, phi)
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite supertranslation field")
    return values.real


def plot_supertranslation(
    alpha,
    *,
    ax=None,
    n_longitude=361,
    n_latitude=181,
    cmap="RdBu_r",
    colorbar_label=r"$\alpha$",
):
    import matplotlib.pyplot as plt

    n_longitude, n_latitude = index(n_longitude), index(n_latitude)
    if n_longitude < 3 or n_latitude < 3:
        raise ValueError("angular grid dimensions must be at least 3")
    longitude = np.linspace(-pi, pi, n_longitude + 1)
    latitude = np.linspace(-pi / 2, pi / 2, n_latitude + 1)
    lon, lat = np.meshgrid(
        0.5 * (longitude[:-1] + longitude[1:]),
        0.5 * (latitude[:-1] + latitude[1:]),
    )
    values = evaluate_supertranslation(alpha, pi / 2 - lat, lon)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.8), subplot_kw={"projection": "mollweide"})
    elif getattr(ax, "name", None) != "mollweide":
        raise ValueError("ax must use the Mollweide projection")
    bound = float(np.max(np.abs(values))) or 1.0
    image = ax.pcolormesh(
        longitude, latitude, values, shading="flat", cmap=cmap,
        vmin=-bound, vmax=bound, rasterized=True,
    )
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\pi/2-\theta$")
    colorbar = ax.figure.colorbar(image, ax=ax, orientation="horizontal", pad=0.12)
    colorbar.set_label(colorbar_label)
    colorbar.formatter.set_useMathText(True)
    colorbar.formatter.set_powerlimits((-3, 4))
    colorbar.update_ticks()
    return ax.figure, ax

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ds_wave_spectrum import direct_mode_power, make_frequency_modes
from ds_wave_target import interpolate_target_power


def make_centered_integer_axis(k_max: int, grid_size: int) -> np.ndarray:
	k_max = max(int(k_max), 1)
	grid_size = max(int(grid_size), 1)
	full_size = 2 * k_max + 1
	if grid_size >= full_size:
		return np.arange(-k_max, k_max + 1, dtype=np.float64)

	negative_count = grid_size // 2
	positive_count = grid_size - negative_count - 1
	negative = np.rint(np.linspace(-k_max, -1, negative_count)).astype(np.int32) if negative_count > 0 else np.empty(0, dtype=np.int32)
	positive = np.rint(np.linspace(1, k_max, positive_count)).astype(np.int32) if positive_count > 0 else np.empty(0, dtype=np.int32)
	axis = np.concatenate([negative, np.array([0], dtype=np.int32), positive])
	if len(np.unique(axis)) != len(axis):
		axis = np.unique(axis)
		if 0 not in axis:
			axis = np.sort(np.concatenate([axis, np.array([0], dtype=np.int32)]))
	return axis.astype(np.float64)

def compute_periodogram_2d(points: np.ndarray, grid_size: int = 96, max_freq: float = 3.0) -> tuple[np.ndarray, tuple[float, float, float, float]]:
	# Equation 2 diagnostic on integer Fourier modes, displayed as k / sqrt(N).
	points = np.asarray(points, dtype=np.float64)
	n_points = points.shape[0]
	if n_points == 0:
		raise ValueError("points must not be empty")
	k_max = max(1, int(math.ceil(max_freq * math.sqrt(float(n_points)))))
	k_values = make_centered_integer_axis(k_max, grid_size)
	kx, ky = np.meshgrid(k_values, k_values, indexing="xy")
	modes = np.stack([kx.ravel(), ky.ravel()], axis=1)
	power = direct_mode_power(points, modes).reshape(kx.shape)
	extent = (
		float(k_values[0] / math.sqrt(float(n_points))),
		float(k_values[-1] / math.sqrt(float(n_points))),
		float(k_values[0] / math.sqrt(float(n_points))),
		float(k_values[-1] / math.sqrt(float(n_points))),
	)
	return power, extent

def compute_radial_psd(
	points: np.ndarray,
	num_bins: int = 80,
	max_freq: float = 3.0,
	mode_limit: int = 8192,
	priority_nu: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	# Paper evaluation averages Equation 2 powers radially for 1D spectra.
	points = np.asarray(points, dtype=np.float64)
	modes = make_frequency_modes(points.shape[0], nu_max=max_freq, mode_limit=mode_limit, seed=0, priority_nu=priority_nu)
	radii = np.linalg.norm(modes, axis=1) / math.sqrt(float(points.shape[0]))
	powers = direct_mode_power(points, modes)
	edges = np.linspace(0.0, max_freq, num_bins + 1)
	centres = 0.5 * (edges[:-1] + edges[1:])
	radial = np.full(num_bins, np.nan, dtype=np.float64)
	for index in range(num_bins):
		mask = (radii >= edges[index]) & (radii < edges[index + 1])
		if np.any(mask):
			radial[index] = np.mean(powers[mask])
	valid = np.isfinite(radial)
	if np.any(valid):
		radial = np.interp(centres, centres[valid], radial[valid], left=radial[valid][0], right=radial[valid][-1])
	else:
		radial[:] = 0.0
	return centres, radial

def compute_low_frequency_mode_powers(
	points: np.ndarray,
	nu_max: float,
	mode_limit: int = 0,
	seed: int = 0,
) -> dict:
	"""Return exact low-frequency Fourier powers sorted from largest to smallest.

	This diagnostic checks whether the low-frequency hole is genuinely leaking,
	instead of relying on a downsampled periodogram image.
	"""
	points = np.asarray(points, dtype=np.float64)
	if points.ndim != 2 or points.shape[0] < 1:
		raise ValueError("points must have shape (n_points, n_dims) with at least one point")
	if nu_max <= 0.0:
		raise ValueError("nu_max must be positive")

	modes = make_frequency_modes(points.shape[0], nu_max=nu_max, mode_limit=mode_limit, seed=seed)
	if modes.shape[0] == 0:
		raise ValueError("no Fourier modes available for the requested n_points and nu_max")
	radii = np.linalg.norm(modes, axis=1) / math.sqrt(float(points.shape[0]))
	powers = direct_mode_power(points, modes)
	order = np.argsort(powers)[::-1]
	modes = modes[order]
	radii = radii[order]
	powers = powers[order]
	return {
		"modes": modes,
		"radii": radii,
		"powers": powers,
		"mean_power": float(np.mean(powers)),
		"median_power": float(np.median(powers)),
		"max_power": float(np.max(powers)),
		"mode_count": int(modes.shape[0]),
	}

def compute_target_mode_powers(
	target: dict,
	n_points: int,
	nu_max: float,
	mode_limit: int = 0,
	seed: int = 0,
) -> dict:
	"""Return target powers at exact integer Fourier mode radii below nu_max."""
	if n_points < 1:
		raise ValueError("n_points must be at least 1")
	if nu_max <= 0.0:
		raise ValueError("nu_max must be positive")

	modes = make_frequency_modes(n_points, nu_max=nu_max, mode_limit=mode_limit, seed=seed)
	if modes.shape[0] == 0:
		raise ValueError("no Fourier modes available for the requested n_points and nu_max")
	radii = np.linalg.norm(modes, axis=1) / math.sqrt(float(n_points))
	mask = radii < nu_max
	modes = modes[mask]
	radii = radii[mask]
	if modes.shape[0] == 0:
		raise ValueError("no Fourier modes below the requested nu_max")

	powers = interpolate_target_power(target, radii)
	order = np.argsort(powers)[::-1]
	modes = modes[order]
	radii = radii[order]
	powers = powers[order]
	return {
		"modes": modes,
		"radii": radii,
		"powers": powers,
		"mean_power": float(np.mean(powers)),
		"median_power": float(np.median(powers)),
		"max_power": float(np.max(powers)),
		"mode_count": int(modes.shape[0]),
	}

def summarise_mode_power_bands(report: dict, bands: list[tuple[float, float]]) -> list[dict]:
	"""Summarise mode-power diagnostics over radial frequency bands."""
	radii = np.asarray(report["radii"], dtype=np.float64)
	powers = np.asarray(report["powers"], dtype=np.float64)
	if radii.shape != powers.shape:
		raise ValueError("report radii and powers must have the same shape")

	summaries = []
	for low, high in bands:
		if high <= low:
			raise ValueError("band high value must be greater than low value")
		mask = (radii >= low) & (radii < high)
		if np.any(mask):
			mean_power = float(np.mean(powers[mask]))
			median_power = float(np.median(powers[mask]))
			max_power = float(np.max(powers[mask]))
		else:
			mean_power = 0.0
			median_power = 0.0
			max_power = 0.0
		summaries.append({
			"range": (float(low), float(high)),
			"count": int(np.sum(mask)),
			"mean_power": mean_power,
			"median_power": median_power,
			"max_power": max_power,
		})
	return summaries

def compute_empirical_pcf(points: np.ndarray, num_bins: int = 80, r_max: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
	# Equation 1 diagnostic: normalized pair-distance histogram in toroidal coordinates.
	points = np.asarray(points, dtype=np.float64)
	n_points = points.shape[0]
	if n_points < 2:
		centres = np.linspace(0.0, r_max, num_bins, endpoint=False)
		return centres, np.zeros_like(centres)

	distances = []
	for index in range(n_points - 1):
		delta = np.abs(points[index + 1:] - points[index])
		delta = np.minimum(delta, 1.0 - delta)
		distances.append(np.linalg.norm(delta, axis=1) * math.sqrt(float(n_points)))
	distances = np.concatenate(distances, axis=0)
	edges = np.linspace(0.0, r_max, num_bins + 1)
	counts, _ = np.histogram(distances, bins=edges)
	annulus_area = math.pi * ((edges[1:] / math.sqrt(float(n_points))) ** 2 - (edges[:-1] / math.sqrt(float(n_points))) ** 2)
	expected = 0.5 * n_points * (n_points - 1) * annulus_area
	pcf = counts / np.maximum(expected, 1e-12)
	centres = 0.5 * (edges[:-1] + edges[1:])
	return centres, pcf

def save_figure(fig: plt.Figure, path: Path) -> Path:
	import matplotlib.pyplot as plt

	path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(path, dpi=160, bbox_inches="tight")
	plt.close(fig)
	return path

def plot_ds_wave_targets(targets: list[dict]) -> plt.Figure:
	import matplotlib.pyplot as plt

	# Figure 9 style diagnostic: target P(nu) and the implied Equation 12 PCF.
	fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
	for target in targets:
		label = f"m0={target['m0']}"
		if not target["success"]:
			continue
		axes[0].plot(target["nu"], target["P"], label=label)
		axes[1].plot(target["r"], target["g"], label=label)
	axes[0].set_title("DS-Wave target power")
	axes[0].set_xlabel("nu")
	axes[0].set_ylabel("P(nu)")
	axes[0].set_ylim(bottom=0.0)
	axes[0].grid(True, alpha=0.25)
	axes[1].set_title("Pair correlation target")
	axes[1].set_xlabel("r")
	axes[1].set_ylabel("g(r)")
	axes[1].set_ylim(bottom=0.0)
	axes[1].grid(True, alpha=0.25)
	axes[0].legend()
	axes[1].legend()
	return fig

def plot_points(points: np.ndarray, title: str = "DS-Wave points") -> plt.Figure:
	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(figsize=(5.5, 5.5))
	ax.scatter(points[:, 0], points[:, 1], s=5, color="#174a7c", alpha=0.85, linewidths=0)
	ax.set_title(title)
	ax.set_xlim(0.0, 1.0)
	ax.set_ylim(0.0, 1.0)
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.grid(True, alpha=0.2)
	return fig

def plot_periodogram(power: np.ndarray, extent: tuple[float, float, float, float], title: str = "2D periodogram") -> plt.Figure:
	import matplotlib.pyplot as plt

	# Paper figures show empirical 2D power spectra from Equation 2.
	display_power = np.array(power, dtype=np.float64, copy=True)
	center_y = display_power.shape[0] // 2
	center_x = display_power.shape[1] // 2
	display_power[center_y, center_x] = 0.0
	log_power = np.log1p(display_power)
	vmin, vmax = np.percentile(log_power, [2.0, 99.0])
	fig, ax = plt.subplots(figsize=(5.6, 5.2))
	image = ax.imshow(
		log_power,
		origin="lower",
		extent=extent,
		cmap="gray",
		vmin=vmin,
		vmax=vmax,
		interpolation="nearest",
	)
	ax.set_title(title)
	ax.set_xlabel("kx / sqrt(N)")
	ax.set_ylabel("ky / sqrt(N)")
	fig.colorbar(image, ax=ax, label="log1p power")
	return fig

def plot_radial_psd_overlay(points: np.ndarray, target: dict, max_freq: float = 3.0) -> plt.Figure:
	import matplotlib.pyplot as plt

	# Compare empirical radial Equation 2 power against the variational target P = F + 1.
	freqs, radial = compute_radial_psd(points, max_freq=max_freq, priority_nu=target.get("nu0"))
	target_power = interpolate_target_power(target, freqs)
	fig, ax = plt.subplots(figsize=(7.0, 4.2))
	ax.plot(freqs, radial, label="empirical", color="#174a7c", linewidth=2.0)
	ax.plot(freqs, target_power, label="target", color="#9a3412", linewidth=2.0)
	ax.set_title("Radial PSD")
	ax.set_xlabel("nu")
	ax.set_ylabel("power")
	ax.set_xlim(0.0, max_freq)
	ax.set_ylim(bottom=0.0)
	ax.grid(True, alpha=0.25)
	ax.legend()
	return fig

def plot_pcf_overlay(points: np.ndarray, target: dict) -> plt.Figure:
	import matplotlib.pyplot as plt

	# Compare empirical Equation 1 PCF against g = H[F] + 1 from Equation 12.
	r, empirical = compute_empirical_pcf(points, r_max=float(target["r"][-1]))
	target_g = np.interp(r, target["r"], target["g"], left=target["g"][0], right=target["g"][-1])
	fig, ax = plt.subplots(figsize=(7.0, 4.2))
	ax.plot(r, empirical, label="empirical", color="#174a7c", linewidth=2.0)
	ax.plot(r, target_g, label="target", color="#9a3412", linewidth=2.0)
	ax.set_title("Pair correlation")
	ax.set_xlabel("r")
	ax.set_ylabel("g(r)")
	ax.set_ylim(bottom=0.0)
	ax.grid(True, alpha=0.25)
	ax.legend()
	return fig

from __future__ import annotations

"""Fourier-mode measurement helpers for DS-Wave diagnostics.

Hosts the shared building blocks used for measurements and diagnostics
make_frequency_modes` (Equation 2 periodogram mode generation with the passband-preserving subsampling of Section 4.2 / Equation 14) and
direct_mode_power (Equation 2 empirical power for explicit mode vectors). These are consumed by ds_wave_diagnostics, ds_wave_psd_eval, and the tests.
"""

import math
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

import numpy as np


def make_frequency_modes(
	n_points: int,
	nu_max: float,
	mode_limit: int = 4096,
	seed: int = 0,
	priority_nu: float | None = None,
) -> np.ndarray:
	# Equation 2 periodogram modes in normalized coordinates: nu = |k| / sqrt(N).
	max_mode = max(1, int(math.ceil(nu_max * math.sqrt(float(n_points)))))
	values = np.arange(-max_mode, max_mode + 1, dtype=np.int32)
	kx, ky = np.meshgrid(values, values, indexing="xy")
	modes = np.stack([kx.ravel(), ky.ravel()], axis=1)
	nonzero = np.any(modes != 0, axis=1)
	modes = modes[nonzero]
	radius = np.linalg.norm(modes.astype(np.float32), axis=1) / math.sqrt(float(n_points))
	modes = modes[radius <= nu_max]
	radius = radius[radius <= nu_max]

	if mode_limit > 0 and modes.shape[0] > mode_limit:
		if priority_nu is not None:
			# Section 4.2 / Equation 14: preserve the low-frequency hole, but never discard the entire passband. When the priority disk alone fills the
			# budget, reserve a fraction for modes above priority_nu so the radial PSD still measures the passband (otherwise it reads ~0 there).
			priority_mask = radius <= priority_nu
			priority_modes = modes[priority_mask]
			priority_radius = radius[priority_mask]
			remaining_modes = modes[~priority_mask]
			rng = np.random.default_rng(seed)
			if priority_modes.shape[0] >= mode_limit:
				# Reserve up to 25% of the budget for an unbiased passband sample.
				passband_budget = min(remaining_modes.shape[0], mode_limit // 4)
				disk_budget = mode_limit - passband_budget
				order = np.argsort(priority_radius, kind="stable")
				kept_disk = priority_modes[order[:disk_budget]]
				if passband_budget > 0 and remaining_modes.shape[0] > 0:
					if remaining_modes.shape[0] > passband_budget:
						choice = rng.choice(remaining_modes.shape[0], size=passband_budget, replace=False)
						remaining_modes = remaining_modes[np.sort(choice)]
					return np.vstack([kept_disk, remaining_modes]).astype(np.float32)
				return kept_disk.astype(np.float32)

			remaining_count = mode_limit - priority_modes.shape[0]
			if remaining_modes.shape[0] > remaining_count:
				choice = rng.choice(remaining_modes.shape[0], size=remaining_count, replace=False)
				remaining_modes = remaining_modes[np.sort(choice)]
			return np.vstack([priority_modes, remaining_modes]).astype(np.float32)

		rng = np.random.default_rng(seed)
		choice = rng.choice(modes.shape[0], size=mode_limit, replace=False)
		modes = modes[np.sort(choice)]
	return modes.astype(np.float32)


def direct_mode_power(points: np.ndarray, modes: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
	# Equation 2 empirical power for explicit Fourier mode vectors.
	points = np.asarray(points, dtype=np.float32)
	modes = np.asarray(modes, dtype=np.float32)
	powers = []
	for start in range(0, modes.shape[0], chunk_size):
		chunk = modes[start:start + chunk_size]
		phase = 2.0 * math.pi * (points @ chunk.T)
		real = np.cos(phase).sum(axis=0)
		imag = np.sin(phase).sum(axis=0)
		powers.append((real * real + imag * imag) / float(points.shape[0]))
	return np.concatenate(powers, axis=0)



def default_zone_plate_alpha(n_pixels: int, nyquist_radius: float = 0.25) -> float:
	if n_pixels < 2:
		raise ValueError("n_pixels must be at least 2")
	if not (0.0 < nyquist_radius <= math.sqrt(2.0)):
		raise ValueError("nyquist_radius must be in (0, sqrt(2)]")
	return math.pi * (float(n_pixels) / 2.0) / float(nyquist_radius)


def zone_plate_value(coords: np.ndarray, alpha: float) -> np.ndarray:
	coords = np.asarray(coords, dtype=np.float32)
	r2 = coords[..., 0] ** 2 + coords[..., 1] ** 2
	return 0.5 + 0.5 * np.cos(alpha * r2)


def sample_zone_plate(points: np.ndarray, alpha: float) -> np.ndarray:
	points = np.asarray(points, dtype=np.float32)
	if points.ndim != 2 or points.shape[1] != 2:
		raise ValueError("points must have shape (n_points, 2)")
	return zone_plate_value(points, alpha)


def zone_plate_reference(n_pixels: int, alpha: float, supersample: int = 8, prefilter_sigma: float = 0.5) -> np.ndarray:
	if n_pixels < 2:
		raise ValueError("n_pixels must be at least 2")
	supersample = max(1, int(supersample))
	fine = n_pixels * supersample
	axis = (np.arange(fine, dtype=np.float32) + 0.5) / float(fine)  # corner origin at (0, 0)
	(xx, yy) = np.meshgrid(axis, axis)  # row index -> y, column index -> x
	hi = zone_plate_value(np.stack([xx, yy], axis=-1), alpha)
	if prefilter_sigma > 0.0:

		hi = gaussian_filter(hi, sigma=prefilter_sigma * supersample, mode="reflect")
	return hi.reshape(n_pixels, supersample, n_pixels, supersample).mean(axis=(1, 3))


def reconstruct_zone_plate(points, values, n_pixels: int, sigma: float = 0.8) -> np.ndarray:
	if n_pixels < 2:
		raise ValueError("n_pixels must be at least 2")
	points = np.asarray(points, dtype=np.float32) % 1.0
	values = np.asarray(values, dtype=np.float32)
	if points.shape[0] != values.shape[0]:
		raise ValueError("points and values must have matching length")
	col = np.clip((points[:, 0] * n_pixels).astype(np.int64), 0, n_pixels - 1)
	row = np.clip((points[:, 1] * n_pixels).astype(np.int64), 0, n_pixels - 1)
	numerator = np.zeros((n_pixels, n_pixels), dtype=np.float32)
	density = np.zeros((n_pixels, n_pixels), dtype=np.float32)
	np.add.at(numerator, (row, col), values)
	np.add.at(density, (row, col), 1.0)
	# Reflect (not wrap): the corner-origin chirp is discontinuous across the image
	# edges, so the reconstruction low-pass must not blend opposite edges together.
	numerator = gaussian_filter(numerator, sigma=sigma, mode="reflect")
	density = gaussian_filter(density, sigma=sigma, mode="reflect")
	return numerator / np.maximum(density, 1e-8)


def render_zone_plate(points, n_pixels: int = 128, alpha: float | None = None, sigma: float = 0.8, supersample: int = 8) -> dict:
	if alpha is None:
		alpha = default_zone_plate_alpha(n_pixels)
	points = np.asarray(points, dtype=np.float32)
	values = sample_zone_plate(points, alpha)
	reconstruction = reconstruct_zone_plate(points, values, n_pixels, sigma=sigma)
	reference = zone_plate_reference(n_pixels, alpha, supersample=supersample)
	return {
		"alpha": float(alpha),
		"reconstruction": reconstruction,
		"reference": reference,
		"error": reconstruction - reference,
		"n_pixels": int(n_pixels),
		"spp": float(points.shape[0]) / float(n_pixels * n_pixels),
	}


def plot_zone_plate(points, n_pixels: int = 128, alpha: float | None = None, sigma: float = 0.8, supersample: int = 8, label: str = ""):
	rendered = render_zone_plate(points, n_pixels=n_pixels, alpha=alpha, sigma=sigma, supersample=supersample)
	reference = rendered["reference"]
	reconstruction = rendered["reconstruction"]
	error = rendered["error"]

	# Single realization error power spectrum; shift DC to the centre.
	spectrum = np.fft.fftshift(np.fft.fft2(error))
	power = np.log1p(np.abs(spectrum) ** 2)
	nyquist = n_pixels / 2.0  # cycles per unit length

	suffix = f" ({label})" if label else ""
	
	# origin="upper" places the (0, 0) origin at the top-left
	(fig, axes) = plt.subplots(1, 4, figsize=(18.0, 4.7))
	axes[0].imshow(reference, origin="upper", cmap="gray")
	axes[0].set_title("Reference (anti-aliased)")
	axes[1].imshow(reconstruction, origin="upper", cmap="gray")
	axes[1].set_title(f"Reconstruction{suffix}")
	image = axes[2].imshow(error, origin="upper", cmap="RdBu_r")
	axes[2].set_title("Error (reconstruction - reference)")
	fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
	(vmin, vmax) = np.percentile(power, [2.0, 99.5])
	axes[3].imshow(power, origin="lower", cmap="magma", extent=(-nyquist, nyquist, -nyquist, nyquist), vmin=vmin, vmax=vmax)
	axes[3].add_patch(plt.Circle((0.0, 0.0), nyquist, fill=False, color="orange", linewidth=1.6))
	axes[3].set_title("Error power spectrum")
	axes[3].set_xlabel("frequency (cycles / unit)")
	for ax in axes[:3]:
		ax.set_xticks([])
		ax.set_yticks([])
	fig.suptitle(f"DS-Wave zone plate{suffix}: N={points.shape[0]}, " f"{n_pixels}x{n_pixels}, {rendered['spp']:.2f} spp, alpha={rendered['alpha']:.1f}")
	fig.tight_layout()
	return fig

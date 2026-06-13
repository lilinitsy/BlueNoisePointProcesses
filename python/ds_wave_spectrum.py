from __future__ import annotations

"""Fourier-mode measurement helpers for DS-Wave diagnostics.

Hosts the shared building blocks used by the measurement/diagnostics stack:
``make_frequency_modes`` (Equation 2 periodogram mode generation with the
passband-preserving subsampling of Section 4.2 / Equation 14) and
``direct_mode_power`` (Equation 2 empirical power for explicit mode vectors).
These are consumed by ds_wave_diagnostics, ds_wave_psd_eval, and the tests.
"""

import math

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
	radius = np.linalg.norm(modes.astype(np.float64), axis=1) / math.sqrt(float(n_points))
	modes = modes[radius <= nu_max]
	radius = radius[radius <= nu_max]

	if mode_limit > 0 and modes.shape[0] > mode_limit:
		if priority_nu is not None:
			# Section 4.2 / Equation 14: preserve the low-frequency hole, but never
			# discard the entire passband. When the priority disk alone fills the
			# budget, reserve a fraction for modes above priority_nu so the radial
			# PSD still measures the passband (otherwise it reads ~0 there).
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
					return np.vstack([kept_disk, remaining_modes]).astype(np.float64)
				return kept_disk.astype(np.float64)

			remaining_count = mode_limit - priority_modes.shape[0]
			if remaining_modes.shape[0] > remaining_count:
				choice = rng.choice(remaining_modes.shape[0], size=remaining_count, replace=False)
				remaining_modes = remaining_modes[np.sort(choice)]
			return np.vstack([priority_modes, remaining_modes]).astype(np.float64)

		rng = np.random.default_rng(seed)
		choice = rng.choice(modes.shape[0], size=mode_limit, replace=False)
		modes = modes[np.sort(choice)]
	return modes.astype(np.float64)


def direct_mode_power(points: np.ndarray, modes: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
	# Equation 2 empirical power for explicit Fourier mode vectors.
	points = np.asarray(points, dtype=np.float64)
	modes = np.asarray(modes, dtype=np.float64)
	powers = []
	for start in range(0, modes.shape[0], chunk_size):
		chunk = modes[start:start + chunk_size]
		phase = 2.0 * math.pi * (points @ chunk.T)
		real = np.cos(phase).sum(axis=0)
		imag = np.sin(phase).sum(axis=0)
		powers.append((real * real + imag * imag) / float(points.shape[0]))
	return np.concatenate(powers, axis=0)

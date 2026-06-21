from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

try:
	import torch
except ImportError:
	torch = None

python_dir = Path(__file__).resolve().parent
if str(python_dir) not in sys.path:
	sys.path.insert(0, str(python_dir))

from ds_wave_targetrdf import (
	SynthesisResult,
	choose_torch_device,
	resolve_targetrdf_resolution,
	make_targetrdf_target_curve,
	compute_targetrdf_force_curve,
	compute_targetrdf_energy,
	smooth_curve_gaussian,
	interpolate_curve_torch,
)


def grid_points(grid: int) -> np.ndarray:
	axis = (np.arange(grid, dtype=np.float64) + 0.5) / float(grid)
	(xx, yy) = np.meshgrid(axis, axis)
	return np.stack([xx.ravel(), yy.ravel()], axis=1)


def linear_gradient_field(low: float = 0.3, high: float = 1.7):
	def field(xy: np.ndarray) -> np.ndarray:
		xy = np.asarray(xy, dtype=np.float64)
		return low + (high - low) * xy[:, 0]
	return field


def radial_bump_field(center=(0.5, 0.5), sigma: float = 0.18, floor: float = 0.2):
	(cx, cy) = center
	def field(xy: np.ndarray) -> np.ndarray:
		xy = np.asarray(xy, dtype=np.float64)
		r2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
		return floor + np.exp(-r2 / (2.0 * sigma * sigma))
	return field


def normalize_density(field, n_points: int, grid: int = 256):
	if n_points < 2:
		raise ValueError("n_points must be at least 2")
	xs = grid_points(grid)
	vals = np.asarray(field(xs), dtype=np.float64)
	if np.any(vals <= 0.0):
		raise ValueError("density field must be strictly positive")
	integral = float(np.mean(vals))
	scale = float(n_points) / integral

	def lam(xy: np.ndarray) -> np.ndarray:
		return np.asarray(field(xy), dtype=np.float64) * scale

	return lam


def seed_points_proportional_to_density(lam, n_points: int, grid: int = 256, seed: int = 0) -> np.ndarray:
	rng = np.random.default_rng(seed)
	xs = grid_points(grid)
	weights = np.asarray(lam(xs), dtype=np.float64)
	weights = weights / np.sum(weights)
	cells = rng.choice(xs.shape[0], size=n_points, replace=True, p=weights)
	centers = xs[cells]
	jitter = (rng.random((n_points, 2)) - 0.5) / float(grid)
	return np.remainder(centers + jitter, 1.0)


def adaptive_rdf(points, q_pts, n_points: int, nbins: int, smoothing: float, chunk_size: int = 256) -> np.ndarray:
	if nbins < 2:
		raise ValueError("nbins must be at least 2")

	sqrt_n = math.sqrt(float(n_points))
	dx = 0.5 / float(nbins)
	counts = torch.zeros((nbins,), dtype=torch.float64, device=points.device)
	with torch.no_grad():
		all_indices = torch.arange(n_points, device=points.device)
		for start in range(0, n_points, chunk_size):
			end = min(start + chunk_size, n_points)

			row_indices = torch.arange(start, end, device=points.device)[:, None]
			delta = points[start:end, None, :] - points[None, :, :]

			# Wrap each displacement to the nearest toroidal image in [-0.5, 0.5].
			delta = delta - torch.round(delta)
			distances = torch.sqrt(torch.sum(delta * delta, dim=2))

			# Warp distances: w_ij = d_ij * q_i * q_j / sqrt(N). q_i is shape (chunk,), q_j is shape (n_points,).
			qi = q_pts[start:end, None] # (chunk, 1)
			qj = q_pts[None, :] # (1, n_points)
			warped = distances * (qi * qj) / sqrt_n

			# Count each unordered pair once (j > i) with warped distance in [0, 0.5).
			mask = (all_indices[None, :] > row_indices) & (warped < 0.5)
			if torch.any(mask):
				bin_indices = torch.floor(warped[mask] / dx).to(dtype=torch.int64)
				bin_indices = torch.clamp(bin_indices, 0, nbins - 1)
				counts = counts + torch.bincount(bin_indices, minlength=nbins).to(dtype=torch.float64)

	# Same normalization as compute_targetrdf_rdf
	scale = float(n_points * (n_points - 1)) * 0.5 * math.pi * dx * dx
	shell_indices = torch.arange(nbins, dtype=torch.float64, device=points.device)
	rdf = counts / (scale * (2.0 * shell_indices + 1.0))
	return smooth_curve_gaussian(rdf.cpu().numpy(), smoothing)


def adaptive_gradients(points, q_pts, n_points: int, force_curve: np.ndarray, chunk_size: int = 256):
	if chunk_size < 1:
		raise ValueError("chunk_size must be positive")

	sqrt_n = math.sqrt(float(n_points))
	force = torch.as_tensor(force_curve, dtype=points.dtype, device=points.device)
	gradients = torch.zeros_like(points)
	with torch.no_grad():
		for start in range(0, n_points, chunk_size):
			end = min(start + chunk_size, n_points)

			# delta[chunk_i, j] is the wrapped vector i->j.
			delta = points[None, :, :] - points[start:end, None, :]
			delta = delta + (delta < -0.5).to(dtype=points.dtype) - (delta > 0.5).to(dtype=points.dtype)
			dist2 = torch.sum(delta * delta, dim=2)

			# Mask out self-pairs and distant pairs by physical distance (toroidal anisotropy cutoff).
			mask = (dist2 > 0.0) & (dist2 <= 0.49 * 0.49)
			distances = torch.sqrt(torch.clamp(dist2, min=1e-20))

			# Warp to get the distance used for force lookup.
			qi = q_pts[start:end, None] # (chunk, 1)
			qj = q_pts[None, :] # (1, n_points)
			warped = distances * (qi * qj) / sqrt_n

			# Look up force at warped distance and zero out masked pairs.
			pair_force = interpolate_curve_torch(force, warped)
			pair_force = torch.where(mask, pair_force, torch.zeros_like(pair_force))

			# Physical delta gives the 2D update direction.
			gradients[start:end] = -torch.sum(delta * pair_force[:, :, None], dim=1)
	max_gradient = torch.sqrt(torch.max(torch.sum(gradients * gradients, dim=1))).item()
	return (gradients, max_gradient)


def synthesize_adaptive_points(
	target,
	lam,
	n_points: int = 1024,
	iterations: int = 100,
	seed: int = 0,
	device: str | None = "auto",
	step_scale: float = 1.0,
	nbins: int | None = None,
	smoothing: float | None = None,
	chunk_size: int = 256,
	initial_points: np.ndarray | None = None,
	log_every: int | None = None,
) -> SynthesisResult:
	if n_points < 2:
		raise ValueError("n_points must be at least 2 for PCF matching")
	if iterations < 0:
		raise ValueError("iterations must be non-negative")
	if nbins is not None and nbins < 2:
		raise ValueError("nbins must be at least 2")
	if smoothing is not None and smoothing < 0.0:
		raise ValueError("smoothing must be non-negative")
	if step_scale <= 0.0:
		raise ValueError("step_scale must be positive")
	if chunk_size < 1:
		raise ValueError("chunk_size must be positive")
	if torch is None:
		raise RuntimeError("torch is required for point synthesis")

	torch_device = choose_torch_device(device)
	torch.manual_seed(seed)
	if torch_device.type == "cuda":
		torch.cuda.manual_seed_all(seed)

	(nbins, smoothing) = resolve_targetrdf_resolution(n_points, nbins, smoothing)
	torch_dtype = torch.float32

	if initial_points is None:
		init_np = seed_points_proportional_to_density(lam, n_points, seed=seed)
		current = torch.as_tensor(init_np, dtype=torch_dtype, device=torch_device)
	else:
		initial_points = np.asarray(initial_points, dtype=np.float32)
		if initial_points.shape != (n_points, 2):
			raise ValueError("initial_points must have shape (n_points, 2)")
		if not np.all(np.isfinite(initial_points)):
			raise ValueError("initial_points must be finite")
		current = torch.as_tensor(initial_points, dtype=torch_dtype, device=torch_device).clone()
		current.remainder_(1.0)

	# Create target RDF (same as uniform), then evaluates DS-Wave g at unit_r * sqrt(N) = (warped radius) w_ij * sqrt(N)
	(target_rdf_np, target_pcf_np, unit_r_np) = make_targetrdf_target_curve(target, n_points, nbins, smoothing)

	# Compute per-point density weights q_i = lam_i^0.25 on the torch device.
	def _compute_q(pts_tensor):
		pts_np = pts_tensor.detach().cpu().numpy().astype(np.float64)
		lam_vals = np.asarray(lam(pts_np), dtype=np.float32)
		lam_vals = np.maximum(lam_vals, 1e-20)
		q_vals = lam_vals ** 0.25
		return torch.as_tensor(q_vals, dtype=torch_dtype, device=torch_device)

	q_pts = _compute_q(current)
	rdf_np = adaptive_rdf(current, q_pts, n_points, nbins, smoothing=smoothing, chunk_size=chunk_size)
	energy = compute_targetrdf_energy(rdf_np, target_rdf_np)

	best = current.clone()
	best_q = q_pts.clone()
	best_rdf_np = rdf_np.copy()
	best_energy = energy
	attempts = 0
	current_step_scale = float(step_scale)
	energy_history = []
	log_cadence = 1 if log_every is None else max(log_every, 1)

	for iteration in range(iterations):
		force_curve_np = compute_targetrdf_force_curve(rdf_np, target_rdf_np, n_points)
		(gradients, max_gradient) = adaptive_gradients(current, q_pts, n_points, force_curve_np, chunk_size=chunk_size)
		if max_gradient > 1e-12:
			step_size = current_step_scale / (math.sqrt(float(n_points)) * max_gradient)
			current = torch.remainder(current + float(step_size) * gradients, 1.0)

			# Recompute density weights at new positions.
			q_pts = _compute_q(current)
			rdf_np = adaptive_rdf(current, q_pts, n_points, nbins, smoothing=smoothing, chunk_size=chunk_size)
			energy = compute_targetrdf_energy(rdf_np, target_rdf_np)
			attempts += 1

			if energy < best_energy:
				best = current.clone()
				best_q = q_pts.clone()
				best_rdf_np = rdf_np.copy()
				best_energy = energy
				attempts = 0
			elif energy > best_energy * 1.2:
				current = best.clone()
				q_pts = best_q.clone()
				rdf_np = best_rdf_np.copy()
				energy = best_energy
				attempts = 5
			if attempts >= 5:
				attempts = 0
				current_step_scale *= 0.9
		if iteration % log_cadence == 0 or iteration == iterations - 1:
			energy_history.append(best_energy)

	if iterations == 0:
		energy_history.append(best_energy)

	return SynthesisResult(
		points=best.detach().cpu().numpy(),
		energy_history=np.array(energy_history, dtype=np.float32),
		r_values=unit_r_np * math.sqrt(float(n_points)),
		target_pcf=target_pcf_np,
		target_rdf=target_rdf_np,
		final_rdf=best_rdf_np,
		iterations_run=iterations,
		nbins=nbins,
		smoothing=smoothing,
		final_step_scale=current_step_scale,
		device=str(torch_device),
	)


def measure_density(points: np.ndarray, grid: int = 128, sigma: float | None = None) -> np.ndarray:
	from scipy.ndimage import gaussian_filter

	points = np.asarray(points, dtype=np.float64)
	if points.ndim != 2 or points.shape[1] != 2:
		raise ValueError("points must be shape (N, 2)")
	n = points.shape[0]
	if sigma is None:
		sigma = max(1.5, float(grid) / math.sqrt(float(max(n, 1))))

	count = np.zeros((grid, grid), dtype=np.float64)
	col = np.clip((points[:, 0] * grid).astype(int), 0, grid - 1)
	row = np.clip((points[:, 1] * grid).astype(int), 0, grid - 1)
	np.add.at(count, (row, col), 1.0)

	# Gaussian blur with toroidal boundary to avoid edge artifacts.
	blurred = gaussian_filter(count, sigma=sigma, mode="wrap")

	# Rescale so mean == N (intensity, not probability density).
	mean_val = float(np.mean(blurred))
	if mean_val > 0.0:
		blurred = blurred * (float(n) / mean_val)
	return blurred


def density_match_error(points: np.ndarray, lam, grid: int = 128) -> float:
	points = np.asarray(points, dtype=np.float64)
	measured = measure_density(points, grid=grid)

	xs = grid_points(grid)
	target_vals = np.asarray(lam(xs), dtype=np.float64).reshape(grid, grid)

	diff_sq = float(np.mean((measured - target_vals) ** 2))
	target_sq = float(np.mean(target_vals ** 2))
	if target_sq == 0.0:
		return 0.0 if diff_sq == 0.0 else float("inf")
	return math.sqrt(diff_sq / target_sq)


def plot_adaptive_result(points: np.ndarray, lam, target):
	import matplotlib
	import matplotlib.pyplot as plt
	from ds_wave_diagnostics import compute_periodogram_2d, compute_radial_psd

	g = 256
	(fig, axes) = plt.subplots(2, 2, figsize=(12, 10))

	ax_a = axes[0, 0]
	xs = grid_points(g)
	density_map = np.asarray(lam(xs), dtype=np.float64).reshape(g, g)
	im_a = ax_a.imshow(density_map, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
	fig.colorbar(im_a, ax=ax_a)
	ax_a.set_title("Target density")
	ax_a.set_xlabel("x")
	ax_a.set_ylabel("y")

	ax_b = axes[0, 1]
	ax_b.scatter(points[:, 0], points[:, 1], s=2, color="#174a7c", linewidths=0)
	ax_b.set_aspect("equal", adjustable="box")
	ax_b.set_xlim(0, 1)
	ax_b.set_ylim(0, 1)
	ax_b.set_title(f"Adaptive DS-Wave points (N={len(points)})")
	ax_b.set_xlabel("x")
	ax_b.set_ylabel("y")

	ax_c = axes[1, 0]
	max_freq = float(getattr(target, "nu0", 1.0)) * 3.0
	(power, extent) = compute_periodogram_2d(points, max_freq=max_freq)
	display_power = np.array(power, dtype=np.float64, copy=True)
	cy = display_power.shape[0] // 2
	cx = display_power.shape[1] // 2
	display_power[cy, cx] = 0.0  # blank DC
	log_power = np.log1p(display_power)
	vmin = float(np.percentile(log_power, 2.0))
	vmax = float(np.percentile(log_power, 99.0))
	ax_c.imshow(
		log_power,
		origin="lower",
		extent=extent,
		cmap="gray",
		vmin=vmin,
		vmax=vmax,
		interpolation="nearest",
	)
	ax_c.set_title("Empirical PSD (2D periodogram)")
	ax_c.set_xlabel("kx / sqrt(N)")
	ax_c.set_ylabel("ky / sqrt(N)")

	ax_d = axes[1, 1]
	d_max = min(5.0, float(target.nu[-1]))
	ax_d.plot(target.nu, target.P, color="#9a3412", linewidth=2.0, label="target P(nu)")
	(g_freq, g_psd) = compute_radial_psd(points, max_freq=d_max, priority_nu=target.nu0)
	ax_d.plot(g_freq, g_psd, color="#174a7c", linewidth=1.5, label="empirical (global)")
	w = 0.3
	lo = 0.5 - 0.5 * w
	window_mask = (
		(points[:, 0] >= lo) & (points[:, 0] < lo + w)
		& (points[:, 1] >= lo) & (points[:, 1] < lo + w)
	)
	window_pts = (points[window_mask] - lo) / w
	if window_pts.shape[0] >= 16:
		(w_freq, w_psd) = compute_radial_psd(window_pts, max_freq=d_max, priority_nu=target.nu0)
		ax_d.plot(w_freq, w_psd, color="#0F6E56", linewidth=1.5, label="empirical (window ρ≈const)")
	ax_d.axvspan(0, target.nu0, alpha=0.15, color="steelblue", label=f"hole [0, {target.nu0})")
	ax_d.axhline(1.0, color="0.55", linestyle=":", linewidth=1.0)
	ax_d.set_xlim(0, d_max)
	target_peak = float(np.nanmax(target.P)) if target.P is not None else 2.0
	ax_d.set_ylim(0, max(2.5, 1.4 * target_peak))
	ax_d.set_title("DS-Wave spectrum: target vs empirical 1D PSD")
	ax_d.set_xlabel("nu")
	ax_d.set_ylabel("P(nu)")
	ax_d.legend(fontsize=8)
	ax_d.grid(True, alpha=0.25)

	fig.suptitle("Adaptive Density DS-Wave")
	fig.tight_layout()
	return fig


def local_pcf_in_window(
	points: np.ndarray,
	lam,
	center: tuple[float, float],
	window: float,
	target=None,
	nbins: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
	points = np.asarray(points, dtype=np.float64)
	n_global = points.shape[0]

	cx, cy = center
	half = window * 0.5
	x_lo, x_hi = cx - half, cx + half
	y_lo, y_hi = cy - half, cy + half
	mask = ((points[:, 0] >= x_lo) & (points[:, 0] < x_hi) & (points[:, 1] >= y_lo) & (points[:, 1] < y_hi))
	win_pts = points[mask]
	n_win = win_pts.shape[0]

	lam_win = np.asarray(lam(win_pts), dtype=np.float64)
	lam_win = np.maximum(lam_win, 1e-20)

	# Per-point weight q_i = lam_i^0.25; warp = d * qi * qj / sqrt(N_global), Normalized radius r_ij = warp * sqrt(N_global) = d * qi * qj.
	q_win = lam_win ** 0.25
	sqrt_n = math.sqrt(float(n_global))

	# Bin range in warped units [0, 0.5] (same as the global synthesizer).
	dx = 0.5 / float(nbins)
	counts = np.zeros(nbins, dtype=np.float64)

	if n_win >= 2:
		# Compute all pairwise toroidal displacements (upper-triangle pairs only).
		idx_i, idx_j = np.triu_indices(n_win, k=1)
		delta = win_pts[idx_i] - win_pts[idx_j] # (n_pairs, 2)
		delta = delta - np.round(delta)
		dist = np.sqrt(np.sum(delta * delta, axis=1)) # (n_pairs,)

		warped = dist * q_win[idx_i] * q_win[idx_j] / sqrt_n # (n_pairs,)

		in_range = warped < 0.5
		bin_indices = np.floor(warped[in_range] / dx).astype(int)
		bin_indices = np.clip(bin_indices, 0, nbins - 1)
		np.add.at(counts, bin_indices, 1.0)

	norm_n = float(n_win) if n_win >= 2 else 1.0
	win_area = float(window) * float(window)
	scale = norm_n * (norm_n - 1.0) * 0.5 * math.pi * dx * dx
	shell_indices = np.arange(nbins, dtype=np.float64)
	denominator = scale * (2.0 * shell_indices + 1.0)
	denominator = np.where(denominator > 0.0, denominator, 1.0)
	g_raw = counts * win_area / denominator
	g_smooth = smooth_curve_gaussian(g_raw.astype(np.float32), sigma=1.0).astype(np.float64)
	bin_centers = (np.arange(nbins, dtype=np.float64) + 0.5) * dx
	r = bin_centers * sqrt_n

	return (r, g_smooth)

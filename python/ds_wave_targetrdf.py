from __future__ import annotations

"""TargetRDF point synthesis for DS-Wave targets.

DS-Wave itself solves for desired second-order statistics: a radial power
spectrum P(nu), shifted spectrum F(nu) = P(nu) - 1, and the corresponding pair
correlation function g(r). This module turns that target into an actual point
set by matching the radial distribution function (RDF), following the
Heck/Schlomer/Deussen TargetRDF optimizer used by the DS-Wave paper.

Coordinate systems used here:
- Unit-square coordinates: point positions and toroidal distances in [0, 1).
  TargetRDF only uses pair distances in [0, 0.5].
- Normalized DS-Wave radius: unit-square distance multiplied by sqrt(N). This
  is the radius used by the DS-Wave target spectrum/PCF.
- RDF bins: a 1D sampled curve over unit-square distances [0, 0.5).
"""

import math

import numpy as np

from ds_wave_target import evaluate_target_pcf, interpolate_target_pcf


try:
	import torch
except ImportError:
	torch = None


def choose_torch_device(device: str | None = None):
	"""Pick a usable torch device.

	`device="auto"` prefers CUDA, but the probe allocation catches cases where
	CUDA is installed but cannot actually execute on the current GPU. Returning
	CPU here is better than letting a long synthesis run fail after setup.
	"""
	if torch is None:
		raise RuntimeError("torch is required for point synthesis")
	if device is None or device == "auto":
		if torch.cuda.is_available():
			candidate = torch.device("cuda")
			try:
				probe = torch.empty((1,), device=candidate)
				probe.fill_(0.0)
				torch.cuda.synchronize()
				return candidate
			except RuntimeError:
				return torch.device("cpu")
		return torch.device("cpu")
	requested = torch.device(device)
	if requested.type == "cuda" and not torch.cuda.is_available():
		return torch.device("cpu")
	if requested.type == "cuda":
		try:
			probe = torch.empty((1,), device=requested)
			probe.fill_(0.0)
			torch.cuda.synchronize()
		except RuntimeError:
			return torch.device("cpu")
	return requested

def smooth_curve_gaussian(values: np.ndarray, sigma: float) -> np.ndarray:
	"""Smooth a 1D curve with a Gaussian measured in bins.

	TargetRDF smooths RDF curves before comparing them. `sigma` is not a
	physical distance; it is a number of histogram bins. For example, sigma=8
	means each bin averages nearby bins with an 8-bin Gaussian scale.
	"""
	values = np.asarray(values, dtype=np.float64)
	if sigma <= 0.0:
		return values.copy()

	smoothed = np.empty_like(values)
	for index in range(values.shape[0]):
		# Use a finite 5-sigma window so this remains O(n * sigma) instead of O(n^2).
		j_min = max(0, int(math.floor(index - 5.0 * sigma)))
		j_max = min(values.shape[0] - 1, int(math.ceil(index + 5.0 * sigma)))
		indices = np.arange(j_min, j_max + 1, dtype=np.float64)
		weights = np.exp(-((indices - float(index)) ** 2) / (2.0 * sigma * sigma))
		smoothed[index] = np.sum(values[j_min:j_max + 1] * weights) / np.sum(weights)
	return smoothed

def make_targetrdf_target_curve(target: dict, n_points: int, nbins: int, smoothing: float) -> tuple[np.ndarray, np.ndarray]:
	"""Convert the solved DS-Wave spectrum into a TargetRDF curve.

	The DS-Wave solver stores its PCF samples on `target["r"]`, but that grid may
	not cover the whole TargetRDF unit-square distance range once multiplied by
	sqrt(N). To avoid clamping to the last stored g(r) value, we evaluate g(r)
	directly from F(nu) at the exact radii TargetRDF needs.

	Shapes:
	- target_rdf: (nbins,), desired g(r) values.
	- unit_r: (nbins,), toroidal distances in unit-square coordinates.
	"""
	# Heck et al. targetrdf works in unit-square distances x in [0, 0.5].
	dx = 0.5 / float(nbins)
	# Bin locations match targetrdf.cc Curve::ToX(index): x0 + index * dx.
	unit_r = np.arange(nbins, dtype=np.float64) * dx
	normalised_r = unit_r * math.sqrt(float(n_points))
	# Evaluate from the spectrum so large unit-square distances do not clamp to target["g"][-1].
	target_g = np.maximum(evaluate_target_pcf(target, normalised_r), 0.0)
	return smooth_curve_gaussian(target_g, smoothing), unit_r

def compute_targetrdf_force_curve(rdf: np.ndarray, target_rdf: np.ndarray, n_points: int) -> np.ndarray:
	"""Build the radial force curve from current-vs-target RDF error.

	Heck et al. 2013 targetrdf.cc, CalcGradients(): cumulative RDF error divided by N*r^2.

	The result is a scalar lookup table. It is not a 2D vector yet:
	`force_curve[k]` says how strongly pairs at distance about `k * dx` should
	push/pull. `compute_targetrdf_gradients()` later multiplies this scalar by
	the actual wrapped pair direction to get a 2D update.
	"""
	rdf = np.asarray(rdf, dtype=np.float64)
	target_rdf = np.asarray(target_rdf, dtype=np.float64)
	if rdf.shape != target_rdf.shape:
		raise ValueError("rdf and target_rdf must have the same shape")
	if rdf.ndim != 1 or rdf.shape[0] < 2:
		raise ValueError("rdf must be a one-dimensional array with at least two samples")
	if n_points < 2:
		raise ValueError("n_points must be at least 2")

	dx = 0.5 / float(rdf.shape[0])
	# Start with per-bin RDF error. Multiplying by dx turns the discrete bin differences into a simple integral contribution over distance.
	force = dx * (rdf - target_rdf)
	# Integrate error from radius 0 outwards. Positive means "too many pairs up to this radius"; negative means "too few pairs up to this radius".
	force = np.cumsum(force)
	# Radius 0 would divide by x^2 below, so keep the zero-distance force finite.
	force[0] = 0.0
	for index in range(1, force.shape[0]):
		x = index * dx
		# TargetRDF normalizes by N and r^2 before using this as a scalar lookup: distance -> pair-force strength. The 2D direction comes later.
		force[index] /= float(n_points) * x * x
	return force

def compute_targetrdf_energy(rdf: np.ndarray, target_rdf: np.ndarray) -> float:
	"""Return the L2 distance between two RDF curves over [0, 0.5].

	This is the optimizer's acceptance metric. Lower means the current point
	set's measured pair-distance statistics are closer to the desired target.
	"""
	diff = np.asarray(rdf, dtype=np.float64) - np.asarray(target_rdf, dtype=np.float64)
	dx = 0.5 / float(diff.shape[0])
	return float(math.sqrt(dx * np.sum(diff * diff)))

def compute_targetrdf_rdf(points, nbins: int, smoothing: float = 8.0, chunk_size: int = 256) -> np.ndarray:
	"""Estimate the toroidal radial distribution function for a point set.

	The RDF measures how many point pairs occur at each distance relative to a
	random uniform point set. Values near 0 mean pairs at that distance are
	suppressed. Values near 1 mean random-like. Values above 1 mean more pairs
	than random at that radius.

	Shapes:
	- points: (n_points, 2), in [0, 1).
	- counts/rdf: (nbins,), radial bins for distances in [0, 0.5).
	"""
	if nbins < 2:
		raise ValueError("nbins must be at least 2")
	if chunk_size < 1:
		raise ValueError("chunk_size must be positive")

	n_points = points.shape[0]
	dx = 0.5 / float(nbins)
	counts = torch.zeros((nbins,), dtype=torch.float64, device=points.device)
	with torch.no_grad():
		all_indices = torch.arange(n_points, device=points.device)
		for start in range(0, n_points, chunk_size):
			end = min(start + chunk_size, n_points)
			# Pairwise chunk against all points: delta/distances have shape (end - start, n_points).
			row_indices = torch.arange(start, end, device=points.device)[:, None]
			delta = points[start:end, None, :] - points[None, :, :]
			# Wrap each displacement to the nearest toroidal image in [-0.5, 0.5].
			delta = delta - torch.round(delta)
			distances = torch.sqrt(torch.sum(delta * delta, dim=2))
			# Count each unordered pair once, matching targetrdf.cc CalcRDF().
			mask = (all_indices[None, :] > row_indices) & (distances < 0.5)
			if torch.any(mask):
				bin_indices = torch.floor(distances[mask] / dx).to(dtype=torch.int64)
				bin_indices = torch.clamp(bin_indices, 0, nbins - 1)
				counts = counts + torch.bincount(bin_indices, minlength=nbins).to(dtype=torch.float64)

	scale = float(n_points * (n_points - 1)) * 0.5 * math.pi * dx * dx
	shell_indices = torch.arange(nbins, dtype=torch.float64, device=points.device)
	# Divide pair counts by expected annulus area and number of unordered pairs. A random/uniform point set should therefore have RDF close to 1.
	rdf = counts / (scale * (2.0 * shell_indices + 1.0))
	rdf_np = rdf.cpu().numpy()
	return smooth_curve_gaussian(rdf_np, smoothing)

def interpolate_curve_torch(curve, x_values, x1: float = 0.5):
	"""Linearly sample a 1D curve tensor at positions in [0, x1].

	`x_values` can be a matrix of pair distances. The returned tensor has the
	same shape as `x_values`, with each distance replaced by an interpolated
	curve value.
	"""
	size = curve.shape[0]
	dx = x1 / float(size)
	xx = torch.clamp(x_values / dx, min=0.0, max=float(size - 1))
	index = torch.floor(xx).to(dtype=torch.int64)
	next_index = torch.clamp(index + 1, max=size - 1)
	alpha = (xx - index.to(dtype=x_values.dtype)).to(dtype=curve.dtype)
	return (1.0 - alpha) * curve[index] + alpha * curve[next_index]

def compute_targetrdf_gradients(points, force_curve: np.ndarray, chunk_size: int = 256):
	"""Compute point updates induced by the radial force curve.

	For each point, this sums contributions from all other points within the
	reliable RDF radius. The force curve supplies a scalar based on pair
	distance; the wrapped displacement supplies the 2D direction.

	Shapes:
	- points: (n_points, 2).
	- force_curve: (nbins,).
	- gradients: (n_points, 2).
	"""
	if chunk_size < 1:
		raise ValueError("chunk_size must be positive")

	n_points = points.shape[0]
	force = torch.as_tensor(force_curve, dtype=points.dtype, device=points.device)
	gradients = torch.zeros_like(points)
	with torch.no_grad():
		for start in range(0, n_points, chunk_size):
			end = min(start + chunk_size, n_points)
			# delta[i, j] is the wrapped vector from chunk point i to point j.
			delta = points[None, :, :] - points[start:end, None, :]
			delta = delta + (delta < -0.5).to(dtype=points.dtype) - (delta > 0.5).to(dtype=points.dtype)
			dist2 = torch.sum(delta * delta, dim=2)
			# targetrdf ignores distances above 0.49 to avoid toroidal anisotropy near the boundary.
			mask = (dist2 > 0.0) & (dist2 <= 0.49 * 0.49)
			distances = torch.sqrt(torch.clamp(dist2, min=1e-20))
			pair_force = interpolate_curve_torch(force, distances)
			pair_force = torch.where(mask, pair_force, torch.zeros_like(pair_force))
			# pair_force is radial. Multiplying by the wrapped pair vector turns it into a 2D update direction, summed over neighbours for each point.
			gradients[start:end] = -torch.sum(delta * pair_force[:, :, None], dim=1)
	# The caller uses this to normalize the largest per-point movement.
	max_gradient = torch.sqrt(torch.max(torch.sum(gradients * gradients, dim=1))).item()
	return gradients, max_gradient

def synthesize_targetrdf_points(
	target: dict,
	n_points: int = 128,
	iterations: int = 100,
	seed: int = 0,
	device: str | None = "auto",
	step_scale: float = 1.0,
	nbins: int | None = None,
	smoothing: float = 8.0,
	chunk_size: int = 256,
	initial_points: np.ndarray | None = None,
	log_every: int | None = None,
) -> dict:
	"""Synthesize points whose RDF follows a solved DS-Wave target.

	This follows the Heck/Schlomer/Deussen TargetRDF force construction, while
	using a fixed iteration budget instead of stopping at a temperature floor.

	High-level loop:
	1. Build the target RDF from the DS-Wave target spectrum.
	2. Measure the current point set's RDF.
	3. Convert RDF error into a radial force lookup curve.
	4. Move points along pairwise forces.
	5. Keep the best RDF match and reduce step scale after failed attempts.
	"""
	if n_points < 2:
		raise ValueError("n_points must be at least 2 for PCF matching")
	if iterations < 0:
		raise ValueError("iterations must be non-negative")
	if nbins is not None and nbins < 2:
		raise ValueError("nbins must be at least 2")
	if smoothing < 0.0:
		raise ValueError("smoothing must be non-negative")
	if step_scale <= 0.0:
		raise ValueError("step_scale must be positive")
	if chunk_size < 1:
		raise ValueError("chunk_size must be positive")
	if torch is None:
		raise RuntimeError("torch is required for point synthesis")

	torch_device = choose_torch_device(device)
	# Torch owns the random initial point set, so seed torch rather than NumPy.
	torch.manual_seed(seed)
	if torch_device.type == "cuda":
		torch.cuda.manual_seed_all(seed)

	nbins = int(n_points if nbins is None else nbins)
	torch_dtype = torch.float32
	if initial_points is None:
		# Start from a random toroidal point set in the unit square.
		current = torch.rand((n_points, 2), dtype=torch_dtype, device=torch_device)
	else:
		initial_points = np.asarray(initial_points, dtype=np.float32)
		if initial_points.shape != (n_points, 2):
			raise ValueError("initial_points must have shape (n_points, 2)")
		if not np.all(np.isfinite(initial_points)):
			raise ValueError("initial_points must be finite")
		current = torch.as_tensor(initial_points, dtype=torch_dtype, device=torch_device).clone()
		# Keep user-provided points in the same toroidal [0, 1) domain.
		current.remainder_(1.0)

	# target_rdf_np and rdf_np live on CPU as NumPy curves. Point updates live on torch tensors.
	target_rdf_np, unit_r_np = make_targetrdf_target_curve(target, n_points, nbins, smoothing)
	# target_pcf_np is returned for plotting/diagnostics. The optimizer itself
	# uses target_rdf_np, computed from the spectrum above.
	target_pcf_np = np.maximum(interpolate_target_pcf(target, unit_r_np * math.sqrt(float(n_points))), 0.0)
	rdf_np = compute_targetrdf_rdf(current, nbins, smoothing=smoothing, chunk_size=chunk_size)
	energy = compute_targetrdf_energy(rdf_np, target_rdf_np)

	# HSD13 tracks the best RDF match and rolls back unstable moves. Keeping this
	# state avoids spatial islanding when a proposed step worsens the RDF too much.
	best = current.clone()
	best_rdf_np = rdf_np.copy()
	best_energy = energy
	attempts = 0
	current_step_scale = float(step_scale)
	energy_history = []
	iterations_run = 0

	for iteration in range(iterations):
		# Recompute forces from the latest accepted/current RDF estimate.
		force_curve_np = compute_targetrdf_force_curve(rdf_np, target_rdf_np, n_points)
		gradients, max_gradient = compute_targetrdf_gradients(current, force_curve_np, chunk_size=chunk_size)
		if max_gradient > 1e-12:
			# Scale the largest point displacement to current_step_scale / sqrt(N).
			step_size = current_step_scale / (math.sqrt(float(n_points)) * max_gradient)
			current = torch.remainder(current + float(step_size) * gradients, 1.0)
			# A move changes all pair distances, so the RDF and energy must be
			# remeasured from the updated point set.
			rdf_np = compute_targetrdf_rdf(current, nbins, smoothing=smoothing, chunk_size=chunk_size)
			energy = compute_targetrdf_energy(rdf_np, target_rdf_np)
			attempts += 1
			if energy < best_energy:
				# Accept improvement and reset the failed-attempt counter.
				best = current.clone()
				best_rdf_np = rdf_np.copy()
				best_energy = energy
				attempts = 0
			elif energy > best_energy * 1.2:
				# Large regression: restore the best point set and force step decay.
				current = best.clone()
				rdf_np = best_rdf_np.copy()
				energy = best_energy
				attempts = 5
			if attempts >= 5:
				# Match targetrdf's temperature decay idea, but without ending early:
				# keep the requested iteration budget and make future moves smaller.
				attempts = 0
				current_step_scale *= 0.9
		iterations_run += 1
		if log_every is None or iteration % max(log_every, 1) == 0 or iteration == iterations - 1:
			energy_history.append(best_energy)

	if iterations == 0:
		energy_history.append(best_energy)

	return {
		"points": best.detach().cpu().numpy(),
		"energy_history": np.array(energy_history, dtype=np.float64),
		"r_values": unit_r_np * math.sqrt(float(n_points)),
		"target_pcf": target_pcf_np,
		"target_rdf": target_rdf_np,
		"final_rdf": best_rdf_np,
		"iterations_run": iterations_run,
		"pcf_algorithm": "targetrdf_force",
		"pcf_num_bins": nbins,
		"pcf_smoothing": smoothing,
		"pcf_step_scale": step_scale,
		"pcf_final_step_scale": current_step_scale,
		"device": str(torch_device),
	}


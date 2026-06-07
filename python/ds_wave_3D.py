from __future__ import annotations

"""Exploratory 3D DS-Wave target-spectrum solver.

This module intentionally does not synthesize 3D point sets. It only solves and
visualizes the 3D analogue of the DS-Wave target curves:

	P(nu) = F(nu) + 1
	g(r) = 1 + 4 pi integral F(nu) sinc(2 pi r nu) nu^2 dnu

The LP structure matches the 2D target solver, but the realizability transform
uses the 3D radial Fourier kernel instead of the 2D J0 kernel.
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

try:
	import torch
except ImportError:
	torch = None

python_dir = Path(__file__).resolve().parent
if str(python_dir) not in sys.path:
	sys.path.insert(0, str(python_dir))

from ds_wave_target import empty_target, linprog_status_name, trapezoid_weights
from ds_wave_targetrdf import (
	choose_torch_device,
	compute_targetrdf_energy,
	interpolate_curve_torch,
	smooth_curve_gaussian,
)
from ds_wave_spectrum import direct_mode_power


def sinc(x: np.ndarray | float) -> np.ndarray:
	"""Return sin(x) / x with sinc(0) = 1.

	NumPy's np.sinc uses sin(pi x) / (pi x), so we keep this explicit.
	"""
	x = np.asarray(x, dtype=np.float64)
	result = np.ones_like(x, dtype=np.float64)
	np.divide(np.sin(x), x, out=result, where=np.abs(x) >= 1e-12)
	return result

def make_hankel_matrix_3d(nu: np.ndarray, r: np.ndarray) -> np.ndarray:
	"""Build the discrete 3D radial Fourier/Hankel transform matrix.

	Shapes:
	- nu: (n_nu,), normalized radial frequencies.
	- r: (n_r,), normalized radial distances.
	- return: (n_r, n_nu), so H @ F evaluates f(r) = g(r) - 1.
	"""
	nu_weights = trapezoid_weights(nu) * nu * nu
	phase = 2.0 * math.pi * np.outer(r, nu)
	return 4.0 * math.pi * sinc(phase) * nu_weights[None, :]

def solve_ds_wave_target_3d(
	nu0: float = 0.85,
	e0: float = 0.0,
	m0: float | str | None = None,
	nu_max: float = 10.0,
	n_nu: int = 1001,
	r_max: float = 4.0,
	n_r: int = 128,
	tail_anchor_count: int = 1,
	m0_tol: float = 0.02,
	max_m0: float = 64.0,
	require_success: bool = True,
) -> dict:
	"""Solve the 3D DS-Wave target-spectrum LP.

	The only mathematical difference from the 2D target solve is the transform
	that maps shifted spectrum F(nu) to shifted PCF f(r).
	"""
	if m0 == "min":
		min_m0 = find_min_m0_3d(
			nu0=nu0,
			e0=e0,
			nu_max=nu_max,
			n_nu=n_nu,
			r_max=r_max,
			n_r=n_r,
			tail_anchor_count=tail_anchor_count,
			m0_tol=m0_tol,
			max_m0=max_m0,
		)
		if min_m0 is None:
			nu = np.linspace(0.0, nu_max, n_nu, dtype=np.float64)
			r = np.linspace(0.0, r_max, n_r, dtype=np.float64)
			H = make_hankel_matrix_3d(nu, r)
			low_mask = nu < nu0
			target = empty_target("infeasible", "No feasible finite m0 found.", nu, r, H, low_mask, m0, nu0=nu0, e0=e0)
			target["dimension"] = 3
			if require_success:
				raise RuntimeError(target["message"])
			return target
		return solve_ds_wave_target_3d(
			nu0=nu0,
			e0=e0,
			m0=min_m0,
			nu_max=nu_max,
			n_nu=n_nu,
			r_max=r_max,
			n_r=n_r,
			tail_anchor_count=tail_anchor_count,
			m0_tol=m0_tol,
			max_m0=max_m0,
			require_success=require_success,
		)

	if nu0 < 0.0:
		raise ValueError("nu0 must be non-negative")
	if e0 < 0.0:
		raise ValueError("e0 must be non-negative")
	if nu_max <= 0.0:
		raise ValueError("nu_max must be positive")
	if n_nu < 3:
		raise ValueError("n_nu must be at least 3")
	if n_r < 1:
		raise ValueError("n_r must be at least 1")
	if tail_anchor_count < 1 or tail_anchor_count >= n_nu:
		raise ValueError("tail_anchor_count must be in [1, n_nu)")
	if m0 is not None and float(m0) < 1.0:
		raise ValueError("m0 must be at least 1.0, None, or 'min'")
	if m0_tol <= 0.0:
		raise ValueError("m0_tol must be positive")

	nu = np.linspace(0.0, nu_max, n_nu, dtype=np.float64)
	r = np.linspace(0.0, r_max, n_r, dtype=np.float64)
	H = make_hankel_matrix_3d(nu, r)
	low_mask = nu < nu0

	n_f_values = n_nu
	n_tv_values = n_nu - 1
	n_variables = n_f_values + n_tv_values

	c = np.zeros(n_variables, dtype=np.float64)
	c[n_f_values:] = 1.0

	bounds = []
	for index in range(n_f_values):
		lower = -1.0
		upper = None
		if m0 is not None:
			upper = float(m0) - 1.0
		if low_mask[index]:
			low_upper = e0 - 1.0
			upper = low_upper if upper is None else min(upper, low_upper)
		if index >= n_f_values - tail_anchor_count:
			lower = 0.0
			upper = 0.0
		bounds.append((lower, upper))
	for _ in range(n_tv_values):
		bounds.append((0.0, None))

	a_rows = []
	b_values = []

	for row in -H:
		full_row = np.zeros(n_variables, dtype=np.float64)
		full_row[:n_f_values] = row
		a_rows.append(full_row)
		b_values.append(1.0)

	for index in range(n_tv_values):
		row = np.zeros(n_variables, dtype=np.float64)
		row[index + 1] = 1.0
		row[index] = -1.0
		row[n_f_values + index] = -1.0
		a_rows.append(row)
		b_values.append(0.0)

		row = np.zeros(n_variables, dtype=np.float64)
		row[index + 1] = -1.0
		row[index] = 1.0
		row[n_f_values + index] = -1.0
		a_rows.append(row)
		b_values.append(0.0)

	A_ub = np.vstack(a_rows)
	b_ub = np.array(b_values, dtype=np.float64)

	result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
	status = linprog_status_name(result.status)
	if not result.success:
		target = empty_target(status, result.message, nu, r, H, low_mask, m0, nu0=nu0, e0=e0, result=result)
		target["dimension"] = 3
		if require_success:
			raise RuntimeError(f"3D DS-Wave target solve {status}: {result.message}")
		return target

	F = result.x[:n_f_values]
	P = F + 1.0
	g = H @ F + 1.0
	return {
		"success": True,
		"status": "optimal",
		"message": result.message,
		"dimension": 3,
		"nu0": nu0,
		"e0": e0,
		"nu": nu,
		"r": r,
		"H": H,
		"low_mask": low_mask,
		"m0": m0,
		"F": F,
		"P": P,
		"g": g,
		"objective": float(result.fun),
		"linprog_result": result,
	}

def find_min_m0_3d(
	nu0: float = 0.85,
	e0: float = 0.0,
	nu_max: float = 10.0,
	n_nu: int = 1001,
	r_max: float = 4.0,
	n_r: int = 128,
	tail_anchor_count: int = 1,
	m0_tol: float = 0.02,
	max_m0: float = 64.0,
) -> float | None:
	"""Find the smallest feasible 3D m0 using bracket expansion plus bisection."""
	if m0_tol <= 0.0:
		raise ValueError("m0_tol must be positive")

	lower = 1.0
	lower_target = solve_ds_wave_target_3d(
		nu0=nu0,
		e0=e0,
		m0=lower,
		nu_max=nu_max,
		n_nu=n_nu,
		r_max=r_max,
		n_r=n_r,
		tail_anchor_count=tail_anchor_count,
		require_success=False,
	)
	if lower_target["success"]:
		return lower

	upper = 2.0
	upper_target = solve_ds_wave_target_3d(
		nu0=nu0,
		e0=e0,
		m0=upper,
		nu_max=nu_max,
		n_nu=n_nu,
		r_max=r_max,
		n_r=n_r,
		tail_anchor_count=tail_anchor_count,
		require_success=False,
	)
	while not upper_target["success"] and upper < max_m0:
		lower = upper
		upper = min(upper * 2.0, max_m0)
		upper_target = solve_ds_wave_target_3d(
			nu0=nu0,
			e0=e0,
			m0=upper,
			nu_max=nu_max,
			n_nu=n_nu,
			r_max=r_max,
			n_r=n_r,
			tail_anchor_count=tail_anchor_count,
			require_success=False,
		)

	if not upper_target["success"]:
		return None

	while upper - lower > m0_tol:
		mid = 0.5 * (lower + upper)
		target = solve_ds_wave_target_3d(
			nu0=nu0,
			e0=e0,
			m0=mid,
			nu_max=nu_max,
			n_nu=n_nu,
			r_max=r_max,
			n_r=n_r,
			tail_anchor_count=tail_anchor_count,
			require_success=False,
		)
		if target["success"]:
			upper = mid
		else:
			lower = mid
	return upper

def evaluate_target_pcf_3d(target: dict, radii: np.ndarray) -> np.ndarray:
	"""Evaluate the 3D target PCF g(r) directly from F at arbitrary radii."""
	if target["F"] is None:
		raise ValueError("target has no solved shifted power spectrum")
	radii = np.asarray(radii, dtype=np.float64)
	nu = np.asarray(target["nu"], dtype=np.float64)
	F = np.asarray(target["F"], dtype=np.float64)
	nu_weights = trapezoid_weights(nu) * nu * nu * F
	phase = 2.0 * math.pi * np.outer(radii, nu)
	return 1.0 + np.sum(4.0 * math.pi * sinc(phase) * nu_weights[None, :], axis=1)

def make_targetrdf_target_curve_3d(target: dict, n_points: int, nbins: int, smoothing: float) -> tuple[np.ndarray, np.ndarray]:
	"""Convert a solved 3D DS-Wave target into a TargetRDF curve.

	Shapes:
	- target_rdf: (nbins,), desired g(r) values.
	- unit_r: (nbins,), toroidal unit-cube distances in [0, 0.5).

	The normalized DS-Wave radius is unit-cube distance times N^(1/3), because
	the characteristic spacing in a unit-volume 3D domain is proportional to
	N^(-1/3).
	"""
	if n_points < 2:
		raise ValueError("n_points must be at least 2")
	if nbins < 2:
		raise ValueError("nbins must be at least 2")
	if smoothing < 0.0:
		raise ValueError("smoothing must be non-negative")

	dx = 0.5 / float(nbins)
	unit_r = np.arange(nbins, dtype=np.float64) * dx
	normalised_r = unit_r * float(n_points) ** (1.0 / 3.0)
	target_g = np.maximum(evaluate_target_pcf_3d(target, normalised_r), 0.0)
	return smooth_curve_gaussian(target_g, smoothing), unit_r

def compute_targetrdf_force_curve_3d(rdf: np.ndarray, target_rdf: np.ndarray, n_points: int) -> np.ndarray:
	"""Build a 3D radial force lookup curve from RDF error.

	This mirrors the 2D TargetRDF force construction, but the radial volume
	scaling changes from r^2 to r^3. The returned scalar curve is later
	multiplied by wrapped 3D pair directions.
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
	force = dx * (rdf - target_rdf)
	force = np.cumsum(force)
	force[0] = 0.0
	for index in range(1, force.shape[0]):
		x = index * dx
		force[index] /= float(n_points) * x * x * x
	return force

def compute_targetrdf_rdf_3d(points, nbins: int, smoothing: float = 8.0, chunk_size: int = 256) -> np.ndarray:
	"""Estimate the toroidal 3D radial distribution function.

	Shapes:
	- points: (n_points, 3), in [0, 1).
	- return: (nbins,), radial bins for distances in [0, 0.5).
	"""
	if torch is None:
		raise RuntimeError("torch is required for point synthesis")
	if points.ndim != 2 or points.shape[1] != 3:
		raise ValueError("points must have shape (n_points, 3)")
	if nbins < 2:
		raise ValueError("nbins must be at least 2")
	if smoothing < 0.0:
		raise ValueError("smoothing must be non-negative")
	if chunk_size < 1:
		raise ValueError("chunk_size must be positive")

	n_points = points.shape[0]
	dx = 0.5 / float(nbins)
	counts = torch.zeros((nbins,), dtype=torch.float64, device=points.device)
	with torch.no_grad():
		all_indices = torch.arange(n_points, device=points.device)
		for start in range(0, n_points, chunk_size):
			end = min(start + chunk_size, n_points)
			row_indices = torch.arange(start, end, device=points.device)[:, None]
			delta = points[start:end, None, :] - points[None, :, :]
			delta = delta - torch.round(delta)
			distances = torch.sqrt(torch.sum(delta * delta, dim=2))
			mask = (all_indices[None, :] > row_indices) & (distances < 0.5)
			if torch.any(mask):
				bin_indices = torch.floor(distances[mask] / dx).to(dtype=torch.int64)
				bin_indices = torch.clamp(bin_indices, 0, nbins - 1)
				counts = counts + torch.bincount(bin_indices, minlength=nbins).to(dtype=torch.float64)

	pair_count = float(n_points * (n_points - 1)) * 0.5
	shell_indices = torch.arange(nbins, dtype=torch.float64, device=points.device)
	shell_volume = (4.0 * math.pi / 3.0) * (dx ** 3) * ((shell_indices + 1.0) ** 3 - shell_indices ** 3)
	rdf = counts / (pair_count * shell_volume)
	rdf_np = rdf.cpu().numpy()
	return smooth_curve_gaussian(rdf_np, smoothing)

def compute_targetrdf_gradients_3d(points, force_curve: np.ndarray, chunk_size: int = 256):
	"""Compute 3D point updates induced by a radial force curve."""
	if torch is None:
		raise RuntimeError("torch is required for point synthesis")
	if points.ndim != 2 or points.shape[1] != 3:
		raise ValueError("points must have shape (n_points, 3)")
	if chunk_size < 1:
		raise ValueError("chunk_size must be positive")

	n_points = points.shape[0]
	force = torch.as_tensor(force_curve, dtype=points.dtype, device=points.device)
	gradients = torch.zeros_like(points)
	with torch.no_grad():
		for start in range(0, n_points, chunk_size):
			end = min(start + chunk_size, n_points)
			delta = points[None, :, :] - points[start:end, None, :]
			delta = delta + (delta < -0.5).to(dtype=points.dtype) - (delta > 0.5).to(dtype=points.dtype)
			dist2 = torch.sum(delta * delta, dim=2)
			mask = (dist2 > 0.0) & (dist2 <= 0.49 * 0.49)
			distances = torch.sqrt(torch.clamp(dist2, min=1e-20))
			pair_force = interpolate_curve_torch(force, distances)
			pair_force = torch.where(mask, pair_force, torch.zeros_like(pair_force))
			gradients[start:end] = -torch.sum(delta * pair_force[:, :, None], dim=1)
	max_gradient = torch.sqrt(torch.max(torch.sum(gradients * gradients, dim=1))).item()
	return gradients, max_gradient

def synthesize_targetrdf_points_3d(
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
	"""Synthesize periodic unit-cube points whose RDF follows a 3D DS-Wave target."""
	if n_points < 2:
		raise ValueError("n_points must be at least 2 for 3D PCF matching")
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
	torch.manual_seed(seed)
	if torch_device.type == "cuda":
		torch.cuda.manual_seed_all(seed)

	nbins = int(n_points if nbins is None else nbins)
	torch_dtype = torch.float32
	if initial_points is None:
		current = torch.rand((n_points, 3), dtype=torch_dtype, device=torch_device)
	else:
		initial_points = np.asarray(initial_points, dtype=np.float32)
		if initial_points.shape != (n_points, 3):
			raise ValueError("initial_points must have shape (n_points, 3)")
		if not np.all(np.isfinite(initial_points)):
			raise ValueError("initial_points must be finite")
		current = torch.as_tensor(initial_points, dtype=torch_dtype, device=torch_device).clone()
		current.remainder_(1.0)

	target_rdf_np, unit_r_np = make_targetrdf_target_curve_3d(target, n_points, nbins, smoothing)
	target_pcf_np = np.maximum(evaluate_target_pcf_3d(target, unit_r_np * float(n_points) ** (1.0 / 3.0)), 0.0)
	rdf_np = compute_targetrdf_rdf_3d(current, nbins, smoothing=smoothing, chunk_size=chunk_size)
	energy = compute_targetrdf_energy(rdf_np, target_rdf_np)

	best = current.clone()
	best_rdf_np = rdf_np.copy()
	best_energy = energy
	attempts = 0
	current_step_scale = float(step_scale)
	energy_history = []
	iterations_run = 0

	for iteration in range(iterations):
		force_curve_np = compute_targetrdf_force_curve_3d(rdf_np, target_rdf_np, n_points)
		gradients, max_gradient = compute_targetrdf_gradients_3d(current, force_curve_np, chunk_size=chunk_size)
		if max_gradient > 1e-12:
			step_size = current_step_scale / (float(n_points) ** (1.0 / 3.0) * max_gradient)
			current = torch.remainder(current + float(step_size) * gradients, 1.0)
			rdf_np = compute_targetrdf_rdf_3d(current, nbins, smoothing=smoothing, chunk_size=chunk_size)
			energy = compute_targetrdf_energy(rdf_np, target_rdf_np)
			attempts += 1
			if energy < best_energy:
				best = current.clone()
				best_rdf_np = rdf_np.copy()
				best_energy = energy
				attempts = 0
			elif energy > best_energy * 1.2:
				current = best.clone()
				rdf_np = best_rdf_np.copy()
				energy = best_energy
				attempts = 5
			if attempts >= 5:
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
		"r_values": unit_r_np * float(n_points) ** (1.0 / 3.0),
		"target_pcf": target_pcf_np,
		"target_rdf": target_rdf_np,
		"final_rdf": best_rdf_np,
		"iterations_run": iterations_run,
		"synthesis_mode": "pcf_matching_3d",
		"pcf_algorithm": "targetrdf_force_3d",
		"pcf_num_bins": nbins,
		"pcf_smoothing": smoothing,
		"pcf_step_scale": step_scale,
		"pcf_final_step_scale": current_step_scale,
		"device": str(torch_device),
	}

def synthesize_toroidal_points_3d(
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
	"""Convenience wrapper for 3D DS-Wave PCF/TargetRDF point synthesis."""
	return synthesize_targetrdf_points_3d(
		target,
		n_points=n_points,
		iterations=iterations,
		seed=seed,
		device=device,
		step_scale=step_scale,
		nbins=nbins,
		smoothing=smoothing,
		chunk_size=chunk_size,
		initial_points=initial_points,
		log_every=log_every,
	)

def make_frequency_modes_3d(
	n_points: int,
	nu_max: float,
	mode_limit: int = 4096,
	seed: int = 0,
	priority_nu: float | None = None,
) -> np.ndarray:
	"""Return integer 3D Fourier modes with nu = |k| / N^(1/3)."""
	if n_points < 1:
		raise ValueError("n_points must be at least 1")
	if nu_max <= 0.0:
		raise ValueError("nu_max must be positive")
	if mode_limit < 0:
		raise ValueError("mode_limit must be non-negative")

	normaliser = float(n_points) ** (1.0 / 3.0)
	max_mode = max(1, int(math.ceil(nu_max * normaliser)))
	values = np.arange(-max_mode, max_mode + 1, dtype=np.int32)
	kx, ky, kz = np.meshgrid(values, values, values, indexing="xy")
	modes = np.stack([kx.ravel(), ky.ravel(), kz.ravel()], axis=1)
	nonzero = np.any(modes != 0, axis=1)
	modes = modes[nonzero]
	radius = np.linalg.norm(modes.astype(np.float64), axis=1) / normaliser
	keep = radius <= nu_max
	modes = modes[keep]
	radius = radius[keep]

	if mode_limit > 0 and modes.shape[0] > mode_limit:
		if priority_nu is not None:
			priority_mask = radius <= priority_nu
			priority_modes = modes[priority_mask]
			priority_radius = radius[priority_mask]
			if priority_modes.shape[0] >= mode_limit:
				order = np.argsort(priority_radius, kind="stable")
				return priority_modes[order[:mode_limit]].astype(np.float64)

			remaining_modes = modes[~priority_mask]
			remaining_count = mode_limit - priority_modes.shape[0]
			rng = np.random.default_rng(seed)
			if remaining_modes.shape[0] > remaining_count:
				choice = rng.choice(remaining_modes.shape[0], size=remaining_count, replace=False)
				remaining_modes = remaining_modes[np.sort(choice)]
			return np.vstack([priority_modes, remaining_modes]).astype(np.float64)

		rng = np.random.default_rng(seed)
		choice = rng.choice(modes.shape[0], size=mode_limit, replace=False)
		modes = modes[np.sort(choice)]
	return modes.astype(np.float64)

def compute_radial_psd_3d(
	points: np.ndarray,
	num_bins: int = 80,
	max_freq: float = 3.0,
	mode_limit: int = 8192,
	priority_nu: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Average Equation 2 Fourier powers over 3D radial frequency shells."""
	points = np.asarray(points, dtype=np.float64)
	if points.ndim != 2 or points.shape[1] != 3:
		raise ValueError("points must have shape (n_points, 3)")
	if num_bins < 1:
		raise ValueError("num_bins must be positive")
	if points.shape[0] < 1:
		raise ValueError("points must not be empty")

	normaliser = float(points.shape[0]) ** (1.0 / 3.0)
	modes = make_frequency_modes_3d(points.shape[0], nu_max=max_freq, mode_limit=mode_limit, seed=0, priority_nu=priority_nu)
	if modes.shape[0] == 0:
		raise ValueError("no Fourier modes available for the requested n_points and max_freq")
	radii = np.linalg.norm(modes, axis=1) / normaliser
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

def plot_radial_psd_overlay_3d(
	points: np.ndarray,
	target: dict,
	num_bins: int = 80,
	max_freq: float = 3.0,
	mode_limit: int = 8192,
):
	"""Plot empirical 3D radial PSD against the optimized 3D target spectrum."""
	import matplotlib.pyplot as plt

	freqs, radial = compute_radial_psd_3d(
		points,
		num_bins=num_bins,
		max_freq=max_freq,
		mode_limit=mode_limit,
		priority_nu=target.get("nu0"),
	)
	target_power = np.interp(freqs, target["nu"], target["P"], left=target["P"][0], right=target["P"][-1])
	fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=120)
	ax.plot(freqs, radial, label="empirical", color="#174a7c", linewidth=2.0)
	ax.plot(freqs, target_power, label="target", color="#9a3412", linewidth=2.0)
	ax.set_title("3D Radial PSD")
	ax.set_xlabel("nu")
	ax.set_ylabel("power")
	ax.set_xlim(0.0, max_freq)
	ax.set_ylim(bottom=0.0)
	ax.grid(True, alpha=0.25)
	ax.legend()
	return fig

def plot_3d_target(target: dict):
	"""Plot the optimized 3D radial power spectrum and implied PCF."""
	import matplotlib.pyplot as plt

	fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
	axes[0].plot(target["nu"], target["P"], label="3D target")
	axes[0].axhline(1.0, color="0.55", linestyle="--", linewidth=1)
	axes[0].set_xlabel("normalised radial frequency")
	axes[0].set_ylabel("P(nu)")
	axes[0].set_title("3D DS-Wave Target Spectrum")
	axes[0].legend()

	axes[1].plot(target["r"], target["g"], label="3D target")
	axes[1].axhline(1.0, color="0.55", linestyle="--", linewidth=1)
	axes[1].axhline(0.0, color="0.55", linestyle=":", linewidth=1)
	axes[1].set_xlabel("normalised radial distance")
	axes[1].set_ylabel("g(r)")
	axes[1].set_title("3D Implied Pair Correlation")
	axes[1].legend()

	fig.tight_layout()
	return fig

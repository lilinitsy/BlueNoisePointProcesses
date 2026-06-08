from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog
from scipy.special import j0


def trapezoid_weights(axis: np.ndarray) -> np.ndarray:
	"""Return integration weights for samples on a strictly increasing 1D axis."""
	if axis.ndim != 1 or axis.shape[0] < 2:
		raise ValueError("axis must be a one-dimensional array with at least two samples")

	widths = np.diff(axis)
	if np.any(widths <= 0.0):
		raise ValueError("axis samples must be strictly increasing")

	weights = np.empty_like(axis, dtype=np.float64)
	weights[0] = 0.5 * widths[0]
	weights[-1] = 0.5 * widths[-1]
	if axis.shape[0] > 2:
		weights[1:-1] = 0.5 * (widths[:-1] + widths[1:])
	return weights

def make_hankel_matrix(nu: np.ndarray, r: np.ndarray) -> np.ndarray:
	"""Build the discrete 2D radial Hankel transform matrix.

	Shapes:
	- nu: (n_nu,), normalized radial frequencies.
	- r: (n_r,), normalized radial distances.
	- return: (n_r, n_nu), so H @ F evaluates the shifted PCF f(r).
	"""
	nu_weights = trapezoid_weights(nu) * nu
	phase = 2.0 * math.pi * np.outer(r, nu)

	# Entries look like: H[r_i, nu_j] = 2 pi * J0(2 pi r_i nu_j) * nu_j * delta_{nu_j}

	return 2.0 * math.pi * j0(phase) * nu_weights[None, :]

def empty_target(
	status: str,
	message: str,
	nu: np.ndarray,
	r: np.ndarray,
	H: np.ndarray,
	low_mask: np.ndarray,
	m0: float | str | None,
	nu0: float | None = None,
	e0: float | None = None,
	result=None,
) -> dict:
	"""Return the target dictionary shape used by failed/infeasible solves."""
	return {
		"success": False,
		"status": status,
		"message": message,
		"nu0": nu0,
		"e0": e0,
		"nu": nu,
		"r": r,
		"H": H,
		"low_mask": low_mask,
		"m0": m0,
		"F": None,
		"P": None,
		"g": None,
		"objective": None,
		"linprog_result": result,
	}

def linprog_status_name(status: int) -> str:
	if status == 0:
		return "optimal"
	if status == 2:
		return "infeasible"
	if status == 3:
		return "unbounded"
	return "failed"

def solve_ds_wave_target(
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
	"""Solve the DS-Wave target-spectrum linear program.

	The solver works with the shifted spectrum F = P - 1 on the frequency grid
	nu. The Hankel matrix maps F to f = g - 1 on the distance grid r.
	"""
	if m0 == "min":
		min_m0 = find_min_m0(
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
			H = make_hankel_matrix(nu, r)
			low_mask = nu < nu0
			target = empty_target("infeasible", "No feasible finite m0 found.", nu, r, H, low_mask, m0, nu0=nu0, e0=e0)
			if require_success:
				raise RuntimeError(target["message"])
			return target
		return solve_ds_wave_target(
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

	# 1D discretization grids from Section 5.4.
	# nu: (n_nu,), r: (n_r,), H: (n_r, n_nu).
	nu = np.linspace(0.0, nu_max, n_nu, dtype=np.float64)
	r = np.linspace(0.0, r_max, n_r, dtype=np.float64)
	H = make_hankel_matrix(nu, r)
	low_mask = nu < nu0

	# LP variable layout:
	# x = [F_0 ... F_{n_nu-1}, tv_0 ... tv_{n_nu-2}]
	# F has length n_nu. tv stores |F[i+1] - F[i]| for Equation 16.
	n_f_values = n_nu
	n_tv_values = n_nu - 1
	n_variables = n_f_values + n_tv_values

	c = np.zeros(n_variables, dtype=np.float64)
	# Equation 16: total variation energy, linearized with auxiliary |dF| variables.
	c[n_f_values:] = 1.0

	bounds = []
	for index in range(n_f_values):
		# Equation 12: P = F + 1 >= 0, hence F >= -1.
		lower = -1.0
		upper = None
		if m0 is not None:
			# Equation 17: high-frequency peak cap |F| <= m0 - 1.
			upper = float(m0) - 1.0
		if low_mask[index]:
			# Equation 14: low-frequency power P = F + 1 <= e0.
			low_upper = e0 - 1.0
			upper = low_upper if upper is None else min(upper, low_upper)
		if index >= n_f_values - tail_anchor_count:
			# Section 5.4 samples nu in [0, 10] because F has converged to 0 by the tail.
			lower = 0.0
			upper = 0.0
		bounds.append((lower, upper))
	for _ in range(n_tv_values):
		bounds.append((0.0, None))

	a_rows = []
	b_values = []

	for row in -H:
		# Equation 12: g = H[F] + 1 >= 0 becomes -H[F] <= 1.
		full_row = np.zeros(n_variables, dtype=np.float64)
		full_row[:n_f_values] = row
		a_rows.append(full_row)
		b_values.append(1.0)

	for index in range(n_tv_values):
		# Equation 16 linearization: t_i >= F_{i+1} - F_i and t_i >= -(F_{i+1} - F_i).
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

	# A_ub has one PCF-realizability row per r sample plus two TV rows per edge.
	A_ub = np.vstack(a_rows)
	b_ub = np.array(b_values, dtype=np.float64)

	result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
	status = linprog_status_name(result.status)
	if not result.success:
		target = empty_target(status, result.message, nu, r, H, low_mask, m0, nu0=nu0, e0=e0, result=result)
		if require_success:
			raise RuntimeError(f"DS-Wave target solve {status}: {result.message}")
		return target


	# This computes the PCF
	F = result.x[:n_f_values]
	P = F + 1.0
	g = H @ F + 1.0


	return {
		"success": True,
		"status": "optimal",
		"message": result.message,
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

def solve_ds_wave_target_for_targetrdf(
	n_points: int,
	nu0: float = 0.85,
	e0: float = 0.0,
	m0: float | str | None = None,
	nu_max: float = 10.0,
	n_nu: int = 1001,
	r_max: float | None = None,
	n_r: int | None = None,
	r_samples_per_unit: float = 32.0,
	tail_anchor_count: int = 1,
	m0_tol: float = 0.02,
	max_m0: float = 64.0,
	require_success: bool = True,
) -> dict:
	"""Solve a DS-Wave target over the full TargetRDF distance domain."""
	if n_points < 2:
		raise ValueError("n_points must be at least 2")
	if r_samples_per_unit <= 0.0:
		raise ValueError("r_samples_per_unit must be positive")

	if r_max is None:
		r_max = 0.5 * math.sqrt(float(n_points))
	if r_max <= 0.0:
		raise ValueError("r_max must be positive")
	if n_r is None:
		n_r = max(2, int(math.ceil(r_samples_per_unit * r_max)))

	return solve_ds_wave_target(
		nu0=nu0,
		e0=e0,
		m0=m0,
		nu_max=nu_max,
		n_nu=n_nu,
		r_max=r_max,
		n_r=n_r,
		tail_anchor_count=tail_anchor_count,
		m0_tol=m0_tol,
		max_m0=max_m0,
		require_success=require_success,
	)

def find_min_m0(
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
	"""Find the smallest feasible m0 using bracket expansion plus bisection."""
	if m0_tol <= 0.0:
		raise ValueError("m0_tol must be positive")

	# Section 5.2: m0 = min is found by searching the feasible nu0/e0/m0 region.
	lower = 1.0
	lower_target = solve_ds_wave_target(
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
	upper_target = solve_ds_wave_target(
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
		upper_target = solve_ds_wave_target(
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
		target = solve_ds_wave_target(
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

def interpolate_target_power(target: dict, radii: np.ndarray) -> np.ndarray:
	"""Interpolate solved target power P(nu) at normalized frequency radii."""
	if target["P"] is None:
		raise ValueError("target has no solved power spectrum")
	powers = np.interp(radii, target["nu"], target["P"], left=target["P"][0], right=target["P"][-1])
	nu0 = target.get("nu0")
	e0 = target.get("e0")
	if nu0 is not None and e0 is not None:
		radii_array = np.asarray(radii, dtype=np.float64)
		powers = np.where(radii_array < float(nu0), np.minimum(powers, float(e0)), powers)
	return powers

def evaluate_target_pcf(target: dict, radii: np.ndarray) -> np.ndarray:
	"""Evaluate g(r) directly from F on arbitrary normalized distance radii.

	This avoids extrapolating the stored target["g"] samples when TargetRDF
	needs distances beyond the solver's r grid.
	"""
	if target["F"] is None:
		raise ValueError("target has no solved shifted power spectrum")
	radii = np.asarray(radii, dtype=np.float64)
	nu = np.asarray(target["nu"], dtype=np.float64)
	F = np.asarray(target["F"], dtype=np.float64)
	nu_weights = trapezoid_weights(nu) * nu * F
	# phase: (len(radii), n_nu). Sum over frequency to get one g(r) per radius.
	phase = 2.0 * math.pi * np.outer(radii, nu)
	return 1.0 + np.sum(2.0 * math.pi * j0(phase) * nu_weights[None, :], axis=1)

def interpolate_target_pcf(target: dict, radii: np.ndarray) -> np.ndarray:
	"""Interpolate precomputed g(r) samples on target["r"].

	Use evaluate_target_pcf() when radii can extend beyond target["r"].
	"""
	if target["g"] is None:
		raise ValueError("target has no solved pair correlation function")
	return np.interp(radii, target["r"], target["g"], left=target["g"][0], right=target["g"][-1])

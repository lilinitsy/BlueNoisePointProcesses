"""
The solver works with the shifted spectrum F = P - 1 on a frequency grid nu,
and a discrete Hankel matrix H that maps F to the shifted pair correlation f = g - 1 on a distance grid r. 
minimize total variation of F (Eq. 16) subject to realizability (P >= 0, g >= 0, Eq. 12), 
the clean low band (P <= e0 below nu0, Eq. 14), 
and the oscillation cap (|F| <= m0 - 1 above nu0, Eq. 17 upper bound).

nu is normalized frequency (|k| / sqrt(N) for integer Fourier modes k of an N-point set on the unit torus); 
r is normalized distance (toroidal distance times sqrt(N)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.special import j0


@dataclass
class DsWaveTarget:
	"""A solved (or failed) DS-Wave target spectrum.

	Example values are from solve_ds_wave_target(nu0=0.85, e0=0.0, m0=1.765625)
	(the minimal feasible m0 found by find_min_m0 for nu0=0.85):

	  success    bool       LP solved? False means F/P/g are None.
	  status     str        "optimal", "infeasible", "unbounded", or "failed".
	  message    str        solver detail, e.g. "Optimization terminated successfully."
	  nu0        float      low-frequency cutoff (normalized frequency), e.g. 0.85
	  e0         float      allowed power below nu0, e.g. 0.0
	  m0         float      oscillation cap = max P(nu), e.g. 1.765625; None = uncapped
	  nu         ndarray    frequency axis, e.g. linspace(0.0, 10.0, 1001)
	  P          ndarray    target radial power spectrum on nu (the decaying square wave): 0.0 below nu0, first plateau at m0, alternating plateaus decaying toward 1.0
	  F          ndarray    shifted spectrum, F = P - 1
	  r          ndarray    distance axis, e.g. linspace(0.0, 4.0, 128)
	  g          ndarray    target pair correlation on r: g(0) = 0 (exclusion zone)
	  H          ndarray    (n_r, n_nu) Hankel matrix; g = H @ F + 1
	  low_mask   ndarray    boolean mask of nu samples below nu0 (LP internals)
	  objective  float      LP objective (total variation of F)
	  linprog_result        raw scipy.optimize.OptimizeResult
	  requested_m0  str     debug: requested target m0 rather than solved
	"""
	success: bool
	status: str
	message: str
	nu0: float
	e0: float
	m0: float | None
	nu: np.ndarray
	r: np.ndarray
	H: np.ndarray
	low_mask: np.ndarray
	F: np.ndarray | None
	P: np.ndarray | None
	g: np.ndarray | None
	objective: float | None
	linprog_result: object
	requested_m0: str = ""
	dimension: int = 2


def trapezoid_weights(axis: np.ndarray) -> np.ndarray:
	"""Return trapezoid integration weights for samples on a strictly increasing 1D axis."""
	if axis.ndim != 1 or axis.shape[0] < 2:
		raise ValueError("axis must be a one-dimensional array with at least two samples")

	widths = np.diff(axis)
	if np.any(widths <= 0.0):
		raise ValueError("axis samples must be strictly increasing")

	weights = np.empty_like(axis, dtype=np.float32)
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

	Entries: H[i, j] = 2*pi * J0(2*pi * r_i * nu_j) * nu_j * w_j, where w_j are trapezoid weights on the nu axis.
	"""
	nu_weights = trapezoid_weights(nu) * nu
	phase = 2.0 * math.pi * np.outer(r, nu)
	return 2.0 * math.pi * j0(phase) * nu_weights[None, :]


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
	m0: float | None = None,
	nu_max: float = 10.0,
	n_nu: int = 1001,
	r_max: float = 4.0,
	n_r: int = 128,
	tail_anchor_count: int = 1,
) -> DsWaveTarget:
	"""Solve the DS-Wave target-spectrum linear program for one fixed m0.

	m0=None places no oscillation cap. To solve at the minimal feasible m0, call find_min_m0() first and pass its result here. 
	Always returns a DsWaveTarget; check .success (an infeasible LP is fine, e.g. m0=1.0 at nu0=0.85).
	"""
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
		raise ValueError("m0 must be at least 1.0, or None for no cap")

	# Discretization grids (Section 5.4). nu: (n_nu,), r: (n_r,), H: (n_r, n_nu).
	nu = np.linspace(0.0, nu_max, n_nu, dtype=np.float32)
	r = np.linspace(0.0, r_max, n_r, dtype=np.float32)
	H = make_hankel_matrix(nu, r)
	low_mask = nu < nu0

	# LP variable layout: x = [F_0 .. F_{n_nu-1}, tv_0 .. tv_{n_nu-2}]. F is the shifted spectrum; tv stores |F[i+1] - F[i]| for Eq. 16.
	n_f_values = n_nu
	n_tv_values = n_nu - 1
	n_variables = n_f_values + n_tv_values

	# Objective (Eq. 16): minimize the total variation, sum of the tv variables.
	c = np.zeros(n_variables, dtype=np.float32)
	c[n_f_values:] = 1.0

	bounds = []
	for index in range(n_f_values):
		# Eq. 12: P = F + 1 >= 0, hence F >= -1.
		lower = -1.0
		upper = None
		if m0 is not None:
			# Eq. 17 (upper bound): P <= m0, hence F <= m0 - 1.
			upper = float(m0) - 1.0
		if low_mask[index]:
			# Eq. 14: low-frequency power P = F + 1 <= e0.
			low_upper = e0 - 1.0
			upper = low_upper if upper is None else min(upper, low_upper)
		if index >= n_f_values - tail_anchor_count:
			# Anchor the tail at F = 0: by nu_max the spectrum has converged to 1.
			lower = 0.0
			upper = 0.0
		bounds.append((lower, upper))
	for _ in range(n_tv_values):
		bounds.append((0.0, None))

	a_rows = []
	b_values = []

	# Eq. 12: g = H @ F + 1 >= 0, i.e. -H @ F <= 1 (one row per r sample).
	for row in -H:
		full_row = np.zeros(n_variables, dtype=np.float32)
		full_row[:n_f_values] = row
		a_rows.append(full_row)
		b_values.append(1.0)

	# Eq. 16 linearization: tv_i >= F_{i+1} - F_i and tv_i >= -(F_{i+1} - F_i).
	for index in range(n_tv_values):
		row = np.zeros(n_variables, dtype=np.float32)
		row[index + 1] = 1.0
		row[index] = -1.0
		row[n_f_values + index] = -1.0
		a_rows.append(row)
		b_values.append(0.0)

		row = np.zeros(n_variables, dtype=np.float32)
		row[index + 1] = -1.0
		row[index] = 1.0
		row[n_f_values + index] = -1.0
		a_rows.append(row)
		b_values.append(0.0)

	A_ub = np.vstack(a_rows)
	b_ub = np.array(b_values, dtype=np.float32)

	result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
	if not result.success:
		return DsWaveTarget(
			success=False,
			status=linprog_status_name(result.status),
			message=result.message,
			nu0=nu0,
			e0=e0,
			m0=m0,
			nu=nu,
			r=r,
			H=H,
			low_mask=low_mask,
			F=None,
			P=None,
			g=None,
			objective=None,
			linprog_result=result,
		)

	F = result.x[:n_f_values]
	P = F + 1.0
	g = H @ F + 1.0

	return DsWaveTarget(
		success=True,
		status="optimal",
		message=result.message,
		nu0=nu0,
		e0=e0,
		m0=m0,
		nu=nu,
		r=r,
		H=H,
		low_mask=low_mask,
		F=F,
		P=P,
		g=g,
		objective=float(result.fun),
		linprog_result=result,
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
	"""Find the smallest feasible m0 by bracket expansion plus bisection.

	Returns None when no m0 <= max_m0 is feasible (Section 5.2: m0 = min is found by searching the feasible nu0/e0/m0 region).
	"""
	if m0_tol <= 0.0:
		raise ValueError("m0_tol must be positive")

	def solve_at(m0: float) -> DsWaveTarget:
		return solve_ds_wave_target(
			nu0=nu0,
			e0=e0,
			m0=m0,
			nu_max=nu_max,
			n_nu=n_nu,
			r_max=r_max,
			n_r=n_r,
			tail_anchor_count=tail_anchor_count,
		)

	lower = 1.0
	if solve_at(lower).success:
		return lower

	upper = 2.0
	upper_target = solve_at(upper)
	while not upper_target.success and upper < max_m0:
		lower = upper
		upper = min(upper * 2.0, max_m0)
		upper_target = solve_at(upper)
	if not upper_target.success:
		return None

	while upper - lower > m0_tol:
		mid = 0.5 * (lower + upper)
		if solve_at(mid).success:
			upper = mid
		else:
			lower = mid
	return upper


def interpolate_target_power(target: DsWaveTarget, radii: np.ndarray) -> np.ndarray:
	"""Interpolate the solved target power P(nu) at normalized frequency radii.

	Below nu0 the spec is P <= e0 exactly (Eq. 14), but linear interpolation of the discrete P grid ramps up across the nu0 step 
	(the last sample below nu0 is 0, the first above is m0). 
	Radii in that straddle region would read a meaningless ramp value, so the spec bound is re-applied there.
	"""
	if target.P is None:
		raise ValueError("target has no solved power spectrum")
	radii = np.asarray(radii, dtype=np.float32)
	powers = np.interp(radii, target.nu, target.P, left=target.P[0], right=target.P[-1])
	return np.where(radii < target.nu0, np.minimum(powers, target.e0), powers)


def evaluate_target_pcf(target: DsWaveTarget, radii: np.ndarray) -> np.ndarray:
	"""Evaluate g(r) directly from F on arbitrary normalized distance radii.

	This avoids extrapolating the stored target.g samples when synthesis needs distances beyond the solver's 
	r grid: g(r) = 1 + 2*pi * sum_j J0(2*pi r nu_j) * nu_j * F_j * w_j with trapezoid weights w on the nu axis.
	"""
	if target.F is None:
		raise ValueError("target has no solved shifted power spectrum")
	radii = np.asarray(radii, dtype=np.float32)
	nu_weights = trapezoid_weights(target.nu) * target.nu * target.F
	phase = 2.0 * math.pi * np.outer(radii, target.nu)
	return 1.0 + np.sum(2.0 * math.pi * j0(phase) * nu_weights[None, :], axis=1)


def interpolate_target_pcf(target: DsWaveTarget, radii: np.ndarray) -> np.ndarray:
	"""Interpolate the precomputed g(r) samples on target.r.

	Use evaluate_target_pcf() when radii can extend beyond target.r.
	"""
	if target.g is None:
		raise ValueError("target has no solved pair correlation function")
	return np.interp(radii, target.r, target.g, left=target.g[0], right=target.g[-1])

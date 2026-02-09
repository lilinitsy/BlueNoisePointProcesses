import numpy as np
from numpy.typing import NDArray # Allows us to annotate with specific dtypes.
from typing import Tuple, Optional



def stair_psd(k: NDArray[np.float32], k0: float, k1: float, P0: float) -> NDArray[np.float32]:
	"""
	Stair blue noise power spectral density (Eq. 6).

	Parameters
	----------
	k  : array of frequencies (non-negative)
	k0 : end of zero region before spike
	k1 : end of mid-frequency stair (k1 >= k0)
	P0 : height of mid-frequency stair (P0 >= 1)

	Returns
	-------
	P(k) : PSD values
	"""
	f_k1 = np.where(k > k1, 1.0, 0.0)
	f_k0 = np.where(k > k0, 1.0, 0.0)
	return f_k1 + P0 * (f_k0 - f_k1)


def step_psd(k: NDArray[np.float32], k0: float) -> NDArray[np.float32]:
	"""Step blue noise PSD (Eq. 4): 0 for k <= k0, 1 for k > k0."""
	return np.where(k > k0, 1.0, 0.0)



def j1_numpy(x: NDArray[np.float32]) -> NDArray[np.float32]:
	"""
	Bessel function of the first kind, order 1, computed via power series.
	Accurate for moderate |x|; for large x we use the asymptotic form.
	"""
	out = np.zeros_like(x, dtype=np.float32)
	small = np.abs(x) < 20.0
	large = ~small

	# Power series for small x
	if np.any(small):
		xs = x[small]
		val = np.zeros_like(xs)
		term = xs / 2.0
		for m in range(30):
			if m == 0:
				term_m = xs / 2.0
			else:
				term_m = term_m * (-xs * xs / 4.0) / (m * (m + 1))
			val += term_m
		out[small] = val

	# Asymptotic for large x
	if np.any(large):
		xl = x[large]
		out[large] = np.sqrt(2.0 / (np.pi * np.abs(xl))) * np.cos(
			np.abs(xl) - 3.0 * np.pi / 4.0
		) * np.sign(xl)

	return out


def jinc(x: NDArray[np.float32]) -> NDArray[np.float32]:
	"""
	Jinc as used in the paper:  J1(x) / x, with Jinc(0) = 1/2.

	This follows from  int_0^a k J0(kr) dk = (a/r) J1(ar),
	and the PCF derivation in Appendix A uses Jinc(rk) = J1(rk)/(rk),
	multiplied by k^2, so the effective term is k * J1(rk) / r.

	We define Jinc(x) = J1(x)/x  with  Jinc(0) = 0.5.
	"""
	safe_x = np.where(np.abs(x) < 1e-15, 1.0, x)
	j1_vals = j1_numpy(safe_x)
	result = j1_vals / safe_x
	# limit: J1(x)/x -> 1/2 as x -> 0
	result = np.where(np.abs(x) < 1e-15, 0.5, result)
	return result


def stair_pcf_closed(
	r: NDArray[np.float32],
	k0: float,
	k1: float,
	P0: float,
	rho: float,
) -> NDArray[np.float32]:
	"""
	Closed-form PCF for Stair blue noise (Appendix A, simplified).

	G(r) = 1 - 1/(2*pi*rho) * [(1-P0)* k1/r * J1(k1*r) + P0 * k0/r * J1(k0*r)]

	which can be rewritten using Jinc(x) = J1(x)/x:

	G(r) = 1 - 1/(2*pi*rho) * [(1-P0)*k1^2 * Jinc(k1*r) + P0*k0^2 * Jinc(k0*r)]
	"""

	return 1.0 - (1.0 / (2.0 * np.pi * rho)) * ((1.0 - P0) * k1**2 * jinc(r * k1) + P0 * k0**2 * jinc(r * k0))


def step_pcf_closed(
	r: NDArray[np.float32],
	k0: float,
	rho: float,
) -> NDArray[np.float32]:
	"""Closed-form PCF for Step blue noise (Eq. 7). Special case P0=1."""
	return 1.0 - (1.0 / (2.0 * np.pi * rho)) * k0**2 * jinc(r * k0)


def min_samples(k0: float, k1: float, P0: float, V: float = 1.0) -> float:
	"""Minimum N for realizability (Eq. 11)."""
	return (V / (4.0 * np.pi)) * ((1.0 - P0) * k1 ** 2 + P0 * k0 ** 2)


def max_k0(N: int, k1: float, P0: float, V: float = 1.0) -> float:
	"""Maximum k0 for realizability (Eq. 13)."""
	val = (4.0 * np.pi * N - V * (1.0 - P0) * k1 ** 2) / (V * P0)
	return np.sqrt(max(val, 0.0))


def min_k1(N: int, k0: float, P0: float, V: float = 1.0) -> float:
	"""Minimum k1 for realizability (Eq. 14)."""
	val = (4.0 * np.pi * N - V * P0 * k0 ** 2) / (V * (1.0 - P0))

	if P0 > 1.0:
		return np.sqrt(max(val, 0.0))
	else:
		return k0  # degenerate: Step blue noise


def min_P0(N: int, k0: float, k1: float, V: float = 1.0) -> float:
	"""Minimum P0 for realizability (Eq. 12)."""
	return (4.0 * np.pi * N - V * k1**2) / (V * (k0**2 - k1**2))


def optimal_stair_params(
	N: int,
	P0: float = 1.5,
	delta: float = 50.0,
	V: float = 1.0,
) -> Tuple[float, float, float]:
	"""
	Given N and desired P0, compute optimal k0 and k1 = k0 + delta
	that maximize k0 while remaining realizable.

	Returns (k0, k1, P0).
	"""
	# From Eq. 11:  N >= V/(4pi) * ((1-P0)*k1^2 + P0*k0^2)
	# with k1 = k0 + delta, solve for k0.
	# (1-P0)*(k0+delta)^2 + P0*k0^2 <= 4*pi*N/V
	# Expand: (1-P0)*(k0^2 + 2*k0*delta + delta^2) + P0*k0^2
	#       = k0^2 + 2*(1-P0)*k0*delta + (1-P0)*delta^2
	# So: k0^2 + 2*(1-P0)*delta*k0 + (1-P0)*delta^2 - 4*pi*N/V <= 0
	a_coeff = 1.0
	b_coeff = 2.0 * (1.0 - P0) * delta
	c_coeff = (1.0 - P0) * delta**2 - 4.0 * np.pi * N / V

	disc = b_coeff**2 - 4.0 * a_coeff * c_coeff
	if disc < 0:
		raise ValueError("No realizable solution for given parameters.")
	k0_opt = (-b_coeff + np.sqrt(disc)) / (2.0 * a_coeff)
	k1_opt = k0_opt + delta
	return float(k0_opt), float(k1_opt), P0



def estimate_pcf(
	points: NDArray[np.float32],
	r_vals: NDArray[np.float32],
	sigma: float = 0.005,
	region_size: Tuple[float, float] = (1.0, 1.0),
) -> NDArray[np.float32]:
	"""
	Unbiased PCF estimator with edge correction (Eq. 15).

	Parameters
	----------
	points      : (N, 2) array of point positions
	r_vals      : radii at which to evaluate the PCF
	sigma       : bandwidth of the Gaussian kernel
	region_size : (width, height) of the sampling region

	Returns
	-------
	G_hat(r) for each r in r_vals
	"""
	N: int = points.shape[0]
	w, h = region_size
	V_W: float = w * h
	S_W: float = 2.0 * (w + h)

	# Pairwise distances
	diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 2)
	dists = np.sqrt(np.sum(diff**2, axis=2))  # (N, N)

	# Mask out diagonal
	mask = ~np.eye(N, dtype=bool)

	G_hat = np.zeros_like(r_vals)
	S_E = 2.0 * np.pi  # circumference factor in 2D (the paper uses S_E = 2*pi*r)

	for idx, r in enumerate(r_vals):
		# Edge correction (Eq. 17)
		gamma_W: float = V_W - (S_W / np.pi) * r
		if gamma_W <= 0:
			G_hat[idx] = 0.0
			continue

		# Gaussian kernel evaluation
		kernel_vals = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-((r - dists)**2) / (2.0 * sigma**2))
		kernel_vals = kernel_vals * mask  # zero out diagonal

		S_E_r = S_E * r  # 2*pi*r
		if S_E_r < 1e-15:
			G_hat[idx] = 0.0
			continue

		G_hat[idx] = (V_W / gamma_W) * (V_W / N) * (
			1.0 / (S_E_r * (N - 1))
		) * np.sum(kernel_vals)

	return G_hat


def estimate_pcf_vectorized(
	points: NDArray[np.float32],
	r_vals: NDArray[np.float32],
	sigma: float = 0.005,
	region_size: Tuple[float, float] = (1.0, 1.0),
) -> NDArray[np.float32]:
	"""
	Vectorized (faster) version of the PCF estimator.
	Uses more memory but is significantly faster for moderate N.
	"""
	N: int = points.shape[0]
	w, h = region_size
	V_W: float = w * h
	S_W: float = 2.0 * (w + h)

	# Pairwise distances
	diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
	dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
	mask = ~np.eye(N, dtype=bool)

	# Edge correction for each r
	gamma_W = V_W - (S_W / np.pi) * r_vals  # (M,)
	gamma_W = np.maximum(gamma_W, 1e-15)

	S_E_r = 2.0 * np.pi * r_vals  # (M,)
	S_E_r = np.maximum(S_E_r, 1e-15)

	# Kernel: (M, N, N) would be too large; process in chunks
	G_hat = np.zeros(len(r_vals))

	# For each pair distance, accumulate kernel contributions across all r
	pair_dists = dists[mask]  # (N*(N-1),)

	for idx, r in enumerate(r_vals):
		kern = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-((r - pair_dists)**2) / (2.0 * sigma**2))
		G_hat[idx] = (V_W / gamma_W[idx]) * (V_W / N) * (1.0 / (S_E_r[idx] * (N - 1))) * np.sum(kern)

	return G_hat



def synthesize_stair_blue_noise(
	N: int,
	k0: float,
	k1: float,
	P0: float,
	num_iterations: int = 50,
	step_size: float = 0.001,
	sigma: float = 0.005,
	r_min: float = 0.0001,
	r_max: float = 0.4,
	r_step: float = 0.0001,
	region_size: Tuple[float, float] = (1.0, 1.0),
	seed: Optional[int] = None,
	verbose: bool = True,
) -> Tuple[NDArray[np.float32], list]:
	"""
	Synthesize a Stair blue noise point set via weighted least-squares
	PCF matching (Section 5.2, Algorithm with Eq. 18).

	Parameters
	----------
	N              : number of points
	k0             : zero-region boundary
	k1             : stair boundary (k1 > k0)
	P0             : stair height (P0 >= 1)
	num_iterations : gradient descent iterations
	step_size      : lambda in the update rule
	sigma          : Gaussian kernel bandwidth for PCF estimation
	r_min, r_max, r_step : discretization of radius
	region_size    : (width, height) of sampling domain
	seed           : random seed
	verbose        : print progress

	Returns
	-------
	points   : (N, 2) final point set
	errors   : list of fitting errors per iteration
	"""
	rng = np.random.default_rng(seed)
	w, h = region_size
	V: float = w * h
	rho: float = N / V

	# Initialize random point set
	points = rng.uniform(0, 1, size=(N, 2)) * np.array([w, h])

	# Discretize radius
	r_vals = np.arange(r_min, r_max + r_step / 2, r_step)
	M: int = len(r_vals)

	# Target PCF
	G_star = stair_pcf_closed(r_vals, k0, k1, P0, rho)

	# Initialize weights uniformly
	weights = np.ones(M)

	errors: list = []

	for iteration in range(num_iterations):
		# Compute pairwise differences and distances
		diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 2)
		dists = np.sqrt(np.sum(diff**2, axis=2))  # (N, N)
		mask = ~np.eye(N, dtype=bool)

		# Estimate current PCF
		G_current = estimate_pcf_vectorized(points, r_vals, sigma, region_size)

		# Fitting error
		residuals = G_current - G_star
		weighted_error = np.sum((residuals / weights)**2)
		errors.append(float(weighted_error))

		if verbose:
			print(f"  Iteration {iteration:4d}/{num_iterations}: error = {weighted_error:.6f}")

		# Update weights adaptively (Section 5.2)
		abs_residuals = np.abs(residuals)
		weights = np.where(abs_residuals > 1e-10, 1.0 / abs_residuals, 1e10)

		# Avoid division by zero on diagonal
		safe_dists = np.where(dists < 1e-15, 1.0, dists)
		# Unit direction vectors: (x_l - x_i) / |x_l - x_i|  -> shape (N, N, 2)
		direction = diff / safe_dists[:, :, np.newaxis]

		gradients = np.zeros_like(points)  # (N, 2)

		for i in range(N):
			d_i = dists[i]  # distances from point i to all others (N,)
			dir_i = -diff[i]  # (x_l - x_i) for all l, but diff = points_i - points_l
			dir_il = -diff[i]  # (N, 2)

			# For each neighbor l != i
			neighbor_mask = np.arange(N) != i
			d_il = d_i[neighbor_mask]  # (N-1,)
			dir_il_masked = dir_il[neighbor_mask]  # (N-1, 2)
			safe_d = np.maximum(d_il, 1e-15)
			unit_dir = dir_il_masked / safe_d[:, np.newaxis]  # (N-1, 2)

			r_broad = r_vals[np.newaxis, :]  # (1, M)
			d_broad = d_il[:, np.newaxis]  # (N-1, 1)

			kern = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-((r_broad - d_broad)**2) / (2.0 * sigma**2))  # (N-1, M)

			# (G(rj) - G*(rj)) / (wj * rj)
			coeff = residuals / (weights * r_vals)  # (M,)
			coeff = coeff[np.newaxis, :]  # (1, M)

			# (|xl - xi| - rj)
			dist_minus_r = d_broad - r_broad  # (N-1, M)

			inner_sum = np.sum(coeff * dist_minus_r * kern, axis=1)  # (N-1,)

			# Gradient contribution
			grad_i = np.sum(unit_dir * inner_sum[:, np.newaxis], axis=0)  # (2,)
			gradients[i] = grad_i

		# Normalize and update
		grad_norms = np.sqrt(np.sum(gradients**2, axis=1, keepdims=True))
		grad_norms = np.maximum(grad_norms, 1e-15)
		normalized_grad = gradients / grad_norms

		points = points - step_size * normalized_grad

		# Clamp to region
		points[:, 0] = np.clip(points[:, 0], 0, w)
		points[:, 1] = np.clip(points[:, 1], 0, h)

	return points, errors


# =============================================================================
# 5. Radial mean power spectrum estimation
# =============================================================================

def estimate_radial_psd(
	points: NDArray[np.float32],
	k_max: float = 500.0,
	k_step: float = 1.0,
	num_angles: int = 64,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
	"""
	Estimate the radially-averaged power spectral density (Eq. 1).

	Parameters
	----------
	points     : (N, 2) point positions
	k_max      : maximum frequency
	k_step     : frequency resolution
	num_angles : number of angles for radial averaging

	Returns
	-------
	k_vals : frequency values
	P_k    : radially averaged PSD
	"""
	N: int = points.shape[0]
	k_vals = np.arange(k_step, k_max + k_step / 2, k_step)
	angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

	P_k = np.zeros(len(k_vals))

	for ki, k_mag in enumerate(k_vals):
		p_sum = 0.0
		for angle in angles:
			kx = k_mag * np.cos(angle)
			ky = k_mag * np.sin(angle)
			# Eq. 1: P(k) = (1/N) |sum_j exp(-2*pi*i*k.x_j)|^2
			phases = 2.0 * np.pi * (kx * points[:, 0] + ky * points[:, 1])
			val = np.sum(np.exp(-1j * phases))
			p_sum += np.abs(val)**2 / N
		P_k[ki] = p_sum / num_angles

	return k_vals, P_k
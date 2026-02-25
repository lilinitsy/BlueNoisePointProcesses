import numpy as np
import torch
from numpy.typing import NDArray
from typing import Tuple, Optional


def stair_psd_gpu(k: NDArray[np.float32], k0: float, k1: float, P0: float) -> NDArray[np.float32]:
	f_k1 = np.where(k > k1, 1.0, 0.0)
	f_k0 = np.where(k > k0, 1.0, 0.0)
	return f_k1 + P0 * (f_k0 - f_k1)


def step_psd_gpu(k: NDArray[np.float32], k0: float) -> NDArray[np.float32]:
	return np.where(k > k0, 1.0, 0.0)


def j1_numpy_gpu(x: NDArray[np.float32]) -> NDArray[np.float32]:
	out = np.zeros_like(x, dtype=np.float32)
	small = np.abs(x) < 20.0
	large = ~small

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

	if np.any(large):
		xl = x[large]
		out[large] = np.sqrt(2.0 / (np.pi * np.abs(xl))) * np.cos(
			np.abs(xl) - 3.0 * np.pi / 4.0
		) * np.sign(xl)

	return out


def jinc_gpu(x: NDArray[np.float32]) -> NDArray[np.float32]:
	safe_x = np.where(np.abs(x) < 1e-15, 1.0, x)
	j1_vals = j1_numpy_gpu(safe_x)
	result = j1_vals / safe_x
	result = np.where(np.abs(x) < 1e-15, 0.5, result)
	return result


def stair_pcf_closed_gpu(
	r: NDArray[np.float32],
	k0: float,
	k1: float,
	P0: float,
	rho: float,
) -> NDArray[np.float32]:
	return 1.0 - (1.0 / (2.0 * np.pi * rho)) * ((1.0 - P0) * k1**2 * jinc_gpu(r * k1) + P0 * k0**2 * jinc_gpu(r * k0))


def step_pcf_closed_gpu(
	r: NDArray[np.float32],
	k0: float,
	rho: float,
) -> NDArray[np.float32]:
	return 1.0 - (1.0 / (2.0 * np.pi * rho)) * k0**2 * jinc_gpu(r * k0)


def min_samples_gpu(k0: float, k1: float, P0: float, V: float = 1.0) -> float:
	return (V / (4.0 * np.pi)) * ((1.0 - P0) * k1 ** 2 + P0 * k0 ** 2)


def max_k0_gpu(N: int, k1: float, P0: float, V: float = 1.0) -> float:
	val = (4.0 * np.pi * N - V * (1.0 - P0) * k1 ** 2) / (V * P0)
	return np.sqrt(max(val, 0.0))


def min_k1_gpu(N: int, k0: float, P0: float, V: float = 1.0) -> float:
	val = (4.0 * np.pi * N - V * P0 * k0 ** 2) / (V * (1.0 - P0))
	if P0 > 1.0:
		return np.sqrt(max(val, 0.0))
	else:
		return k0


def min_P0_gpu(N: int, k0: float, k1: float, V: float = 1.0) -> float:
	return (4.0 * np.pi * N - V * k1**2) / (V * (k0**2 - k1**2))


def optimal_stair_params_gpu(
	N: int,
	P0: float = 1.5,
	delta: float = 50.0,
	V: float = 1.0,
) -> Tuple[float, float, float]:
	a_coeff = 1.0
	b_coeff = 2.0 * (1.0 - P0) * delta
	c_coeff = (1.0 - P0) * delta**2 - 4.0 * np.pi * N / V

	disc = b_coeff**2 - 4.0 * a_coeff * c_coeff
	if disc < 0:
		raise ValueError("No realizable solution for given parameters.")
	k0_opt = (-b_coeff + np.sqrt(disc)) / (2.0 * a_coeff)
	k1_opt = k0_opt + delta
	return float(k0_opt), float(k1_opt), P0


def estimate_pcf_gpu(
	points: torch.Tensor,
	r_vals: torch.Tensor,
	sigma: float = 0.005,
	region_size: Tuple[float, float] = (1.0, 1.0),
) -> torch.Tensor:
	"""
	GPU PCF estimator for toroidal domain (Eq. 15).
	All inputs and outputs are torch tensors on the same device.
	"""
	N: int = points.shape[0]
	w, h = region_size
	V_W: float = w * h

	# Toroidal pairwise distances
	diff = points.unsqueeze(1) - points.unsqueeze(0)  # (N, N, 2)
	diff[:, :, 0] = diff[:, :, 0] - w * torch.round(diff[:, :, 0] / w)
	diff[:, :, 1] = diff[:, :, 1] - h * torch.round(diff[:, :, 1] / h)
	dists = torch.sqrt((diff ** 2).sum(dim=2))

	# Extract off-diagonal pair distances: (N*(N-1),)
	mask = ~torch.eye(N, dtype=torch.bool, device=points.device)
	pair_dists = dists[mask]

	# No edge correction needed on torus: gamma_W = V_W
	S_E_r = torch.clamp(2.0 * torch.pi * r_vals, min=1e-15)

	M = r_vals.shape[0]
	P = pair_dists.shape[0]
	chunk_size = max(1, min(M, int(2e9 / (P * 4))))

	G_hat = torch.zeros(M, device=points.device)

	for start in range(0, M, chunk_size):
		end = min(start + chunk_size, M)
		r_chunk = r_vals[start:end]

		kern = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * torch.exp(
			-((r_chunk.unsqueeze(1) - pair_dists.unsqueeze(0)) ** 2) / (2.0 * sigma ** 2)
		)

		kern_sums = kern.sum(dim=1)

		G_hat[start:end] = (V_W / N) * (
			1.0 / (S_E_r[start:end] * (N - 1))
		) * kern_sums

	return G_hat


def compute_gradients_gpu(
	points: torch.Tensor,
	diff: torch.Tensor,
	dists: torch.Tensor,
	r_vals: torch.Tensor,
	residuals: torch.Tensor,
	weights: torch.Tensor,
	sigma: float,
) -> torch.Tensor:
	"""
	GPU gradient computation (Eq. 18).
	Returns gradient for each point: (N, 2).
	"""
	N: int = points.shape[0]
	M: int = r_vals.shape[0]
	device = points.device

	mask = ~torch.eye(N, dtype=torch.bool, device=device)

	# Coefficients for all r: (M,)
	coeff = residuals / (weights * r_vals)

	# Direction vectors: (x_l - x_i) = -diff[i,l] since diff = points_i - points_l
	safe_dists = torch.where(dists < 1e-15, torch.ones_like(dists), dists)
	unit_dir = -diff / safe_dists.unsqueeze(2)  # (N, N, 2), direction from i to l

	# Get (N, N-1) distances and (N, N-1, 2) unit directions
	dists_masked = dists[mask].reshape(N, N - 1)  # (N, N-1)
	unit_dir_masked = unit_dir[mask.unsqueeze(2).expand_as(unit_dir)].reshape(N, N - 1, 2)  # (N, N-1, 2)

	# Inner sum over r for each (i, l) pair
	chunk_size = max(1, min(M, int(2e9 / (N * (N - 1) * 4))))

	inner_sum = torch.zeros(N, N - 1, device=device)  # (N, N-1)

	for start in range(0, M, chunk_size):
		end = min(start + chunk_size, M)
		r_chunk = r_vals[start:end]  # (C,)
		coeff_chunk = coeff[start:end]  # (C,)

		# (N, N-1, 1) - (1, 1, C) -> (N, N-1, C)
		d_minus_r = dists_masked.unsqueeze(2) - r_chunk.unsqueeze(0).unsqueeze(0)

		kern = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * torch.exp(
			-((r_chunk.unsqueeze(0).unsqueeze(0) - dists_masked.unsqueeze(2)) ** 2) / (2.0 * sigma ** 2)
		)  # (N, N-1, C)

		inner_sum += (coeff_chunk.unsqueeze(0).unsqueeze(0) * d_minus_r * kern).sum(dim=2)  # (N, N-1)

	# Gradient: sum over neighbors
	gradients = (unit_dir_masked * inner_sum.unsqueeze(2)).sum(dim=1)  # (N, 2)

	return gradients


def synthesize_stair_blue_noise_gpu(
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
	device: str = 'cuda',
) -> Tuple[NDArray[np.float32], list]:
	"""
	GPU-accelerated Stair blue noise synthesis via weighted least-squares
	PCF matching (Section 5.2, Eq. 18).

	Returns numpy arrays for compatibility with existing code.
	"""
	if device == 'cuda' and not torch.cuda.is_available():
		print("CUDA not available, falling back to CPU")
		device = 'cpu'

	rng = np.random.default_rng(seed)
	(width, height) = region_size
	V: float = width * height
	rho: float = N / V

	# Initialize random point set
	points_np = rng.uniform(0, 1, size=(N, 2)).astype(np.float32) * np.array([width, height], dtype=np.float32)
	points = torch.from_numpy(points_np).to(device)

	# Discretize radius
	r_vals_np = np.arange(r_min, r_max + r_step / 2, r_step, dtype=np.float32)
	r_vals = torch.from_numpy(r_vals_np).to(device)
	M: int = len(r_vals)

	# Target PCF
	G_star_np = stair_pcf_closed_gpu(r_vals_np, k0, k1, P0, rho).astype(np.float32)
	G_star = torch.from_numpy(G_star_np).to(device)

	# Initialize weights
	weights = torch.ones(M, device=device)

	errors: list = []

	for iteration in range(num_iterations):
		# Toroidal pairwise differences and distances
		diff = points.unsqueeze(1) - points.unsqueeze(0)  # (N, N, 2)
		diff[:, :, 0] = diff[:, :, 0] - width * torch.round(diff[:, :, 0] / width)
		diff[:, :, 1] = diff[:, :, 1] - height * torch.round(diff[:, :, 1] / height)
		dists = torch.sqrt((diff ** 2).sum(dim=2))  # (N, N)

		# Estimate current PCF
		G_current = estimate_pcf_gpu(points, r_vals, sigma, region_size)

		# Fitting error
		residuals = G_current - G_star
		weighted_error = ((residuals / weights) ** 2).sum().item()
		errors.append(weighted_error)

		if verbose:
			print(f"  Iteration {iteration:4d}/{num_iterations}: error = {weighted_error:.6f}")

		# Update weights adaptively
		abs_residuals = torch.abs(residuals)
		weights = torch.where(abs_residuals > 1e-10, 1.0 / abs_residuals, torch.tensor(1e10, device=device))

		# Compute gradients
		gradients = compute_gradients_gpu(points, diff, dists, r_vals, residuals, weights, sigma)

		# Normalize and update
		grad_norms = torch.sqrt((gradients ** 2).sum(dim=1, keepdim=True))
		grad_norms = torch.clamp(grad_norms, min=1e-15)
		normalized_grad = gradients / grad_norms

		points = points - step_size * normalized_grad

		# Wrap to region (toroidal)
		points[:, 0] = torch.fmod(points[:, 0], width)
		points[:, 1] = torch.fmod(points[:, 1], height)
		# fmod can return negative values, fix that
		points[:, 0] = torch.where(points[:, 0] < 0, points[:, 0] + width, points[:, 0])
		points[:, 1] = torch.where(points[:, 1] < 0, points[:, 1] + height, points[:, 1])

	return points.cpu().numpy(), errors


def estimate_pcf_vectorized_gpu(
	points: NDArray[np.float32],
	r_vals: NDArray[np.float32],
	sigma: float = 0.005,
	region_size: Tuple[float, float] = (1.0, 1.0),
) -> NDArray[np.float32]:
	N: int = points.shape[0]
	width, height = region_size
	V_W: float = width * height
	S_W: float = 2.0 * (width + height)

	diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
	dists = np.sqrt(np.sum(diff ** 2, axis=2))
	mask = ~np.eye(N, dtype=bool)

	gamma_W = V_W - (S_W / np.pi) * r_vals
	gamma_W = np.maximum(gamma_W, 1e-15)

	S_E_r = 2.0 * np.pi * r_vals
	S_E_r = np.maximum(S_E_r, 1e-15)

	G_hat = np.zeros(len(r_vals))
	pair_dists = dists[mask]

	for idx, r in enumerate(r_vals):
		kern = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-((r - pair_dists)**2) / (2.0 * sigma**2))
		G_hat[idx] = (V_W / gamma_W[idx]) * (V_W / N) * (1.0 / (S_E_r[idx] * (N - 1))) * np.sum(kern)

	return G_hat


def estimate_radial_psd_gpu(
	points: NDArray[np.float32],
	k_max: float = 500.0,
	k_step: float = 1.0,
	num_angles: int = 64,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
	N: int = points.shape[0]
	k_vals = np.arange(k_step, k_max + k_step / 2, k_step)
	angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

	P_k = np.zeros(len(k_vals))

	for ki, k_mag in enumerate(k_vals):
		p_sum = 0.0
		for angle in angles:
			kx = k_mag * np.cos(angle)
			ky = k_mag * np.sin(angle)
			phases = 2.0 * np.pi * (kx * points[:, 0] + ky * points[:, 1])
			val = np.sum(np.exp(-1j * phases))
			p_sum += np.abs(val)**2 / N
		P_k[ki] = p_sum / num_angles

	return k_vals, P_k
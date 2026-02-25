import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def grid_sigma(n_points: int, n_dims: int, base_sigma: float = 1.0) -> float:
	return base_sigma * (n_points ** (-1.0 / n_dims))


def auto_n_periods(sigma: float) -> int:
	# Eqn 21-22 
	return max(1, int(np.ceil(4.0 * sigma)))


def pairwise_toroidal_sq_dist(points: torch.Tensor) -> torch.Tensor:
	diff = points.unsqueeze(1) - points.unsqueeze(0)
	diff = diff - torch.round(diff)
	return (diff * diff).sum(dim=-1)


def energy_and_gradient(
	points: torch.Tensor,
	sigma: float,
	n_periods: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	(N, d) = points.shape
	device = points.device
	dtype = points.dtype
	inv_2s2 = 1.0 / (2.0 * sigma * sigma)

	r = torch.arange(-n_periods, n_periods + 1, device=device, dtype=dtype)	# (2P+1,)

	# Eq 22
	x_i = points.unsqueeze(1)
	x_j = points.unsqueeze(0)
	diff = x_i - x_j
	diff = diff - torch.round(diff) # wraps to toroidal [-0.5, 0.5]
	k = torch.arange(-n_periods, n_periods + 1, device = device, dtype = dtype)
	xi_minus_xj_minus_k = diff.unsqueeze(-1) - k

	gaussian = torch.exp(-xi_minus_xj_minus_k ** 2 * inv_2s2)
	g        = gaussian.sum(dim=-1)
	dg_dxi   = (-inv_2s2 * xi_minus_xj_minus_k * gaussian).sum(dim=-1)

	log_E_ij = torch.log(g.clamp(min=1e-38)).sum(dim=-1)
	E_ij     = torch.exp(log_E_ij)

	# Exclude self-interaction (i == j diagonal, Eq. 15)
	E_ij = E_ij * (1.0 - torch.eye(N, device=device, dtype=dtype))

	# Energy [Eq. 15]
	energy = E_ij.sum() / (2.0 * N)

	# Gradient: dE/dxi = (1/2) * sum_j E_ij * (dg_dxi[i,j] / g[i,j])
	# Factor of 2: E_ij == E_ji and dg_dxi[i,j] == -dg_dxi[j,i], so both halves contribute equally
	grad = (E_ij.unsqueeze(-1) * (dg_dxi / g.clamp(min=1e-38))).sum(dim=1) / 2

	return energy, grad



def optimize_uniform_blue_noise(
	n_points: int,
	n_dims: int = 2,
	base_sigma: float = 1.0,
	n_iterations: int = 10_000,
	step_size: float = 0.5,
	seed: int | None = None,
	device: str | None = None,
	log_every: int = 1000,
	verbose: bool = True,
) -> dict:
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if verbose:
		print(f"[GBN] device={device}, N={n_points}, d={n_dims}, "
		      f"sigma_base={base_sigma}, iters={n_iterations}")

	if seed is not None:
		torch.manual_seed(seed)
		np.random.seed(seed)

	sigma = grid_sigma(n_points, n_dims, base_sigma)
	n_periods = auto_n_periods(sigma)
	# step_size is a fraction of sigma, making convergence rate independent of N.
	# The gradient is normalised to unit RMS before scaling, so raw gradient
	# magnitude (which shrinks as sigma/N) never affects how far points move.
	effective_step = step_size * sigma

	if verbose:
		print(f"[GBN] sigma={sigma:.6f}  n_periods={n_periods}  effective_step={effective_step:.6f}")

	points = torch.rand(n_points, n_dims, device=device, dtype=torch.float32)

	energy_log = []

	for iteration in range(1, n_iterations + 1):
		(energy, grad) = energy_and_gradient(points, sigma, n_periods)

		# Normalise to unit RMS then scale by effective_step  [Algorithm 1, line 21]
		grad = grad / grad.pow(2).mean().sqrt().clamp(min=1e-12)
		points = points - effective_step * grad
		points = points % 1.0

		if verbose and (iteration % log_every == 0 or iteration == 1):
			e_val = energy.item()
			energy_log.append((iteration, e_val))
			print(f"  iter {iteration:>7d}  energy={e_val:.6e}")

	points_np = points.detach().cpu().numpy()
	return {
		'points': points_np,
		'energy': energy_log,
		'sigma': sigma,
		'n_dims': n_dims,
		'n_points': n_points,
		'n_iterations': n_iterations,
	}


def compute_shaping_factors(
	points: torch.Tensor,
	sigma: float,
	n_iter: int = 15,
) -> torch.Tensor:
	N = points.shape[0]
	device = points.device

	# Initialise: uniform density assumption  [Algorithm 2, line 1]
	a = torch.ones(N, device=device, dtype=torch.float32)

	# Precompute pairwise distances (reused across iterations)
	sq_dist = pairwise_toroidal_sq_dist(points)				# (N, N)
	diag_mask = 1.0 - torch.eye(N, device=device, dtype=torch.float32)

	for _ in range(n_iter):
		# Accumulated density at each point k  [Algorithm 2, line 3]
		# d_k = sum_{l!=k} a_l * exp(-a_l * ||x_k - x_l||^2 / (2*sigma^2))
		a_exp = a.unsqueeze(0) * torch.exp(-a.unsqueeze(0) * sq_dist / (2.0 * sigma * sigma))
		a_exp = a_exp * diag_mask
		d = a_exp.sum(dim=1)
		a = d

		mean_a = a.mean()
		if mean_a > 1e-10:
			a = a / mean_a

	return a


def sample_density(
	points: torch.Tensor,
	density_map: torch.Tensor,
) -> torch.Tensor:
	(H, W) = density_map.shape
	xy = points * 2.0 - 1.0
	grid = xy.view(1, -1, 1, 2)
	dm = density_map.view(1, 1, H, W)
	sampled = torch.nn.functional.grid_sample(
		dm, grid, mode='bilinear', padding_mode='border', align_corners=False,
	)
	a = sampled.view(-1).clamp(min=1e-6)
	return a / a.mean()


def adaptive_energy_and_gradient(
	points: torch.Tensor,
	amplitudes: torch.Tensor,
	sigma: float,
	n_periods: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	# Eqn 30-32

	(N, d) = points.shape
	device = points.device
	dtype = points.dtype

	ak = amplitudes.unsqueeze(1)                        # (N, 1)
	al = amplitudes.unsqueeze(0)                        # (1, N)
	a_kl = 2.0 * ak * al / (ak + al + 1e-10)           # (N, N)  [Eq. 32]

	x_i = points.unsqueeze(1)
	x_j = points.unsqueeze(0)
	diff = x_i - x_j
	diff = diff - torch.round(diff)                     # (N, N, d)  toroidal [-0.5, 0.5]
	k = torch.arange(-n_periods, n_periods + 1, device=device, dtype=dtype)
	xi_minus_xj_minus_k = diff.unsqueeze(-1) - k       # (N, N, d, 2P+1)

	# Adaptive Gaussian: width scaled by a_kl per pair  [Eq. 30]
	inv_2s2_kl = a_kl.unsqueeze(-1).unsqueeze(-1) / (2.0 * sigma * sigma)
	gaussian = torch.exp(-xi_minus_xj_minus_k ** 2 * inv_2s2_kl)  # (N, N, d, 2P+1)
	g      = gaussian.sum(dim=-1)                                  # (N, N, d)
	dg_dxi = (-inv_2s2_kl * xi_minus_xj_minus_k * gaussian).sum(dim=-1)  # (N, N, d)

	log_E_ij = torch.log(g.clamp(min=1e-38)).sum(dim=-1)  # (N, N)
	E_ij     = torch.exp(log_E_ij)

	E_ij = E_ij * (1.0 - torch.eye(N, device=device, dtype=dtype))

	energy = (a_kl * E_ij).sum() / (2.0 * N)

	grad = (E_ij.unsqueeze(-1) * (dg_dxi / g.clamp(min=1e-38))).sum(dim=1) / 2

	return energy, grad


def optimize_adaptive_blue_noise(
	density_map: np.ndarray,
	n_points: int,
	base_sigma: float = 1.0,
	n_iterations: int = 10_000,
	step_size: float = 0.5,
	resample_every: int = 100,
	seed: int | None = None,
	device: str | None = None,
	verbose: bool = True,
) -> dict:
	assert density_map.ndim == 2, "density_map must be a 2D array"

	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if seed is not None:
		torch.manual_seed(seed)
		np.random.seed(seed)

	(height, width) = density_map.shape
	sigma = grid_sigma(n_points, 2, base_sigma)
	n_periods = auto_n_periods(sigma)
	effective_step = step_size * sigma

	density = density_map.astype(np.float32)
	density = density / density.sum()

	'''
	flat = density.ravel()
	indices = np.random.choice(len(flat), size=n_points, replace=True, p=flat)
	(rows, cols) = np.unravel_index(indices, (height, width))
	init_y = (rows + np.random.rand(n_points)) / height
	init_x = (cols + np.random.rand(n_points)) / width
	init_pts = np.stack([init_x, init_y], axis=1).astype(np.float32)
	points = torch.tensor(init_pts, device=device, dtype=torch.float32)
	'''
	flat = density.ravel()
	indices = np.random.choice(len(flat), size=n_points, replace=True, p=flat)
	(rows, cols) = np.unravel_index(indices, (height, width))
	init_y = (rows + np.random.rand(n_points)) / height
	init_x = (cols + np.random.rand(n_points)) / width
	init_pts = np.stack([init_x, init_y], axis=1).astype(np.float32)

	points = torch.tensor(init_pts, device=device, dtype=torch.float32)

	
	density_t = torch.tensor(density_map / density_map.mean(), dtype=torch.float32, device=device)
	amplitudes = sample_density(points, density_t)

	for iteration in range(1, n_iterations + 1):
		if iteration % resample_every == 0:
			amplitudes = sample_density(points, density_t)
		energy, grad = adaptive_energy_and_gradient(points, amplitudes, sigma, n_periods)
		#grad = grad / grad.norm(dim=1, keepdim=True).clamp(min=1e-12)
		grad = grad / grad.pow(2).mean().sqrt().clamp(min=1e-12)
		points = points - effective_step * grad
		points = points % 1.0

		if verbose and iteration % 1000 == 0:
			print(f"  [adaptive] iter {iteration:>7d}  energy={energy.item():.6e}")

	return {
		'points': points.detach().cpu().numpy(),
		'amplitudes': amplitudes.detach().cpu().numpy(),
	}





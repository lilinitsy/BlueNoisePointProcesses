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


# For algorithm 2


import torch
import torch.nn.functional as F
import numpy as np


def _wrap_toroidal(diff: torch.Tensor) -> torch.Tensor:
	# wrap to [-0.5, 0.5] like your uniform code
	return diff - torch.round(diff)


def _make_negative_particles_from_density(
	density_map: torch.Tensor,
	n_points: int,
	neg_res: int | tuple[int, int] = 128,
	device: str | torch.device = "cpu",
	dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Build a *fixed* set of negative particles (positions + weights) from a density map.

	- We downsample density_map to neg_res (default 128x128) for tractability.
	- Particle weights are proportional to density and normalized so sum(weights)=n_points.
	- These particles act as negative kernels (attraction).
	"""
	if density_map.ndim != 2:
		raise ValueError("density_map must be a 2D tensor [H, W].")

	d = density_map.to(device=device, dtype=dtype).clamp(min=0)

	# Downsample for manageable M
	d4 = d[None, None, :, :]  # (1,1,H,W)
	if isinstance(neg_res, int):
		neg_h = neg_w = neg_res
	else:
		neg_h, neg_w = neg_res

	ds = F.interpolate(d4, size=(neg_h, neg_w), mode="bilinear", align_corners=False)[0, 0]
	flat = ds.reshape(-1)
	total = flat.sum()
	if total <= 0:
		raise ValueError("density_map sum must be > 0 (after clamping).")

	# weights sum to N (to roughly enforce zero-mean: +N from points, -N from pixels)
	w_pos = (flat / total) * float(n_points)  # positive magnitudes
	w_neg = -w_pos  # negative "charges"

	# positions at pixel centers in [0,1]^2
	yy, xx = torch.meshgrid(
		torch.arange(neg_h, device=device, dtype=dtype),
		torch.arange(neg_w, device=device, dtype=dtype),
		indexing="ij",
	)
	pos = torch.stack([(xx + 0.5) / neg_w, (yy + 0.5) / neg_h], dim=-1).reshape(-1, 2)
	return pos, w_neg


def _sample_points_from_density(
	density_map: torch.Tensor,
	n_points: int,
	seed: int | None,
	device: str | torch.device,
	dtype: torch.dtype,
) -> torch.Tensor:
	"""
	Initialize points by sampling pixels proportional to density, with jitter inside pixel.
	"""
	if seed is not None:
		torch.manual_seed(seed)

	d = density_map.to(device=device, dtype=dtype).clamp(min=0)
	H, W = d.shape
	flat = d.reshape(-1)
	total = flat.sum()
	if total <= 0:
		# fallback to uniform
		return torch.rand(n_points, 2, device=device, dtype=dtype)

	probs = flat / total
	# sample pixel indices
	idx = torch.multinomial(probs, num_samples=n_points, replacement=True)
	yy = (idx // W).to(dtype=dtype)
	xx = (idx % W).to(dtype=dtype)

	# jitter inside pixel
	jx = torch.rand(n_points, device=device, dtype=dtype)
	jy = torch.rand(n_points, device=device, dtype=dtype)

	x = (xx + jx) / W
	y = (yy + jy) / H
	return torch.stack([x, y], dim=-1)


def _algorithm2_update_a(
	points: torch.Tensor,
	a: torch.Tensor,
	sigma: float,
	periodic: bool,
) -> torch.Tensor:
	"""
	Algorithm 2 (one iteration): compute local accumulated density d_k and set a_k=d_k,
	then normalize mean(a)=1.  :contentReference[oaicite:4]{index=4}

	d_k = sum_{l!=k} a_l * exp( -a_l * ||x_k-x_l||^2 / (2 sigma^2) )
	"""
	N, d = points.shape
	assert a.shape == (N,)

	diff = points.unsqueeze(1) - points.unsqueeze(0)  # (N,N,d)
	if periodic:
		diff = _wrap_toroidal(diff)

	dist2 = (diff * diff).sum(dim=-1)  # (N,N)

	# source-shape is a_l (column)
	a_l = a.unsqueeze(0)  # (1,N)
	inv_2s2 = 1.0 / (2.0 * sigma * sigma)

	K = torch.exp(-a_l * dist2 * inv_2s2)

	# exclude diagonal
	K = K * (1.0 - torch.eye(N, device=points.device, dtype=points.dtype))

	d_k = (a_l * K).sum(dim=1)  # (N,)

	# avoid collapse
	d_k = d_k.clamp(min=1e-8)
	d_k = d_k / d_k.mean().clamp(min=1e-8)
	return d_k


def _adaptive_repulsion_energy_grad(
	points: torch.Tensor,
	a: torch.Tensor,
	sigma: float,
	periodic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Adaptive repulsion based on Eq. (30–32) (bounded implementation, no replica sums). :contentReference[oaicite:5]{index=5}

	Energy_ij ~ a_ij * exp( -a_ij * ||diff||^2 / (2 sigma^2) )
	Grad_i    ~ -sum_j a_ij^2 * exp(...) * diff / sigma^2
	"""
	N, d = points.shape
	diff = points.unsqueeze(1) - points.unsqueeze(0)  # (N,N,d)
	if periodic:
		diff = _wrap_toroidal(diff)
	dist2 = (diff * diff).sum(dim=-1)  # (N,N)

	a_i = a.unsqueeze(1)  # (N,1)
	a_j = a.unsqueeze(0)  # (1,N)
	a_ij = (2.0 * a_i * a_j) / (a_i + a_j).clamp(min=1e-8)  # (N,N)

	inv_2s2 = 1.0 / (2.0 * sigma * sigma)
	K = torch.exp(-a_ij * dist2 * inv_2s2)

	mask = (1.0 - torch.eye(N, device=points.device, dtype=points.dtype))
	K = K * mask
	a_ij = a_ij * mask

	E_ij = a_ij * K  # (N,N)

	energy = E_ij.sum() / (2.0 * N)

	# gradient of (a_ij * exp(-a_ij dist2/(2s2))) wrt x_i:
	# = - a_ij^2 * exp(...) * diff / s2
	inv_s2 = 1.0 / (sigma * sigma)
	grad = -(a_ij * a_ij * K).unsqueeze(-1) * diff * inv_s2
	# as in your uniform code, use symmetry: divide by N (not super critical due to RMS normalization)
	grad = grad.sum(dim=1) / N

	return energy, grad


def _density_attraction_energy_grad(
	points: torch.Tensor,
	a: torch.Tensor,
	neg_pos: torch.Tensor,
	neg_w: torch.Tensor,
	sigma: float,
	periodic: bool,
	chunk: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Point-to-negative-particles interaction:
	Each negative particle has:
	  - position neg_pos[m]
	  - weight  neg_w[m] (negative), sum(neg_w) ~= -N

	We use mutual shaping between point (a_i) and pixel-kernel shape b=1:
	  a_ip = 2 a_i * 1 / (a_i + 1)

	Energy term per pair:  (neg_w) * a_ip * exp( -a_ip * ||diff||^2 / (2 sigma^2) )
	Gradient wrt point:    neg_w * (-a_ip^2 * exp(...) * diff / sigma^2)

	Because neg_w is negative, this pulls points toward high-density regions.
	"""
	N, d = points.shape
	M = neg_pos.shape[0]
	device = points.device
	dtype = points.dtype

	inv_2s2 = 1.0 / (2.0 * sigma * sigma)
	inv_s2 = 1.0 / (sigma * sigma)

	# mutual shaping with b=1
	a_ip = (2.0 * a) / (a + 1.0).clamp(min=1e-8)  # (N,)

	total_energy = torch.zeros((), device=device, dtype=dtype)
	total_grad = torch.zeros_like(points)

	for start in range(0, M, chunk):
		end = min(M, start + chunk)
		pos = neg_pos[start:end]              # (C,2)
		w = neg_w[start:end].to(dtype=dtype)  # (C,)

		diff = points.unsqueeze(1) - pos.unsqueeze(0)  # (N,C,2)
		if periodic:
			diff = _wrap_toroidal(diff)

		dist2 = (diff * diff).sum(dim=-1)  # (N,C)

		a_pair = a_ip.unsqueeze(1)  # (N,1)
		K = torch.exp(-a_pair * dist2 * inv_2s2)  # (N,C)

		# energy: sum_{i,c} w[c] * a_pair[i] * K[i,c]
		E = (w.unsqueeze(0) * a_pair * K)  # (N,C)
		total_energy = total_energy + E.sum() / N

		# grad: sum_{c} w[c] * ( -a^2 * K * diff / s^2 )
		g = (w.unsqueeze(0) * (-(a_pair * a_pair) * K) * inv_s2).unsqueeze(-1) * diff  # (N,C,2)
		total_grad = total_grad + g.sum(dim=1) / N

	return total_energy, total_grad


def optimize_adaptive_blue_noise(
	density_map: torch.Tensor,
	n_points: int,
	base_sigma: float = 1.0,
	n_iterations: int = 10_000,
	step_size: float = 0.5,
	attract_strength: float = 1.0,
	neg_res: int | tuple[int, int] = 128,
	shape_updates_per_iter: int = 1,
	shape_damping: float = 1.0,  # 1.0 = pure Alg2 assignment; try 0.1–0.5 if it oscillates
	periodic: bool = False,      # for density maps, bounded is usually what you want
	seed: int | None = None,
	device: str | None = None,
	log_every: int = 1000,
	verbose: bool = True,
	local_step: bool = True,
	a_clip: tuple[float, float] | None = (0.25, 8.0),
	bound_mode: str = "reflect",   # "reflect" or "clamp"
) -> dict:
	"""
	Adaptive GBN optimizer based on Section 5.5:
	- kernel shaping g_k(x)=a_k exp(-a_k ||x-xk||^2/(2 sigma^2))  :contentReference[oaicite:6]{index=6}
	- mutual shaping a_kl = 2 a_k a_l / (a_k + a_l)               :contentReference[oaicite:7]{index=7}
	- alternate between position optimization and Algorithm-2 shaping updates
	  (paper: one shaping iter per optimization step).             :contentReference[oaicite:8]{index=8}
	- density map drives attraction via fixed negative particles (from the paper text). :contentReference[oaicite:9]{index=9}
	"""
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'

	if seed is not None:
		torch.manual_seed(seed)
		np.random.seed(seed)

	if verbose:
		print(f"[GBN adaptive] device={device}, N={n_points}, iters={n_iterations}, "
		      f"sigma_base={base_sigma}, periodic={periodic}, neg_res={neg_res}")

	# Same sigma heuristic as uniform: sigma ~ grid spacing * base_sigma
	n_dims = 2
	sigma = base_sigma * (n_points ** (-1.0 / n_dims))
	effective_step = step_size * sigma

	# initialize points from density map
	points = _sample_points_from_density(
		density_map=density_map,
		n_points=n_points,
		seed=seed,
		device=device,
		dtype=torch.float32
	)

	# fixed negative particles from density map
	neg_pos, neg_w = _make_negative_particles_from_density(
		density_map=density_map,
		n_points=n_points,
		neg_res=neg_res,
		device=device,
		dtype=torch.float32,
	)

	# shaping factors
	a = torch.ones(n_points, device=device, dtype=torch.float32)

	energy_log = []

	if verbose:
		print(f"[GBN adaptive] sigma={sigma:.6f}  effective_step={effective_step:.6f}  "
		      f"neg_particles={neg_pos.shape[0]}  attract_strength={attract_strength}")

	for iteration in range(1, n_iterations + 1):

		# --- Algorithm 2 shaping (one iter per step recommended) ---
		for _ in range(max(0, int(shape_updates_per_iter))):
			new_a = _algorithm2_update_a(points, a, sigma=sigma, periodic=periodic)
			a = (1.0 - shape_damping) * a + shape_damping * new_a

		# --- Repulsion among points (adaptive kernels) ---
		E_rep, grad_rep = _adaptive_repulsion_energy_grad(points, a, sigma=sigma, periodic=periodic)

		# --- Attraction to density map (negative particles) ---
		E_att, grad_att = _density_attraction_energy_grad(
			points, a, neg_pos, neg_w, sigma=sigma, periodic=periodic, chunk=8192
		)

		energy = E_rep + attract_strength * E_att
		grad = grad_rep + attract_strength * grad_att

		# Normalize to unit RMS, then step (same as your uniform code)
		if a_clip is not None:
			a_min, a_max = a_clip
			a = a.clamp(a_min, a_max)
			a = a / a.mean().clamp(min=1e-8)   # keep mean(a)=1 like Alg. 2 :contentReference[oaicite:3]{index=3}

		grad = grad / grad.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-12)

		# Per-point step: smaller steps where a is large (high density)
		if local_step:
			step = (effective_step / torch.sqrt(a).clamp(min=1e-8)).unsqueeze(-1)  # (N,1)
		else:
			step = effective_step

		points = points - step * grad

		# keep in domain
		if periodic:
			points = points % 1.0
		else:
			if bound_mode == "reflect":
				# reflection avoids “sticky” clamping artifacts near edges
				points = torch.remainder(points, 2.0)
				points = torch.where(points <= 1.0, points, 2.0 - points)
			else:
				points = points.clamp(0.0, 1.0)
		if verbose and (iteration % log_every == 0 or iteration == 1):
			e_val = float(energy.detach().cpu())
			energy_log.append((iteration, e_val))
			print(f"  iter {iteration:>7d}  energy={e_val:.6e}  "
			      f"mean(a)={float(a.mean()):.3f}  a[min,max]=({float(a.min()):.3f},{float(a.max()):.3f})")

	points_np = points.detach().cpu().numpy()
	a_np = a.detach().cpu().numpy()

	return {
		'points': points_np,
		'a': a_np,
		'energy': energy_log,
		'sigma': sigma,
		'n_points': n_points,
		'n_iterations': n_iterations,
		'periodic': periodic,
		'neg_particles': int(neg_pos.shape[0]),
	}
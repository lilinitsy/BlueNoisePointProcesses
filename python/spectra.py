import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import signal
from scipy.ndimage import gaussian_filter
import torch


def generate_zone_plate(width: int, height: int, alpha = np.pi) -> np.ndarray:
	x = np.linspace(0, 1, width)
	y = np.linspace(0, 1, height)
	X, Y = np.meshgrid(x, y)
	
	r_squared = X**2 + Y**2
	
	zone_plate = (1 + np.cos(alpha * r_squared)) / 2
	
	return zone_plate


def sample_and_reconstruct_zone_plate(zone_plate: np.ndarray, points: np.ndarray, domain_size: Tuple[float, float], kernel_radius: float = 3.0) -> np.ndarray:
	(width, height) = domain_size
	(img_height, img_width) = zone_plate.shape
	
	points_scaled = points.copy()
	points_scaled[:, 0] = points_scaled[:, 0] * (img_width / width)
	points_scaled[:, 1] = points_scaled[:, 1] * (img_height / height)
	
	sampled_values = np.zeros(len(points_scaled))
	for (idx, point) in enumerate(points_scaled):
		sampled_values[idx] = zone_plate[int(point[1]), int(point[0])]
	
	reconstructed = np.zeros((img_height, img_width))
	
	for i in range(img_height):
		for j in range(img_width):
			pixel_pos = np.array([j, i])
			distances = np.linalg.norm(points_scaled - pixel_pos, axis=1)			
			weights = np.exp(-(distances**2) / (2 * kernel_radius**2))
			
			if np.sum(weights) > 1e-10:
				reconstructed[i, j] = np.sum(weights * sampled_values) / np.sum(weights)
			else:
				reconstructed[i, j] = 0.5  

	return reconstructed



def sample_and_reconstruct_zone_plate_tiled(zone_plate: np.ndarray, 
                                 tile_points: np.ndarray,
                                 tile_size: int = 64,
                                 kernel_radius: float = 2.0) -> np.ndarray:
	"""
	Reconstruct zone plate using tiled blue noise sampling.
	
	Parameters:
	-----------
	zone_plate : np.ndarray
		Zone plate to sample
	tile_points : np.ndarray
		Sample points from a single tile (Nx2 array of x,y coordinates)
	tile_size : int
		Size of the tile that tile_points came from
	kernel_radius : float
		Gaussian reconstruction kernel radius
	"""
	height, width = zone_plate.shape
	
	# Tile the points across the entire image
	all_points = []
	
	for ty in range(0, height, tile_size):
		for tx in range(0, width, tile_size):
			# Offset points for this tile
			offset_points = tile_points.copy()
			offset_points[:, 0] += tx
			offset_points[:, 1] += ty
			
			# Clip to image bounds
			valid = (offset_points[:, 0] < width) & (offset_points[:, 1] < height)
			offset_points = offset_points[valid]
			
			all_points.append(offset_points)
	
	all_points = np.vstack(all_points)
	
	# Sample values at all points
	sampled_values = zone_plate[all_points[:, 1].astype(int), 
	                            all_points[:, 0].astype(int)]
	
	# Reconstruct using Gaussian splatting
	reconstructed = np.zeros((height, width))
	
	for i in range(height):
		for j in range(width):
			pixel_pos = np.array([j, i])
			distances = np.linalg.norm(all_points - pixel_pos, axis=1)
			weights = np.exp(-(distances**2) / (2 * kernel_radius**2))
			
			if np.sum(weights) > 1e-10:
				reconstructed[i, j] = np.sum(weights * sampled_values) / np.sum(weights)
			else:
				reconstructed[i, j] = 0.5
	
	return reconstructed




def compute_1d_power_spectrum(points: np.ndarray, domain_size: float = 1.0, num_bins: int = 100, max_freq: float = 5.0):
	points = np.asarray(points, dtype=np.float64)
	if points.ndim != 2 or points.shape[1] < 2:
		raise ValueError("points must be an (N,2) array for 2D power spectrum.")

	# Normalize to unit torus [0,1)^2 with per-axis scaling (important!)
	if isinstance(domain_size, tuple) or isinstance(domain_size, list) or isinstance(domain_size, np.ndarray):
		L = np.asarray(domain_size, dtype = np.float64)
		if L.size != 2:
			raise ValueError("domain_size tuple/list must have length 2 for 2D points.")
		points_norm = points[:, :2] / L[None, :]
	else:
		L = float(domain_size)
		points_norm = points[:, :2] / L

	points_norm = np.mod(points_norm, 1.0)  # torus wrap
	N = points_norm.shape[0]
	if N == 0:
		return np.linspace(0, max_freq, num_bins), np.zeros(num_bins)

	# Normalization used in the paper: ν = |k| / sqrt(N) for unit area
	freq_scale = np.sqrt(N) 

	K = int(np.ceil(max_freq * freq_scale))
	K = max(K, 1)

	max_lattice_side = 512
	full_side = 2 * K + 1
	step = int(np.ceil(full_side / max_lattice_side)) if full_side > max_lattice_side else 1

	k_vals = np.arange(-K, K + 1, step, dtype = np.int32)
	kx, ky = np.meshgrid(k_vals, k_vals, indexing="xy")
	kx = kx.ravel()
	ky = ky.ravel()

	# Drop DC
	nonzero = ~((kx == 0) & (ky == 0))
	kx = kx[nonzero]
	ky = ky[nonzero]

	# Normalized radial frequency per lattice vector
	k_mag = np.sqrt(kx.astype(np.float64) ** 2 + ky.astype(np.float64) ** 2)
	nu = k_mag / freq_scale  # ν = |k| / sqrt(N)

	keep = (nu > 0) & (nu <= max_freq)
	kx = kx[keep]
	ky = ky[keep]
	nu = nu[keep]

	# Compute P(k) in chunks: P(k) = (1/N) |sum_j exp(-2πi k·x_j)|^2
	def _power_for_kvectors(kx_arr, ky_arr, chunk = 4096):
		out = np.empty(kx_arr.shape[0], dtype=np.float64)
		x = points_norm[:, 0]
		y = points_norm[:, 1]
		two_pi = 2.0 * np.pi

		for start in range(0, kx_arr.shape[0], chunk):
			end = min(start + chunk, kx_arr.shape[0])
			kx_c = kx_arr[start:end].astype(np.float64)
			ky_c = ky_arr[start:end].astype(np.float64)

			# phases: (N, Mchunk) = x[:,None]*kx[None,:] + y[:,None]*ky[None,:]
			ph = two_pi * (x[:, None] * kx_c[None, :] + y[:, None] * ky_c[None, :])
			S = np.exp(-1j * ph).sum(axis=0)
			out[start:end] = (S.real * S.real + S.imag * S.imag) / N
		return out

	Pk = _power_for_kvectors(kx, ky)

	# Radial binning in normalized ν
	bins = np.linspace(0.0, max_freq, num_bins + 1)
	bin_centers = 0.5 * (bins[:-1] + bins[1:])
	power = np.zeros(num_bins, dtype = np.float64)
	counts = np.zeros(num_bins, dtype = np.int64)

	inds = np.searchsorted(bins, nu, side="right") - 1
	valid = (inds >= 0) & (inds < num_bins)
	inds = inds[valid]
	Pk_v = Pk[valid]

	np.add.at(power, inds, Pk_v)
	np.add.at(counts, inds, 1)

	nonempty = counts > 0
	power[nonempty] /= counts[nonempty]

	return bin_centers, power


def compute_2d_power_spectrum(points: np.ndarray, domain_size: float = 1.0, grid_size: int = 128):
	points = np.asarray(points, dtype=np.float64)
	if points.ndim != 2 or points.shape[1] < 2:
		raise ValueError("points must be an (N,2) array for 2D power spectrum.")

	if isinstance(domain_size, tuple) or isinstance(domain_size, list) or isinstance(domain_size, np.ndarray):
		L = np.asarray(domain_size, dtype = np.float64)
		if L.size != 2:
			raise ValueError("domain_size tuple/list must have length 2 for 2D points.")
		points_norm = points[:, :2] / L[None, :]
	else:
		L = float(domain_size)
		points_norm = points[:, :2] / L

	points_norm = np.mod(points_norm, 1.0)
	N = points_norm.shape[0]
	if N == 0:
		return np.zeros((grid_size, grid_size), dtype = np.float64), (-5.0, 5.0, -5.0, 5.0)

	max_freq = 5.0
	freq_scale = np.sqrt(N) 
	K = int(np.ceil(max_freq * freq_scale))
	K = max(K, 1)

	k1d = np.rint(np.linspace(-K, K, grid_size)).astype(np.int32)
	kx_grid, ky_grid = np.meshgrid(k1d, k1d, indexing="xy")
	kx_flat = kx_grid.ravel()
	ky_flat = ky_grid.ravel()

	power_flat = np.empty(kx_flat.shape[0], dtype = np.float64)
	x = points_norm[:, 0]
	y = points_norm[:, 1]
	two_pi = 2.0 * np.pi

	chunk = 4096
	for start in range(0, kx_flat.shape[0], chunk):
		end = min(start + chunk, kx_flat.shape[0])
		kx_c = kx_flat[start:end].astype(np.float64)
		ky_c = ky_flat[start:end].astype(np.float64)

		ph = two_pi * (x[:, None] * kx_c[None, :] + y[:, None] * ky_c[None, :])
		S = np.exp(-1j * ph).sum(axis = 0)
		power_flat[start:end] = (S.real * S.real + S.imag * S.imag) / N

	power_spectrum_2d = power_flat.reshape((grid_size, grid_size))

	c = grid_size // 2
	ix0 = int(np.argmin(np.abs(k1d)))
	iy0 = ix0
	neighbors = power_spectrum_2d[max(0, iy0 - 1):min(grid_size, iy0 + 2),
	                              max(0, ix0 - 1):min(grid_size, ix0 + 2)].copy()
	neighbors = neighbors.ravel()
	neighbors = neighbors[neighbors != power_spectrum_2d[iy0, ix0]]
	power_spectrum_2d[iy0, ix0] = np.mean(neighbors) if neighbors.size else np.mean(power_spectrum_2d)

	k_norm = k1d.astype(np.float64) / freq_scale
	freq_extent = (k_norm[0], k_norm[-1], k_norm[0], k_norm[-1])

	return power_spectrum_2d, freq_extent




def visualize_sampling_analysis(points: np.ndarray, domain_size = (256, 256), zone_plate_alpha = np.pi * 64.0, kernel_radius = 1.5):
	(fig, axes) = plt.subplots(1, 4, figsize=(20, 5))
	
	# Left: Point distribution
	axes[0].scatter(points[:, 0], points[:, 1], c='darkblue', s=10, alpha=0.7)
	axes[0].set_xlim(0, domain_size[0])
	axes[0].set_ylim(0, domain_size[1])
	axes[0].set_aspect('equal')
	axes[0].set_title(f'Point Distribution\n({len(points)} samples)')
	axes[0].set_xlabel('X')
	axes[0].set_ylabel('Y')
	axes[0].grid(True, alpha=0.3)
	
	# Left-Middle: Original zone plate
	zone_plate = generate_zone_plate(int(domain_size[0]), int(domain_size[1]), alpha = zone_plate_alpha)
	axes[1].imshow(zone_plate, cmap='gray')
	axes[1].set_title(f'Original Zone Plate\n(alpha = {zone_plate_alpha:.2f})')
	axes[1].axis('off')
	
	# Right-Middle: Reconstructed sampled zone plate
	#reconstructed_zone = sample_and_reconstruct_zone_plate(zone_plate, points, domain_size, kernel_radius = kernel_radius)
	reconstructed_zone = sample_and_reconstruct_zone_plate_tiled(zone_plate, points, tile_size = 64, kernel_radius = kernel_radius)

	axes[2].imshow(reconstructed_zone, cmap='gray')
	axes[2].set_title(f'Sampled Zone Plate\n({len(points)} samples)')
	axes[2].axis('off')
	
	# Right: Power spectrum with 2D inset
	max_dim = max(domain_size)
	(freqs_1d, power_1d) = compute_1d_power_spectrum(points, max_dim)
	(power_2d, extent) = compute_2d_power_spectrum(points, max_dim, grid_size = 256)

	P = power_2d / np.nanmean(power_2d) # normalize to 0-1

	# Ignore DC
	c = P.shape[0] // 2
	P[c, c] = np.nan

	# % clip
	(vmin, vmax) = np.nanpercentile(P, [5, 99.5])
	
	axes[3].plot(freqs_1d, power_1d, linewidth=2)
	axes[3].set_xlabel('Frequency', fontsize=12)
	axes[3].set_ylabel('Power', fontsize=12)
	axes[3].set_title('Power Spectrum with 2D Inset', fontsize=14)
	axes[3].grid(True, alpha=0.3)
	axes[3].set_ylim([0, 4])
	axes[3].set_xlim([0, 5])
	
	axins = inset_axes(axes[3], width="40%", height="40%", loc='upper right')
	
	# Normalize to [0, 1] for better contrast
	power_2d_norm = power_2d / (np.max(power_2d) + 1e-10)
	axins.imshow(P, cmap='gray', extent=extent, origin='lower', vmin=0, vmax=1)
	axins.set_xticks([])
	axins.set_yticks([])
	axins.set_title('2D', fontsize=8)
	
	plt.tight_layout()
	return fig




def visualize_zone_plate(width: int = 256, height: int = 256, alpha = np.pi):
	"""
	Visualize the zone plate test pattern to understand what we're sampling.
	"""
	zone_plate = generate_zone_plate(width, height, alpha)
	
	fig, ax = plt.subplots(1, 1, figsize=(8, 8))
	im = ax.imshow(zone_plate, cmap='gray', extent=[0, width, 0, height])
	ax.set_title(f'Zone Plate Pattern (α={alpha:.2f})\nLow frequencies at edges → High frequencies at center', fontsize=14)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	plt.colorbar(im, ax=ax, label='Intensity')
	plt.tight_layout()
	plt.show()
	return fig




'''
CUDA VERSIONS

'''

def sample_and_reconstruct_zone_plate_tiled_gpu(
    zone_plate: np.ndarray,
    tile_points: np.ndarray,
    tile_size: int = 64,
    kernel_radius: float = 2.0,
    device: str = 'cuda',
    batch_size: int = 64
) -> np.ndarray:
	if device == 'cuda' and not torch.cuda.is_available():
		print("CUDA not available, falling back to CPU")
		device = 'cpu'

	height, width = zone_plate.shape

	# Convert inputs to torch tensors
	zone_plate_torch = torch.from_numpy(zone_plate).float().to(device)
	tile_points_torch = torch.from_numpy(tile_points).float().to(device)

	# Tile the points across the entire image
	all_points = []

	for ty in range(0, height, tile_size):
		for tx in range(0, width, tile_size):
			offset_points = tile_points_torch.clone()
			offset_points[:, 0] += tx
			offset_points[:, 1] += ty
			
			# Clip to image bounds
			valid = (offset_points[:, 0] < width) & (offset_points[:, 1] < height)
			offset_points = offset_points[valid]
			
			all_points.append(offset_points)

	all_points = torch.cat(all_points, dim = 0)  # Shape: (N, 2) where N is total number of points

	sample_y = all_points[:, 1].long().clamp(0, height - 1)
	sample_x = all_points[:, 0].long().clamp(0, width - 1)
	sampled_values = zone_plate_torch[sample_y, sample_x]  # Shape: (N,)

	reconstructed = torch.zeros((height, width), device=device)
	kernel_const = -1.0 / (2 * kernel_radius**2)

	for start_row in range(0, height, batch_size):
		end_row = min(start_row + batch_size, height)
		num_rows = end_row - start_row
		
		y_grid = torch.arange(start_row, end_row, device=device)[:, None].expand(num_rows, width)
		x_grid = torch.arange(width, device=device)[None, :].expand(num_rows, width)
		pixel_positions = torch.stack([x_grid, y_grid], dim = 2)  

		pixel_positions = pixel_positions.reshape(num_rows * width, 1, 2)  # Shape: (num_rows * width, 1, 2)
		all_points_broadcast = all_points[None, :, :]  # Shape: (1, N, 2)
		
		distances_sq = ((pixel_positions - all_points_broadcast) ** 2).sum(dim = 2)
		
		weights = torch.exp(distances_sq * kernel_const)  # Shape: (num_rows*width, N)
		
		weight_sum = weights.sum(dim=1, keepdim=True)  # Shape: (num_rows*width, 1)
		weighted_values = (weights * sampled_values[None, :]).sum(dim=1)  # Shape: (num_rows*width,)
		
		# Avoid division by zero
		result = torch.where(
			weight_sum.squeeze() > 1e-10,
			weighted_values / weight_sum.squeeze(),
			torch.tensor(0.5, device=device)
		)
		
		reconstructed[start_row:end_row, :] = result.reshape(num_rows, width)

	return reconstructed.cpu().numpy()


def sample_and_reconstruct_zone_plate_tiled_optimized(
    zone_plate: np.ndarray,
    tile_points: np.ndarray,
    tile_size: int = 64,
    kernel_radius: float = 2.0,
    device: str = 'cuda',
    cutoff_radius: Optional[float] = None
) -> np.ndarray:
	if device == 'cuda' and not torch.cuda.is_available():
		print("CUDA not available, falling back to CPU")
		device = 'cpu'

	if cutoff_radius is None:
		cutoff_radius = 3 * kernel_radius

	height, width = zone_plate.shape

	# Convert inputs to torch tensors
	zone_plate_torch = torch.from_numpy(zone_plate).float().to(device)
	tile_points_torch = torch.from_numpy(tile_points).float().to(device)

	# Tile the points across the entire image
	all_points = []

	for ty in range(0, height, tile_size):
		for tx in range(0, width, tile_size):
			offset_points = tile_points_torch.clone()
			offset_points[:, 0] += tx
			offset_points[:, 1] += ty
			
			valid = (offset_points[:, 0] < width) & (offset_points[:, 1] < height)
			offset_points = offset_points[valid]
			
			all_points.append(offset_points)

	all_points = torch.cat(all_points, dim=0)

	# Sample values
	sample_y = all_points[:, 1].long().clamp(0, height - 1)
	sample_x = all_points[:, 0].long().clamp(0, width - 1)
	sampled_values = zone_plate_torch[sample_y, sample_x]

	# Reconstruct with spatial cutoff
	reconstructed = torch.zeros((height, width), device=device)
	kernel_const = -1.0 / (2 * kernel_radius**2)
	cutoff_radius_sq = cutoff_radius ** 2

	# Process each pixel
	for i in range(height):
		for j in range(width):
			pixel_pos = torch.tensor([j, i], dtype=torch.float32, device=device)
			
			distances_sq = ((all_points - pixel_pos) ** 2).sum(dim=1)
			nearby_mask = distances_sq < cutoff_radius_sq
			
			if nearby_mask.any():
				nearby_distances_sq = distances_sq[nearby_mask]
				nearby_values = sampled_values[nearby_mask]
				
				weights = torch.exp(nearby_distances_sq * kernel_const)
				weight_sum = weights.sum()
				
				if weight_sum > 1e-10:
					reconstructed[i, j] = (weights * nearby_values).sum() / weight_sum
				else:
					reconstructed[i, j] = 0.5
			else:
				reconstructed[i, j] = 0.5

	return reconstructed.cpu().numpy()


def sample_and_reconstruct_zone_plate_tiled_fastest(
    zone_plate: np.ndarray,
    tile_points: np.ndarray,
    tile_size: int = 64,
    kernel_radius: float = 2.0,
    device: str = 'cuda'
) -> np.ndarray:

	if device == 'cuda' and not torch.cuda.is_available():
		print("CUDA not available, falling back to CPU")
		device = 'cpu'

	height, width = zone_plate.shape

	# Convert inputs to torch tensors
	zone_plate_torch = torch.from_numpy(zone_plate).float().to(device)
	tile_points_torch = torch.from_numpy(tile_points).float().to(device)

	# Tile the points
	all_points = []
	for ty in range(0, height, tile_size):
		for tx in range(0, width, tile_size):
			offset_points = tile_points_torch.clone()
			offset_points[:, 0] += tx
			offset_points[:, 1] += ty
			
			valid = (offset_points[:, 0] < width) & (offset_points[:, 1] < height)
			all_points.append(offset_points[valid])

	all_points = torch.cat(all_points, dim=0)  # Shape: (N, 2)

	sample_y = all_points[:, 1].long().clamp(0, height - 1)
	sample_x = all_points[:, 0].long().clamp(0, width - 1)
	sampled_values = zone_plate_torch[sample_y, sample_x]  # Shape: (N,)

	y_coords, x_coords = torch.meshgrid(
		torch.arange(height, device=device),
		torch.arange(width, device=device),
		indexing='ij'
	)
	pixel_positions = torch.stack([x_coords, y_coords], dim=2)  # Shape: (H, W, 2)

	pixel_positions = pixel_positions.reshape(height * width, 1, 2)  # (H*W, 1, 2)
	all_points_broadcast = all_points[None, :, :]  # (1, N, 2)

	distances_sq = ((pixel_positions - all_points_broadcast) ** 2).sum(dim=2)  # (H*W, N)

	kernel_const = -1.0 / (2 * kernel_radius**2)
	weights = torch.exp(distances_sq * kernel_const)  # (H*W, N)

	weight_sum = weights.sum(dim=1, keepdim=True)  # (H*W, 1)
	weighted_values = (weights * sampled_values[None, :]).sum(dim=1)  # (H*W,)

	result = torch.where(
		weight_sum.squeeze() > 1e-10,
		weighted_values / weight_sum.squeeze(),
		torch.tensor(0.5, device=device)
	)

	reconstructed = result.reshape(height, width)

	return reconstructed.cpu().numpy()


def visualize_sampling_analysis_gpu(points, domain_size = (256, 256), zone_plate_alpha = np.pi * 64.0, 
                                kernel_radius = 1.5, device = 'cuda', batch_size = 64):
	# Convert torch tensor to numpy if needed
	if isinstance(points, torch.Tensor):
		points_np = points.cpu().numpy()
	else:
		points_np = points

	# Check device availability
	if device == 'cuda' and not torch.cuda.is_available():
		print("CUDA not available, falling back to CPU")
		device = 'cpu'

	fig, axes = plt.subplots(1, 4, figsize=(20, 5))

	# Left: Point distribution
	axes[0].scatter(points_np[:, 0], points_np[:, 1], c='darkblue', s=10, alpha=0.7)
	axes[0].set_xlim(0, domain_size[0])
	axes[0].set_ylim(0, domain_size[1])
	axes[0].set_aspect('equal')
	axes[0].set_title(f'Point Distribution\n({len(points_np)} samples)')
	axes[0].set_xlabel('X')
	axes[0].set_ylabel('Y')
	axes[0].grid(True, alpha=0.3)

	# Left-Middle: Original zone plate
	zone_plate = generate_zone_plate(int(domain_size[0]), int(domain_size[1]), alpha=zone_plate_alpha)
	axes[1].imshow(zone_plate, cmap='gray')
	axes[1].set_title(f'Original Zone Plate\n(alpha = {zone_plate_alpha:.2f})')
	axes[1].axis('off')

	# Right-Middle: Reconstructed sampled zone plate (GPU accelerated!)
	reconstructed_zone = sample_and_reconstruct_zone_plate_tiled_gpu(
		zone_plate, 
		points_np, 
		tile_size=64, 
		kernel_radius=kernel_radius,
		device=device,
		batch_size=batch_size
	)

	axes[2].imshow(reconstructed_zone, cmap='gray')
	axes[2].set_title(f'Sampled Zone Plate\n({len(points_np)} samples, GPU)')
	axes[2].axis('off')

	# Right: Power spectrum with 2D inset
	max_dim = max(domain_size)
	freqs_1d, power_1d = compute_1d_power_spectrum(points_np, max_dim)
	power_2d, extent = compute_2d_power_spectrum(points_np, max_dim, grid_size=256)

	P = power_2d / np.nanmean(power_2d)  # normalize

	# Ignore DC
	c = P.shape[0] // 2
	P[c, c] = np.nan

	# Clip percentiles
	vmin, vmax = np.nanpercentile(P, [5, 99.5])

	axes[3].plot(freqs_1d, power_1d, linewidth=2)
	axes[3].set_xlabel('Frequency', fontsize=12)
	axes[3].set_ylabel('Power', fontsize=12)
	axes[3].set_title('Power Spectrum with 2D Inset', fontsize=14)
	axes[3].grid(True, alpha=0.3)
	axes[3].set_ylim([0, 4])
	axes[3].set_xlim([0, 5])

	axins = inset_axes(axes[3], width="40%", height="40%", loc='upper right')

	axins.imshow(P, cmap='gray', extent=extent, origin='lower', vmin=vmin, vmax=vmax)
	axins.set_xticks([])
	axins.set_yticks([])
	axins.set_title('2D', fontsize=8)

	plt.tight_layout()
	return fig
import torch
import numpy as np
from typing import Tuple

def void_and_cluster_gpu(dimensions: Tuple[int, int], sigma: float = 1.9, device: str = 'cuda') -> torch.Tensor:
	"""
	GPU-accelerated void and cluster algorithm using PyTorch.
	"""
	width, height = dimensions
	num_pixels = width * height

	if device == 'cuda' and not torch.cuda.is_available():
		print("CUDA not available, falling back to CPU")
		device = 'cpu'

	rank = torch.zeros((height, width), dtype=torch.int32, device=device)

	kernel_size = int(6 * sigma + 1)
	if kernel_size % 2 == 0:
		kernel_size += 1

	y, x = torch.meshgrid(
		torch.arange(kernel_size, device=device),
		torch.arange(kernel_size, device=device),
		indexing = 'ij'
	)

	center = kernel_size // 2
	kernel = torch.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
	kernel_sum = float(kernel.sum())

	print("Step 1: Generating initial binary pattern...")
	initial_ones = max(int(num_pixels * 0.1), 1)
	binary_pattern, energy_lut = generate_initial_pattern_gpu(width, height, sigma, initial_ones, kernel, kernel_size, device)
	prototype = binary_pattern.clone()
	prototype_lut = energy_lut.clone()

	print("Step 2: Assigning ranks by removing clusters...")
	binary_pattern = prototype.clone()
	energy_lut = prototype_lut.clone()

	count = int(binary_pattern.sum())
	while count > 0:
		cluster_y, cluster_x = find_cluster_gpu(binary_pattern, energy_lut)
		count -= 1
		rank[cluster_y, cluster_x] = count
		binary_pattern[cluster_y, cluster_x] = 0
		update_energy_lut_gpu(energy_lut, cluster_y, cluster_x, -1, kernel, kernel_size, height, width)

	print("Step 3: Adding points until half full...")
	binary_pattern = prototype.clone()
	energy_lut = prototype_lut.clone()

	count = int(binary_pattern.sum())
	target = num_pixels // 2
	while count < target:
		void_y, void_x = find_void_gpu(binary_pattern, energy_lut)
		rank[void_y, void_x] = count
		binary_pattern[void_y, void_x] = 1
		update_energy_lut_gpu(energy_lut, void_y, void_x, 1, kernel, kernel_size, height, width)
		count += 1

	print("Step 4: Adding remaining points...")

	while count < num_pixels:
		inverted_energy = kernel_sum - energy_lut
		masked_energy = torch.where(binary_pattern == 0, inverted_energy, torch.tensor(float('-inf'), device=device))
		idx = torch.argmax(masked_energy)
		void_y, void_x = idx // width, idx % width
		
		rank[void_y, void_x] = count
		binary_pattern[void_y, void_x] = 1
		update_energy_lut_gpu(energy_lut, void_y, void_x, 1, kernel, kernel_size, height, width)
		count += 1

	dither_array = (rank.float() * 255 / num_pixels).to(torch.uint8)
	return dither_array


def update_energy_lut_gpu(energy_lut: torch.Tensor, py: int, px: int, sign: int, 
                      kernel: torch.Tensor, kernel_size: int, height: int, width: int) -> None:
	half_size = kernel_size // 2

	# Calculate wrapped indices for toroidal topology
	ys = (py + torch.arange(kernel_size, device=energy_lut.device) - half_size) % height
	xs = (px + torch.arange(kernel_size, device=energy_lut.device) - half_size) % width

	if (kernel_size <= height) and (kernel_size <= width):
		# Simple case: kernel fits without wrapping issues
		energy_lut[ys[:, None], xs[None, :]] += sign * kernel
	else:
		# Handle wrapping by accumulating
		y_grid, x_grid = torch.meshgrid(ys, xs, indexing='ij')
		energy_lut.index_put_((y_grid, x_grid), sign * kernel, accumulate=True)


def generate_initial_pattern_gpu(width: int, height: int, sigma: float, num_ones: int,
                             kernel: torch.Tensor, kernel_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
	binary_pattern = torch.zeros((height, width), dtype=torch.int32, device=device)
	energy_lut = torch.zeros((height, width), dtype=torch.float32, device=device)

	# Randomly place initial ones
	placed = 0
	while placed < num_ones:
		y, x = torch.randint(0, height, (1,), device=device).item(), torch.randint(0, width, (1,), device=device).item()
		if binary_pattern[y, x] == 0:
			binary_pattern[y, x] = 1
			update_energy_lut_gpu(energy_lut, y, x, 1, kernel, kernel_size, height, width)
			placed += 1

	# Remove tightest cluster, add to largest void
	iteration = 0
	while True:
		cluster_y, cluster_x = find_cluster_gpu(binary_pattern, energy_lut)
		binary_pattern[cluster_y, cluster_x] = 0
		update_energy_lut_gpu(energy_lut, cluster_y, cluster_x, -1, kernel, kernel_size, height, width)
		
		void_y, void_x = find_void_gpu(binary_pattern, energy_lut)
		binary_pattern[void_y, void_x] = 1
		update_energy_lut_gpu(energy_lut, void_y, void_x, 1, kernel, kernel_size, height, width)
		
		if (cluster_y, cluster_x) == (void_y, void_x):
			break
		
		iteration += 1
		if iteration > 10000:
			print("Warning: Initial pattern generation didn't converge after 10000 iterations")
			break

	print(f"Initial pattern converged after {iteration} iterations")
	return binary_pattern, energy_lut


def find_cluster_gpu(binary_pattern: torch.Tensor, energy_lut: torch.Tensor) -> Tuple[int, int]:
	"""Find the tightest cluster (highest energy where pattern = 1)."""
	height, width = binary_pattern.shape
	masked_energy = torch.where(binary_pattern == 1, energy_lut, torch.tensor(float('-inf'), device=binary_pattern.device))
	idx = torch.argmax(masked_energy)
	y, x = idx // width, idx % width
	return int(y), int(x)


def find_void_gpu(binary_pattern: torch.Tensor, energy_lut: torch.Tensor) -> Tuple[int, int]:
	"""Find the largest void (lowest energy where pattern = 0)."""
	height, width = binary_pattern.shape
	masked_energy = torch.where(binary_pattern == 0, energy_lut, torch.tensor(float('inf'), device=binary_pattern.device))
	idx = torch.argmin(masked_energy)
	y, x = idx // width, idx % width
	return int(y), int(x)


def void_cluster_to_points_gpu(dither_array: torch.Tensor, num_samples: int) -> torch.Tensor:
	height, width = dither_array.shape
	total_pixels = height * width
	threshold = int((num_samples / total_pixels) * 255)

	mask = dither_array < threshold
	y_coords, x_coords = torch.where(mask)

	if len(x_coords) > num_samples:
		values = dither_array[y_coords, x_coords]
		indices = torch.argsort(values)[:num_samples]
		x_coords = x_coords[indices]
		y_coords = y_coords[indices]

	points = torch.stack([x_coords.float(), y_coords.float()], dim=1)

	# jitter
	points += torch.rand_like(points) - 0.5

	# Clamp to domain
	points[:, 0] = torch.clamp(points[:, 0], 0, width - 1)
	points[:, 1] = torch.clamp(points[:, 1], 0, height - 1)

	return points


def void_cluster_to_points_gpu(dither_array: torch.Tensor, num_samples: int) -> torch.Tensor:
    height, width = dither_array.shape
    total_pixels = height * width
    threshold = int((num_samples / total_pixels) * 255)
    
    mask = dither_array < threshold
    y_coords, x_coords = torch.where(mask)
    
    if len(x_coords) > num_samples:
        values = dither_array[y_coords, x_coords]
        indices = torch.argsort(values)[:num_samples]
        x_coords = x_coords[indices]
        y_coords = y_coords[indices]
    
    points = torch.stack([x_coords.float(), y_coords.float()], dim=1)
    
    # Add uniform jitter in [-0.5, 0.5] to center samples within pixels
    points += torch.rand_like(points) - 0.5
    
    # Clamp to domain
    points[:, 0] = torch.clamp(points[:, 0], 0, width - 1)
    points[:, 1] = torch.clamp(points[:, 1], 0, height - 1)
    
    return points
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

# https://blog.demofox.org/2019/06/25/generating-blue-noise-textures-with-void-and-cluster/
def void_and_cluster(dimensions: Tuple[int, int], sigma: float = 1.9) -> np.ndarray:
	(width, height) = (dimensions[0], dimensions[1])
	num_pixels = width * height
	rank = np.zeros((height, width), dtype=np.int32)

	# Gaussian kernel
	kernel_size = int(6 * sigma + 1)
	if kernel_size % 2 == 0:
		kernel_size += 1

	(y, x) = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size), indexing='ij')
	center = kernel_size // 2
	kernel = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
	kernel_sum = float(np.sum(kernel))

	# Step 1: Generate initial binary pattern
	print("Step 1: Generating initial binary pattern...")
	initial_ones = max(int(num_pixels * 0.1), 1)
	binary_pattern, energy_lut = generate_initial_pattern(width, height, sigma, initial_ones, kernel, kernel_size)
	prototype = binary_pattern.copy()
	prototype_lut = energy_lut.copy()

	# Step 2: Remove points to assign ranks
	print("Step 2: Assigning ranks by removing clusters...")
	binary_pattern = prototype.copy()
	energy_lut = prototype_lut.copy()

	count = int(np.sum(binary_pattern))
	while count > 0:
		(cluster_y, cluster_x) = find_cluster(binary_pattern, energy_lut)
		count -= 1
		rank[cluster_y, cluster_x] = count
		binary_pattern[cluster_y, cluster_x] = 0
		update_energy_lut(energy_lut, cluster_y, cluster_x, -1, kernel, kernel_size, height, width)

	# Step 3: Add points until half full
	print("Step 3: Adding points until half full...")
	binary_pattern = prototype.copy()
	energy_lut = prototype_lut.copy()

	count = int(np.sum(binary_pattern))
	target = num_pixels // 2
	while count < target:
		(void_y, void_x) = find_void(binary_pattern, energy_lut)
		rank[void_y, void_x] = count
		binary_pattern[void_y, void_x] = 1
		update_energy_lut(energy_lut, void_y, void_x, 1, kernel, kernel_size, height, width)
		count += 1

	# Step 4: Add remaining points
	print("Step 4: Adding remaining points...")
	
	while count < num_pixels:
		inverted_energy = kernel_sum - energy_lut
		masked_energy = np.where(binary_pattern == 0, inverted_energy, -np.inf)
		idx = np.argmax(masked_energy)
		void_y, void_x = idx // width, idx % width
		
		rank[void_y, void_x] = count
		binary_pattern[void_y, void_x] = 1
		update_energy_lut(energy_lut, void_y, void_x, 1, kernel, kernel_size, height, width)
		count += 1

	dither_array = (rank * 255 // num_pixels).astype(np.uint8)
	return dither_array


def update_energy_lut(energy_lut: np.ndarray, py: int, px: int, sign: int, kernel: np.ndarray, kernel_size: int, height: int, width: int) -> None:
	half_size = kernel_size // 2

	# Calculate wrapped indices for toroidal topology
	ys = (py + np.arange(kernel_size) - half_size) % height
	xs = (px + np.arange(kernel_size) - half_size) % width

	if (kernel_size <= height) and (kernel_size <= width):
		energy_lut[np.ix_(ys, xs)] += (sign * kernel)
		return

	(y_grid, x_grid) = np.meshgrid(ys, xs, indexing = 'ij')
	np.add.at(energy_lut, (y_grid, x_grid), (sign * kernel))


def generate_initial_pattern(width: int, height: int, sigma: float, num_ones: int, kernel: np.ndarray, kernel_size: int) -> Tuple[np.ndarray, np.ndarray]:
	binary_pattern = np.zeros((height, width), dtype=np.int32)
	energy_lut = np.zeros((height, width), dtype=np.float64)
	
	# Randomly place initial ones
	placed = 0
	while placed < num_ones:
		y, x = np.random.randint(0, height), np.random.randint(0, width)
		if binary_pattern[y, x] == 0:
			binary_pattern[y, x] = 1
			update_energy_lut(energy_lut, y, x, 1, kernel, kernel_size, height, width)
			placed += 1
	
	# remove tightest cluster, add to largest void
	iteration = 0
	while True:
		(cluster_y, cluster_x) = find_cluster(binary_pattern, energy_lut)
		binary_pattern[cluster_y, cluster_x] = 0
		update_energy_lut(energy_lut, cluster_y, cluster_x, -1, kernel, kernel_size, height, width)
		
		(void_y, void_x) = find_void(binary_pattern, energy_lut)
		binary_pattern[void_y, void_x] = 1
		update_energy_lut(energy_lut, void_y, void_x, 1, kernel, kernel_size, height, width)
		
		if (cluster_y, cluster_x) == (void_y, void_x):
			break
		
		iteration += 1
		if iteration > 10000:
			print("Warning: Initial pattern generation didn't converge after 10000 iterations")
			break
	
	print(f"Initial pattern converged after {iteration} iterations")
	return binary_pattern, energy_lut


def find_cluster(binary_pattern: np.ndarray, energy_lut: np.ndarray) -> Tuple[int, int]:
	(height, width) = binary_pattern.shape
	masked_energy = np.where(binary_pattern == 1, energy_lut, -np.inf)
	idx = np.argmax(masked_energy)
	(y, x) = idx // width, idx % width
	return (y, x)

def find_void(binary_pattern: np.ndarray, energy_lut: np.ndarray) -> Tuple[int, int]:
	(height, width) = binary_pattern.shape
	masked_energy = np.where(binary_pattern == 0, energy_lut, np.inf)
	idx = np.argmin(masked_energy)
	(y, x) = (idx // width, idx % width)
	return (y, x)

def void_cluster_to_points(dither_array: np.ndarray, num_samples: int) -> np.ndarray:

	height, width = dither_array.shape
	total_pixels = height * width
	threshold = int((num_samples / total_pixels) * 255)
	
	mask = dither_array < threshold
	(y_coords, x_coords) = np.where(mask)
	
	if len(x_coords) > num_samples:
		indices = np.argsort(dither_array[y_coords, x_coords])[:num_samples]
		x_coords = x_coords[indices]
		y_coords = y_coords[indices]
	
	points = np.column_stack([x_coords.astype(float), y_coords.astype(float)])

	# Add uniform jitter in [-0.5, 0.5] to center samples within pixels
	points += np.random.uniform(-0.5, 0.5, size=points.shape)
	
	# Clamp to domain
	points[:, 0] = np.clip(points[:, 0], 0, width - 1)
	points[:, 1] = np.clip(points[:, 1], 0, height - 1)

	return points





def visualize_void_and_cluster_noise(dither: np.ndarray, threshold: int = 128, title: str = "Void-and-Cluster", filename = "tmp.png") -> None:
	dither_u8 = dither.astype(np.uint8, copy=False)
	binary = (dither_u8 < np.uint8(threshold)).astype(np.uint8)

	fig, ax = plt.subplots(1, 3, figsize=(12, 4))

	ax[0].imshow(dither_u8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
	ax[0].set_title(f"{title} (8-bit rank)")
	ax[0].axis("off")

	ax[1].imshow(binary, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
	ax[1].set_title(f"Binary preview (t < {threshold})")
	ax[1].axis("off")

	f = np.fft.fftshift(np.fft.fft2(binary.astype(np.float64)))
	mag = np.log1p(np.abs(f))

	ax[2].imshow(mag, cmap="gray", interpolation="nearest")
	ax[2].set_title("FFT magnitude (log)")
	ax[2].axis("off")

	plt.tight_layout()
	plt.savefig(filename, dpi=150, bbox_inches='tight')
	plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


def compute_base_moments(image: np.ndarray, fixation_point: Tuple[int, int], alpha: float = 0.1, base_pooling_size = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	mean_texture = np.zeros_like(image, dtype = np.float32)
	variance_texture = np.zeros_like(image, dtype = np.float32)
	skew_texture = np.zeros_like(image, dtype = np.float32)

	(width, height, _) = image.shape
	(x, y) = np.ogrid[:width, :height]
	distances = np.sqrt((x - fixation_point[0]) ** 2 + (y - fixation_point[1]) ** 2)
	pooling_sizes = base_pooling_size + alpha * distances
	pooling_sizes = pooling_sizes.astype(int)
	half_pools = pooling_sizes // 2

	x_mins = np.clip(x - half_pools, 0, width - 1).flatten()
	x_maxes = np.clip(x + half_pools, 0, width - 1).flatten()
	y_mins = np.clip(y - half_pools, 0, height - 1).flatten()
	y_maxes = np.clip(y + half_pools, 0, height - 1).flatten()

	region_sizes = (x_maxes - x_mins, y_maxes - y_mins)
	regions = [
		image[x_min:x_max, y_min:y_max]
		for x_min, x_max, y_min, y_max in zip(x_mins, x_maxes, y_mins, y_maxes)
	]

	mean_texture_flat = [np.mean(region) for region in regions]
	variance_texture_flat = [np.var(region) for region in regions]
	skew_texture_flat = [
		0 if np.std(region) == 0 else np.mean((region - np.mean(region)) ** 3) / np.std(region) ** 3
		for region in regions
	]	

	mean_texture = np.array(mean_texture_flat).reshape(width, height)
	variance_texture = np.array(variance_texture_flat).reshape(width, height)
	skew_texture = np.array(skew_texture_flat).reshape(width, height)
	

	return (mean_texture, variance_texture, skew_texture)


def compute_basemoments_gaussian_pyramids(mean_texture: np.ndarray, variance_texture: np.ndarray, skew_texture: np.ndarray, num_levels: int = 5) -> Dict[str, List[np.ndarray]]:
	gaussian_pyramids = {
		'mean': [mean_texture],
		'variance': [variance_texture],
		'skew': [skew_texture]
	}

	for i in range(1, num_levels):
		gaussian_pyramids['mean'].append(cv2.pyrDown(gaussian_pyramids['mean'][i - 1]))
		gaussian_pyramids['variance'].append(cv2.pyrDown(gaussian_pyramids['variance'][i - 1]))
		gaussian_pyramids['skew'].append(cv2.pyrDown(gaussian_pyramids['skew'][i - 1]))

	return gaussian_pyramids


def compute_basemoments_laplacian_pyramids(gaussian_pyramids: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
	laplacian_pyramids = {
		'mean': [],
		'variance': [],
		'skew': []
    }

	for moment in gaussian_pyramids.keys():
		num_levels = len(gaussian_pyramids[moment])
		for i in range(num_levels - 1):
			size = (gaussian_pyramids[moment][i].shape[1], gaussian_pyramids[moment][i].shape[0])
			expanded = cv2.pyrUp(gaussian_pyramids[moment][i+1], dstsize=size)
			laplacian = cv2.subtract(gaussian_pyramids[moment][i], expanded)
			laplacian_pyramids[moment].append(laplacian)
		laplacian_pyramids[moment].append(gaussian_pyramids[moment][-1])

	return laplacian_pyramids


def visualize_base_moments(mean_texture: np.ndarray, variance_texture: np.ndarray, skew_texture: np.ndarray) -> None:
	fig, axs = plt.subplots(1, 3, figsize=(15, 5))
	
	# Mean texture visualization
	axs[0].imshow(mean_texture, cmap = 'gray')
	axs[0].set_title("Mean base moment")
	axs[0].axis("off")

	# Variance texture visualization
	axs[1].imshow(variance_texture, cmap = 'gray')
	axs[1].set_title(f"Variance base moment")
	axs[1].axis("off")

	axs[2].imshow(skew_texture, cmap = 'gray')
	axs[2].set_title("Skew base moment")
	axs[2].axis("off")
	
	plt.tight_layout()
	plt.show()
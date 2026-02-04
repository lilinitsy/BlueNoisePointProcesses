import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def poisson_disk_sampling(dimensions: Tuple[float, float], min_radius: float, max_attempts: int = 30, seed: int = 12) -> np.ndarray:
	'''
	Poisson disk sampling using brute force distance checking.
	This method checks all existing points for each candidate, which is slower but may produce visually better results
	since it doesn't enforce one placement per grid cell.

	:param dimensions: float tuple corresponding to (width, height)
	:type dimensions: Tuple[float, float]
	:param min_radius: distance between points
	:type min_radius: float
	:param max_attempts: # trials it should take to retry placing a point
	:type max_attempts: int
	:param seed: random seed
	:type seed: int
	:return: points, ndarray of shape (n, 2) containing the poisson points
	:rtype: ndarray[_AnyShape, dtype[Any]]
	'''

	# Initialization
	(width, height) = dimensions
	np.random.seed(seed)

	points = []
	attempts_count = 0

	while attempts_count < max_attempts:
		# Place a point
		candidate = np.array([np.random.uniform(0, width), np.random.uniform(0, height)])

		# Check against all existing points
		valid_placement = True
		if len(points) > 0:
			points_array = np.array(points)
			distances = np.linalg.norm(points_array - candidate, axis=1)
			
			if np.any(distances <= min_radius):
				valid_placement = False

		if valid_placement:
			points.append(candidate)
			attempts_count = 0
		else:
			attempts_count += 1

	if len(points) <= 0:
		return np.empty((0, 2))

	return np.array(points)



# This uses a spatial grid to check neighbours quicker
def poisson_disk_sampling_grid(dimensions: Tuple[float, float], min_radius: float, max_attempts: int = 30, seed: int = 12) -> np.ndarray:
	'''
	Poisson disk sampling using a spatial grid to check neighbours. 
	It slightly deviates from a poisson disk method that doesn't used a grid, since it enforces one placement per cell, and could lead to differeing point placements
	and placements

	:param dimensions: float tuple corresponding to (width, height)
	:type dimensions: Tuple[float, float]
	:param min_radius: distance between points
	:type min_radius: float
	:param max_attempts: # trials it should take to retry placing a point
	:type max_attempts: int
	:param seed: random seed
	:type seed: int
	:return: points, ndarray of shape (n, 2) containing the poisson points
	:rtype: ndarray[_AnyShape, dtype[Any]]
	'''

	# Initialization
	(width, height) = dimensions
	cell_size = min_radius / np.sqrt(2) # enforce each cell contains only 1 point
	(grid_width, grid_height) = (int(np.ceil(width / cell_size)), int(np.ceil(height / cell_size)))
	grid = np.full((grid_height, grid_width), -1, dtype = int)
	np.random.seed(seed)

	points = []
	attempts_count = 0

	while attempts_count < max_attempts:
		# Place a point
		candidate = np.array([np.random.uniform(0, width), np.random.uniform(0, height)])

		# Current grid location of candidate; let's see if there's anyone in this
		(grid_cell_x, grid_cell_y) = (int(candidate[0] / cell_size), int(candidate[1] / cell_size))

		# Check nearby cells
		valid_placement = True
		for dy in range(-2, 3, 1):
			for dx in range(-2, 3, 1):
				(neighbour_x, neighbour_y) = (grid_cell_x + dx, grid_cell_y + dy) # get neighbour cell to inspect for candidates
				
				# Bounds checking
				if neighbour_x < 0 or neighbour_x >= grid_width or neighbour_y < 0 or neighbour_y >= grid_height:
					continue
				
				# np is backwards, so [y, x]
				neighbour_candidate = grid[neighbour_y, neighbour_x]

				# There's a neighbouring candidate, so evaluate the distance to it
				if neighbour_candidate >= 0:
					distance = np.linalg.norm(candidate - points[neighbour_candidate])

					if distance <= min_radius:
						valid_placement = False
						break
			if not valid_placement:
				break

		if valid_placement:
			point_index = len(points)
			points.append(candidate)
			grid[grid_cell_y, grid_cell_x] = point_index
			attempts_count = 0
		else:
			attempts_count += 1

	if len(points) <= 0:
		return np.empty((0, 2))

	return np.array(points)



def visualize_poisson_disk_samples(dimensions: Tuple[float, float], min_radius: float, points: np.ndarray, title: str = "Poisson Disk Sampling") -> None:
	'''
	:param dimensions: float tuple corresponding to (width, height)
	:type dimensions: Tuple[float, float]
	:param min_radius: distance between points
	:type min_radius: float
	:param points: ndarray of shape (n, 2) containing the poisson points
	:type points: np.ndarray
	:param title: Title for graph
	:type title: str
	'''

	(width, height) = dimensions
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
	
	# Left plot: Points with exclusion zones
	ax1.set_xlim(0, width)
	ax1.set_ylim(0, height)
	ax1.set_aspect('equal')
	ax1.set_title(f'{title}\n({len(points)} points with min radius = {min_radius})', fontsize=14)
	ax1.set_xlabel('X')
	ax1.set_ylabel('Y')
	ax1.grid(True, alpha=0.3)
	
	# Draw exclusion circles around each point
	for point in points:
		circle = Circle(point, min_radius, color='lightblue', alpha=0.3, ec='blue', linewidth=0.5)
		ax1.add_patch(circle)
	
	# Draw the points on top
	ax1.scatter(points[:, 0], points[:, 1], c='red', s=50, zorder=5, alpha=0.8, edgecolors='darkred')
	
	# Right plot: Just the points (cleaner view)
	ax2.set_xlim(0, width)
	ax2.set_ylim(0, height)
	ax2.set_aspect('equal')
	ax2.set_title(f'Point Distribution\n(Density: {len(points)/(width*height):.4f} points/unit²)', fontsize=14)
	ax2.set_xlabel('X')
	ax2.set_ylabel('Y')
	ax2.grid(True, alpha=0.3)
	
	ax2.scatter(points[:, 0], points[:, 1], c='darkblue', s=30, alpha=0.7)
	
	plt.tight_layout()
	return fig

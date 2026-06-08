from __future__ import annotations

import math

import numpy as np

from ds_wave_target import interpolate_target_power
from ds_wave_targetrdf import choose_torch_device


try:
	import torch
except ImportError:
	torch = None


def make_frequency_modes(
	n_points: int,
	nu_max: float,
	mode_limit: int = 4096,
	seed: int = 0,
	priority_nu: float | None = None,
) -> np.ndarray:
	# Equation 2 periodogram modes in normalized coordinates: nu = |k| / sqrt(N).
	max_mode = max(1, int(math.ceil(nu_max * math.sqrt(float(n_points)))))
	values = np.arange(-max_mode, max_mode + 1, dtype=np.int32)
	kx, ky = np.meshgrid(values, values, indexing="xy")
	modes = np.stack([kx.ravel(), ky.ravel()], axis=1)
	nonzero = np.any(modes != 0, axis=1)
	modes = modes[nonzero]
	radius = np.linalg.norm(modes.astype(np.float64), axis=1) / math.sqrt(float(n_points))
	modes = modes[radius <= nu_max]
	radius = radius[radius <= nu_max]

	if mode_limit > 0 and modes.shape[0] > mode_limit:
		if priority_nu is not None:
			# Section 4.2 / Equation 14: preserve all modes in the low-frequency hole.
			priority_mask = radius <= priority_nu
			priority_modes = modes[priority_mask]
			priority_radius = radius[priority_mask]
			if priority_modes.shape[0] >= mode_limit:
				order = np.argsort(priority_radius, kind="stable")
				return priority_modes[order[:mode_limit]].astype(np.float64)

			remaining_modes = modes[~priority_mask]
			remaining_count = mode_limit - priority_modes.shape[0]
			rng = np.random.default_rng(seed)
			if remaining_modes.shape[0] > remaining_count:
				choice = rng.choice(remaining_modes.shape[0], size=remaining_count, replace=False)
				remaining_modes = remaining_modes[np.sort(choice)]
			return np.vstack([priority_modes, remaining_modes]).astype(np.float64)

		rng = np.random.default_rng(seed)
		choice = rng.choice(modes.shape[0], size=mode_limit, replace=False)
		modes = modes[np.sort(choice)]
	return modes.astype(np.float64)

def synthesize_spectrum_matching_points(
	target: dict,
	n_points: int = 128,
	iterations: int = 100,
	seed: int = 0,
	device: str | None = "auto",
	learning_rate: float = 0.03,
	mode_limit: int = 12000,
	mode_chunk_size: int = 512,
	low_frequency_weight: float = 50.0,
	log_every: int | None = None,
) -> dict:
	if n_points < 1:
		raise ValueError("n_points must be at least 1")
	if iterations < 0:
		raise ValueError("iterations must be non-negative")
	if mode_chunk_size < 1:
		raise ValueError("mode_chunk_size must be positive")
	if low_frequency_weight < 1.0:
		raise ValueError("low_frequency_weight must be at least 1")
	if torch is None:
		raise RuntimeError("torch is required for point synthesis")

	torch_device = choose_torch_device(device)
	torch.manual_seed(seed)
	if torch_device.type == "cuda":
		torch.cuda.manual_seed_all(seed)

	nu_max = float(target["nu"][-1])
	priority_nu = target.get("nu0")
	# The paper synthesizes with PCF matching [HSD13]; this prototype matches Equation 2
	# Fourier powers directly, while protecting the Equation 14 low-frequency disk.
	modes_np = make_frequency_modes(n_points, nu_max=nu_max, mode_limit=mode_limit, seed=seed, priority_nu=priority_nu)
	if modes_np.shape[0] == 0:
		raise ValueError("no Fourier modes available for the requested n_points and nu_max")
	mode_radii = np.linalg.norm(modes_np, axis=1) / math.sqrt(float(n_points))
	target_power_np = interpolate_target_power(target, mode_radii)

	torch_dtype = torch.float32
	points = torch.rand((n_points, 2), dtype=torch_dtype, device=torch_device, requires_grad=True)
	modes = torch.as_tensor(modes_np, dtype=torch_dtype, device=torch_device)
	mode_radii_t = torch.as_tensor(mode_radii, dtype=torch_dtype, device=torch_device)
	target_power = torch.as_tensor(target_power_np, dtype=torch_dtype, device=torch_device)
	weights = 1.0 / torch.clamp(target_power, min=0.05)
	if priority_nu is not None:
		# Equation 14 is visually critical; without this weight the zero-power hole is diluted by high-frequency modes.
		weights = torch.where(mode_radii_t <= float(priority_nu), weights * low_frequency_weight, weights)
	optimizer = torch.optim.Adam([points], lr=learning_rate)
	energy_history = []

	for iteration in range(iterations):
		optimizer.zero_grad()
		energy_sum = torch.zeros((), dtype=torch_dtype, device=torch_device)
		weight_sum = torch.zeros((), dtype=torch_dtype, device=torch_device)
		for start in range(0, modes.shape[0], mode_chunk_size):
			end = min(start + mode_chunk_size, modes.shape[0])
			mode_chunk = modes[start:end]
			target_chunk = target_power[start:end]
			weight_chunk = weights[start:end]
			phase = 2.0 * math.pi * (points @ mode_chunk.T)
			real = torch.cos(phase).sum(dim=0)
			imag = torch.sin(phase).sum(dim=0)
			# Equation 2 empirical power: P(k) = |sum_j exp(-2*pi*i*k.x_j)|^2 / N.
			power = (real * real + imag * imag) / float(n_points)
			energy_sum = energy_sum + torch.sum(weight_chunk * (power - target_chunk) ** 2)
			weight_sum = weight_sum + torch.sum(weight_chunk)
		energy = energy_sum / weight_sum
		energy.backward()
		optimizer.step()
		with torch.no_grad():
			points.remainder_(1.0)
		if log_every is None or iteration % max(log_every, 1) == 0 or iteration == iterations - 1:
			energy_history.append(float(energy.detach().cpu()))

	if iterations == 0:
		with torch.no_grad():
			energy_sum = torch.zeros((), dtype=torch_dtype, device=torch_device)
			weight_sum = torch.zeros((), dtype=torch_dtype, device=torch_device)
			for start in range(0, modes.shape[0], mode_chunk_size):
				end = min(start + mode_chunk_size, modes.shape[0])
				mode_chunk = modes[start:end]
				target_chunk = target_power[start:end]
				weight_chunk = weights[start:end]
				phase = 2.0 * math.pi * (points @ mode_chunk.T)
				real = torch.cos(phase).sum(dim=0)
				imag = torch.sin(phase).sum(dim=0)
				# Equation 2 empirical power for the zero-iteration diagnostic path.
				power = (real * real + imag * imag) / float(n_points)
				energy_sum = energy_sum + torch.sum(weight_chunk * (power - target_chunk) ** 2)
				weight_sum = weight_sum + torch.sum(weight_chunk)
			energy = energy_sum / weight_sum
			energy_history.append(float(energy.detach().cpu()))

	return {
		"points": points.detach().cpu().numpy(),
		"energy_history": np.array(energy_history, dtype=np.float64),
		"modes": modes_np,
		"target_power": target_power_np,
		"device": str(torch_device),
		"synthesis_mode": "spectrum_matching",
	}


def refine_low_frequency_modes(
	initial_points: np.ndarray,
	target: dict,
	iterations: int = 250,
	device: str | None = "auto",
	learning_rate: float = 0.01,
	low_frequency_nu: float | None = None,
	mode_limit: int = 0,
	mode_chunk_size: int = 512,
	log_every: int | None = None,
) -> dict:
	if torch is None:
		raise RuntimeError("torch is required for low-frequency refinement")
	if iterations < 0:
		raise ValueError("iterations must be non-negative")
	if learning_rate <= 0.0:
		raise ValueError("learning_rate must be positive")
	if mode_chunk_size < 1:
		raise ValueError("mode_chunk_size must be positive")

	initial_points = np.asarray(initial_points, dtype=np.float32)
	if initial_points.ndim != 2 or initial_points.shape[1] != 2 or initial_points.shape[0] < 1:
		raise ValueError("initial_points must have shape (n_points, 2)")

	n_points = int(initial_points.shape[0])
	if low_frequency_nu is None:
		low_frequency_nu = float(target["nu0"])
	else:
		low_frequency_nu = float(low_frequency_nu)
	if low_frequency_nu <= 0.0:
		raise ValueError("low_frequency_nu must be positive")

	# Practical, non-pure DS-Wave cleanup: suppress residual Fourier leakage in
	# the low-frequency hole after another synthesis method has produced points.
	modes_np = make_frequency_modes(
		n_points,
		nu_max=low_frequency_nu,
		mode_limit=mode_limit,
		seed=0,
		priority_nu=low_frequency_nu,
	)
	if modes_np.shape[0] == 0:
		raise ValueError("no Fourier modes available for low-frequency refinement")

	torch_device = choose_torch_device(device)
	torch_dtype = torch.float32
	wrapped_initial = np.remainder(initial_points, 1.0)
	points = torch.tensor(wrapped_initial, dtype=torch_dtype, device=torch_device, requires_grad=True)
	modes = torch.as_tensor(modes_np, dtype=torch_dtype, device=torch_device)
	optimizer = torch.optim.Adam([points], lr=learning_rate)
	energy_history = []
	history = []

	def compute_energy() -> torch.Tensor:
		energy_sum = torch.zeros((), dtype=torch_dtype, device=torch_device)
		mode_count = 0
		for start in range(0, modes.shape[0], mode_chunk_size):
			end = min(start + mode_chunk_size, modes.shape[0])
			mode_chunk = modes[start:end]
			phase = 2.0 * math.pi * (points @ mode_chunk.T)
			real = torch.cos(phase).sum(dim=0)
			imag = torch.sin(phase).sum(dim=0)
			power = (real * real + imag * imag) / float(n_points)
			energy_sum = energy_sum + torch.sum(power)
			mode_count += end - start
		return energy_sum / float(mode_count)

	for iteration in range(iterations):
		optimizer.zero_grad()
		energy = compute_energy()
		energy.backward()
		optimizer.step()
		with torch.no_grad():
			points.remainder_(1.0)
		if log_every is None or iteration % max(log_every, 1) == 0 or iteration == iterations - 1:
			energy_value = float(energy.detach().cpu())
			energy_history.append(energy_value)
			history.append({"iteration": iteration, "energy": energy_value})

	if iterations == 0:
		with torch.no_grad():
			energy_value = float(compute_energy().detach().cpu())
			energy_history.append(energy_value)
			history.append({"iteration": 0, "energy": energy_value})

	final_points = points.detach().cpu().numpy()
	initial_powers = direct_mode_power(wrapped_initial, modes_np, chunk_size=mode_chunk_size)
	final_powers = direct_mode_power(final_points, modes_np, chunk_size=mode_chunk_size)
	mode_radii = np.linalg.norm(modes_np, axis=1) / math.sqrt(float(n_points))
	metadata = {
		"low_frequency_nu": low_frequency_nu,
		"mode_count": int(modes_np.shape[0]),
		"mode_limit": mode_limit,
		"mode_chunk_size": mode_chunk_size,
		"learning_rate": learning_rate,
		"iterations": iterations,
		"device": str(torch_device),
		"synthesis_mode": "low_frequency_refine",
	}

	return {
		"points": final_points,
		"energy_history": np.array(energy_history, dtype=np.float64),
		"history": history,
		"metadata": metadata,
		"iterations_run": iterations,
		"modes": modes_np,
		"mode_radii": mode_radii,
		"mode_count": int(modes_np.shape[0]),
		"initial_powers": initial_powers,
		"final_powers": final_powers,
		"initial_mean_power": float(np.mean(initial_powers)),
		"final_mean_power": float(np.mean(final_powers)),
		"initial_max_power": float(np.max(initial_powers)),
		"final_max_power": float(np.max(final_powers)),
		"low_frequency_nu": low_frequency_nu,
		"device": str(torch_device),
		"synthesis_mode": "low_frequency_refine",
	}


def direct_mode_power(points: np.ndarray, modes: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
	# Equation 2 empirical power for explicit Fourier mode vectors.
	points = np.asarray(points, dtype=np.float64)
	modes = np.asarray(modes, dtype=np.float64)
	powers = []
	for start in range(0, modes.shape[0], chunk_size):
		chunk = modes[start:start + chunk_size]
		phase = 2.0 * math.pi * (points @ chunk.T)
		real = np.cos(phase).sum(axis=0)
		imag = np.sin(phase).sum(axis=0)
		powers.append((real * real + imag * imag) / float(points.shape[0]))
	return np.concatenate(powers, axis=0)

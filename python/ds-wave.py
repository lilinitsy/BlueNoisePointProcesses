from __future__ import annotations

import argparse
import math
import sys
from enum import Enum
from pathlib import Path

import numpy as np
try:
	import torch
except ImportError:
	torch = None


sys.dont_write_bytecode = True
python_dir = Path(__file__).resolve().parent
if str(python_dir) not in sys.path:
	sys.path.insert(0, str(python_dir))

from ds_wave_target import (
	empty_target,
	evaluate_target_pcf,
	find_min_m0,
	interpolate_target_pcf,
	interpolate_target_power,
	linprog_status_name,
	make_hankel_matrix,
	solve_ds_wave_target,
	trapezoid_weights,
)


from ds_wave_targetrdf import (
	choose_torch_device,
	compute_targetrdf_energy,
	compute_targetrdf_force_curve,
	compute_targetrdf_gradients,
	compute_targetrdf_rdf,
	interpolate_curve_torch,
	make_targetrdf_target_curve,
	smooth_curve_gaussian,
	synthesize_targetrdf_points,
)


from ds_wave_spectrum import (
	direct_mode_power,
	make_frequency_modes,
	synthesize_spectrum_matching_points,
)


from ds_wave_diagnostics import (
	compute_empirical_pcf,
	compute_periodogram_2d,
	compute_radial_psd,
	make_centered_integer_axis,
	plot_ds_wave_targets,
	plot_pcf_overlay,
	plot_periodogram,
	plot_points,
	plot_radial_psd_overlay,
	save_figure,
)


class SynthesisMode(Enum):
	SPECTRUM_MATCHING = "spectrum_matching"
	PCF_MATCHING = "pcf_matching"


def parse_synthesis_mode(value: SynthesisMode | str) -> SynthesisMode:
	if isinstance(value, SynthesisMode):
		return value
	if isinstance(value, str):
		normalised = value.strip().lower().replace("-", "_")
		for mode in SynthesisMode:
			if normalised == mode.value or normalised == mode.name.lower():
				return mode
	raise ValueError(f"unknown synthesis mode: {value!r}")


# Section 5.4: discretizes integrals with the trapezoidal rule.
# Sections 5.1-5.4: isotropic 2D Fourier transforms reduce to Hankel transforms.
# Eq12: maps fourier F = P - 1 to f = g - 1
def synthesize_toroidal_points(
	target: dict,
	n_points: int = 128,
	iterations: int = 100,
	seed: int = 0,
	device: str | None = "auto",
	synthesis_mode: SynthesisMode | str = SynthesisMode.SPECTRUM_MATCHING,
	initial_points: np.ndarray | None = None,
	log_every: int | None = None,
	spectrum_learning_rate: float = 0.03,
	mode_limit: int = 12000,
	mode_chunk_size: int = 512,
	low_frequency_weight: float = 50.0,
	pcf_step_scale: float = 1.0,
	pcf_num_bins: int | None = None,
	pcf_smoothing: float = 8.0,
	pcf_chunk_size: int = 256,
) -> dict:
	mode = parse_synthesis_mode(synthesis_mode)
	if mode is SynthesisMode.SPECTRUM_MATCHING:
		return synthesize_spectrum_matching_points(
			target,
			n_points=n_points,
			iterations=iterations,
			seed=seed,
			device=device,
			learning_rate=spectrum_learning_rate,
			mode_limit=mode_limit,
			mode_chunk_size=mode_chunk_size,
			low_frequency_weight=low_frequency_weight,
			log_every=log_every,
		)
	if mode is SynthesisMode.PCF_MATCHING:
		result = synthesize_targetrdf_points(
			target,
			n_points=n_points,
			iterations=iterations,
			seed=seed,
			device=device,
			step_scale=pcf_step_scale,
			nbins=pcf_num_bins,
			smoothing=pcf_smoothing,
			chunk_size=pcf_chunk_size,
			initial_points=initial_points,
			log_every=log_every,
		)
		result["synthesis_mode"] = SynthesisMode.PCF_MATCHING.value
		return result
	raise ValueError(f"unknown synthesis mode: {synthesis_mode!r}")


def parse_m0_values(values_text: list[str]) -> list[float | str]:
	values = []
	for text in values_text:
		for item in text.split(","):
			item = item.strip()
			if not item:
				continue
			if item == "min":
				values.append(item)
			else:
				values.append(float(item))
	return values


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Solve and smoke-test DS-Wave non-adaptive targets.")
	parser.add_argument("--nu0", type=float, default=0.85)
	parser.add_argument("--e0", type=float, default=0.0)
	parser.add_argument("--nu-max", type=float, default=10.0)
	parser.add_argument("--n-nu", type=int, default=1001)
	parser.add_argument("--n-r", type=int, default=128)
	parser.add_argument("--n-points", type=int, default=128)
	parser.add_argument("--iterations", type=int, default=100)
	parser.add_argument("--seed", type=int, default=1234)
	parser.add_argument("--m0-values", nargs="+", default=["1", "2", "min"])
	parser.add_argument("--output-dir", type=Path, default=python_dir / "ds_wave_outputs")
	parser.add_argument("--device", default="auto")
	parser.add_argument("--synthesis-mode", choices=[mode.value for mode in SynthesisMode], default=SynthesisMode.SPECTRUM_MATCHING.value)
	parser.add_argument("--spectrum-learning-rate", type=float, default=0.03)
	parser.add_argument("--pcf-step-scale", type=float, default=1.0)
	parser.add_argument("--pcf-num-bins", type=int, default=None)
	parser.add_argument("--pcf-smoothing", type=float, default=8.0)
	parser.add_argument("--pcf-chunk-size", type=int, default=256)
	return parser.parse_args()


def run_cli(args: argparse.Namespace) -> list[Path]:
	import matplotlib

	matplotlib.use("Agg", force=True)

	paths = []
	targets = []
	for m0 in parse_m0_values(args.m0_values):
		target = solve_ds_wave_target(
			nu0=args.nu0,
			e0=args.e0,
			m0=m0,
			nu_max=args.nu_max,
			n_nu=args.n_nu,
			n_r=args.n_r,
			require_success=False,
		)
		target["requested_m0"] = m0
		targets.append(target)
		print(f"m0={m0}: {target['status']} - {target['message']}")

	if any(target["success"] for target in targets):
		paths.append(save_figure(plot_ds_wave_targets(targets), args.output_dir / "targets.png"))

	synthesis_target = None
	for target in targets:
		if target["success"] and target.get("requested_m0") == "min":
			synthesis_target = target
	for target in targets:
		if synthesis_target is None and target["success"] and target["m0"] != 1.0:
			synthesis_target = target

	if synthesis_target is None:
		print("No feasible target available for point synthesis.")
		return paths

	result = synthesize_toroidal_points(
		synthesis_target,
		n_points=args.n_points,
		iterations=args.iterations,
		seed=args.seed,
		device=args.device,
		synthesis_mode=args.synthesis_mode,
		spectrum_learning_rate=args.spectrum_learning_rate,
		pcf_step_scale=args.pcf_step_scale,
		pcf_num_bins=args.pcf_num_bins,
		pcf_smoothing=args.pcf_smoothing,
		pcf_chunk_size=args.pcf_chunk_size,
	)
	points = result["points"]
	paths.append(save_figure(plot_points(points), args.output_dir / "points.png"))
	power, extent = compute_periodogram_2d(points, max_freq=args.nu_max)
	paths.append(save_figure(plot_periodogram(power, extent), args.output_dir / "periodogram.png"))
	paths.append(save_figure(plot_radial_psd_overlay(points, synthesis_target, max_freq=args.nu_max), args.output_dir / "radial_psd.png"))
	paths.append(save_figure(plot_pcf_overlay(points, synthesis_target), args.output_dir / "pcf.png"))
	np.save(args.output_dir / "points.npy", points)
	return paths


if __name__ == "__main__":
	run_cli(parse_args())

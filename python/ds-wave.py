"""DS-Wave command line driver.

Solves DS-Wave target spectra (ds_wave_target.py) for one or more m0 values,
synthesizes a point set whose RDF matches the chosen target
(ds_wave_targetrdf.py), and writes diagnostic plots.

This module is also the convenience namespace the notebook imports: the
re-exports below are the public surface (each is used by ds-wave-testing.ipynb
or the tests).
"""
from __future__ import annotations

import argparse
import sys
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
	DsWaveTarget,
	evaluate_target_pcf,
	find_min_m0,
	interpolate_target_pcf,
	interpolate_target_power,
	solve_ds_wave_target,
)

from ds_wave_targetrdf import (
	SynthesisResult,
	resolve_targetrdf_resolution,
	synthesize_targetrdf_points,
)

from ds_wave_diagnostics import (
	compute_empirical_pcf,
	compute_low_frequency_mode_powers,
	compute_periodogram_2d,
	compute_radial_psd,
	compute_target_mode_powers,
	plot_ds_wave_targets,
	plot_pcf_overlay,
	plot_periodogram,
	plot_points,
	plot_radial_psd_overlay,
	save_figure,
	summarise_mode_power_bands,
)


def parse_m0_values(values_text: list[str]) -> list[float | str]:
	"""Parse CLI m0 arguments: floats plus the literal "min".

	"min" only exists at this CLI boundary; run_cli translates it into an
	explicit find_min_m0() call before solving.
	"""
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
	parser.add_argument("--step-scale", type=float, default=1.0)
	parser.add_argument("--num-bins", type=int, default=None)
	parser.add_argument("--smoothing", type=float, default=None)
	parser.add_argument("--chunk-size", type=int, default=256)
	return parser.parse_args()


def run_cli(args: argparse.Namespace) -> list[Path]:
	import matplotlib

	matplotlib.use("Agg", force=True)

	paths = []
	# One solved DsWaveTarget per requested m0 (see the DsWaveTarget docstring
	# for the record contents).
	solved_targets = []
	for requested in parse_m0_values(args.m0_values):
		if requested == "min":
			m0 = find_min_m0(
				nu0=args.nu0,
				e0=args.e0,
				nu_max=args.nu_max,
				n_nu=args.n_nu,
				n_r=args.n_r,
			)
			if m0 is None:
				print("m0=min: infeasible - no feasible finite m0 found.")
				continue
		else:
			m0 = requested
		solved_target = solve_ds_wave_target(
			nu0=args.nu0,
			e0=args.e0,
			m0=m0,
			nu_max=args.nu_max,
			n_nu=args.n_nu,
			n_r=args.n_r,
		)
		solved_target.requested_m0 = str(requested)
		solved_targets.append(solved_target)
		print(f"m0={requested}: {solved_target.status} - {solved_target.message}")

	if any(solved_target.success for solved_target in solved_targets):
		paths.append(save_figure(plot_ds_wave_targets(solved_targets), args.output_dir / "targets.png"))

	# Pick the target to synthesize from: prefer the m0="min" solve, otherwise
	# any feasible solve that actually permits oscillation (m0 != 1).
	synthesis_target = None
	for solved_target in solved_targets:
		if solved_target.success and solved_target.requested_m0 == "min":
			synthesis_target = solved_target
	for solved_target in solved_targets:
		if synthesis_target is None and solved_target.success and solved_target.m0 != 1.0:
			synthesis_target = solved_target

	if synthesis_target is None:
		print("No feasible target available for point synthesis.")
		return paths

	result = synthesize_targetrdf_points(
		synthesis_target,
		n_points=args.n_points,
		iterations=args.iterations,
		seed=args.seed,
		device=args.device,
		step_scale=args.step_scale,
		nbins=args.num_bins,
		smoothing=args.smoothing,
		chunk_size=args.chunk_size,
	)
	points = result.points
	paths.append(save_figure(plot_points(points), args.output_dir / "points.png"))
	(power, extent) = compute_periodogram_2d(points, max_freq=args.nu_max)
	paths.append(save_figure(plot_periodogram(power, extent), args.output_dir / "periodogram.png"))
	paths.append(save_figure(plot_radial_psd_overlay(points, synthesis_target, max_freq=args.nu_max), args.output_dir / "radial_psd.png"))
	paths.append(save_figure(plot_pcf_overlay(points, synthesis_target), args.output_dir / "pcf.png"))
	np.save(args.output_dir / "points.npy", points)
	return paths


if __name__ == "__main__":
	run_cli(parse_args())

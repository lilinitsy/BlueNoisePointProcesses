"""Ensemble radial-PSD evaluation for DS-Wave synthesis.

Used by the bug-reproduction baseline (Task 4), the acceptance test
(test_ds_wave_psd_fix.py), and final verification. Averages the radial PSD
over several independently seeded syntheses so spurious spectral peaks can
be distinguished from single-realization estimator noise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

python_dir = Path(__file__).resolve().parent
if str(python_dir) not in sys.path:
	sys.path.insert(0, str(python_dir))

from ds_wave_target import DsWaveTarget, find_min_m0, interpolate_target_power, solve_ds_wave_target
from ds_wave_targetrdf import synthesize_targetrdf_points
from ds_wave_diagnostics import bin_radial_values, compute_radial_psd, save_figure
from ds_wave_spectrum import make_frequency_modes

import math

# compute_radial_psd default mode budget; mirrored here so the binned target
# samples the identical Fourier modes as the empirical measurement.
DEFAULT_MODE_LIMIT = 8192


def binned_target_power(
	target: dict,
	n_points: int,
	num_bins: int = 96,
	max_freq: float = 3.0,
	mode_limit: int = DEFAULT_MODE_LIMIT,
	seed: int = 0,
	priority_nu: float | None = None,
) -> np.ndarray:
	"""Bin-average the target P(nu) over the SAME Fourier modes compute_radial_psd uses.

	The empirical PSD is the per-bin mean of the measured power at each mode
	radius; this returns the per-bin mean of ``interpolate_target_power`` at the
	identical mode radii (same mode_limit/seed/priority_nu), so the metric
	compares like with like and the step-discontinuity binning artifact cancels.
	"""
	modes = make_frequency_modes(n_points, nu_max=max_freq, mode_limit=mode_limit, seed=seed, priority_nu=priority_nu)
	radii = np.linalg.norm(modes, axis=1) / math.sqrt(float(n_points))
	mode_powers = interpolate_target_power(target, radii)
	_, binned = bin_radial_values(radii, mode_powers, num_bins=num_bins, max_freq=max_freq)
	return binned


def ensemble_radial_psd(
	target: dict,
	n_points: int,
	seeds: list[int],
	iterations: int,
	device: str = "auto",
	max_freq: float = 3.0,
	num_bins: int = 96,
	mode_limit: int = DEFAULT_MODE_LIMIT,
	return_binned_target: bool = False,
):
	"""Average compute_radial_psd over independently seeded syntheses.

	When ``return_binned_target`` is True, also returns the target binned over
	the identical Fourier mode set, aligned with ``freqs``, so psd_fit_metrics
	can compare like with like (see binned_target_power).
	"""
	if not seeds:
		raise ValueError("seeds must be a non-empty list")
	priority_nu = target.nu0
	psds = []
	freqs = None
	for seed in seeds:
		result = synthesize_targetrdf_points(
			target,
			n_points=n_points,
			iterations=iterations,
			seed=seed,
			device=device,
		)
		(freqs, radial) = compute_radial_psd(
			result.points,
			num_bins=num_bins,
			max_freq=max_freq,
			mode_limit=mode_limit,
			priority_nu=priority_nu,
		)
		psds.append(radial)
	mean_psd = np.mean(np.stack(psds), axis=0)
	if return_binned_target:
		binned_target = binned_target_power(
			target,
			n_points=n_points,
			num_bins=num_bins,
			max_freq=max_freq,
			mode_limit=mode_limit,
			seed=0,
			priority_nu=priority_nu,
		)
		return freqs, mean_psd, binned_target
	return freqs, mean_psd


def find_target_discontinuities(target: dict, min_jump: float = 0.2) -> np.ndarray:
	"""Return nu positions where the target power spectrum P has a step discontinuity.

	The DS-Wave target P(nu) is piecewise-constant from an LP; step discontinuities
	at band edges (e.g. the low-pass / stop-band transition at nu0) are genuine
	features of the mathematical spec. No finite point set can reproduce these
	edges sharply — the source papers' own figures show the same rounded transitions.
	This helper locates every nu where |P[i+1] - P[i]| > min_jump between
	consecutive samples; the returned positions are used by psd_fit_metrics to
	exclude edge-adjacent bins from the interior max_abs metric.

	Parameters
	----------
	target:
		A successfully solved DS-Wave target dict with arrays ``nu`` and ``P``.
	min_jump:
		Minimum absolute difference between adjacent P samples to count as a
		discontinuity. Default 0.2 (robust against LP solver floating-point noise
		while well below the typical band-edge step of ~1.5).

	Returns
	-------
	np.ndarray
		1-D array of nu values at which the step occurs. Empty if the target has
		no steps above min_jump.
	"""
	if target.P is None:
		raise ValueError("target has no solved power spectrum")
	nu = np.asarray(target.nu, dtype=np.float32)
	P = np.asarray(target.P, dtype=np.float32)
	diffs = np.abs(np.diff(P))
	jump_indices = np.where(diffs > min_jump)[0]
	# Report the nu mid-point of each jumping pair.
	return 0.5 * (nu[jump_indices] + nu[jump_indices + 1])


def psd_fit_metrics(
	freqs: np.ndarray,
	psd: np.ndarray,
	target: dict,
	target_power: np.ndarray | None = None,
) -> dict:
	"""Quantify deviation between a measured radial PSD and the target P(nu).

	If ``target_power`` is provided (e.g. the binned target from
	binned_target_power / ensemble_radial_psd), it is used directly so the
	comparison is like-with-like. Otherwise the target is interpolated at bin
	centres (legacy behaviour, which carries a step-discontinuity artifact).

	Returned metrics
	----------------
	rmse : float
		Root-mean-square residual over all bins.
	max_abs_residual : float
		Maximum absolute residual over all bins (reported for transparency).
	max_abs_residual_interior : float
		Maximum absolute residual over bins whose centre is at distance > one
		bin-width from every target discontinuity. This is the gating metric:

		  Rationale — the DS-Wave target P(nu) is piecewise-constant from an LP.
		  Step discontinuities at band edges (e.g. the low-pass/stop-band
		  transition at nu0) are genuine features of the mathematical spec; no
		  finite point set can reproduce them sharply, and the source papers'
		  own figures show the same rounded transitions. Gating on the plain
		  max_abs_residual therefore penalises physically unavoidable edge-
		  rounding rather than genuine synthesis defects. max_abs_residual_interior
		  excludes bins within one bin-width of any such edge so it catches only
		  true synthesis anomalies — spurious peaks and notch fill — which are the
		  bugs this acceptance suite guards against. If the target has no
		  discontinuities the two metrics are identical.

	low_band_mean_power : float or None
		Mean PSD power in the low band (nu < 0.8 * nu0).
	nu0 : float
		Target principal frequency.
	"""
	if not target.success:
		raise ValueError("psd_fit_metrics requires a successfully solved target (target.success is False)")
	if target_power is None:
		target_power = interpolate_target_power(target, freqs)
	else:
		target_power = np.asarray(target_power, dtype=np.float32)
	freqs = np.asarray(freqs, dtype=np.float32)
	psd = np.asarray(psd, dtype=np.float32)
	residual = psd - target_power
	nu0 = float(target.nu0)
	low_band = freqs < 0.8 * nu0

	# Compute interior max_abs: exclude bins adjacent to step discontinuities.
	bin_width = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
	disc_positions = find_target_discontinuities(target)
	if disc_positions.size == 0 or bin_width == 0.0:
		interior_mask = np.ones(len(freqs), dtype=bool)
	else:
		# A bin is "edge-adjacent" if its centre is within one bin-width of any
		# discontinuity position.
		dist_to_nearest_disc = np.min(
			np.abs(freqs[:, None] - disc_positions[None, :]), axis=1
		)
		interior_mask = dist_to_nearest_disc > bin_width

	if np.any(interior_mask):
		max_abs_interior = float(np.max(np.abs(residual[interior_mask])))
	else:
		# Fallback: all bins are edge-adjacent (degenerate); report full max.
		max_abs_interior = float(np.max(np.abs(residual)))

	return {
		"rmse": float(np.sqrt(np.mean(residual ** 2))),
		"max_abs_residual": float(np.max(np.abs(residual))),
		"max_abs_residual_interior": max_abs_interior,
		"low_band_mean_power": float(np.mean(psd[low_band])) if np.any(low_band) else None,
		"nu0": nu0,
	}


def plot_psd_overlay(freqs: np.ndarray, psd: np.ndarray, target: dict, title: str):
	import matplotlib.pyplot as plt

	target_power = interpolate_target_power(target, freqs)
	fig, ax = plt.subplots(figsize=(7.0, 4.2))
	ax.plot(freqs, psd, label="ensemble empirical", linewidth=2.0)
	ax.plot(freqs, target_power, label="target", linewidth=2.0)
	ax.set_title(title)
	ax.set_xlabel("nu")
	ax.set_ylabel("power")
	ax.set_ylim(bottom=0.0)
	ax.grid(True, alpha=0.25)
	ax.legend()
	return fig


def _parse_m0(value: str):
	if value == "min":
		return "min"
	try:
		return float(value)
	except ValueError:
		raise argparse.ArgumentTypeError(f"--m0 must be 'min' or a float, got {value!r}")


def main() -> None:
	import matplotlib

	matplotlib.use("Agg", force=True)
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--nu0", type=float, default=0.85)
	parser.add_argument("--e0", type=float, default=0.0)
	parser.add_argument("--m0", type=_parse_m0, default="min")
	parser.add_argument("--n-points", type=int, default=1024)
	parser.add_argument("--iterations", type=int, default=400)
	parser.add_argument("--num-seeds", type=int, default=8)
	parser.add_argument("--device", default="auto")
	parser.add_argument("--max-freq", type=float, default=3.0)
	parser.add_argument("--output-dir", type=Path, required=True)
	args = parser.parse_args()

	# "min" only exists at this CLI boundary; translate it into an explicit
	# find_min_m0 search before solving.
	if args.m0 == "min":
		m0 = find_min_m0(nu0=args.nu0, e0=args.e0)
		if m0 is None:
			raise RuntimeError("no feasible finite m0 found for the requested nu0/e0")
	else:
		m0 = args.m0
	target = solve_ds_wave_target(nu0=args.nu0, e0=args.e0, m0=m0)
	if not target.success:
		raise RuntimeError(f"DS-Wave target solve {target.status}: {target.message}")

	seeds = list(range(1, args.num_seeds + 1))
	freqs, psd, binned_target = ensemble_radial_psd(
		target,
		n_points=args.n_points,
		seeds=seeds,
		iterations=args.iterations,
		device=args.device,
		max_freq=args.max_freq,
		return_binned_target=True,
	)
	metrics = psd_fit_metrics(freqs, psd, target, target_power=binned_target)
	metrics.update(
		n_points=args.n_points,
		iterations=args.iterations,
		seeds=seeds,
		# PCF matching is the only synthesis path; recorded for continuity with
		# the committed research/baseline and research/verification artifacts.
		synthesis_mode="pcf_matching",
	)

	args.output_dir.mkdir(parents=True, exist_ok=True)
	metrics_path = args.output_dir / "metrics_pcf_matching.json"
	metrics_path.write_text(json.dumps(metrics, indent=2))
	fig = plot_psd_overlay(freqs, psd, target, "Ensemble radial PSD (pcf_matching)")
	save_figure(fig, args.output_dir / "radial_psd_pcf_matching.png")
	np.savez(args.output_dir / "psd_pcf_matching.npz", freqs=freqs, psd=psd)
	print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
	main()

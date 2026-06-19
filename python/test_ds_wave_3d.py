"""Standalone 3D DS-Wave tests (no pytest). Run: python test_ds_wave_3d.py"""
import math
import sys
from pathlib import Path

import numpy as np

python_dir = Path(__file__).resolve().parent
if str(python_dir) not in sys.path:
	sys.path.insert(0, str(python_dir))

import ds_wave_3D as d3
from ds_wave_target import make_hankel_matrix, DsWaveTarget


def test_2d_kernel_reproduces_gaussian():
	# A Gaussian is its own d-dimensional Fourier transform: f(r)=H @ F with F=exp(-pi nu^2).
	nu = np.linspace(0.0, 30.0, 60001)
	r = np.array([0.0, 0.3, 0.7, 1.0, 1.5])
	H = make_hankel_matrix(nu, r)
	f = H @ np.exp(-math.pi * nu ** 2)
	assert np.max(np.abs(f - np.exp(-math.pi * r ** 2))) < 1e-3


def test_3d_kernel_reproduces_gaussian():
	nu = np.linspace(0.0, 30.0, 60001)
	r = np.array([0.0, 0.3, 0.7, 1.0, 1.5])
	H = d3.make_hankel_matrix_3d(nu, r)
	f = H @ np.exp(-math.pi * nu ** 2)
	assert np.max(np.abs(f - np.exp(-math.pi * r ** 2))) < 1e-3


def test_3d_target_solves_and_is_realizable():
	target = d3.solve_ds_wave_target_3d(nu0=0.85, e0=0.0, m0="min", nu_max=2.5, n_nu=60, n_r=40, m0_tol=0.08)
	assert isinstance(target, DsWaveTarget)
	assert target.success is True
	assert target.dimension == 3
	assert np.isfinite(target.m0) and target.m0 > 1.0
	assert np.all(target.P[target.low_mask] <= target.e0 + 1e-6)   # clean low band
	assert np.all(target.g >= -1e-6)                                # g(r) >= 0
	assert abs(target.g[0]) < 0.2                                   # exclusion near r=0
	assert abs(target.P[-1] - 1.0) < 0.1                            # converges toward 1


def test_3d_infeasible_returns_clean_dataclass():
	target = d3.solve_ds_wave_target_3d(nu0=0.85, e0=0.0, m0=1.0, nu_max=2.5, n_nu=40, n_r=28, require_success=False)
	assert isinstance(target, DsWaveTarget)
	assert target.success is False
	assert target.dimension == 3
	assert target.F is None and target.P is None and target.g is None


def test_evaluate_pcf_3d_matches_solver_grid():
	target = d3.solve_ds_wave_target_3d(nu0=0.85, e0=0.0, m0="min", nu_max=2.5, n_nu=60, n_r=40, m0_tol=0.08)
	g_eval = d3.evaluate_target_pcf_3d(target, target.r)
	assert np.allclose(g_eval, target.g, atol=1e-5)


def test_resolution_3d_decouples_nbins_and_calibrates_smoothing():
	# nbins=None resolves to max(4096, n_points), decoupled from N (the RC-1 fix).
	# At N=1024 the old coupling gave 1024; now it is floored at 4096.
	(nbins, smoothing) = d3.resolve_targetrdf_resolution_3d(1024, None, None)
	assert nbins == 4096
	# smoothing=None holds a fixed normalized sigma (sigma_rnorm=0.04) via N^(1/3):
	# sigma_rnorm = sigma_bins * (0.5/nbins) * N^(1/3).
	sigma_rnorm = smoothing * (0.5 / nbins) * 1024 ** (1.0 / 3.0)
	assert abs(sigma_rnorm - d3.DEFAULT_TARGETRDF_SIGMA_RNORM_3D) < 1e-6
	# Above the floor, nbins follows n_points.
	(nbins_big, _) = d3.resolve_targetrdf_resolution_3d(8000, None, None)
	assert nbins_big == 8000
	# Explicit values pass through unchanged.
	assert d3.resolve_targetrdf_resolution_3d(1024, 256, 3.0) == (256, 3.0)


def test_3d_synthesis_smoke_returns_dataclass():
	if d3.torch is None:
		print("SKIP test_3d_synthesis_smoke_returns_dataclass: torch unavailable")
		return
	from ds_wave_targetrdf import SynthesisResult
	target = d3.solve_ds_wave_target_3d(nu0=0.85, e0=0.0, m0=2.0, nu_max=2.5, n_nu=30, n_r=20)
	result = d3.synthesize_toroidal_points_3d(target, n_points=16, iterations=2, seed=123, device="cpu", nbins=16, smoothing=2.0)
	assert isinstance(result, SynthesisResult)
	assert result.points.shape == (16, 3)
	assert np.all(np.isfinite(result.points))
	assert np.all(result.points >= 0.0) and np.all(result.points < 1.0)
	assert np.isfinite(result.energy_history[-1])


def _run_all():
	tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
	failures = 0
	for t in tests:
		try:
			t()
			print(f"PASS {t.__name__}")
		except Exception as exc:  # noqa: BLE001
			failures += 1
			print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
	print(f"\n{len(tests) - failures}/{len(tests)} passed")
	return failures


if __name__ == "__main__":
	sys.exit(1 if _run_all() else 0)

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest


python_dir = Path(__file__).resolve().parent
module_path = python_dir / "ds-wave.py"


def load_ds_wave():
	spec = importlib.util.spec_from_file_location("ds_wave", module_path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def test_module_has_no_forbidden_local_imports():
	tree = ast.parse(module_path.read_text(encoding="utf-8"))
	for node in ast.walk(tree):
		if isinstance(node, ast.Import):
			names = {alias.name.split(".")[0] for alias in node.names}
		elif isinstance(node, ast.ImportFrom):
			names = {node.module.split(".")[0]} if node.module else set()
		else:
			continue

		forbidden = {
			"gaussian_blue_noise_gpu",
			"spectra",
			"stair_blue_noise",
			"stair_blue_noise_gpu",
		}
		assert names.isdisjoint(forbidden)


def test_solver_satisfies_constraints_on_small_grid():
	ds_wave = load_ds_wave()

	target = ds_wave.solve_ds_wave_target(
		nu0=0.85,
		e0=0.0,
		m0=2.0,
		nu_max=2.5,
		n_nu=36,
		n_r=30,
		tail_anchor_count=2,
	)

	assert target["status"] == "optimal"
	assert target["success"] is True
	assert np.all(target["P"] >= -1e-8)
	assert np.all(target["g"] >= -1e-8)
	assert np.all(target["P"][target["low_mask"]] <= 1e-8)
	assert np.all(target["P"] <= 2.0 + 1e-8)
	assert np.allclose(target["F"][-2:], 0.0, atol=1e-8)


def test_solver_reports_infeasible_m0_cleanly():
	ds_wave = load_ds_wave()

	target = ds_wave.solve_ds_wave_target(
		nu0=0.85,
		e0=0.0,
		m0=1.0,
		nu_max=2.5,
		n_nu=36,
		n_r=30,
		require_success=False,
	)

	assert target["success"] is False
	assert target["status"] == "infeasible"
	assert target["F"] is None
	assert "infeasible" in target["message"].lower()


def test_solver_rejects_invalid_m0_and_m0_tol():
	ds_wave = load_ds_wave()

	with pytest.raises(ValueError, match="m0"):
		ds_wave.solve_ds_wave_target(m0=0.5, n_nu=12, n_r=8)

	with pytest.raises(ValueError, match="m0_tol"):
		ds_wave.solve_ds_wave_target(m0="min", m0_tol=0.0, n_nu=12, n_r=8)


def test_min_m0_search_returns_finite_feasible_target():
	ds_wave = load_ds_wave()

	target = ds_wave.solve_ds_wave_target(
		nu0=0.85,
		e0=0.0,
		m0="min",
		nu_max=2.5,
		n_nu=34,
		n_r=28,
		m0_tol=0.08,
	)

	assert target["status"] == "optimal"
	assert target["success"] is True
	assert np.isfinite(target["m0"])
	assert target["m0"] > 1.0
	assert np.max(target["P"]) <= target["m0"] + 1e-8
	assert np.all(target["g"] >= -1e-8)


def test_point_synthesis_smoke_outputs_finite_toroidal_points():
	ds_wave = load_ds_wave()
	if ds_wave.torch is None:
		pytest.skip("torch is not available")
	target = ds_wave.solve_ds_wave_target(
		nu0=0.85,
		e0=0.0,
		m0=2.0,
		nu_max=2.5,
		n_nu=30,
		n_r=24,
	)

	result = ds_wave.synthesize_toroidal_points(
		target,
		n_points=16,
		iterations=2,
		seed=123,
		device="cpu",
		mode_limit=48,
	)

	points = result["points"]
	assert points.shape == (16, 2)
	assert np.all(np.isfinite(points))
	assert np.all(points >= 0.0)
	assert np.all(points < 1.0)
	assert np.isfinite(result["energy_history"][-1])
	assert result["synthesis_mode"] == ds_wave.SynthesisMode.SPECTRUM_MATCHING.value


def test_synthesis_mode_enum_has_requested_modes():
	ds_wave = load_ds_wave()

	assert ds_wave.SynthesisMode.SPECTRUM_MATCHING.value == "spectrum_matching"
	assert ds_wave.SynthesisMode.PCF_MATCHING.value == "pcf_matching"
	assert ds_wave.parse_synthesis_mode("SPECTRUM_MATCHING") is ds_wave.SynthesisMode.SPECTRUM_MATCHING
	assert ds_wave.parse_synthesis_mode("pcf_matching") is ds_wave.SynthesisMode.PCF_MATCHING


def test_pcf_matching_synthesis_smoke_outputs_finite_toroidal_points():
	ds_wave = load_ds_wave()
	if ds_wave.torch is None:
		pytest.skip("torch is not available")
	target = ds_wave.solve_ds_wave_target(
		nu0=0.85,
		e0=0.0,
		m0=2.0,
		nu_max=2.5,
		n_nu=30,
		n_r=24,
	)

	result = ds_wave.synthesize_toroidal_points(
		target,
		n_points=16,
		iterations=2,
		seed=123,
		device="cpu",
		synthesis_mode=ds_wave.SynthesisMode.PCF_MATCHING,
		pcf_num_bins=16,
		pcf_smoothing=2.0,
	)

	points = result["points"]
	assert points.shape == (16, 2)
	assert np.all(np.isfinite(points))
	assert np.all(points >= 0.0)
	assert np.all(points < 1.0)
	assert np.isfinite(result["energy_history"][-1])
	assert result["synthesis_mode"] == ds_wave.SynthesisMode.PCF_MATCHING.value


def test_targetrdf_force_curve_integrates_rdf_error():
	ds_wave = load_ds_wave()
	rdf = np.array([0.0, 2.0, 2.0, 1.0], dtype=np.float64)
	target = np.ones_like(rdf)

	force = ds_wave.compute_targetrdf_force_curve(rdf, target, n_points=16)

	assert force.shape == rdf.shape
	assert force[0] == 0.0
	assert force[2] > 0.0


def test_targetrdf_target_curve_uses_unit_square_radius():
	ds_wave = load_ds_wave()
	target = ds_wave.solve_ds_wave_target(
		nu0=0.85,
		e0=0.0,
		m0=2.0,
		nu_max=2.5,
		n_nu=30,
		n_r=24,
	)

	target_rdf, unit_r = ds_wave.make_targetrdf_target_curve(
		target,
		n_points=16,
		nbins=8,
		smoothing=0.0,
	)

	assert target_rdf.shape == (8,)
	assert unit_r.shape == (8,)
	assert unit_r[0] == pytest.approx(0.0)
	assert unit_r[-1] < 0.5
	assert np.all(np.isfinite(target_rdf))


def test_targetrdf_target_curve_evaluates_spectrum_beyond_solver_r_grid():
	ds_wave = load_ds_wave()
	target = {
		"nu": np.linspace(0.0, 1.0, 8, dtype=np.float64),
		"F": np.zeros(8, dtype=np.float64),
		"r": np.array([0.0, 1.0], dtype=np.float64),
		"g": np.array([0.25, 0.25], dtype=np.float64),
	}

	target_rdf, unit_r = ds_wave.make_targetrdf_target_curve(
		target,
		n_points=64,
		nbins=8,
		smoothing=0.0,
	)

	assert unit_r[-1] > target["r"][-1] / np.sqrt(64.0)
	assert np.allclose(target_rdf, 1.0)


def test_targetrdf_rdf_counts_normalise_random_pairs():
	ds_wave = load_ds_wave()
	if ds_wave.torch is None:
		pytest.skip("torch is not available")
	points = ds_wave.torch.tensor(
		[
			[0.0, 0.0],
			[0.25, 0.0],
			[0.0, 0.25],
			[0.25, 0.25],
		],
		dtype=ds_wave.torch.float32,
	)

	rdf = ds_wave.compute_targetrdf_rdf(points, nbins=8, smoothing=0.0, chunk_size=2)

	assert rdf.shape == (8,)
	assert np.all(np.isfinite(rdf))
	assert np.max(rdf) > 0.0


def test_pcf_matching_runs_requested_iterations_when_logging_every_step():
	ds_wave = load_ds_wave()
	if ds_wave.torch is None:
		pytest.skip("torch is not available")
	target = ds_wave.solve_ds_wave_target(
		nu0=0.85,
		e0=0.0,
		m0=2.0,
		nu_max=2.5,
		n_nu=30,
		n_r=24,
	)

	result = ds_wave.synthesize_toroidal_points(
		target,
		n_points=16,
		iterations=7,
		seed=123,
		device="cpu",
		synthesis_mode=ds_wave.SynthesisMode.PCF_MATCHING,
		pcf_num_bins=16,
		pcf_smoothing=2.0,
		log_every=1,
	)

	assert result["iterations_run"] == 7
	assert result["energy_history"].shape[0] == 7


def test_pcf_matching_tracks_best_energy_during_fixed_iterations():
	ds_wave = load_ds_wave()
	if ds_wave.torch is None:
		pytest.skip("torch is not available")
	target = ds_wave.solve_ds_wave_target(
		nu0=0.85,
		e0=0.0,
		m0=2.0,
		nu_max=2.5,
		n_nu=30,
		n_r=24,
	)

	result = ds_wave.synthesize_toroidal_points(
		target,
		n_points=64,
		iterations=20,
		seed=123,
		device="cpu",
		synthesis_mode=ds_wave.SynthesisMode.PCF_MATCHING,
		pcf_num_bins=64,
		pcf_smoothing=2.0,
		log_every=1,
	)

	deltas = np.diff(result["energy_history"])
	assert np.all(deltas <= 1e-12)


def test_pcf_matching_accepts_initial_points():
	ds_wave = load_ds_wave()
	if ds_wave.torch is None:
		pytest.skip("torch is not available")
	target = ds_wave.solve_ds_wave_target(
		nu0=0.85,
		e0=0.0,
		m0=2.0,
		nu_max=2.5,
		n_nu=30,
		n_r=24,
	)
	initial_points = np.full((16, 2), 0.25, dtype=np.float32)

	result = ds_wave.synthesize_toroidal_points(
		target,
		n_points=16,
		iterations=1,
		seed=123,
		device="cpu",
		synthesis_mode=ds_wave.SynthesisMode.PCF_MATCHING,
		initial_points=initial_points,
		pcf_num_bins=16,
		pcf_smoothing=2.0,
	)

	assert result["points"].shape == (16, 2)
	assert result["iterations_run"] == 1


def test_pcf_matching_rejects_invalid_radius_support():
	ds_wave = load_ds_wave()
	if ds_wave.torch is None:
		pytest.skip("torch is not available")
	target = ds_wave.solve_ds_wave_target(
		nu0=0.85,
		e0=0.0,
		m0=2.0,
		nu_max=2.5,
		n_nu=30,
		n_r=24,
	)

	with pytest.raises(ValueError, match="at least 2"):
		ds_wave.synthesize_toroidal_points(
			target,
			n_points=1,
			iterations=1,
			device="cpu",
			synthesis_mode=ds_wave.SynthesisMode.PCF_MATCHING,
		)


def test_frequency_modes_preserve_low_frequency_hole_modes():
	ds_wave = load_ds_wave()
	n_points = 2048
	nu0 = 0.85
	nu_max = 10.0
	mode_limit = 8192

	all_modes = ds_wave.make_frequency_modes(n_points, nu_max=nu_max, mode_limit=0, seed=0)
	all_radii = np.linalg.norm(all_modes, axis=1) / np.sqrt(float(n_points))
	expected_low_count = int(np.sum(all_radii <= nu0))

	modes = ds_wave.make_frequency_modes(
		n_points,
		nu_max=nu_max,
		mode_limit=mode_limit,
		seed=1234,
		priority_nu=nu0,
	)
	radii = np.linalg.norm(modes, axis=1) / np.sqrt(float(n_points))

	assert expected_low_count < mode_limit
	assert int(np.sum(radii <= nu0)) == expected_low_count


def test_point_synthesis_rejects_empty_mode_set():
	ds_wave = load_ds_wave()
	if ds_wave.torch is None:
		pytest.skip("torch is not available")
	target = ds_wave.solve_ds_wave_target(
		nu0=0.0,
		e0=0.0,
		m0=2.0,
		nu_max=0.01,
		n_nu=12,
		n_r=8,
	)

	with pytest.raises(ValueError, match="no Fourier modes"):
		ds_wave.synthesize_toroidal_points(
			target,
			n_points=4,
			iterations=1,
			seed=123,
			device="cpu",
		)

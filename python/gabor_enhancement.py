import math
import numpy as np
import torch
from PIL import Image

PI      = math.pi
EPSILON = 1e-6

_THIBOS_ECC = np.array([0.0,  5.0,  10.0, 15.0, 20.0, 25.0, 30.0], dtype=np.float64)
_THIBOS_TL  = np.array([60.0, 27.0, 10.5,  8.0,  5.5,  4.8,  4.0], dtype=np.float64)
_THIBOS_TH  = np.array([60.0, 40.0, 26.0, 24.0, 23.0, 21.0, 20.5], dtype=np.float64)

def _thibos_limits_cpp(ecc_deg: np.ndarray, ppd: float):
	T_L_cpd = np.interp(ecc_deg, _THIBOS_ECC, _THIBOS_TL)
	T_H_cpd = np.interp(ecc_deg, _THIBOS_ECC, _THIBOS_TH)
	return (T_L_cpd / ppd).astype(np.float32), (T_H_cpd / ppd).astype(np.float32)

def wang_hash(s: int) -> int:
	s = int(np.uint32(s))
	s = int(np.uint32(s ^ np.uint32(61)) ^ np.uint32(s >> 16))
	s = int(np.uint32(s) * np.uint32(9))
	s = int(np.uint32(s) ^ np.uint32(s >> 4))
	s = int(np.uint32(s) * np.uint32(0x27d4eb2d))
	s = int(np.uint32(s) ^ np.uint32(s >> 15))
	return s

def rand_float(state: int):
	state = wang_hash(state)
	return state, float(np.uint32(state)) / 4294967296.0

def rand_int(state: int, lo: int, hi: int):
	state, f = rand_float(state)
	return state, lo + int(f * (hi - lo))

def pixels_per_degree(screen_width_px: float, screen_width_cm: float, distance_cm: float) -> float:
	cm_per_pixel = screen_width_cm / screen_width_px
	size_cm      = 2.0 * distance_cm * math.tan(math.radians(1.0) / 2.0)
	return size_cm / cm_per_pixel

def luminance_np(rgb: np.ndarray) -> np.ndarray:
	return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])

def estimate_orientation_image(lum: np.ndarray) -> np.ndarray:
	p  = np.pad(lum, 1, mode='edge')
	tl = p[:-2, :-2];  tc = p[:-2, 1:-1];  tr = p[:-2, 2:]
	ml = p[1:-1, :-2];                       mr = p[1:-1, 2:]
	bl = p[2:,   :-2];  bc = p[2:,  1:-1];  br = p[2:,  2:]
	gx = -tl + tr - 2.0*ml + 2.0*mr - bl + br
	gy = -tl - 2.0*tc - tr  + bl + 2.0*bc + br
	return np.arctan2(gy, gx).astype(np.float32)

def build_laplacian_pyramid(image_rgb: np.ndarray, num_levels: int = 5):
	from scipy.ndimage import zoom as _zoom
	gaussian = [image_rgb.astype(np.float32)]
	for _ in range(num_levels):
		prev   = gaussian[-1]
		h, w   = prev.shape[:2]
		nh, nw = max(1, h // 2), max(1, w // 2)
		down   = _zoom(prev, (nh / h, nw / w, 1), order=1)
		gaussian.append(down.astype(np.float32))
	laplacian = []
	for k in range(num_levels):
		curr   = gaussian[k]
		nxt    = gaussian[k + 1]
		h, w   = curr.shape[:2]
		nh, nw = nxt.shape[:2]
		up     = _zoom(nxt, (h / nh, w / nw, 1), order=1)
		laplacian.append((curr - up).astype(np.float32))
	laplacian.append(gaussian[-1])
	return laplacian, gaussian

def precompute_impulses(tex_w: int, tex_h: int, num_cells: int, num_impulses: int, rng_seed: int):
	cell_h = tex_h // num_cells
	cell_w = tex_w // num_cells

	imp_x  = np.empty((num_cells, num_cells, num_impulses), dtype=np.float32)
	imp_y  = np.empty((num_cells, num_cells, num_impulses), dtype=np.float32)
	imp_rn = np.empty((num_cells, num_cells, num_impulses), dtype=np.float32)

	for cy in range(num_cells):
		for cx in range(num_cells):
			cell_seed = int(np.uint32(rng_seed + cy * num_cells + cx))
			state     = wang_hash(cell_seed)

			y0 = cy * cell_h;  y1 = min(tex_h, y0 + cell_h)
			x0 = cx * cell_w;  x1 = min(tex_w, x0 + cell_w)

			for i in range(num_impulses):
				state, iy = rand_int(state, y0, y1)
				state, ix = rand_int(state, x0, x1)
				state, r1 = rand_float(state)
				state, r2 = rand_float(state)
				r1 = max(r1, 1e-12)
				rn = math.sqrt(-2.0 * math.log(r1)) * math.cos(2.0 * PI * r2)
				imp_x [cy, cx, i] = float(ix)
				imp_y [cy, cx, i] = float(iy)
				imp_rn[cy, cx, i] = float(rn)

	return imp_x, imp_y, imp_rn

def accumulate_noise(
	imp_x_t:     torch.Tensor,
	imp_y_t:     torch.Tensor,
	imp_rn_t:    torch.Tensor,
	pixel_cx:    torch.Tensor,
	pixel_cy:    torch.Tensor,
	px_x:        torch.Tensor,
	px_y:        torch.Tensor,
	F_L_map:     torch.Tensor,
	F_H_map:     torch.Tensor,
	sigma_n_map: torch.Tensor,
	amp_map:     torch.Tensor,
	theta_map:   torch.Tensor,
	num_cells:   int,
	active_mask: torch.Tensor,
) -> torch.Tensor:

	noise    = torch.zeros_like(px_x)

	F_L_e      = F_L_map.unsqueeze(-1)
	F_H_e      = F_H_map.unsqueeze(-1)
	amp_e      = amp_map.unsqueeze(-1)
	sigma_n_e  = sigma_n_map.unsqueeze(-1)
	cos_th     = torch.cos(theta_map).unsqueeze(-1)
	sin_th     = torch.sin(theta_map).unsqueeze(-1)
	px_x_e     = px_x.unsqueeze(-1)
	px_y_e     = px_y.unsqueeze(-1)

	log_mean = 0.5 * (torch.log(torch.clamp(F_L_e, min=EPSILON)) +
	                  torch.log(torch.clamp(F_H_e, min=EPSILON)))

	for dy in range(-1, 2):
		for dx in range(-1, 2):
			ncy_raw = pixel_cy + dy
			ncx_raw = pixel_cx + dx
			valid   = (
				(ncy_raw >= 0) & (ncy_raw < num_cells) &
				(ncx_raw >= 0) & (ncx_raw < num_cells) &
				active_mask
			)

			ncy = ncy_raw.clamp(0, num_cells - 1)
			ncx = ncx_raw.clamp(0, num_cells - 1)

			imp_ix = imp_x_t [ncy, ncx, :]
			imp_iy = imp_y_t [ncy, ncx, :]
			imp_rn = imp_rn_t[ncy, ncx, :]

			ox = px_x_e - imp_ix
			oy = px_y_e - imp_iy

			fcpp     = torch.exp(log_mean + sigma_n_e * imp_rn)
			fcpp     = torch.clamp(fcpp, min=F_L_e, max=F_H_e)

			kr       = 3.5 / fcpp
			in_range = ((ox*ox + oy*oy) <= (kr * kr)) & valid.unsqueeze(-1)

			rx       = ox * cos_th + oy * sin_th
			ry       = -ox * sin_th + oy * cos_th
			sigma    = 0.5 / fcpp
			gaussian = torch.exp(-(rx*rx + ry*ry) / (2.0 * sigma * sigma))
			sinusoid = torch.cos(2.0 * PI * fcpp * rx)

			noise   += (amp_e * gaussian * sinusoid * in_range).sum(dim=-1)

	return noise

def gabor_foveated_enhance(
	image_path: str,
	foveation_center: tuple[float, float] = (0.5, 0.5),
	radius_fovea: float                   = 0.1,
	radius_periphery: float               = 0.2,
	screen_width_cm: float                = 60.0,
	screen_height_cm: float               = 30.0,
	distance_from_screen_cm: float        = 71.0,
	blur_rate_arcmin_per_degree: float    = 0.34,
	s_k: float                            = 21.02,
	s_f: float                            = 2.21,
	cells: int                            = 64,
	impulses_per_cell: int                = 32,
	seed: int                             = 10,
	frequency_scale: float                = 1.0,
	region_mode: int                      = 0,
	fovea_suppress_radius: float          = 8.0,
	amp_override: float | None            = None,
	device: str | None                    = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	dev = torch.device(device)
	print(f"Device : {dev}")

	img  = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
	H, W = img.shape[:2]
	print(f"Image  : {W}×{H}")

	lum       = luminance_np(img)
	theta_map = estimate_orientation_image(lum)

	print("Building Laplacian pyramid …")
	lap_pyr, _ = build_laplacian_pyramid(img, num_levels=5)

	xs = np.arange(W, dtype=np.float32)
	ys = np.arange(H, dtype=np.float32)
	px_x_np, px_y_np = np.meshgrid(xs, ys)

	fov_cx, fov_cy = foveation_center
	gaze_px        = np.array([fov_cx * W, fov_cy * H], dtype=np.float32)
	ppd            = pixels_per_degree(W, screen_width_cm, distance_from_screen_cm)

	uv_x           = (px_x_np + 0.5) / W
	uv_y           = (px_y_np + 0.5) / H
	screen_dims_cm = np.array([screen_width_cm, screen_height_cm], dtype=np.float32)
	fov_uv         = np.array([fov_cx, fov_cy], dtype=np.float32)
	uv_grid        = np.stack([uv_x, uv_y], axis=-1)
	phys_diff      = (uv_grid - fov_uv) * screen_dims_cm
	ecc_deg_map    = np.linalg.norm(
	                   np.degrees(np.arctan(phys_diff / distance_from_screen_cm)),
	                   axis=-1).astype(np.float32)

	dist_px   = np.sqrt((px_x_np - gaze_px[0])**2 + (px_y_np - gaze_px[1])**2)
	sigma_map = np.maximum(EPSILON,
	            (blur_rate_arcmin_per_degree * (dist_px / ppd) / 60.0 * ppd) / 2.355
	            ).astype(np.float32)

	T_L_cpp, T_H_cpp = _thibos_limits_cpp(ecc_deg_map, ppd)
	sigma_safe       = np.maximum(EPSILON, sigma_map)
	gauss_cutoff_cpp = (3.0 / (2.0 * PI * sigma_safe)).astype(np.float32)
	F_L_map          = np.maximum(T_L_cpp, gauss_cutoff_cpp)
	F_H_map          = np.minimum(T_H_cpp, np.float32(0.5))
	bad              = F_L_map >= F_H_map
	F_H_map          = np.where(bad, F_L_map * 1.001, F_H_map).astype(np.float32)
	F_L_map          = (F_L_map * frequency_scale).astype(np.float32)
	F_H_map          = (F_H_map * frequency_scale).astype(np.float32)

	mu_n    = 0.5 * (np.log(np.maximum(EPSILON, F_L_map)) +
	                 np.log(np.maximum(EPSILON, F_H_map)))
	sigma_n = (0.5 * s_f * (mu_n - np.log(np.maximum(EPSILON, F_L_map)))).astype(np.float32)

	f_c_map = np.sqrt(-np.log(0.25)) / (PI * sigma_safe)
	l_a_map = (-np.log2(np.maximum(EPSILON, f_c_map)) - 0.5).astype(np.float32)
	l0_map  = np.clip(np.floor(l_a_map).astype(np.int32), 0, 3)
	l1_map  = np.clip(l0_map + 1, 0, 3)
	t_map   = (l_a_map - np.floor(l_a_map)).astype(np.float32)

	def _lap_full(lvl: int) -> np.ndarray:
		arr = lap_pyr[lvl]
		if arr.shape[0] != H or arr.shape[1] != W:
			from scipy.ndimage import zoom as _zoom
			arr = _zoom(arr, (H / arr.shape[0], W / arr.shape[1], 1), order=1)
		return np.linalg.norm(arr, axis=-1).astype(np.float32)

	lap_levels = [_lap_full(k) for k in range(4)]
	A_map      = np.zeros((H, W), dtype=np.float32)
	for lvl in range(4):
		A_map[l0_map == lvl] += lap_levels[lvl][l0_map == lvl] * (1.0 - t_map[l0_map == lvl])
		A_map[l1_map == lvl] += lap_levels[lvl][l1_map == lvl] * t_map[l1_map == lvl]

	amp_map = (s_k * A_map).astype(np.float32)

	print(f"[diag] sigma_px  mean={sigma_map.mean():.3f}  max={sigma_map.max():.3f}")
	print(f"[diag] F_L cpp   mean={F_L_map.mean():.4f}  F_H cpp mean={F_H_map.mean():.4f}")
	print(f"[diag] sigma_n   mean={sigma_n.mean():.4f}  max={sigma_n.max():.4f}")
	print(f"[diag] A_map     mean={A_map.mean():.6f}  max={A_map.max():.6f}")
	print(f"[diag] amp_map   mean={amp_map.mean():.6f}  max={amp_map.max():.6f}")

	if amp_override is not None:
		print(f"[diag] amp_override={amp_override}")
		amp_map = np.full((H, W), float(amp_override), dtype=np.float32)

	fovea_mask = ecc_deg_map <= fovea_suppress_radius

	if region_mode == 0:
		region_mask = np.ones((H, W), dtype=bool)
	elif region_mode == 1:
		region_mask = uv_x <= 0.5
	else:
		region_mask = uv_x > 0.5

	active_mask_np = (~fovea_mask) & region_mask

	print(f"Precomputing impulses ({cells}×{cells} cells × {impulses_per_cell}) …")
	imp_x_np, imp_y_np, imp_rn_np = precompute_impulses(W, H, cells, impulses_per_cell, seed)

	def _t(a): return torch.from_numpy(a).to(dev)

	cell_h   = H // cells
	cell_w   = W // cells
	pixel_cy = torch.from_numpy(np.clip(px_y_np.astype(np.int64) // cell_h, 0, cells - 1)).to(dev)
	pixel_cx = torch.from_numpy(np.clip(px_x_np.astype(np.int64) // cell_w, 0, cells - 1)).to(dev)

	print("Accumulating noise …")
	noise_t = accumulate_noise(
		imp_x_t     = _t(imp_x_np),
		imp_y_t     = _t(imp_y_np),
		imp_rn_t    = _t(imp_rn_np),
		pixel_cx    = pixel_cx,
		pixel_cy    = pixel_cy,
		px_x        = _t(px_x_np),
		px_y        = _t(px_y_np),
		F_L_map     = _t(F_L_map),
		F_H_map     = _t(F_H_map),
		sigma_n_map = _t(sigma_n),
		amp_map     = _t(amp_map),
		theta_map   = _t(theta_map),
		num_cells   = cells,
		active_mask = torch.from_numpy(active_mask_np).to(dev),
	)
	noise_np = noise_t.cpu().numpy()

	output_image                 = img.copy()
	output_image[fovea_mask]     = 0.5
	Y      = luminance_np(img)
	Yn     = np.clip(Y + noise_np, 0.0, 1.0)
	scale  = (Yn + EPSILON) / (Y + EPSILON)
	output_image[active_mask_np] = np.clip(img * scale[..., None], 0.0, 1.0)[active_mask_np].astype(np.float32)

	all_ix = imp_x_np.reshape(-1).astype(np.int32).clip(0, W - 1)
	all_iy = imp_y_np.reshape(-1).astype(np.int32).clip(0, H - 1)
	impulse_map = np.zeros((H, W), dtype=np.float32)
	impulse_map[all_iy, all_ix] = 1.0

	print("Done.")
	return output_image, noise_np, impulse_map
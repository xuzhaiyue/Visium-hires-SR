from sr02_degrade_benchmark import *
SIGMA_RANGE=(1.5, 1.75)
def degrade_real2(gt: np.ndarray, lr_size: int, rng) -> tuple[np.ndarray, dict]:
    """v2: no global fading; JPEG moved before downsampling (real order: scanner stores JPEG → Space Ranger downsamples to PNG).
    Blur, defocus and noise ranges are those of v1 (sr09_scale65_calibrate fits the blur)."""
    od = -np.log(np.clip(gt.astype(np.float32) / 255, 1e-6, 1))
    alpha, beta = rng.uniform(0.85, 1.1, 3), rng.uniform(-0.03, 0.03, 3)
    hed = od.reshape(-1, 3) @ HED_FROM_RGB * alpha + beta
    img = np.clip(np.exp(-(hed @ RGB_FROM_HED)).reshape(gt.shape) * 255, 0, 255).astype(np.uint8)
    prm = {"stain_alpha": alpha.round(3).tolist(), "stain_beta": beta.round(3).tolist()}
    sigma_um = rng.uniform(*SIGMA_RANGE)
    img = cv2.GaussianBlur(img, (0, 0), sigma_um / GT_MPP)
    prm["blur_sigma_um"] = round(sigma_um, 3)
    if rng.random() < 0.3:
        r = max(1, int(round(rng.uniform(0.5, 1.5) / GT_MPP)))
        k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        cv2.circle(k, (r, r), r, 1, -1)
        img = cv2.filter2D(img, -1, k / k.sum())
        prm["defocus_r_um"] = r * GT_MPP
    q = int(rng.integers(60, 96))
    ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    prm["jpeg_q_at_0.5um"] = q
    img = cv2.resize(img, (lr_size, lr_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    noise = rng.uniform(1, 5)
    img += rng.normal(0, noise, img.shape)
    prm["noise_sd"] = round(noise, 2)
    return np.clip(img, 0, 255).astype(np.uint8), prm

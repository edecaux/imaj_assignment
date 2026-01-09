from PIL import Image
import numpy as np

def quantize_image_wu_rmse(image_path, RMSE_THRESHOLD=3.3, max_colors=25, sample_size=10000):
    """
    Quantifies an image using Wu's algorithm, increasing n_colors until RMSE < threshold.
    Returns (best_img, best_palette, best_k, k_list, rmse_list)
    best_palette shape = (k,4) columns = R,G,B,percentage (0-100)
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    pixels = img_np.reshape(-1, 3)

    rmse_list = []
    k_list = []
    best_img = None
    best_palette = None
    best_k = None

    for n_colors in range(2, max_colors + 1):
        img_quant = img.quantize(colors=n_colors, method=Image.FASTOCTREE)  # P-mode
        img_quant_np = np.array(img_quant.convert("RGB"))
        quant_pixels = img_quant_np.reshape(-1, 3)
        if quant_pixels.shape != pixels.shape:
            quant_pixels = quant_pixels[:pixels.shape[0], :]
        mse = np.mean((pixels - quant_pixels) ** 2)
        rmse = np.sqrt(mse)
        rmse = 100 * rmse / 255  # Normalize RMSE to [0, 100]
        rmse_list.append(rmse)
        k_list.append(n_colors)

        # counts par index palette (image P-mode)
        indices = np.array(img_quant).ravel()
        counts = np.bincount(indices, minlength=n_colors)
        total = counts.sum()
        if total > 0:
            percentages = 100.0 * counts[:n_colors] / total
        else:
            percentages = np.zeros(n_colors, dtype=float)

        if rmse < RMSE_THRESHOLD:
            best_img = img_quant.convert("RGB")
            palette_raw = img_quant.getpalette()[:n_colors*3]
            palette_rgb = np.array(palette_raw).reshape(-1, 3)
            best_palette = np.hstack([palette_rgb.astype(float), percentages.reshape(-1, 1)])
            best_k = n_colors
            break

    if best_img is None:
        best_img = img_quant.convert("RGB")
        palette_raw = img_quant.getpalette()[:n_colors*3]
        palette_rgb = np.array(palette_raw).reshape(-1, 3)
        indices = np.array(img_quant).ravel()
        counts = np.bincount(indices, minlength=n_colors)
        total = counts.sum()
        if total > 0:
            percentages = 100.0 * counts[:n_colors] / total
        else:
            percentages = np.zeros(n_colors, dtype=float)
        best_palette = np.hstack([palette_rgb.astype(float), percentages.reshape(-1, 1)])
        best_k = n_colors

    return best_img, best_palette, best_k, k_list, rmse_list
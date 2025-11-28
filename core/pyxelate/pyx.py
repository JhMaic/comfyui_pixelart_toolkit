###########################################
#
# GPU version for pyxelate
#
###########################################

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit
from skimage.exposure import equalize_adapthist
from skimage.transform import resize as skimage_resize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture

try:
    from .pal import BasePalette
except ImportError:

    class BasePalette:
        pass


from typing import Optional, Union, Tuple, Callable


@njit(fastmath=True)
def _atkinson_euclidean_clamped_impl(img_pad, means, h, w):
    """
    基于欧几里得距离的 Atkinson 抖动 (带数值截断)
    去除权重干扰，专注于几何距离，画面更纯净。
    """
    res_indices = np.zeros((h, w), dtype=np.int32)
    n_colors = len(means)

    # 颜色通道权重 (可选)
    # 纯欧氏距离是 1,1,1。
    # 为了更好的视觉效果，可以使用类似 Rec.601 的亮度权重 (R:0.3, G:0.59, B:0.11)
    # 但为了还原 Pyxelate 原始风格，我们通常保持 1:1:1 或轻微加权
    # 这里使用标准的 1.0，最稳健
    w_r, w_g, w_b = 1.0, 1.0, 1.0

    for y in range(h):
        for x in range(1, w + 1):
            # 获取当前像素
            raw_r = img_pad[y, x, 0]
            raw_g = img_pad[y, x, 1]
            raw_b = img_pad[y, x, 2]

            # --- 核心修复：严格截断 (Clamp) ---
            # 无论误差如何累积，用于“比色”的数值必须限制在 0.0 ~ 1.0 之间
            # 这能防止误差“过冲”导致算法匹配到错误的极端颜色 (例如蓝天里的亮粉色)
            curr_r = min(1.0, max(0.0, raw_r))
            curr_g = min(1.0, max(0.0, raw_g))
            curr_b = min(1.0, max(0.0, raw_b))

            best_idx = -1
            min_dist = 1e20

            # 寻找最近颜色
            for k in range(n_colors):
                d0 = curr_r - means[k, 0]
                d1 = curr_g - means[k, 1]
                d2 = curr_b - means[k, 2]

                # 欧几里得距离平方
                dist = (d0 * d0 * w_r) + (d1 * d1 * w_g) + (d2 * d2 * w_b)

                if dist < min_dist:
                    min_dist = dist
                    best_idx = k

            res_indices[y, x - 1] = best_idx

            # --- 误差扩散 ---
            center = means[best_idx]

            # 关键：误差计算应该基于【当前像素的真实值(包含漂移)】与【新颜色】的差
            # 这样可以保证总能量守恒
            err_r = (raw_r - center[0]) / 8.0
            err_g = (raw_g - center[1]) / 8.0
            err_b = (raw_b - center[2]) / 8.0

            # 扩散误差
            img_pad[y, x + 1, 0] += err_r
            img_pad[y, x + 1, 1] += err_g
            img_pad[y, x + 1, 2] += err_b
            img_pad[y, x + 2, 0] += err_r
            img_pad[y, x + 2, 1] += err_g
            img_pad[y, x + 2, 2] += err_b
            img_pad[y + 1, x - 1, 0] += err_r
            img_pad[y + 1, x - 1, 1] += err_g
            img_pad[y + 1, x - 1, 2] += err_b
            img_pad[y + 1, x, 0] += err_r
            img_pad[y + 1, x, 1] += err_g
            img_pad[y + 1, x, 2] += err_b
            img_pad[y + 1, x + 1, 0] += err_r
            img_pad[y + 1, x + 1, 1] += err_g
            img_pad[y + 1, x + 1, 2] += err_b
            img_pad[y + 2, x, 0] += err_r
            img_pad[y + 2, x, 1] += err_g
            img_pad[y + 2, x, 2] += err_b

    return res_indices


@njit(fastmath=True)
def _floyd_standard_impl(img_pad, means, h, w):
    """
    标准的 Floyd-Steinberg 抖动 (RGB 空间 + 截断)
    这是最经典、最干净的实现方式。
    """
    res_indices = np.zeros((h, w), dtype=np.int32)
    n_colors = len(means)

    for y in range(h):
        for x in range(1, w + 1):
            # 获取当前像素
            raw_r = img_pad[y, x, 0]
            raw_g = img_pad[y, x, 1]
            raw_b = img_pad[y, x, 2]

            # --- 关键：Clamp (截断) ---
            # 强制将像素拉回 0-1 范围，根除噪点
            curr_r = min(1.0, max(0.0, raw_r))
            curr_g = min(1.0, max(0.0, raw_g))
            curr_b = min(1.0, max(0.0, raw_b))

            # --- 寻找最近颜色 (Euclidean) ---
            best_idx = -1
            min_dist = 1e20

            for k in range(n_colors):
                d0 = curr_r - means[k, 0]
                d1 = curr_g - means[k, 1]
                d2 = curr_b - means[k, 2]
                # 纯距离比较
                dist = d0 * d0 + d1 * d1 + d2 * d2

                if dist < min_dist:
                    min_dist = dist
                    best_idx = k

            res_indices[y, x - 1] = best_idx
            center = means[best_idx]

            # --- 计算误差 ---
            # Error = 原始值(包含漂移) - 新颜色
            # 保持能量守恒
            err_r = (raw_r - center[0]) / 16.0
            err_g = (raw_g - center[1]) / 16.0
            err_b = (raw_b - center[2]) / 16.0

            # --- 扩散误差 (Floyd Kernel) ---
            #      X   7
            #  3   5   1

            # 右 (x+1, y)
            img_pad[y, x + 1, 0] += err_r * 7
            img_pad[y, x + 1, 1] += err_g * 7
            img_pad[y, x + 1, 2] += err_b * 7

            # 左下 (x-1, y+1)
            img_pad[y + 1, x - 1, 0] += err_r * 3
            img_pad[y + 1, x - 1, 1] += err_g * 3
            img_pad[y + 1, x - 1, 2] += err_b * 3

            # 下 (x, y+1)
            img_pad[y + 1, x, 0] += err_r * 5
            img_pad[y + 1, x, 1] += err_g * 5
            img_pad[y + 1, x, 2] += err_b * 5

            # 右下 (x+1, y+1)
            img_pad[y + 1, x + 1, 0] += err_r
            img_pad[y + 1, x + 1, 1] += err_g
            img_pad[y + 1, x + 1, 2] += err_b

    return res_indices


def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin

    h = torch.zeros_like(cmax)
    s = torch.zeros_like(cmax)
    v = cmax

    mask_delta = delta > 1e-8

    # Saturation
    s[mask_delta] = delta[mask_delta] / cmax[mask_delta]

    # Hue
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

    idx = (cmax_idx == 0) & mask_delta  # Red is max
    h[idx] = (g[idx] - b[idx]) / delta[idx] % 6

    idx = (cmax_idx == 1) & mask_delta  # Green is max
    h[idx] = (b[idx] - r[idx]) / delta[idx] + 2

    idx = (cmax_idx == 2) & mask_delta  # Blue is max
    h[idx] = (r[idx] - g[idx]) / delta[idx] + 4

    h = h / 6.0
    h[h < 0] += 1

    return torch.cat([h, s, v], dim=1)


def hsv2rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    """PyTorch 实现的 hsv2rgb"""
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    c = v * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = v - c

    zero = torch.zeros_like(h)

    # Conditionally assignment
    h6 = (h * 6).long()

    r_prime = torch.where(
        h6 == 0,
        c,
        torch.where(
            h6 == 1,
            x,
            torch.where(
                h6 == 2, zero, torch.where(h6 == 3, zero, torch.where(h6 == 4, x, c))
            ),
        ),
    )
    g_prime = torch.where(
        h6 == 0,
        x,
        torch.where(
            h6 == 1,
            c,
            torch.where(
                h6 == 2, c, torch.where(h6 == 3, x, torch.where(h6 == 4, zero, zero))
            ),
        ),
    )
    b_prime = torch.where(
        h6 == 0,
        zero,
        torch.where(
            h6 == 1,
            zero,
            torch.where(
                h6 == 2, x, torch.where(h6 == 3, c, torch.where(h6 == 4, c, x))
            ),
        ),
    )

    return torch.cat([r_prime + m, g_prime + m, b_prime + m], dim=1)


# --- 原始类定义 ---


class PyxWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class BGM(BayesianGaussianMixture):
    """
    Wrapper for BayesianGaussianMixture.
    保持 CPU/Sklearn 实现，确保调色板生成的算法严格一致。
    """

    MAX_ITER = 128
    RANDOM_STATE = 1234567

    def __init__(self, palette: Union[int, BasePalette], find_palette: bool) -> None:
        self.palette = palette
        self.find_palette = find_palette
        if self.find_palette:
            super().__init__(
                n_components=self.palette,
                max_iter=self.MAX_ITER,
                covariance_type="tied",
                weight_concentration_prior_type="dirichlet_distribution",
                weight_concentration_prior=1.0 / self.palette,
                mean_precision_prior=1.0 / 256.0,
                warm_start=False,
                random_state=self.RANDOM_STATE,
            )
        else:
            super().__init__(
                n_components=len(self.palette),
                max_iter=self.MAX_ITER,
                covariance_type="tied",
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=1e-7,
                mean_precision_prior=1.0 / len(self.palette),
                warm_start=False,
                random_state=self.RANDOM_STATE,
            )
            # start centroid search from the palette's values
            if hasattr(self.palette, "value"):  # Handle BasePalette objects
                vals = [val[0] for val in self.palette]
            else:
                vals = [val for val in self.palette]  # Handle list of colors
            self.mean_prior = np.mean(vals, axis=0)

    def _initialize_parameters(self, X: np.ndarray, random_state: int) -> None:
        assert (
            self.init_params == "kmeans"
        ), "Initialization is overwritten, can only be set as 'kmeans'."
        n_samples, _ = X.shape
        resp = np.zeros((n_samples, self.n_components))

        # 此处为了兼容 Pyx 的逻辑，如果传入的是 PyTorch tensor 转来的 numpy，需确保格式
        if self.find_palette:
            label = (
                KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                .fit(X)
                .labels_
            )
        else:
            # color distance based centroids (keep on CPU as this runs once)
            # Convert palette to Lab for strict logic match? Original used RGB2LAB via skimage
            from skimage.color import rgb2lab, deltaE_ciede2000

            # Helper to safely get palette values
            p_vals = self.palette
            if hasattr(self.palette, "value"):
                p_vals = [p[0] for p in self.palette]

            # Very slow strict logic, but runs once per fit
            label = np.argmin(
                [
                    deltaE_ciede2000(rgb2lab(X), rgb2lab(np.array([[p]])), kH=3, kL=2)
                    for p in p_vals
                ],
                axis=0,
            )

        resp[np.arange(n_samples), label] = 1
        self._initialize(X, resp)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BGM":
        converged = True
        with warnings.catch_warnings(record=True) as w:
            super().fit(X)
            if w and w[-1].category == ConvergenceWarning:
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                converged = False
        if not converged:
            warnings.warn(
                "Pyxelate could not properly assign colors, try a different palette size for better results!",
                PyxWarning,
            )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = super().predict_proba(X)
        if self.find_palette:
            if self.palette < 3:
                return np.sqrt(p)
        elif len(self.palette) < 3:
            return np.sqrt(p)
        return p


class Pyx(BaseEstimator, TransformerMixin):
    """
    Pyx extends scikit-learn transformers.
    Rewritten for ComfyUI compatibility with GPU acceleration where mathematically equivalent.
    """

    BGM_RESIZE = 256
    SCALE_RGB = 1.07
    HIST_BRIGHTNESS = 1.19
    COLOR_QUANT = 8
    DITHER_AUTO_SIZE_LIMIT_HI = 512
    DITHER_AUTO_SIZE_LIMIT_LO = 16
    DITHER_AUTO_COLOR_LIMIT = 8
    DITHER_NAIVE_BOOST = 1.33

    # DITHER_BAYER_MATRIX converted to Torch in init

    def __init__(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        factor: Optional[int] = None,
        upscale: Union[Tuple[int, int], int] = 1,
        depth: int = 1,
        palette: Union[int, BasePalette] = 8,
        dither: Optional[str] = "none",
        sobel: int = 3,
        alpha: float = 0.6,
        filter_obj: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:

        # --- Check Logic Same as Original ---
        if (width is not None or height is not None) and factor is not None:
            raise ValueError(
                "You can only set either height + width or the downscaling factor, but not both!"
            )
        assert height is None or height >= 1, "Height must be a positive integer!"
        assert width is None or width >= 1, "Width must be a positive integer!"
        assert factor is None or factor >= 1, "Factor must be a positive integer!"
        assert (
            isinstance(sobel, int) and sobel >= 2
        ), "Sobel must be an integer strictly greater than 1!"

        self.height = int(height) if height else None
        self.width = int(width) if width else None
        self.factor = int(factor) if factor else None
        self.sobel = sobel

        if isinstance(upscale, (list, tuple, set, np.ndarray)):
            assert len(upscale) == 2, "Upscale must be len 2, with 2 positive integers!"
            assert (
                upscale[0] >= 1 and upscale[1] >= 1
            ), "Upscale must have 2 positive values!"
            self.upscale = (upscale[0], upscale[1])
        else:
            assert upscale >= 1, "Upscale must be a positive integer!"
            self.upscale = (upscale, upscale)

        assert depth > 0 and isinstance(depth, int), "Depth must be a positive integer!"
        if depth > 2:
            warnings.warn(
                "Depth too high, it will probably take really long to finish!",
                PyxWarning,
            )
        self.depth = depth
        self.filter_obj = filter_obj
        self.palette = palette
        self.find_palette = isinstance(self.palette, (int, float))

        if self.find_palette and palette < 2:
            raise ValueError("The minimum number of colors in a palette is 2")
        elif not self.find_palette and len(palette) < 2:
            raise ValueError("The minimum number of colors in a palette is 2")

        assert dither in (
            None,
            "none",
            "naive",
            "bayer",
            "floyd",
            "atkinson",
        ), "Unknown dithering algorithm!"
        self.dither = dither
        self.alpha = float(alpha)

        # Instantiate BGM model (CPU based)
        self.model = BGM(self.palette, self.find_palette)
        self.is_fitted = False
        self.palette_cache = None

        # GPU Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dither_bayer_matrix_torch = torch.tensor(
            [
                [-0.5, 0.0, -0.375, 0.125],
                [0.25, -0.25, 0.375, -0.125],
                [-0.3125, 0.1875, -0.4375, 0.0625],
                [0.4375, -0.0625, 0.3125, -0.1875],
            ],
            device=self.device,
            dtype=torch.float32,
        )

    def _get_size(self, original_height: int, original_width: int) -> Tuple[int, int]:
        if self.height is not None and self.width is not None:
            return self.height, self.width
        elif self.height is not None:
            return self.height, int(self.height / original_height * original_width)
        elif self.width is not None:
            return int(self.width / original_width * original_height), self.width
        elif self.factor is not None:
            return original_height // self.factor, original_width // self.factor
        else:
            return original_height, original_width

    # Removed _image_to_float/int as we handle tensors mainly

    @property
    def colors(self) -> np.ndarray:
        """Get colors in palette (0 - 255 range). Cached on CPU."""
        if self.palette_cache is None:
            if self.find_palette:
                assert self.is_fitted, "Call 'fit(image)' first!"
                # Use torch for speedier hsv conversion if means are on GPU, but they are in sklearn model
                means = self.model.means_.reshape(-1, 1, 3)  # [N, 1, 3]

                # Using skimage here to perfectly match original logic for palette generation
                from skimage.color import rgb2hsv, hsv2rgb

                c = rgb2hsv(means)
                c[:, :, 1:] *= self.SCALE_RGB
                c = hsv2rgb(c)

                c = np.clip(
                    c * 255 // self.COLOR_QUANT * self.COLOR_QUANT, 0, 255
                ).astype(int)
                c[c < self.COLOR_QUANT * 2] = 0
                c[c > 255 - self.COLOR_QUANT * 2] = 255
                self.palette_cache = c

                # Redundancy check
                if len(np.unique([f"{pc[0]}" for pc in self.palette_cache])) != len(c):
                    warnings.warn("Some colors are redundant!", PyxWarning)
            else:
                # Handle passed palette
                if hasattr(self.palette, "value"):
                    p = np.array([x[0] for x in self.palette.value])
                elif isinstance(self.palette, (list, tuple)):
                    # Assuming list of ints or floats
                    p = np.array(self.palette)
                    if p.max() <= 1.0:
                        p = p * 255
                else:
                    p = np.array(self.palette)
                self.palette_cache = np.clip(p, 0, 255).astype(int)

        return self.palette_cache

    def fit(self, X: Union[np.ndarray, torch.Tensor], y=None) -> "Pyx":
        """
        Fits BGM model.
        X: Can be numpy [H,W,C] or Tensor [B,H,W,C].
        """
        # Data preparation
        if torch.is_tensor(X):
            # If batch, we fit on the whole batch flattened, or just take first image?
            # Original fits on one image. Let's fit on the first image to be safe and fast.
            # Convert to numpy for BGM (sklearn)
            if X.dim() == 4:
                X_np = X[0].cpu().numpy()
            else:
                X_np = X.cpu().numpy()
        else:
            X_np = X

        h, w, d = X_np.shape

        # Preprocessing for Fit (CPU - Sklearn/Skimage)
        if d > 3:
            # Alpha handling logic from original
            # Note: _dilate is implemented on GPU later, but here we need CPU for fit prep
            # To strictly follow fit logic, we resize first.
            from skimage.transform import resize

            # Simple dilation for alpha fit
            mask = X_np[:, :, 3]
            # Since we are just fitting colors, we can skip the complex dilation here or do a simple one
            # Original: _dilate(X) then filter by alpha.
            # Let's just resize and filter to save time in fit.
            X_ = resize(
                X_np[:, :, :3],
                (min(h, self.BGM_RESIZE), min(w, self.BGM_RESIZE)),
                anti_aliasing=False,
            )
            # We skip accurate alpha handling in FIT for speed, or strictness?
            # Strictness: we need valid colors.
            X_ = X_.reshape(-1, 3)
        else:
            X_ = skimage_resize(
                X_np[:, :, :3],
                (min(h, self.BGM_RESIZE), min(w, self.BGM_RESIZE)),
                anti_aliasing=False,
            )
            X_ = X_.reshape(-1, 3)

        # Scale RGB logic
        if self.find_palette:
            X_ = ((X_ - 0.5) * self.SCALE_RGB) + 0.5

        self.model.fit(X_)
        self.is_fitted = True
        return self

    def _pad_gpu(self, x: torch.Tensor, pad_size: int) -> torch.Tensor:
        """Mirror padding for GPU tensors [B, C, H, W]"""
        h, w = x.shape[2], x.shape[3]
        h_pad = (pad_size - (h % pad_size)) % pad_size
        w_pad = (pad_size - (w % pad_size)) % pad_size

        # Pad right and bottom
        if h_pad > 0 or w_pad > 0:
            # F.pad order: (left, right, top, bottom)
            return F.pad(
                x, (0, w_pad, 0, h_pad), mode="replicate"
            )  # 'edge' in numpy ~= replicate
        return x

    def _pyxelate_gpu(self, x: torch.Tensor) -> torch.Tensor:
        """
        GPU accelerated version of _pyxelate (Adaptive Downsampling).
        x: [B, C, H, W]
        """
        # 1. Sobel Gradients
        # Define kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device, dtype=x.dtype
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device, dtype=x.dtype
        ).view(1, 1, 3, 3)

        # Apply per channel
        # Reshape to [B*C, 1, H, W] for efficient convolution
        b, c, h, w = x.shape
        x_reshaped = x.view(-1, 1, h, w)

        gx = F.conv2d(x_reshaped, sobel_x, padding=1)
        gy = F.conv2d(x_reshaped, sobel_y, padding=1)

        sobel_mag = torch.sqrt(gx**2 + gy**2) + 1e-8  # [B*C, 1, H, W]

        # 2. Block Processing (replacing view_as_blocks)
        # We need to sum (pixel * sobel) and sum (sobel) in each block

        # Pad first
        pad_h = (self.sobel - (h % self.sobel)) % self.sobel
        pad_w = (self.sobel - (w % self.sobel)) % self.sobel

        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            sobel_padded = F.pad(
                sobel_mag.view(b, c, h, w), (0, pad_w, 0, pad_h), mode="replicate"
            )
        else:
            x_padded = x
            sobel_padded = sobel_mag.view(b, c, h, w)

        # Weighted sum via Average Pooling (Scaled by kernel area) or just Unfold
        # Logic: sum_prod / sobel_norm

        kernel_size = (self.sobel, self.sobel)
        stride = (self.sobel, self.sobel)

        # Multiply image by weights
        weighted_img = x_padded * sobel_padded

        # Sum pooling
        sum_prod = F.avg_pool2d(
            weighted_img, kernel_size, stride=stride, divisor_override=1
        )
        sobel_norm = F.avg_pool2d(
            sobel_padded, kernel_size, stride=stride, divisor_override=1
        )

        result = sum_prod / sobel_norm
        return result

    def _median_gpu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Custom median filter on HSV channels.
        Strictly follows: RGB -> Pad -> HSV -> Median(3x3) -> RGB -> Unpad
        """
        b, c, h, w = x.shape

        # Pad (Edge/Replicate)
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")

        # RGB to HSV
        # x is [B, 3, H, W] -> permute to [B, H, W, 3] for internal func, or rewrite func
        # My util uses [B, 3] usually, let's adjust util or shape.
        # Let's adapt data to [N, 3] for the color func

        x_flat = x_pad.permute(0, 2, 3, 1).reshape(-1, 3)
        hsv_flat = rgb2hsv_torch(x_flat)
        hsv = hsv_flat.view(b, x_pad.shape[2], x_pad.shape[3], 3).permute(0, 3, 1, 2)

        # Median Filtering via Unfold
        # Extract 3x3 patches
        # Input: [B, 3, H_pad, W_pad]
        # Output: [B, 3*3*3, L]

        # We only need valid area, so we don't pad again (we already padded x)
        # Unfold creates [B, C*K*K, L]
        patches = F.unfold(hsv, kernel_size=3)
        # Reshape to [B, C, 9, L]
        patches = patches.view(b, 3, 9, -1)
        # Median over dim 2
        median_res, _ = torch.median(patches, dim=2)

        # Reshape back to image
        median_res = median_res.view(b, 3, h, w)  # Size should match original h,w

        # HSV to RGB
        res_flat = median_res.permute(0, 2, 3, 1).reshape(-1, 3)
        rgb_flat = hsv2rgb_torch(res_flat)
        rgb_res = rgb_flat.view(b, h, w, 3).permute(0, 3, 1, 2)

        return rgb_res

    def _dilate_gpu(self, x: torch.Tensor) -> torch.Tensor:
        """Dilate semi-transparent edges. Using MaxPool as Dilation."""
        # x: [B, 4, H, W]
        if x.shape[1] < 4:
            return x

        b, c, h, w = x.shape
        rgb = x[:, :3]
        alpha = x[:, 3:4]

        # 1. Pad for operation
        # Original: square(3) -> 3x3 dilation
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")

        # Dilation on each channel = MaxPool2d with stride 1
        dilated = F.max_pool2d(x_pad, kernel_size=3, stride=1)

        # Logic: X_[:, :, :3][mask < self.alpha] = alter[mask < self.alpha]
        mask = alpha
        alter_rgb = dilated[:, :3]

        # Replace RGB where alpha is low
        is_transparent = mask < self.alpha

        # Expand mask for RGB
        mask_rgb = is_transparent.repeat(1, 3, 1, 1)

        rgb_new = torch.where(mask_rgb, alter_rgb, rgb)

        return torch.cat([rgb_new, alpha], dim=1)

    def transform(self, X: Union[np.ndarray, torch.Tensor], y=None) -> torch.Tensor:
        """
        Transform image.
        Accepts [B, H, W, C] (Comfy) or [H, W, C] (Numpy).
        Returns [B, H, W, C] (Comfy Tensor).
        """
        assert self.is_fitted, "Call 'fit' first!"

        # --- 1. Input Handling (Standardize to GPU [B, C, H, W]) ---
        if not torch.is_tensor(X):
            X_t = torch.from_numpy(X).float().to(self.device)
            if X_t.dim() == 3:
                X_t = X_t.unsqueeze(0)  # [1, H, W, C]
            X_t = X_t.permute(0, 3, 1, 2)  # [B, C, H, W]
        else:
            X_t = X.float().to(self.device)
            if X_t.dim() == 3:  # [H, W, C] -> [1, C, H, W]
                X_t = X_t.permute(2, 0, 1).unsqueeze(0)
            elif X_t.dim() == 4 and X_t.shape[3] in [3, 4]:  # [B, H, W, C]
                X_t = X_t.permute(0, 3, 1, 2)

        # Normalize to 0-1 if needed
        if X_t.max() > 1.0:
            X_t = X_t / 255.0

        b, c, h, w = X_t.shape

        # --- 2. Calculate Size ---
        new_h, new_w = self._get_size(h, w)

        # --- 3. Pre-processing (Dilation for Alpha) ---
        if c > 3:
            X_t = self._dilate_gpu(X_t)
            # Resize alpha mask separately
            alpha_mask = F.interpolate(
                X_t[:, 3:4], size=(new_h, new_w), mode="bilinear", align_corners=False
            )
        else:
            alpha_mask = None

        # --- 4. Iteration Loop Size Adjustment ---
        final_processing_h, final_processing_w = new_h, new_w
        if self.depth:
            final_processing_h = new_h * (self.sobel**self.depth)
            final_processing_w = new_w * (self.sobel**self.depth)

        # Resize RGB
        # Use bicubic/bilinear to match skimage anti_aliasing=True roughly
        X_curr = F.interpolate(
            X_t[:, :3],
            size=(final_processing_h, final_processing_w),
            mode="bicubic",
            align_corners=False,
        )
        X_curr = torch.clamp(X_curr, 0, 1)

        # --- 5. CLAHE (CPU Bottleneck - Strict Algorithm) ---
        # Need to move to CPU, apply CLAHE, move back
        X_np = X_curr.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, 3]
        X_clahe = []
        for i in range(b):
            # equalize_adapthist expects [H, W, C] double/float
            img_eq = equalize_adapthist(X_np[i])
            X_clahe.append(img_eq)
        X_curr = torch.tensor(
            np.stack(X_clahe), device=self.device, dtype=torch.float32
        ).permute(0, 3, 1, 2)

        # --- 6. Brightness Adjust (GPU) ---
        # RGB -> HSV
        X_flat = X_curr.permute(0, 2, 3, 1).reshape(-1, 3)
        X_hsv = (
            rgb2hsv_torch(X_flat)
            .view(b, final_processing_h, final_processing_w, 3)
            .permute(0, 3, 1, 2)
        )

        # Adjust V
        X_hsv[:, 2:3, :, :] *= self.HIST_BRIGHTNESS

        # HSV -> RGB
        X_flat_2 = torch.clamp(X_hsv, 0, 1).permute(0, 2, 3, 1).reshape(-1, 3)
        X_curr = (
            hsv2rgb_torch(X_flat_2)
            .view(b, final_processing_h, final_processing_w, 3)
            .permute(0, 3, 1, 2)
        )

        # --- 7. Apply Filter ---
        if self.filter_obj is not None:
            X_curr = self.filter_obj(X_curr)
            X_curr = torch.clamp(X_curr, 0.0, 1.0)

        # --- 8. Pyxelate Loop (GPU) ---
        for _ in range(self.depth):
            if c >= 3:  # Always true here
                X_curr = self._median_gpu(X_curr)
            X_curr = self._pyxelate_gpu(X_curr)

        final_h, final_w = X_curr.shape[2], X_curr.shape[3]

        # --- 9. Dithering ---

        # Prepare data for BGM prediction (move to CPU/Numpy)
        X_reshaped = X_curr.permute(0, 2, 3, 1).reshape(-1, 3)  # [N, 3]

        # Apply Palette Adjust (fit logic)
        if self.find_palette:
            X_reshaped = ((X_reshaped - 0.5) * self.SCALE_RGB) + 0.5

        # We need probabilities. BGM is CPU based.
        # To support batching efficiently, we process the whole flattened batch
        X_reshaped_np = X_reshaped.cpu().numpy()

        # Get Palette Colors (0-255) from cache
        colors_np = self.colors  # [K, 1, 3] usually
        colors_flat = self._palette_float()  # [K, 3] 0-1 float
        colors_gpu = torch.tensor(colors_flat, device=self.device, dtype=torch.float32)

        # Dither Logic
        result_flat = None

        if self.dither in [None, "none"]:
            # Just predict (Hard assignment)
            # Use GPU distance instead of BGM.predict for speed if strictly allowable?
            # Original uses `model.predict`. For strictness we should use it.
            # But `predict` on BGM tied/spherical is equiv to Mahalanobis/Euclidean.
            # Let's use BGM predict to be safe on logic.
            probs = self.model.predict(X_reshaped_np)  # [N] indices
            result_flat = (
                torch.tensor(self.colors[probs], device=self.device).float() / 255.0
            )

        elif self.dither == "naive":
            # 1. 获取概率 (CPU -> GPU)
            # predict_proba 返回的是 numpy，需要转为 Tensor
            probs_np = self.model.predict_proba(X_reshaped_np)  # [N, C]
            probs = torch.from_numpy(probs_np).to(self.device).float()

            # 2. 找出第一选择 (p1) 和 第二选择 (p2)
            # 获取最大概率和对应索引 (Best color)
            prob_p1, p1_idx = torch.max(probs, dim=1)

            # --- 关键修正 1：计算阈值前，必须基于“第二大”概率 ---
            # 将最大概率位置置 0，以便找到第二大的概率
            probs_temp = probs.clone()
            probs_temp.scatter_(1, p1_idx.unsqueeze(1), 0.0)

            # 获取第二大概率和对应索引 (Second best color)
            prob_p2, p2_idx = torch.max(probs_temp, dim=1)

            # 3. 计算阈值
            # 原版逻辑：v1 和 v2 是判断“第二选择”的概率是否足够大
            n_colors = len(colors_flat)
            threshold_v1 = 1.0 / (n_colors + 1)
            threshold_v2 = 1.0 / (n_colors * self.DITHER_NAIVE_BOOST + 1)

            v1 = prob_p2 > threshold_v1
            v2 = prob_p2 > threshold_v2

            # 4. 初始化结果为最佳颜色
            # 此时 X_final 全是 p1 颜色
            X_final = colors_gpu[p1_idx]

            # 5. 向量化棋盘格逻辑 (Checkerboard Pattern)
            N = len(X_final)

            # 生成基础索引：0, 2, 4, ... (模拟 range(0, N, 2))
            base_indices = torch.arange(0, N, 2, device=self.device)

            # 计算行号和行的奇偶性 (m)
            # 原版：m = (i // final_w) % 2。这里的 i 对应 base_indices
            rows = base_indices // final_w
            m = rows % 2  # 1 为奇数行，0 为偶数行

            # 判断是否需要补位偏移 (pad)
            # 原版：pad = not bool(final_w % 2)。如果宽度是偶数，则 pad 为 True
            pad = final_w % 2 == 0

            # --- 关键修正 2：确定实际操作的目标索引 ---
            # 原版：if pad: i += m。
            # 意思是：如果宽度是偶数且当前在奇数行，索引 +1
            shifts = (pad & (m == 1)).long()
            target_indices = base_indices + shifts

            # 边界安全检查 (防止索引越界)
            valid_mask = target_indices < N
            target_indices = target_indices[valid_mask]
            m = m[valid_mask]  # 对应的 m 也需要筛选

            # 6. 应用抖动决策
            # 我们需要检查目标位置的 v1/v2 条件
            # 这里的逻辑是：如果是奇数行(m=1)，检查 v1；如果是偶数行(m=0)，检查 v2
            v1_vals = v1[target_indices]
            v2_vals = v2[target_indices]

            # 决策掩码：满足条件则替换为 p2
            should_swap = (m == 1) & v1_vals
            should_swap |= (m == 0) & v2_vals

            # 筛选出需要替换的索引
            swap_indices = target_indices[should_swap]

            # 执行替换：将这些位置的颜色换成 p2
            X_final[swap_indices] = colors_gpu[p2_idx[swap_indices]]

            result_flat = X_final

        elif self.dither == "bayer":
            # GPU Accelerated Bayer
            probs = self.model.predict_proba(X_reshaped_np)  # [N, n_colors]
            probs_t = torch.from_numpy(probs).to(self.device).float()  # [N, C]

            # Convolve logic
            # Reshape probs to [B, n_colors, H, W]
            n_colors = probs_t.shape[1]
            probs_img = probs_t.T.reshape(b, n_colors, final_h, final_w)

            # Bayer Matrix (Pre-loaded in init)
            matrix = (
                self.DITHER_BAYER_MATRIX.to(self.device)
                if hasattr(self, "DITHER_BAYER_MATRIX")
                else self.dither_bayer_matrix_torch
            )
            matrix = matrix.view(1, 1, 4, 4)

            # Pad reflection for convolution
            # The matrix is 4x4. We need to tile it or convolve?
            # Original: convolve(probs, matrix, mode='reflect')
            # This is standard convolution.
            # Since matrix is 4x4, pad=1 or 2.
            # Scipy convolve with mode='reflect' centers the kernel.
            # 4x4 kernel is even, alignment is tricky.
            # Pyxelate uses fixed 4x4 matrix centered at 0.5 offsets.

            # Perform convolution per channel (color prob)
            # Reshape to [B*C, 1, H, W]
            probs_grouped = probs_img.view(-1, 1, final_h, final_w)

            # Pad input to maintain size
            # 4x4 kernel needs padding of 3?? No, conv2d reduces size.
            # To keep size same with 4x4, we need specific padding.
            # Let's pad 2 on all sides and crop?
            probs_padded = F.pad(
                probs_grouped, (1, 2, 1, 2), mode="reflect"
            )  # Asymmetric for 4x4

            convolved = F.conv2d(probs_padded, matrix)
            # Check size
            # [H+3, W+3] * 4x4 -> [H, W] roughly

            convolved = convolved.view(b, n_colors, final_h, final_w)

            # Argmin
            indices = torch.argmin(convolved, dim=1)  # [B, H, W]
            result_flat = colors_gpu[indices.view(-1)]

        elif self.dither in ["floyd", "atkinson"]:
            # Strict CPU fallback using Numba
            # Warn alpha
            if c > 3:
                self._warn_on_dither_with_alpha(c)

            # Must run sequentially per image
            res_list = []
            for i in range(b):
                img_flat = X_reshaped_np[
                    i * final_h * final_w : (i + 1) * final_h * final_w
                ]
                if self.dither == "floyd":
                    out = self._dither_floyd(img_flat, (final_h, final_w))
                else:
                    out = self._dither_atkinson(img_flat, (final_h, final_w))
                res_list.append(torch.from_numpy(out).to(self.device).float() / 255.0)

            result_flat = torch.cat(res_list, dim=0)

        # Reshape result back
        X_final = result_flat.view(b, final_h, final_w, 3).permute(0, 3, 1, 2)

        # --- 10. Restore Alpha & Combine ---
        if alpha_mask is not None:
            alpha_mask = (alpha_mask >= self.alpha).float()  # Threshold 0/1
            X_final = torch.cat([X_final, alpha_mask], dim=1)

        # --- 11. Upscale (GPU) ---
        X_final = torch.repeat_interleave(X_final, self.upscale[0], dim=2)
        X_final = torch.repeat_interleave(X_final, self.upscale[1], dim=3)

        # Return [B, H, W, C] for ComfyUI
        return X_final.permute(0, 2, 3, 1)

    def _palette_float(self):
        """Helper to get palette as 0-1 float [N, 3]"""
        c = self.colors.reshape(-1, 3)
        return c.astype(float) / 255.0

    def _warn_on_dither_with_alpha(self, d: int) -> None:
        if d > 3 and self.dither in ("bayer", "floyd", "atkinson"):
            warnings.warn(
                "Images with transparency can have unwanted artifacts around the edges with this dithering method. Use 'naive' instead.",
                PyxWarning,
            )

    # --- Numba optimized CPU Dithering Methods (Strict Copy) ---

    def _dither_floyd(
        self, reshaped: np.ndarray, final_shape: Tuple[int, int]
    ) -> np.ndarray:
        final_h, final_w = final_shape

        # 准备数据
        X_ = reshaped.reshape(final_h, final_w, 3)

        # Pad: Floyd 需要右边和下边的空间
        # 我们使用统一的 padding 方式，确保涵盖边缘
        X_pad = np.pad(X_, ((0, 2), (1, 2), (0, 0)), "reflect").astype(np.float64)

        # 获取调色板
        means = self.model.means_.astype(np.float64)

        # 执行标准 Floyd 算法
        indices = _floyd_standard_impl(X_pad, means, final_h, final_w)

        return self.colors[indices.reshape(final_h * final_w)]

    def _dither_atkinson(
        self, reshaped: np.ndarray, final_shape: Tuple[int, int]
    ) -> np.ndarray:
        final_h, final_w = final_shape

        # 准备数据
        X_ = reshaped.reshape(final_h, final_w, 3)
        # Pad
        X_pad = np.pad(X_, ((0, 2), (1, 2), (0, 0)), "reflect").astype(np.float64)

        # 只需中心点
        means = self.model.means_.astype(np.float64)

        # 调用新的 Numba 函数
        indices = _atkinson_euclidean_clamped_impl(X_pad, means, final_h, final_w)

        return self.colors[indices.reshape(final_h * final_w)]

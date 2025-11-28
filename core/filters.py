import torch
import torch.nn.functional as F


class FilterWrapper:
    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Filter must implement __call__")


class KuwaharaFilter(FilterWrapper):
    def __init__(self, radius):
        self.radius = radius

    def __call__(self, image_tensor):
        # 确保它能接受 [B, C, H, W] 格式
        return self._kuwahara_gpu(image_tensor, self.radius)

    def _kuwahara_gpu(self, x: torch.Tensor, radius: int = 2) -> torch.Tensor:
        """
        Kuwahara 滤波器 GPU 实现 (最适合像素画的平滑算法)
        radius: 窗口半径，例如 2 表示 5x5 的窗口 ((2*2)+1)
        """
        # x: [B, C, H, W]
        b, c, h, w = x.shape

        # 卷积核大小
        kernel_size = 2 * radius + 1

        # 1. 预计算均值 (Mean) 和 平方均值 (Mean of Squares)
        # 使用平均池化作为盒式滤波器 (Box Filter)
        # 我们需要分别对四个象限进行计算，但用一个大的卷积配合 padding 更高效
        # 这里为了逻辑清晰和通用性，我们采用 Unfold 或者 均值池化模拟

        # 为了避免循环，我们使用 pad 和 avg_pool 来计算四个区域
        # 区域定义: TL(左上), TR(右上), BL(左下), BR(右下)
        # 每个区域的大小是 (radius + 1)

        # 首先 pad 图像，以处理边缘
        x_pad = F.pad(x, (radius, radius, radius, radius), mode="reflect")

        # 定义一个简单的均值计算函数
        def get_mean_std(img_pad, k_size):
            # 均值
            mean = F.avg_pool2d(
                img_pad, kernel_size=(k_size, k_size), stride=1, padding=0
            )
            # 平方均值
            sq_mean = F.avg_pool2d(
                img_pad**2, kernel_size=(k_size, k_size), stride=1, padding=0
            )
            # 方差 = E[X^2] - (E[X])^2
            variance = sq_mean - mean**2
            return mean, variance

        # 由于 Kuwahara 需要以当前像素为中心，取四个角的窗口
        # 这可以通过对整图做一次均值滤波，然后 shift (偏移) 结果来实现

        # 这是一个优化的实现方式：计算一次大图的局部的均值和方差，然后通过切片取四个角
        # 这种方法比标准的 Kuwahara 稍有不同（重叠区域），但效果一致且速度快

        # 窗口大小 (radius + 1)
        w_size = radius + 1

        # 计算整个 padded 图像的滑动窗口均值和方差
        # 这里的 avg_pool 窗口是子区域的大小
        mean_all = F.avg_pool2d(x_pad, kernel_size=w_size, stride=1)
        sq_mean_all = F.avg_pool2d(x_pad**2, kernel_size=w_size, stride=1)
        var_all = sq_mean_all - mean_all**2

        # mean_all 的尺寸现在大概是 [B, C, H+radius, W+radius]
        # 我们需要从中切出四个对应的区域
        # 原始图像的 (0,0) 点在 x_pad 的 (radius, radius)
        # 对应的子窗口均值中心需要偏移

        # Top-Left: 覆盖 [y-r, x-r] 到 [y, x]
        # 在 mean_all 中，对应索引 0:h, 0:w
        m0 = mean_all[:, :, 0:h, 0:w]
        v0 = var_all[:, :, 0:h, 0:w]

        # Top-Right: 覆盖 [y-r, x] 到 [y, x+r]
        # 在 mean_all 中，对应索引 0:h, r:w+r
        m1 = mean_all[:, :, 0:h, radius : w + radius]
        v1 = var_all[:, :, 0:h, radius : w + radius]

        # Bottom-Left: 覆盖 [y, x-r] 到 [y+r, x]
        # 在 mean_all 中，对应索引 r:h+r, 0:w
        m2 = mean_all[:, :, radius : h + radius, 0:w]
        v2 = var_all[:, :, radius : h + radius, 0:w]

        # Bottom-Right: 覆盖 [y, x] 到 [y+r, x+r]
        # 在 mean_all 中，对应索引 r:h+r, r:w+r
        m3 = mean_all[:, :, radius : h + radius, radius : w + radius]
        v3 = var_all[:, :, radius : h + radius, radius : w + radius]

        # 2. 寻找最小方差的索引
        # 将方差求和（跨通道，RGB总方差），因为我们希望 RGB 作为一个整体选择同一个区域
        # [B, 1, H, W]
        v0_sum = torch.sum(v0, dim=1, keepdim=True)
        v1_sum = torch.sum(v1, dim=1, keepdim=True)
        v2_sum = torch.sum(v2, dim=1, keepdim=True)
        v3_sum = torch.sum(v3, dim=1, keepdim=True)

        # 堆叠方差 [B, 4, H, W]
        stack_var = torch.cat([v0_sum, v1_sum, v2_sum, v3_sum], dim=1)

        # 找到最小方差的索引 [B, 1, H, W]
        best_idx = torch.argmin(stack_var, dim=1, keepdim=True)

        # 3. 根据索引取对应的均值
        # 扩展 mask 到 RGB 通道 [B, 3, H, W]
        mask = best_idx.repeat(1, c, 1, 1)

        # 堆叠均值 [B, 4, C, H, W] -> [B, C, H, W, 4] 方便 gather
        stack_mean = torch.stack([m0, m1, m2, m3], dim=4)

        # 使用 gather 取值
        # gather 需要 index 维度匹配，我们在 dim=4 上取值
        # mask 变形为 [B, C, H, W, 1]
        mask_gather = mask.unsqueeze(-1)

        output = torch.gather(stack_mean, 4, mask_gather).squeeze(-1)

        return output

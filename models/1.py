import torch
import torch.nn.functional as F

class WaveletTransform1D:
    def __init__(self, wavelet='haar', device='cuda'):
        # 初始化小波滤波器
        if wavelet == 'haar':
            self.low_pass_filter = torch.tensor([0.707, 0.707], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            self.high_pass_filter = torch.tensor([-0.707, 0.707], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        elif wavelet == 'db2':
            self.low_pass_filter = torch.tensor([0.48296, 0.8365, 0.2241, -0.1294], dtype=torch.float32,
                                                device=device).unsqueeze(0).unsqueeze(0)
            self.high_pass_filter = torch.tensor([-0.1294, -0.2241, 0.8365, -0.48296], dtype=torch.float32,
                                                 device=device).unsqueeze(0).unsqueeze(0)
        elif wavelet == 'db3':
            self.low_pass_filter = torch.tensor([0.3327, 0.8069, 0.4599, -0.1350, -0.0854, 0.0352], dtype=torch.float32,
                                                device=device).unsqueeze(0).unsqueeze(0)
            self.high_pass_filter = torch.tensor([-0.0352, -0.0854, 0.1350, 0.4599, -0.8069, 0.3327],
                                                 dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        elif wavelet == 'db4':
            self.low_pass_filter = torch.tensor([0.2304, 0.7148, 0.6309, -0.0279, -0.1870, 0.0308, 0.0329, -0.0106],
                                                dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            self.high_pass_filter = torch.tensor([-0.0106, -0.0329, 0.0308, 0.1870, -0.0279, -0.6309, 0.7148, -0.2304],
                                                 dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        else:
            raise NotImplementedError("Only 'haar', 'db2', 'db3', and 'db4' wavelets are implemented for this example.")
            # 根据输入通道数调整滤波器大小
        self.low_pass_filter = self.low_pass_filter.expand(7, 1, self.low_pass_filter.size(-1))
        self.high_pass_filter = self.high_pass_filter.expand(7, 1, self.high_pass_filter.size(-1))

    def dwt_step(self, x):
        """
        单级小波分解
        :param x: 输入信号，形状 [B, C, T]
        :return: 低频和高频分量
        """
        low_pass = F.conv1d(x, self.low_pass_filter, stride=2, groups=x.size(1), padding=(self.low_pass_filter.size(-1) - 1) // 2)
        high_pass = F.conv1d(x, self.high_pass_filter, stride=2, groups=x.size(1), padding=(self.high_pass_filter.size(-1) - 1) // 2)
        return low_pass, high_pass

    def idwt_step(self, low_pass, high_pass):
        """
        单级小波重构
        :param low_pass: 低频分量
        :param high_pass: 高频分量
        :return: 重构的信号
        """
        low_reconstructed = F.conv_transpose1d(low_pass, self.low_pass_filter, stride=2, groups=low_pass.size(1), padding=(self.low_pass_filter.size(-1) - 1) // 2)
        high_reconstructed = F.conv_transpose1d(high_pass, self.high_pass_filter, stride=2, groups=high_pass.size(1), padding=(self.high_pass_filter.size(-1) - 1) // 2)
        return low_reconstructed + high_reconstructed

    def multi_level_decompose(self, x, level):
        """
        多级小波分解
        :param x: 输入信号，形状 [B, C, T]
        :param level: 分解层数
        :return: 各级分解的低频和高频分量
        """
        coefficients = []
        current_signal = x
        for i in range(level):
            low_pass, high_pass = self.dwt_step(current_signal)
            coefficients.append((low_pass, high_pass))  # 保存每级低频和高频分量
            current_signal = low_pass  # 继续分解低频分量
        return coefficients

    def multi_level_reconstruct(self, coefficients):
        """
        多级小波重构
        :param coefficients: 各级分解的低频和高频分量
        :return: 每一级别的重构信号列表和最终重构信号
        """
        reconstructed_levels = []  # 保存每一级重构信号
        low_freq_info = []  # 保存每一级低频信息
        high_freq_info = []  # 保存每一级高频信息
        current_signal = None

        # 从最高层开始逐级重构
        for low_pass, high_pass in reversed(coefficients):
            print(f"low_pass shape: {low_pass.shape}, high_pass shape: {high_pass.shape}")
            if current_signal is None:
                # 初始重构
                current_signal = self.idwt_step(low_pass, high_pass)
            else:
                # 在每一级的基础上加入高频分量进行重构
                current_signal = self.idwt_step(current_signal, high_pass)

            # 保存当前级别的重构信号
            reconstructed_levels.append(current_signal)
            low_freq_info.append(low_pass)  # 保存低频信息
            high_freq_info.append(high_pass)  # 保存高频信息
        print(current_signal.shape)
        # 返回重构信号列表、低频和高频信息
        return reconstructed_levels, current_signal, low_freq_info, high_freq_info

# 示例用法
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 输入信号形状 [B, C, T] (B: batch size, C: channels, T: time steps)
input_tensor = torch.randn(32, 7, 128).to(device)  # [32, 1, 128] 的输入信号

# 初始化小波变换类
wavelet_transform = WaveletTransform1D(wavelet='db4', device=device)

# 进行 3 级小波分解
coeffs = wavelet_transform.multi_level_decompose(input_tensor, level=3)

# 重构所有级别，并保存每一级的重构分量和高频、低频信息
reconstructed_levels, final_reconstructed, low_freq_info, high_freq_info = wavelet_transform.multi_level_reconstruct(coeffs)

# 输出每一级的重构结果形状
for level, reconstructed_signal in enumerate(reconstructed_levels):
    print(f"Level {level + 1} reconstructed signal shape: {reconstructed_signal.shape}")

# 输出每一级的高频和低频信息的形状
for level, (low_pass, high_pass) in enumerate(zip(low_freq_info, high_freq_info)):
    print(f"Level {level + 1} - Low Pass Shape: {low_pass.shape}, High Pass Shape: {high_pass.shape}")

# 输出最终重构的信号形状
print(f"Final reconstructed signal shape: {final_reconstructed.shape}")

# 检查原始信号与最终重构信号的差异
diff = torch.abs(input_tensor - final_reconstructed)
print(f"Difference between original and reconstructed signal: {diff.mean().item()}")

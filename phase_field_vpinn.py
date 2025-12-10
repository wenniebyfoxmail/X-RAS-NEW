"""
变分物理信息神经网络 (VPINN) / 深度Ritz方法 (DRM) 断裂力学求解器
基于相场模型 (Phase-Field Fracture)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt


# ============================================================================
# §1 神经网络定义
# ============================================================================

class DisplacementNetwork(nn.Module):
    """位移场网络 u_theta(x) -> (u, v)"""
    
    def __init__(self, layers=[2, 64, 64, 64, 2]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x):
        """
        Args:
            x: (N, 2) 坐标 [x, y]
        Returns:
            u: (N, 2) 位移 [u, v]
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.tanh(layer(x))
        u = self.layers[-1](x)
        return u


class DamageNetwork(nn.Module):
    """损伤场网络 d_phi(x) -> d，输出范围 [0, 1]"""
    
    def __init__(self, layers=[2, 64, 64, 64, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x):
        """
        Args:
            x: (N, 2) 坐标 [x, y]
        Returns:
            d: (N, 1) 损伤场，范围 [0, 1]
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.tanh(layer(x))
        d = self.layers[-1](x)
        # 相场损伤约束在 [0,1]
        d = torch.sigmoid(d)
        return d


# ============================================================================
# §2 自动微分模块
# ============================================================================

def compute_strain(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    计算应变张量 ε(u) (平面应变)

    ε = [ε_xx, ε_yy, 2*ε_xy]^T

    Args:
        u: (N, 2) 位移场 [u, v]
        x: (N, 2) 坐标，需要 requires_grad=True
    Returns:
        epsilon: (N, 3)
    """
    # 保证 x 具有梯度
    if not x.requires_grad:
        x.requires_grad_(True)

    u_x = u[:, 0:1]
    u_y = u[:, 1:2]

    # 自动微分
    grads_u_x = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0]
    grads_u_y = torch.autograd.grad(
        u_y, x, grad_outputs=torch.ones_like(u_y),
        create_graph=True, retain_graph=True
    )[0]

    u_x_x = grads_u_x[:, 0:1]
    u_x_y = grads_u_x[:, 1:2]
    u_y_x = grads_u_y[:, 0:1]
    u_y_y = grads_u_y[:, 1:2]

    # 平面应变 (工程应变记号)
    epsilon_xx = u_x_x
    epsilon_yy = u_y_y
    epsilon_xy = u_x_y + u_y_x  # 工程剪应变 γ_xy = ∂u/∂y + ∂v/∂x

    epsilon = torch.cat([epsilon_xx, epsilon_yy, epsilon_xy], dim=1)
    return epsilon


def compute_energy_split(epsilon: torch.Tensor, E: float, nu: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算能量分解 ψ+(ε) 和 ψ-(ε) (Amor et al. 2009)

    基于应变的正负分解：
    ε+ = <ε_i>_+ * n_i ⊗ n_i
    ψ+ = λ/2 <tr(ε)>²_+ + μ tr(ε+²)
    ψ- = λ/2 <tr(ε)>²_- + μ tr(ε-²)

    简化版本（仅考虑迹分解）：
    ψ+ = (λ/2 + μ) <tr(ε)>²_+
    ψ- = (λ/2 + μ) <tr(ε)>²_- + μ (ε_dev : ε_dev)

    Args:
        epsilon: (N, 3) 应变 [ε_xx, ε_yy, 2*ε_xy]
        E: 杨氏模量
        nu: 泊松比
    Returns:
        psi_plus: (N, 1) 拉伸能量密度
        psi_minus: (N, 1) 压缩能量密度
    """
    # Lamé 参数
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))  # 平面应变
    mu = E / (2 * (1 + nu))

    # 应变分量
    epsilon_xx = epsilon[:, 0:1]
    epsilon_yy = epsilon[:, 1:2]
    epsilon_xy = epsilon[:, 2:3] / 2.0  # 工程剪应变转换为张量剪应变

    # 迹
    trace_epsilon = epsilon_xx + epsilon_yy

    # Macaulay 括号 <x>_+ = max(x, 0)
    trace_plus = torch.relu(trace_epsilon)
    trace_minus = trace_epsilon - trace_plus

    # 偏应变
    epsilon_dev_xx = epsilon_xx - trace_epsilon / 2
    epsilon_dev_yy = epsilon_yy - trace_epsilon / 2
    epsilon_dev_xy = epsilon_xy

    # 偏应变范数
    dev_norm_sq = epsilon_dev_xx**2 + epsilon_dev_yy**2 + 2 * epsilon_dev_xy**2

    # 能量分解
    psi_plus = 0.5 * lam * trace_plus**2 + mu * dev_norm_sq
    psi_minus = 0.5 * lam * trace_minus**2

    return psi_plus, psi_minus


def compute_d_gradient(d: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    计算损伤梯度 ∇d

    Args:
        d: (N, 1) 损伤场
        x: (N, 2) 坐标
    Returns:
        grad_d: (N, 2) 损伤梯度 [∂d/∂x, ∂d/∂y]
    """
    if not x.requires_grad:
        x.requires_grad_(True)

    grad_d = torch.autograd.grad(
        d, x, grad_outputs=torch.ones_like(d),
        create_graph=True, retain_graph=True
    )[0]
    return grad_d


def compute_degradation_function(d: torch.Tensor, k: float = 1e-6) -> torch.Tensor:
    """
    退化函数 g(d) = (1 - d)^2 + k

    Args:
        d: (N, 1)
        k: 残余刚度系数
    """
    return (1 - d) ** 2 + k


def compute_crack_density(d: torch.Tensor) -> torch.Tensor:
    """
    裂纹密度函数 w(d) = d^2 (AT2 模型)
    """
    return d ** 2


# ============================================================================
# §3 深度 Ritz 损失函数 (DRM / VPINN 形式)
# ============================================================================

class DRMLoss:
    """深度Ritz方法损失函数（修正版：经典 AT2，无 history）"""

    def __init__(self, E: float, nu: float, G_c: float, l: float,
                 c_0: float = 2 / 3, k: float = 1e-6):
        """
        Args:
            E: 杨氏模量
            nu: 泊松比
            G_c: 断裂能
            l: 长度尺度参数
            c_0: 裂纹密度归一化常数 (默认 2/3 对应 w(d)=d²)
            k: 残余刚度参数
        """
        self.E = E
        self.nu = nu
        self.G_c = G_c
        self.l = l
        self.c_0 = c_0
        self.k = k

    def compute_energy_loss(
            self,
            x_domain: torch.Tensor,
            u_net: nn.Module,
            d_net: nn.Module,
            d_prev: Optional[torch.Tensor] = None,  # ← 新增
    ) -> torch.Tensor:
        """
        计算能量泛函损失（经典 AT2，无 history）

        采用张拉-压缩能量分裂：
        ψ(ε,d) = g(d)·ψ⁺(ε) + ψ⁻(ε)

        总能量：
        E[u,d] = ∫_Ω [ψ(ε(u), d) + (G_c/c_0)(w(d)/l + l|∇d|²)] dΩ

        修正说明：
        - 移除了 H 参数（history 场）
        - 仅退化拉伸能量 ψ⁺，压缩能量 ψ⁻ 不退化
        - 避免在受压区域出现非物理裂纹，是更常用的 AT2 实现
        """
        x_domain.requires_grad_(True)

        # 前向传播
        # u = u_net(x_domain)
        # d = d_net(x_domain)
        """
        计算 DRM 形式的总能量（AT2，无 history，但这里支持外部传入的 d_prev 做硬不可逆）
        """
        # 1) 位移场
        u = u_net(x_domain)  # (N, 2)

        # 2) 当前网络给出的损伤场
        d_raw = d_net(x_domain)  # (N, 1)

        # 3) 方案 B：如果提供了历史场，就做一次硬不可逆 clamp
        if d_prev is not None:
            # 确保在同一 device、同一形状
            d_prev = d_prev.to(d_raw.device)
            if d_prev.shape != d_raw.shape:
                raise RuntimeError(
                    f"d_prev shape {d_prev.shape} != d_raw shape {d_raw.shape}"
                )
            d = torch.max(d_raw, d_prev)
        else:
            d = d_raw

        # 应变和能量
        epsilon = compute_strain(u, x_domain)
        psi_plus, psi_minus = compute_energy_split(epsilon, self.E, self.nu)

        # ✅ 关键修正：仅退化拉伸能量 ψ+，保留压缩能量 ψ-
        # 经典做法：ψ(ε,d) = g(d) ψ⁺(ε) + ψ⁻(ε)
        # 这样可以避免在受压区域错误地产生或扩展裂纹
        # 损伤梯度
        grad_d = compute_d_gradient(d, x_domain)
        grad_d_norm_sq = (grad_d ** 2).sum(dim=1, keepdim=True)

        # 退化函数和裂纹密度
        g_d = compute_degradation_function(d, self.k)
        w_d = compute_crack_density(d)

        # ✅ 弹性能：g(d) * ψ_plus + ψ_minus（张拉-压缩分裂，仅退化张拉部分）
        elastic_energy = g_d * psi_plus + psi_minus

        # === AT1 nucleation threshold ===
        psi_plus_clamped = torch.clamp(psi_plus, min=0.0)
        threshold = self.G_c / self.l

        # Prevent damage when below threshold
        mask = (psi_plus_clamped < threshold).float()

        # Equivalent to d=0 region (pinning)
        d_effective = d * (1 - mask)
        # Tanne 2018 - nucleation threshold

        # 裂纹能：(G_c/c_0) * (w(d)/l + l*|∇d|²)
        # crack_energy = (self.G_c / self.c_0) * (w_d / self.l + self.l * grad_d_norm_sq)
        crack_energy = self.G_c * (d_effective / self.l + self.l * grad_d_norm_sq)

        # 总能量密度
        energy_density = elastic_energy + crack_energy

        loss_energy = (elastic_energy + 0.1 * crack_energy).mean()

        return loss_energy

    def compute_bc_loss(self, x_bc: torch.Tensor, u_bc_true: torch.Tensor,
                        u_net: nn.Module, weight: float = 1.0) -> torch.Tensor:
        """边界条件损失（保持不变）"""
        u_bc_pred = u_net(x_bc)
        loss = weight * torch.mean((u_bc_pred - u_bc_true) ** 2)
        return loss

    def compute_irreversibility_loss(self, x_domain: torch.Tensor, d_net: nn.Module,
                                     d_prev: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
        """不可逆性罚函数（保持不变）"""
        d_current = d_net(x_domain)
        violation = torch.relu(d_prev - d_current)
        loss = weight * torch.mean(violation ** 2)
        return loss



# ============================================================================
# §4 训练器 (Algorithm 1)
# ============================================================================

class PhaseFieldSolver:
    """相场断裂 VPINN/DRM 求解器（修正版：无 history）"""

    def __init__(self, problem_config: Dict, u_net: nn.Module, d_net: nn.Module):
        """
        Args:
            problem_config: 问题配置字典
            u_net: 位移网络
            d_net: 损伤网络
        """
        self.config = problem_config
        self.u_net = u_net
        self.d_net = d_net

        # 提取材料参数
        self.E = problem_config['E']
        self.nu = problem_config['nu']
        self.G_c = problem_config['G_c']
        self.l = problem_config['l']

        # 损失函数
        self.drm_loss = DRMLoss(self.E, self.nu, self.G_c, self.l)

        # 设备
        self.device = problem_config.get('device', 'cpu')
        self.u_net.to(self.device)
        self.d_net.to(self.device)

        # 优化器
        self.optimizer_u = torch.optim.Adam(self.u_net.parameters(),
                                            lr=problem_config.get('lr_u', 1e-3))
        self.optimizer_d = torch.optim.Adam(self.d_net.parameters(),
                                            lr=problem_config.get('lr_d', 1e-3))

        # ✅ 移除 self.H，只保留 d_prev 用于不可逆性
        self.d_prev = None

    def initialize_fields(self, x_domain: torch.Tensor):
        """初始化损伤场（用于不可逆性）"""
        self.u_net.train()
        self.d_net.train()

        with torch.no_grad():
            d_init = self.d_net(x_domain.to(self.device))
            self.d_prev = d_init.detach().clone()

        # ✅ 不再初始化 H
        print("  Initialized d_prev for irreversibility constraint")

    def train_step(self, x_domain: torch.Tensor, x_bc: torch.Tensor,
                   u_bc: torch.Tensor, n_epochs: int = 1000,
                   weight_bc: float = 100.0, weight_irrev: float = 0.1,  # ✅ 默认关闭不可逆性
                   verbose: bool = True):
        """
        单步训练（经典 AT2，无 history）

        Args:
            x_domain: (N, 2) 域内采样点
            x_bc: (M, 2) 边界点
            u_bc: (M, 2) 边界条件
            n_epochs: epoch 数
            weight_bc: 边界条件权重
            weight_irrev: 不可逆性权重（建议初期设为 0）
            verbose: 是否打印信息
        """
        x_domain = x_domain.to(self.device)
        x_bc = x_bc.to(self.device)
        u_bc = u_bc.to(self.device)

        self.u_net.train()
        self.d_net.train()

        for epoch in range(n_epochs):
            # 清零梯度
            self.optimizer_u.zero_grad()
            self.optimizer_d.zero_grad()

            # ✅ 关键：不再传 H
            L_energy = self.drm_loss.compute_energy_loss(
                x_domain, self.u_net, self.d_net
            )

            L_bc = self.drm_loss.compute_bc_loss(x_bc, u_bc, self.u_net, weight_bc)

            # 可选的不可逆性损失
            if weight_irrev > 0 and self.d_prev is not None:
                L_irrev = self.drm_loss.compute_irreversibility_loss(
                    x_domain, self.d_net, self.d_prev, weight_irrev
                )
            else:
                L_irrev = torch.tensor(0.0, device=self.device)

            # 总损失
            loss = L_energy + L_bc + L_irrev

            # 反向传播
            loss.backward()
            self.optimizer_u.step()
            self.optimizer_d.step()

            # 打印 + 损伤统计（便于诊断是否出现全域高平台）
            if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
                with torch.no_grad():
                    d_vals = self.d_net(x_domain)
                    d_min = d_vals.min().item()
                    d_max = d_vals.max().item()
                    d_mean = d_vals.mean().item()
                print(f"  Epoch {epoch:4d} | Loss: {loss.item():.6e} | "
                      f"Energy: {L_energy.item():.6e} | BC: {L_bc.item():.6e} | "
                      f"Irrev: {L_irrev.item():.6e} | "
                      f"d_min={d_min:.3f}, d_max={d_max:.3f}, d_mean={d_mean:.3f}")

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测位移和损伤场"""
        self.u_net.eval()
        self.d_net.eval()
        x = x.to(self.device)
        with torch.no_grad():
            u = self.u_net(x)
            d = self.d_net(x)
        return u, d


# ============================================================================
# §5 工具函数：网格生成与可视化
# ============================================================================

def generate_domain_points(nx: int, ny: int,
                           x_range=(0.0, 1.0),
                           y_range=(0.0, 1.0)) -> torch.Tensor:
    """
    生成规则网格上的采样点
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y)
    pts = np.stack([X.flatten(), Y.flatten()], axis=1)
    return torch.tensor(pts, dtype=torch.float32)


def visualize_solution(solver: PhaseFieldSolver,
                       x_grid: torch.Tensor,
                       nx: int, ny: int,
                       save_path: str = None):
    """
    可视化位移场和损伤场
    """
    solver.u_net.eval()
    solver.d_net.eval()

    x_grid = x_grid.to(solver.device)
    with torch.no_grad():
        u_pred = solver.u_net(x_grid)
        d_pred = solver.d_net(x_grid)

    u_pred = u_pred.cpu().numpy()
    d_pred = d_pred.cpu().numpy().reshape(ny, nx)

    X = x_grid[:, 0].cpu().numpy().reshape(ny, nx)
    Y = x_grid[:, 1].cpu().numpy().reshape(ny, nx)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # U_x
    im0 = axes[0].contourf(X, Y, u_pred[:, 0].reshape(ny, nx), levels=50)
    axes[0].set_title("u_x")
    plt.colorbar(im0, ax=axes[0])

    # U_y
    im1 = axes[1].contourf(X, Y, u_pred[:, 1].reshape(ny, nx), levels=50)
    axes[1].set_title("u_y")
    plt.colorbar(im1, ax=axes[1])

    # Damage
    im2 = axes[2].contourf(X, Y, d_pred, levels=50, vmin=0.0, vmax=1.0)
    axes[2].set_title("Damage d")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")

    plt.close()


if __name__ == "__main__":
    print("Phase-Field VPINN/DRM Solver Loaded Successfully!")
    print("This is a library module. Please run 'run_experiments.py' instead.")
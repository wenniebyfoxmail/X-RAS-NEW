"""
X-RAS-PINN: 域分解 + 自适应采样的高性能相场断裂求解器
Extended Physics-Informed Neural Network with Residual-based Adaptive Sampling

关键创新:
1. 域分解 (XPINN): 将域分解为 Ω_sing (裂纹区) 和 Ω_far (远场)
2. 接口损失: 确保子域间的位移和牵引力连续性
3. 自适应采样 (RAS): 基于应变能密度和损伤梯度的融合指标
4. 三阶段训练: 预训练 → 聚焦 → 联合微调

理论基础: §3.2 (X-RAS-PINN Method)
- Eq. (xpinn_total_loss_corrected): 总损失 = 两个子域能量和 + BC + 接口
- Eq. (interface_loss): 接口损失 = 位移连续 + 牵引力连续
- Eq. (sed_indicator): 融合指标 = (1-β)·SED + β·|∇d|
- Eq. (adaptive_pdf): 自适应概率密度函数
- Algorithm 2: 三阶段训练流程
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os

# 从原始模块导入基础组件
from phase_field_vpinn import (
    DisplacementNetwork, DamageNetwork,
    compute_strain, compute_energy_split, compute_d_gradient,
    compute_degradation_function, compute_crack_density,
    generate_domain_points
)


# ============================================================================
# §1 域分解模块
# ============================================================================

def partition_domain(x: torch.Tensor,
                     crack_center: torch.Tensor,
                     r_sing: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    域分解函数: 将点标记为 Ω_sing 或 Ω_far

    策略: 以裂纹中心为原点,半径 r_sing 内为奇异区

    Args:
        x: (N, 2) 坐标点
        crack_center: (2,) 裂纹中心 [x_c, y_c]
        r_sing: 奇异区半径

    Returns:
        mask_sing: (N,) bool, True 表示在 Ω_sing 中
        mask_far: (N,) bool, True 表示在 Ω_far 中
    """
    distances = torch.norm(x - crack_center.unsqueeze(0), dim=1)
    mask_sing = distances <= r_sing
    mask_far = ~mask_sing

    return mask_sing, mask_far


@dataclass
class SubdomainModels:
    """子域模型容器"""
    u_net_1: nn.Module  # Ω_sing 位移网络
    d_net_1: nn.Module  # Ω_sing 损伤网络
    u_net_2: nn.Module  # Ω_far 位移网络
    d_net_2: nn.Module  # Ω_far 损伤网络


# ============================================================================
# §2 接口损失函数 (Eq. interface_loss)
# ============================================================================

def compute_interface_loss(
    x_interface: torch.Tensor,
    u_net_1: nn.Module,
    u_net_2: nn.Module,
    E: float,
    nu: float,
    weight_u: float = 1.0,
    weight_trac: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 保证有梯度，并且不污染外部变量
    x = x_interface.clone().detach().requires_grad_(True)

    # 1. 在这里计算位移
    u1 = u_net_1(x)
    u2 = u_net_2(x)

    # 2. 位移连续性
    L_u = weight_u * torch.mean((u1 - u2) ** 2)

    # 3. “牵引力连续性”的近似（用能量差）
    epsilon_1 = compute_strain(u1, x)
    epsilon_2 = compute_strain(u2, x)
    psi_plus_1, psi_minus_1 = compute_energy_split(epsilon_1, E, nu)
    psi_plus_2, psi_minus_2 = compute_energy_split(epsilon_2, E, nu)

    stress_diff = torch.abs((psi_plus_1 + psi_minus_1) - (psi_plus_2 + psi_minus_2))
    L_trac = weight_trac * torch.mean(stress_diff)

    return L_u, L_trac



# ============================================================================
# §3 自适应采样模块 (RAS) (Eq. sed_indicator, Eq. adaptive_pdf)
# ============================================================================
def compute_indicator(
    x: torch.Tensor,
    u_net: nn.Module,
    d_net: nn.Module,
    E: float,
    nu: float,
    beta: float = 0.5
) -> torch.Tensor:
    # 这里不需要保留梯度给外部，只是做一个指标 ⇒ no_grad 也可以
    x = x.clone().detach().requires_grad_(True)

    u = u_net(x)
    d = d_net(x)

    epsilon = compute_strain(u, x)
    psi_plus, psi_minus = compute_energy_split(epsilon, E, nu)
    sed = psi_plus + psi_minus

    grad_d = compute_d_gradient(d, x)
    grad_d_norm = torch.norm(grad_d, dim=1, keepdim=True)

    sed_normalized = sed / (sed.max() + 1e-8)
    grad_d_normalized = grad_d_norm / (grad_d_norm.max() + 1e-8)

    indicator = (1 - beta) * sed_normalized + beta * grad_d_normalized
    return indicator.squeeze()


def resample(
    x_current: torch.Tensor,
    indicator_values: torch.Tensor,
    N_add: int,
    domain_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    temperature: float = 2.0
) -> torch.Tensor:
    """
    基于指标的自适应重采样 (§3.2.4, Eq. adaptive_pdf)

    构建概率密度函数:
    p(x) ∝ η_fused(x)^T

    其中 T 是温度参数,控制采样的集中程度:
    - T → 0: 只在最高指标处采样 (过度集中)
    - T → ∞: 均匀采样 (退化为随机)
    - T ∈ [1, 3]: 合理平衡

    Args:
        x_current: (N, 2) 当前采样点
        indicator_values: (N,) 每个点的指标值
        N_add: 新增点数
        domain_bounds: ((x_min, x_max), (y_min, y_max))
        temperature: 温度参数

    Returns:
        x_new: (N + N_add, 2) 更新后的采样点
    """
    # ✅ 保证维度干净：x_current 是 (N, 2)，indicator_values 是 (N,)
    x_current = x_current.view(-1, 2)
    indicator_values = indicator_values.view(-1)

    # 1. 构建概率分布 (Eq. adaptive_pdf)
    # 使用温度缩放避免数值溢出
    max_val = indicator_values.max()
    indicator_scaled = indicator_values / (max_val + 1e-8)
    weights = torch.pow(indicator_scaled, temperature)
    weights = weights / weights.sum()  # 归一化

    # 2. 重要性采样
    # 在高指标区域附近生成新点
    n_current = x_current.shape[0]
    indices = torch.multinomial(weights, N_add, replacement=True)

    # 3. 在选中点附近添加扰动
    x_selected = x_current[indices]

    # 自适应扰动半径: 在高指标区域使用更小的扰动
    local_indicator = indicator_values[indices]
    noise_scale = 0.02 * (1.0 - local_indicator / (local_indicator.max() + 1e-8))
    noise_scale = noise_scale.unsqueeze(1)

    noise = torch.randn_like(x_selected) * noise_scale
    x_sampled = x_selected + noise

    # 4. 裁剪到域内
    (x_min, x_max), (y_min, y_max) = domain_bounds
    x_sampled[:, 0] = torch.clamp(x_sampled[:, 0], x_min, x_max)
    x_sampled[:, 1] = torch.clamp(x_sampled[:, 1], y_min, y_max)

    # 5. 合并
    x_new = torch.cat([x_current, x_sampled], dim=0)

    return x_new


# ============================================================================
# §4 X-RAS-PINN 损失函数 (Eq. xpinn_total_loss_corrected)
# ============================================================================

class XRASPINNLoss:
    """
    X-RAS-PINN 损失函数

    关键修正 (相对于强形式PINN):
    总损失不是强形式残差 MSE_{f,q},而是两个子域能量的总和:

    L_total = L_energy(u^(1), d^(1))_Ω_sing
            + L_energy(u^(2), d^(2))_Ω_far
            + L_BC
            + L_interface

    这是变分形式 (DRM/VPINN) 而非残差形式
    """

    def __init__(self, E: float, nu: float, G_c: float, l: float,
                 c_0: float = 8/3, k: float = 1e-6):
        """
        Args:
            E: 杨氏模量
            nu: 泊松比
            G_c: 断裂能
            l: 长度尺度参数
            c_0: 裂纹密度归一化常数
            k: 残余刚度参数
        """
        self.E = E
        self.nu = nu
        self.G_c = G_c
        self.l = l
        self.c_0 = c_0
        self.k = k

    def compute_subdomain_energy(
        self,
        x_subdomain: torch.Tensor,
        u_net: nn.Module,
        d_net: nn.Module
    ) -> torch.Tensor:
        """
        计算子域能量 (与原始DRM相同,但应用于子域)

        E[u,d] = ∫_Ω [g(d)·ψ⁺(ε) + ψ⁻(ε) + (G_c/c_0)(w(d)/l + l|∇d|²)] dΩ
        """
        x = x_subdomain.clone().detach().requires_grad_(True)

        # 前向传播
        u = u_net(x)
        d = d_net(x)

        # 应变和能量
        epsilon = compute_strain(u, x)
        psi_plus, psi_minus = compute_energy_split(epsilon, self.E, self.nu)

        # 损伤梯度
        grad_d = compute_d_gradient(d, x)
        grad_d_norm_sq = (grad_d ** 2).sum(dim=1, keepdim=True)

        # 退化函数和裂纹密度
        g_d = compute_degradation_function(d, self.k)
        w_d = compute_crack_density(d)

        # 弹性能: g(d) * ψ_plus + ψ_minus
        elastic_energy = g_d * psi_plus + psi_minus

        # 裂纹能
        crack_energy = (self.G_c / self.c_0) * (w_d / self.l + self.l * grad_d_norm_sq)

        # 总能量密度
        energy_density = elastic_energy + crack_energy

        # 蒙特卡洛积分
        loss = energy_density.mean()

        return loss

    def compute_bc_loss(
        self,
        x_bc: torch.Tensor,
        u_bc_true: torch.Tensor,
        u_net_1: nn.Module,
        u_net_2: nn.Module,
        mask_bc_1: torch.Tensor,
        mask_bc_2: torch.Tensor,
        weight: float = 1.0
    ) -> torch.Tensor:
        """
        边界条件损失 (需要处理两个子域)

        Args:
            mask_bc_1: 边界点属于子域1的掩码
            mask_bc_2: 边界点属于子域2的掩码
        """
        loss = 0.0

        if mask_bc_1.sum() > 0:
            u_bc_pred_1 = u_net_1(x_bc[mask_bc_1])
            loss += torch.mean((u_bc_pred_1 - u_bc_true[mask_bc_1]) ** 2)

        if mask_bc_2.sum() > 0:
            u_bc_pred_2 = u_net_2(x_bc[mask_bc_2])
            loss += torch.mean((u_bc_pred_2 - u_bc_true[mask_bc_2]) ** 2)

        return weight * loss


# ============================================================================
# §5 X-RAS-PINN 求解器 (Algorithm 2: 三阶段训练)
# ============================================================================

class XRASPINNSolver:
    """
    X-RAS-PINN 求解器: 域分解 + 自适应采样

    训练流程 (Algorithm 2):
    Phase 1 (预训练): 在粗网格上训练两个子域,无接口约束
    Phase 2 (聚焦): 添加接口约束,在高指标区域加密采样
    Phase 3 (联合微调): 在密集网格上联合优化
    """

    def __init__(
        self,
        problem_config: Dict,
        models: SubdomainModels,
        crack_center: torch.Tensor,
        r_sing: float = 0.15
    ):
        """
        Args:
            problem_config: 问题配置
            models: 子域模型容器
            crack_center: 裂纹中心坐标
            r_sing: 奇异区半径
        """
        self.config = problem_config
        self.models = models
        self.crack_center = crack_center
        self.r_sing = r_sing

        # 提取参数
        self.E = problem_config['E']
        self.nu = problem_config['nu']
        self.G_c = problem_config['G_c']
        self.l = problem_config['l']

        # 损失函数
        self.loss_fn = XRASPINNLoss(self.E, self.nu, self.G_c, self.l)

        # 设备
        self.device = problem_config.get('device', 'cpu')
        self._move_models_to_device()

        # 优化器
        self._initialize_optimizers(problem_config)

        # 训练状态
        self.current_phase = 0
        self.history = []

    def _move_models_to_device(self):
        """将所有模型移动到设备"""
        self.models.u_net_1.to(self.device)
        self.models.d_net_1.to(self.device)
        self.models.u_net_2.to(self.device)
        self.models.d_net_2.to(self.device)

    def _initialize_optimizers(self, config):
        """初始化优化器"""
        lr_u = config.get('lr_u', 1e-3)
        lr_d = config.get('lr_d', 1e-3)

        # 子域1
        self.optimizer_u1 = torch.optim.Adam(
            self.models.u_net_1.parameters(), lr=lr_u)
        self.optimizer_d1 = torch.optim.Adam(
            self.models.d_net_1.parameters(), lr=lr_d)

        # 子域2
        self.optimizer_u2 = torch.optim.Adam(
            self.models.u_net_2.parameters(), lr=lr_u)
        self.optimizer_d2 = torch.optim.Adam(
            self.models.d_net_2.parameters(), lr=lr_d)

    def partition_points(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        划分点集

        Returns:
            x_sing: 奇异区点
            x_far: 远场点
            mask_sing: 奇异区掩码
            mask_far: 远场掩码
        """
        mask_sing, mask_far = partition_domain(x, self.crack_center, self.r_sing)
        x_sing = x[mask_sing]
        x_far = x[mask_far]
        return x_sing, x_far, mask_sing, mask_far

    def get_interface_points(
        self, x_domain: torch.Tensor, n_interface: int = 200
    ) -> torch.Tensor:
        """
        生成接口点 (在 r = r_sing 的圆周上)
        """
        theta = torch.linspace(0, 2*np.pi, n_interface)
        x_c, y_c = self.crack_center[0], self.crack_center[1]

        x_interface = torch.stack([
            x_c + self.r_sing * torch.cos(theta),
            y_c + self.r_sing * torch.sin(theta)
        ], dim=1)

        return x_interface

    def train_phase1_pretrain(
        self,
        x_domain: torch.Tensor,
        x_bc: torch.Tensor,
        u_bc: torch.Tensor,
        n_epochs: int = 1000,
        weight_bc: float = 100.0,
        verbose: bool = True
    ):
        """
        Phase 1: 预训练 (Algorithm 2, lines 1-5)

        在粗网格上独立训练两个子域,不考虑接口约束
        目标: 快速获得合理的初始解
        """
        self.current_phase = 1
        print("\n" + "="*70)
        print("  Phase 1: Pretrain on Coarse Grid (No Interface Constraints)")
        print("="*70)

        x_domain = x_domain.to(self.device)
        x_bc = x_bc.to(self.device)
        u_bc = u_bc.to(self.device)

        # 划分域
        x_sing, x_far, _, _ = self.partition_points(x_domain)
        mask_bc_sing, mask_bc_far = partition_domain(x_bc, self.crack_center, self.r_sing)

        print(f"  Domain partition: Ω_sing={x_sing.shape[0]}, Ω_far={x_far.shape[0]}")
        print(f"  BC partition: BC_sing={mask_bc_sing.sum()}, BC_far={mask_bc_far.sum()}")

        # 训练
        for epoch in range(n_epochs):
            # 清零梯度
            self.optimizer_u1.zero_grad()
            self.optimizer_d1.zero_grad()
            self.optimizer_u2.zero_grad()
            self.optimizer_d2.zero_grad()

            # 子域1能量
            L_energy_1 = self.loss_fn.compute_subdomain_energy(
                x_sing, self.models.u_net_1, self.models.d_net_1)

            # 子域2能量
            L_energy_2 = self.loss_fn.compute_subdomain_energy(
                x_far, self.models.u_net_2, self.models.d_net_2)

            # 边界条件
            L_bc = self.loss_fn.compute_bc_loss(
                x_bc, u_bc,
                self.models.u_net_1, self.models.u_net_2,
                mask_bc_sing, mask_bc_far,
                weight_bc
            )

            # 总损失 (无接口项)
            loss = L_energy_1 + L_energy_2 + L_bc

            # 反向传播
            loss.backward()
            self.optimizer_u1.step()
            self.optimizer_d1.step()
            self.optimizer_u2.step()
            self.optimizer_d2.step()

            # 打印
            if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
                print(f"  Epoch {epoch:4d} | Loss: {loss.item():.6e} | "
                      f"E1: {L_energy_1.item():.6e} | E2: {L_energy_2.item():.6e} | "
                      f"BC: {L_bc.item():.6e}")

        print("  Phase 1 完成!")

    def train_phase2_focused(
        self,
        x_domain: torch.Tensor,
        x_bc: torch.Tensor,
        u_bc: torch.Tensor,
        n_epochs: int = 1000,
        n_interface: int = 200,
        weight_bc: float = 100.0,
        weight_interface: float = 50.0,
        N_add: int = 1000,
        beta: float = 0.5,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Phase 2: 聚焦训练 + 自适应采样 (Algorithm 2, lines 6-12)

        添加接口约束,在高指标区域自适应加密

        Returns:
            x_domain_new: 加密后的采样点
        """
        self.current_phase = 2
        print("\n" + "="*70)
        print("  Phase 2: Focused Training with Interface + Adaptive Sampling")
        print("="*70)

        x_domain = x_domain.to(self.device)
        x_bc = x_bc.to(self.device)
        u_bc = u_bc.to(self.device)

        # 生成接口点
        x_interface = self.get_interface_points(x_domain, n_interface).to(self.device)
        print(f"  Interface points: {x_interface.shape[0]}")

        # 划分域
        x_sing, x_far, _, _ = self.partition_points(x_domain)
        mask_bc_sing, mask_bc_far = partition_domain(x_bc, self.crack_center, self.r_sing)

        # 训练
        for epoch in range(n_epochs):
            self.optimizer_u1.zero_grad()
            self.optimizer_d1.zero_grad()
            self.optimizer_u2.zero_grad()
            self.optimizer_d2.zero_grad()

            # 子域能量
            L_energy_1 = self.loss_fn.compute_subdomain_energy(
                x_sing, self.models.u_net_1, self.models.d_net_1)
            L_energy_2 = self.loss_fn.compute_subdomain_energy(
                x_far, self.models.u_net_2, self.models.d_net_2)

            # 边界条件
            L_bc = self.loss_fn.compute_bc_loss(
                x_bc, u_bc,
                self.models.u_net_1, self.models.u_net_2,
                mask_bc_sing, mask_bc_far,
                weight_bc
            )

            # 接口损失
            u1_interface = self.models.u_net_1(x_interface)
            u2_interface = self.models.u_net_2(x_interface)
            L_u, L_trac = compute_interface_loss(
                x_interface,
                self.models.u_net_1, self.models.u_net_2,
                self.E, self.nu
            )
            L_interface = weight_interface * (L_u + L_trac)

            # 总损失
            loss = L_energy_1 + L_energy_2 + L_bc + L_interface

            # 反向传播
            loss.backward()
            self.optimizer_u1.step()
            self.optimizer_d1.step()
            self.optimizer_u2.step()
            self.optimizer_d2.step()

            # 打印
            if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
                print(f"  Epoch {epoch:4d} | Loss: {loss.item():.6e} | "
                      f"Interface: {L_interface.item():.6e} | "
                      f"L_u: {L_u.item():.6e} | L_trac: {L_trac.item():.6e}")

        # 自适应采样 (Algorithm 2, line 11-12)
        print("\n  Computing indicators for adaptive sampling...")

            # 在奇异区计算指标 (裂纹区更需要加密)
        indicator_sing = compute_indicator(
            x_sing, self.models.u_net_1, self.models.d_net_1,
            self.E, self.nu, beta
        )
        indicator_far = compute_indicator(
            x_far, self.models.u_net_2, self.models.d_net_2,
            self.E, self.nu, beta
        )
        # 重采样
        domain_bounds = (
            (self.config.get('x_min', 0.0), self.config.get('x_max', 1.0)),
            (self.config.get('y_min', 0.0), self.config.get('y_max', 1.0))
        )

        # 优先在奇异区加密
        N_add_sing = int(N_add * 0.7)
        N_add_far = N_add - N_add_sing

        x_sing_new = resample(
            x_sing.cpu(), indicator_sing.detach().cpu(), N_add_sing, domain_bounds)
        x_far_new = resample(
            x_far.cpu(), indicator_far.detach().cpu(), N_add_far, domain_bounds)

        x_domain_new = torch.cat([x_sing_new, x_far_new], dim=0)

        print(f"  Adaptive sampling: {x_domain.shape[0]} → {x_domain_new.shape[0]} points")
        print(f"    Ω_sing: {x_sing.shape[0]} → {x_sing_new.shape[0]}")
        print(f"    Ω_far: {x_far.shape[0]} → {x_far_new.shape[0]}")

        # 可视化指标分布
        self._visualize_indicator(x_domain,
                                  torch.cat([indicator_sing.cpu(), indicator_far.cpu()]))

        print("  Phase 2 完成!")

        return x_domain_new

    def train_phase3_joint_finetune(
        self,
        x_domain: torch.Tensor,
        x_bc: torch.Tensor,
        u_bc: torch.Tensor,
        n_epochs: int = 1000,
        n_interface: int = 200,
        weight_bc: float = 100.0,
        weight_interface: float = 50.0,
        verbose: bool = True
    ):
        """
        Phase 3: 联合微调 (Algorithm 2, lines 13-15)

        在加密后的网格上联合优化所有损失项
        """
        self.current_phase = 3
        print("\n" + "="*70)
        print("  Phase 3: Joint Fine-tuning on Dense Grid")
        print("="*70)

        x_domain = x_domain.to(self.device)
        x_bc = x_bc.to(self.device)
        u_bc = u_bc.to(self.device)

        # 生成接口点
        x_interface = self.get_interface_points(x_domain, n_interface).to(self.device)

        # 划分域
        x_sing, x_far, _, _ = self.partition_points(x_domain)
        mask_bc_sing, mask_bc_far = partition_domain(x_bc, self.crack_center, self.r_sing)

        print(f"  Final grid: Ω_sing={x_sing.shape[0]}, Ω_far={x_far.shape[0]}")

        # 训练
        for epoch in range(n_epochs):
            self.optimizer_u1.zero_grad()
            self.optimizer_d1.zero_grad()
            self.optimizer_u2.zero_grad()
            self.optimizer_d2.zero_grad()

            # 子域能量
            L_energy_1 = self.loss_fn.compute_subdomain_energy(
                x_sing, self.models.u_net_1, self.models.d_net_1)
            L_energy_2 = self.loss_fn.compute_subdomain_energy(
                x_far, self.models.u_net_2, self.models.d_net_2)

            # 边界条件
            L_bc = self.loss_fn.compute_bc_loss(
                x_bc, u_bc,
                self.models.u_net_1, self.models.u_net_2,
                mask_bc_sing, mask_bc_far,
                weight_bc
            )

            # 接口损失
            # u1_interface = self.models.u_net_1(x_interface)
            # u2_interface = self.models.u_net_2(x_interface)
            L_u, L_trac = compute_interface_loss(
                x_interface,
                self.models.u_net_1,
                self.models.u_net_2,
                self.E,
                self.nu
            )
            L_interface = weight_interface * (L_u + L_trac)
            # 总损失
            loss = L_energy_1 + L_energy_2 + L_bc + L_interface

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.models.u_net_1.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.models.d_net_1.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.models.u_net_2.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.models.d_net_2.parameters(), 1.0)

            self.optimizer_u1.step()
            self.optimizer_d1.step()
            self.optimizer_u2.step()
            self.optimizer_d2.step()

            # 打印
            if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
                with torch.no_grad():
                    d1_vals = self.models.d_net_1(x_sing)
                    d2_vals = self.models.d_net_2(x_far)
                    d_all = torch.cat([d1_vals, d2_vals])
                    d_mean = d_all.mean().item()
                    d_max = d_all.max().item()

                print(f"  Epoch {epoch:4d} | Loss: {loss.item():.6e} | "
                      f"Interface: {L_interface.item():.6e} | "
                      f"d_mean: {d_mean:.3f} | d_max: {d_max:.3f}")

        print("  Phase 3 完成!")

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测位移和损伤场 (自动选择子域)

        Args:
            x: (N, 2) 查询点

        Returns:
            u: (N, 2) 位移
            d: (N, 1) 损伤
        """
        self.models.u_net_1.eval()
        self.models.d_net_1.eval()
        self.models.u_net_2.eval()
        self.models.d_net_2.eval()

        x = x.to(self.device)
        mask_sing, mask_far = partition_domain(x, self.crack_center, self.r_sing)

        u = torch.zeros(x.shape[0], 2, device=self.device)
        d = torch.zeros(x.shape[0], 1, device=self.device)

        with torch.no_grad():
            if mask_sing.sum() > 0:
                u[mask_sing] = self.models.u_net_1(x[mask_sing])
                d[mask_sing] = self.models.d_net_1(x[mask_sing])

            if mask_far.sum() > 0:
                u[mask_far] = self.models.u_net_2(x[mask_far])
                d[mask_far] = self.models.d_net_2(x[mask_far])

        return u, d

    def _visualize_indicator(self, x: torch.Tensor, indicator: torch.Tensor):
        """可视化指标分布 (用于诊断)"""

        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            x_np = x.cpu().numpy()
            indicator_np = indicator.cpu().numpy()

            scatter = ax.scatter(
                x_np[:, 0], x_np[:, 1],
                c=indicator_np, cmap='hot', s=5, alpha=0.6
            )

            # 标记裂纹中心
            ax.plot(self.crack_center[0].item(), self.crack_center[1].item(),
                   'g*', markersize=15, label='Crack Center')

            # 标记奇异区边界
            circle = plt.Circle(
                (self.crack_center[0].item(), self.crack_center[1].item()),
                self.r_sing, color='blue', fill=False, linestyle='--',
                linewidth=2, label=f'Ω_sing (r={self.r_sing})'
            )
            ax.add_patch(circle)

            plt.colorbar(scatter, ax=ax, label='Indicator η_fused')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Adaptive Sampling Indicator')
            ax.legend()
            ax.set_aspect('equal')

            out_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, "indicator_distribution.png")
            plt.savefig(save_path, dpi=150)
            print(f"    Indicator visualization saved to {save_path}")


        except Exception as e:
            print(f"    Warning: Could not visualize indicator: {e}")


# ============================================================================
# §6 可视化工具
# ============================================================================

def visualize_xpinn_solution(
    solver: XRASPINNSolver,
    x_grid: torch.Tensor,
    nx: int, ny: int,
    save_path: str = None
):
    """
    可视化 X-RAS-PINN 解 (显示域分解)
    """
    solver.models.u_net_1.eval()
    solver.models.d_net_1.eval()
    solver.models.u_net_2.eval()
    solver.models.d_net_2.eval()

    u_pred, d_pred = solver.predict(x_grid)

    u_pred = u_pred.cpu().numpy()
    d_pred = d_pred.cpu().numpy().reshape(ny, nx)

    X = x_grid[:, 0].cpu().numpy().reshape(ny, nx)
    Y = x_grid[:, 1].cpu().numpy().reshape(ny, nx)

    # 子域掩码
    mask_sing, _ = partition_domain(x_grid, solver.crack_center, solver.r_sing)
    mask_sing = mask_sing.cpu().numpy().reshape(ny, nx)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # U_x
    im0 = axes[0, 0].contourf(X, Y, u_pred[:, 0].reshape(ny, nx), levels=50)
    axes[0, 0].contour(X, Y, mask_sing, levels=[0.5], colors='white',
                       linewidths=2, linestyles='--')
    axes[0, 0].set_title("u_x (dashed = Ω_sing boundary)")
    plt.colorbar(im0, ax=axes[0, 0])

    # U_y
    im1 = axes[0, 1].contourf(X, Y, u_pred[:, 1].reshape(ny, nx), levels=50)
    axes[0, 1].contour(X, Y, mask_sing, levels=[0.5], colors='white',
                       linewidths=2, linestyles='--')
    axes[0, 1].set_title("u_y")
    plt.colorbar(im1, ax=axes[0, 1])

    # Damage
    im2 = axes[1, 0].contourf(X, Y, d_pred, levels=50, vmin=0.0, vmax=1.0, cmap='Reds')
    axes[1, 0].contour(X, Y, mask_sing, levels=[0.5], colors='blue',
                       linewidths=2, linestyles='--')
    axes[1, 0].plot(solver.crack_center[0].item(), solver.crack_center[1].item(),
                    'g*', markersize=15)
    axes[1, 0].set_title("Damage d")
    plt.colorbar(im2, ax=axes[1, 0])

    # Domain partition
    im3 = axes[1, 1].contourf(X, Y, mask_sing.astype(float), levels=[0, 0.5, 1],
                              colors=['lightblue', 'lightcoral'])
    axes[1, 1].plot(solver.crack_center[0].item(), solver.crack_center[1].item(),
                    'g*', markersize=15, label='Crack Center')
    axes[1, 1].set_title("Domain Partition (red=Ω_sing, blue=Ω_far)")
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Solution saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    print("="*70)
    print("  X-RAS-PINN Solver for Phase-Field Fracture")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Domain Decomposition (XPINN)")
    print("  ✓ Interface Constraints (Displacement + Traction)")
    print("  ✓ Residual-based Adaptive Sampling (RAS)")
    print("  ✓ Three-Phase Training (Algorithm 2)")
    print("\nReady for use!")
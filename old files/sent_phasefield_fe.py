# sent_phasefield_fe_vertical.py

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import numpy as np
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem, NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

# ✅ 新增：统一读取 config
from config import create_config, print_config


def run_sent_phasefield_vertical(
    debug: bool = False,
    cfg: dict | None = None,
    npz_path: str = "data/fe_sent_phasefield.npz",
    xdmf_path: str = "data/fe_sent_phasefield.xdmf",
):
    """
    SENT 相场 (竖直拉伸) 的 FEniCSx 基准解
    - 与 Phase-1 / Phase-2 使用同一份 config
    - 输出 .npz + .xdmf，供 PINN/X-RAS 对比使用
    """

    # -------------------------
    # 1. 统一配置
    # -------------------------
    if cfg is None:
        cfg = create_config(debug=debug)
    print_config(cfg)

    # ✅ 和 PINN 完全一致的参数名
    L = float(cfg["L"])                # 试样长度 (x方向)
    H = float(cfg["H"])                # 试样高度 (y方向)
    notch_length = float(cfg["notch_length"])

    E = float(cfg["E"])
    nu = float(cfg["nu"])
    Gc = float(cfg["G_c"])
    ell = float(cfg["l"])

    n_steps = int(cfg["n_loading_steps"])
    max_disp = float(cfg["max_displacement"])

    # FE 网格分辨率可以放在 config 里，也可以先给默认值
    nx_fe = int(cfg.get("nx_fe", 120))
    ny_fe = int(cfg.get("ny_fe", 60))


    # -------------------------
    # 2. 几何 & 网格
    # -------------------------
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        points=((0.0, 0.0), (L, H)),
        n=(nx_fe, ny_fe),
        cell_type=mesh.CellType.triangle,
    )
    dx = ufl.Measure("dx", domain=msh)
    ds = ufl.Measure("ds", domain=msh)

    # 向量位移 + 标量损伤
    Vu = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    Vd = fem.functionspace(msh, ("Lagrange", 1))

    u = fem.Function(Vu, name="u")
    du = ufl.TestFunction(Vu)

    d = fem.Function(Vd, name="d")
    eta = ufl.TestFunction(Vd)

    # -------------------------
    # # notch 损伤种子初始化（仿照 PINN）
    # # -------------------------
    # coords_d = Vd.tabulate_dof_coordinates()
    # x_d = coords_d[:, 0]
    # y_d = coords_d[:, 1]
    #
    # notch_length = float(cfg["notch_length"])
    # H = float(cfg["H"])
    # initial_d = float(cfg["initial_d"]) # 用来控制 tip 区的强度
    # seed_radius = float(cfg["notch_seed_radius"]) # 这里既用作 tip 半径，也用作裂缝带半厚度
    #
    # # 距离 notch tip 的半径
    # r = np.sqrt((x_d - notch_length) ** 2 + (y_d - H / 2.0) ** 2)
    #
    # # 高斯型种子
    # d_seed = initial_d * np.exp(-(r / seed_radius) ** 2)
    #
    # # 尖端非常近的点再抬高到接近 1
    # tip_core_radius = 0.3 * seed_radius
    # tip_mask = r < tip_core_radius
    # d_seed[tip_mask] = np.maximum(d_seed[tip_mask], 0.95)
    #
    # # 远场削到几乎 0
    # far_mask = r > 3.0 * seed_radius
    # d_seed[far_mask] = 0.0
    #
    # # 写入到 d 中
    # d_vec = d.x.array
    # d_vec[:] = d_seed.astype(d_vec.dtype)
    # d.x.array[:] = d_vec
    #
    # # 不可逆历史场同步
    # d_old = fem.Function(Vd)
    # d_old.x.array[:] = d.x.array


    # -------------------------
    # notch 损伤种子初始化：整条预裂缝线 + 尖端平滑区
    # -------------------------
    coords_d = Vd.tabulate_dof_coordinates()
    x_d = coords_d[:, 0]
    y_d = coords_d[:, 1]

    notch_length = float(cfg["notch_length"])
    H = float(cfg["H"])
    initial_d = float(cfg["initial_d"])          # 用来控制 tip 区的强度
    seed_radius = float(cfg["notch_seed_radius"])  # 这里既用作 tip 半径，也用作裂缝带半厚度

    # 先全部置零
    d_seed = np.zeros_like(x_d, dtype=float)

    # -------------------------
    # 1) 预裂缝线：y = H/2 附近的一条“带”，0 <= x <= notch_length
    #    这代表已经完全断开的裂缝面，直接设置 d = 1
    # -------------------------
    band_half = seed_radius          # 裂缝带半厚度（可在 config 里通过 notch_seed_radius 控制）
    line_mask = (x_d <= notch_length) & (np.abs(y_d - H / 2.0) <= band_half)
    d_seed[line_mask] = 1.0

    # -------------------------
    # 2) 裂尖平滑区：在 (notch_length, H/2) 附近做一个高斯过渡
    #    让裂缝从“已断开”平滑过渡到“尚未损伤”，方便数值收敛
    # -------------------------
    r_tip = np.sqrt((x_d - notch_length) ** 2 + (y_d - H / 2.0) ** 2)

    # 高斯型 tip 种子（初值不一定是 1，可以用 initial_d 控制）
    tip_gauss = initial_d * np.exp(-(r_tip / seed_radius) ** 2)

    # 只对 tip 前方和周围一定范围内的点激活高斯
    tip_region_mask = (x_d >= notch_length) & (r_tip <= 3.0 * seed_radius)
    d_seed[tip_region_mask] = np.maximum(d_seed[tip_region_mask], tip_gauss[tip_region_mask])

    # 尖端非常近的点再抬高到接近 1，增强裂尖种子
    tip_core_radius = 0.5 * seed_radius
    tip_core_mask = r_tip < tip_core_radius
    d_seed[tip_core_mask] = np.maximum(d_seed[tip_core_mask], 0.95)

    # -------------------------
    # 3) 远场保持 0（默认就是 0，这里可不额外处理）
    # -------------------------

    # 写回 FEniCS 函数
    d_vec = d.x.array
    d_vec[:] = d_seed.astype(d_vec.dtype)
    d.x.array[:] = d_vec

    # 不可逆历史场同步
    d_old = fem.Function(Vd)
    d_old.x.array[:] = d.x.array

    # -------------------------
    # 3. 材料 & 相场能量
    # -------------------------
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def eps(v):
        return ufl.sym(ufl.grad(v))

    def sigma_0(v):
        return 2.0 * mu * eps(v) + lam * ufl.tr(eps(v)) * ufl.Identity(len(v))

    # 未降阶的 Cauchy 应力
    def sigma_0(v):
        return 2.0 * mu * eps(v) + lam * ufl.tr(eps(v)) * ufl.Identity(len(v))

    # 含相场降阶的有效应力 σ_eff = g(d) σ⁺ + σ⁻
    def sigma_eff(u, d):
        eps_u = eps(u)
        tr_eps = ufl.tr(eps_u)
        eps_dev = eps_u - tr_eps / 2 * ufl.Identity(2)

        tr_plus = 0.5 * (tr_eps + abs(tr_eps))
        tr_minus = tr_eps - tr_plus

        # 体积应变正负部分
        eps_vol_plus = tr_plus / 2 * ufl.Identity(2)
        eps_vol_minus = tr_minus / 2 * ufl.Identity(2)

        eps_plus = eps_dev + eps_vol_plus
        eps_minus = eps_dev + eps_vol_minus

        sigma_plus = lam * ufl.tr(eps_plus) * ufl.Identity(2) + 2.0 * mu * eps_plus
        sigma_minus = lam * ufl.tr(eps_minus) * ufl.Identity(2) + 2.0 * mu * eps_minus

        g_d_loc = (1.0 - d) ** 2 + kappa
        return g_d_loc * sigma_plus + sigma_minus


    kappa = 1e-8  # 残余刚度

    # --- strain tensor ---
    eps_u = eps(u)
    tr_eps = ufl.tr(eps_u)
    eps_dev = eps_u - tr_eps / 2 * ufl.Identity(2)

    # --- positive/negative splits (Miehe 2010) ---
    tr_plus = 0.5 * (tr_eps + abs(tr_eps))
    tr_minus = tr_eps - tr_plus

    psi_plus = 0.5 * lam * tr_plus ** 2 + mu * ufl.inner(eps_dev, eps_dev)
    psi_minus = 0.5 * lam * tr_minus ** 2 + mu * ufl.inner(eps_dev, eps_dev)

    # --- degradation ---
    g_d = (1.0 - d) ** 2 + kappa

    # --- AT1 crack density ---
    # w_AT1(d) = d
    # crack_density_AT1 = d/ell + ell*|∇d|²
    crack_density_AT1 = d / ell + ell * ufl.inner(ufl.grad(d), ufl.grad(d))

    # --- total AT1 energy ---
    energy = (
            (g_d * psi_plus + psi_minus) * dx
            + Gc * crack_density_AT1 * dx
    )


    # ====== U 子问题：非线性 (Newton) ======
    F_u = ufl.derivative(energy, u, du)
    J_u = ufl.derivative(F_u, u)

    # ====== D 子问题：线性 AT1 方程 (手工推导) ======
    # 强形式：
    # -2(1-d) ψ⁺ + Gc/ℓ - 2 Gc ℓ Δd = 0
    # 展开：
    # 2 ψ⁺ d + 2 Gc ℓ (-Δd) + (Gc/ℓ - 2 ψ⁺) = 0
    # 变分形式：
    # ∫ (2 ψ⁺ d v + 2 Gc ℓ ∇d·∇v) dx = ∫ (2 ψ⁺ - Gc/ℓ) v dx

    # AT1: 2 psi_plus d v + 2 Gc ell ∇d·∇v = (2 psi_plus - Gc/ell) v

    d_trial = ufl.TrialFunction(Vd)
    v_d = ufl.TestFunction(Vd)

    a_d = (
        2.0 * psi_plus * d_trial * v_d
        + 2.0 * Gc * ell * ufl.inner(ufl.grad(d_trial), ufl.grad(v_d))
    ) * dx

    L_d = ((2.0 * psi_plus - Gc / ell) * v_d) * dx



    # -------------------------
    # 4. 边界条件：竖直拉伸
    # -------------------------
    def bottom(x):
        return np.isclose(x[1], 0.0)

    def top(x):
        return np.isclose(x[1], H)

    facets_bottom = mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, bottom
    )
    facets_top = mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, top
    )

    dofs_bottom = fem.locate_dofs_topological(Vu, msh.topology.dim - 1, facets_bottom)
    dofs_top = fem.locate_dofs_topological(Vu, msh.topology.dim - 1, facets_top)

    # 底部固支：u = (0, 0)
    bc_bottom = fem.dirichletbc(
        np.array((0.0, 0.0), dtype=ScalarType), dofs_bottom, Vu
    )

    # 顶部竖直位移：u_y = disp, u_x = 0
    # 这里位移值每个 load step 更新
    top_disp = fem.Function(Vu)
    bc_top = fem.dirichletbc(top_disp, dofs_top)

    bcs_u = [bc_bottom, bc_top]

    # 损伤边界：自然边界 (Neumann) → 这里不用显式 BC

    # -------------------------
    # 5. 求解器准备 (交替最小化)
    # -------------------------
    # 1) 准备 U (位移) 的求解器
    # -------------------------------------------------
    # 必须显式定义 J_u (UFL 形式)，否则会报 UnboundLocalError
    J_u = ufl.derivative(F_u, u)

    # 1) 准备 U (位移) 的求解器 —— 非线性 Newton
    problem_u = NewtonSolverNonlinearProblem(F_u, u, bcs=bcs_u, J=J_u)
    solver_u = NewtonSolver(MPI.COMM_WORLD, problem_u)
    solver_u.convergence_criterion = "incremental"
    solver_u.rtol = 1e-8
    solver_u.max_it = 30

    # 2) 准备 D (损伤) 的求解器 —— 线性 AT1 方程
    problem_d = LinearProblem(
        a_d,
        L_d,
        bcs=[],
        u=d,  # 解直接写入现有的 d
        petsc_options={},
        petsc_options_prefix="d_",  # 随便一个前缀即可
    )

    # 载荷路径
    load_steps = np.linspace(0.0, max_disp, n_steps)

    # 存储历史
    coords_u = Vu.tabulate_dof_coordinates()
    coords_d = Vd.tabulate_dof_coordinates()
    n_dofs_u = Vu.dofmap.index_map.size_local * msh.geometry.dim
    n_dofs_d = Vd.dofmap.index_map.size_local

    u_hist = []
    d_hist = []
    reaction_hist = []


    for i_step, disp in enumerate(load_steps):
        if msh.comm.rank == 0:
            print(f"\n=== Load step {i_step+1}/{n_steps}, disp = {disp:.6e} ===")

        # 更新顶部位移
        top_disp.x.array[:] = 0.0
        top_disp.x.array[1::2] = disp  # uy 分量

        # 交替迭代
        for it in range(5):  # 先给一个小的迭代数，可调
            # 5.1 解 u
            solver_u.solve(u)

            # 5.2 解 d（线性 AT1 损伤方程）
            problem_d.solve()  # 解出来直接写入 d

            # 不可逆性 + 0 ≤ d ≤ 1 投影
            d_vec = d.x.array
            d_old_vec = d_old.x.array

            # 先做不可逆性：不能比历史小
            d_vec[:] = np.maximum(d_vec, d_old_vec)
            # 再做物理约束：0 ≤ d ≤ 1
            d_vec[:] = np.minimum(d_vec, 1.0)
            d_vec[:] = np.maximum(d_vec, 0.0)

            d.x.array[:] = d_vec

        # 更新历史场
        d_old.x.array[:] = d.x.array

        # 计算反力（顶边反力合力）
        # 计算竖直方向合反力（这里对整条边界积分；
        # 由于除顶边外其余为自由边或固支，理论上只在顶边有非零反力）
        n = ufl.FacetNormal(msh)
        ey = ufl.as_vector((0.0, 1.0))

        n = ufl.FacetNormal(msh)
        ey = ufl.as_vector((0.0, 1.0))

        # 如果你已经有顶边标记，就用 ds(top_id)；否则暂时先用整个 ds
        traction_form = fem.form(ufl.dot(ufl.dot(sigma_eff(u, d), n), ey) * ds)
        traction_y = fem.assemble_scalar(traction_form)

        reaction_hist.append(traction_y)

        # # 存储场
        u_hist.append(u.x.array.real.copy())
        d_hist.append(d.x.array.real.copy())

    # rank-0 输出 npz

    if msh.comm.rank == 0:
        u_hist = np.stack(u_hist, axis=-1)  # (Ndof_u, n_steps)
        d_hist = np.stack(d_hist, axis=-1)  # (Ndof_d, n_steps)
        reaction_hist = np.array(reaction_hist)

        np.savez(
            npz_path,
            coords_u=coords_u,
            coords_d=coords_d,
            u=u_hist,
            d=d_hist,
            load_steps=load_steps,
            reactions=reaction_hist,
            L=L,
            H=H,
            notch_length=notch_length,
            E=E,
            nu=nu,
            G_c=Gc,
            l=ell,
        )
        print(f"✅ FE phase-field baseline saved to {npz_path}")

    # 写 XDMF（最后一步场）
    with io.XDMFFile(msh.comm, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(u, 0.0)
        xdmf.write_function(d, 0.0)

    if msh.comm.rank == 0:
        print(f"✅ XDMF written to {xdmf_path}")


if __name__ == "__main__":
    run_sent_phasefield_vertical(debug=False)

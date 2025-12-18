
from mpi4py import MPI                              # MPI 并行计算支持
from petsc4py.PETSc import ScalarType               # PETSc 标量类型（用于边界条件值）
import numpy as np
import ufl                                          # 统一形式语言，用于定义变分形式
from dolfinx import mesh, fem, io                   # FEniCSx 核心模块
from dolfinx.fem.petsc import LinearProblem, NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver          # 非线性 Newton 求解器

from config import create_config, print_config     # 用户自定义配置模块
from datetime import datetime


def run_sent_phasefield_vertical(
        debug: bool = False,
        cfg: dict | None = None,
        npz_path: str = "data/fe_sent_phasefield.npz",
        xdmf_path: str = "data/fe_sent_phasefield.xdmf",
):
    """
    SENT 相场 (竖直拉伸) 的 FEniCSx 基准解 - 最终修正版
    修复了 Step 0 全红问题和反力计算问题。
    """

    # -------------------------
    # 1. 统一配置
    # -------------------------
    if cfg is None:
        cfg = create_config(debug=debug)

    if MPI.COMM_WORLD.rank == 0:
        print_config(cfg)

    # 几何参数
    L_val = float(cfg["L"])  # 试样长度
    H_val = float(cfg["H"])  # 试样高度
    notch_length = float(cfg["notch_length"])  # 预制缺口长度

    # 材料参数
    E_val = float(cfg["E"])  # 杨氏模量
    nu_val = float(cfg["nu"])  # 泊松比
    Gc_val = float(cfg["G_c"])  # 断裂能（临界能量释放率）
    ell_val = float(cfg["l"])  # 相场正则化长度尺度

    # 加载参数
    n_steps = int(cfg["n_loading_steps"])  # 加载步数
    max_disp = float(cfg["max_displacement"])  # 最大施加位移

    # 网格参数
    nx_fe = int(cfg.get("nx_fe", 640))  # x 方向网格数
    ny_fe = int(cfg.get("ny_fe", 640))  # y 方向网格数

    # -------------------------
    # 2. 几何 & 网格
    # -------------------------
    # 创建矩形网格，左下角 (0,0)，右上角 (L, H)
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        points=((0.0, 0.0), (L_val, H_val)),
        n=(nx_fe, ny_fe),
        cell_type=mesh.CellType.triangle, # 使用三角形单元
    )

    # 定义顶部边界（用于施加位移和计算反力）
    def top_boundary(x):
        return np.isclose(x[1], H_val)

    fdim = msh.topology.dim - 1   # 边界维度 = 1（二维问题的边是线）
    top_facets = mesh.locate_entities_boundary(msh, fdim, top_boundary)

    # # 为顶部边界创建标记（tag=1），用于积分测度
    top_tag = 1
    sorted_top = np.argsort(top_facets)
    mt = mesh.meshtags(msh, fdim, top_facets[sorted_top], np.full(len(top_facets), top_tag, dtype=np.int32))

    # 定义积分测度
    dx = ufl.Measure("dx", domain=msh)     # 体积积分
    ds_top = ufl.Measure("ds", domain=msh, subdomain_data=mt, subdomain_id=top_tag)  # 顶部边界积分

    # -------------------------
    # 3. 函数空间
    # -------------------------
    # 位移场函数空间：向量值 P1 元（线性拉格朗日）
    Vu = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    # 损伤场函数空间：标量 P1 元
    Vd = fem.functionspace(msh, ("Lagrange", 1))

    # 位移场相关函数
    u = fem.Function(Vu, name="u")  # 位移解
    du = ufl.TestFunction(Vu)       # 位移测试函数

    # 损伤场相关函数
    d = fem.Function(Vd, name="d")  # 损伤解
    v_d = ufl.TestFunction(Vd)      # 损伤测试函数
    d_trial = ufl.TrialFunction(Vd) # 损伤试探函数（用于雅可比矩阵）

    # 历史场 H（与 d 同一空间，便于实现）存储最大拉伸能量密度（驱动损伤演化的关键）,保证不可逆
    H = fem.Function(Vd, name="H")
    H_old = fem.Function(Vd, name="H_old")
    H.x.array[:] = 0.0
    H_old.x.array[:] = 0.0

    # -------------------------
    # 4. 初始损伤 (Notch)
    # -------------------------
    coords_d = Vd.tabulate_dof_coordinates()  # 获取所有自由度坐标
    x_d = coords_d[:, 0]  # x 坐标
    y_d = coords_d[:, 1]  # y 坐标

    initial_d = float(cfg["initial_d"])  # 初始损伤值
    seed_radius = float(cfg["notch_seed_radius"])  # 缺口/裂尖的种子半径

    d_vec = d.x.array
    d_vec[:] = 0.0

    # 预制裂缝带：x ≤ notch_length 且 y 在中线附近（半高度处）
    # 这模拟了 SENT 试样左侧的预制缺口
    line_mask = (x_d <= notch_length) & (np.abs(y_d - H_val / 2.0) <= seed_radius)
    d_vec[line_mask] = 1.0   # 缺口区域完全损伤

    # 裂尖平滑过渡：用高斯函数使裂尖处损伤平滑衰减
    # 避免数值上的尖锐不连续
    r_tip = np.sqrt((x_d - notch_length) ** 2 + (y_d - H_val / 2.0) ** 2)
    tip_mask = (x_d > notch_length - seed_radius) & (r_tip < 3.0 * seed_radius)
    d_gauss = initial_d * np.exp(-(r_tip[tip_mask] / seed_radius) ** 2)
    d_vec[tip_mask] = np.maximum(d_vec[tip_mask], d_gauss)

    d.x.array[:] = d_vec

    # 保存初始损伤作为"旧值"，用于保证不可逆性
    d_old = fem.Function(Vd)
    d_old.x.array[:] = d.x.array

    # -------------------------
    # 5. 能量与方程
    # -------------------------
    # Lamé 常数（由 E 和 ν 计算）
    mu = E_val / (2.0 * (1.0 + nu_val))  # 剪切模量
    lam = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))  # 第一 Lamé 常数
    kappa = 1e-5  # 小参数，防止完全损伤时刚度为零（数值稳定性）

    # 应变张量
    def eps(v):
        return ufl.sym(ufl.grad(v)) # 对称梯度 = 线性应变

    eps_u = eps(u)
    tr_eps = ufl.tr(eps_u)  # 应变迹 = 体积应变

    # 应变偏量（2D 平面应变）
    eps_dev = eps_u - tr_eps / 2 * ufl.Identity(2)

    # 应变迹的正负分解（拉/压分解）
    # tr⁺ = max(tr(ε), 0)，tr⁻ = tr(ε) - tr⁺
    tr_plus = 0.5 * (tr_eps + ufl.sqrt(tr_eps ** 2 + 1e-14)) # 正部分（）
    tr_minus = tr_eps - tr_plus           # 负部分（压缩）

    # 弹性能量密度的拉压分解
    # ψ⁺：拉伸能量（会驱动裂纹扩展）
    # ψ⁻：压缩能量（不驱动裂纹，但仍贡献应力）
    psi_plus = 0.5 * lam * tr_plus ** 2 + mu * ufl.inner(eps_dev, eps_dev) #拉伸+剪切
    psi_minus = 0.5 * lam * tr_minus ** 2

    # 退化函数 g(d) = (1-d)² + κ
    # d=0 时 g≈1（完整材料），d=1 时 g≈κ（几乎无刚度）
    g_d = (1.0 - d) ** 2 + kappa

    # =============================================
    # 裂纹表面能密度 —— 这是 AT1 模型！
    # =============================================
    # AT1: γ = Gc * (d/ℓ + ℓ|∇d|²)     ← 这里用的
    # AT2: γ = Gc * (d²/ℓ + ℓ|∇d|²)
    crack_energy = Gc_val * (d / ell_val + ell_val * ufl.inner(ufl.grad(d), ufl.grad(d)))

    # 总势能
    energy = (g_d * psi_plus + psi_minus) * dx + crack_energy * dx

    # 位移方程的弱形式（对总能量关于 u 求变分）
    F_u = ufl.derivative(energy, u, du)  # 残差
    J_u = ufl.derivative(F_u, u)  # 雅可比矩阵（切线刚度）

    # =============================================
    # 损伤方程（使用历史场 H 而非实时 ψ⁺）
    # =============================================
    # 损伤能量泛函：用 H 代替 ψ⁺
    damage_energy = (g_d * H + Gc_val * (d / ell_val + ell_val * ufl.inner(ufl.grad(d), ufl.grad(d)))) * dx

    F_d = ufl.derivative(damage_energy, d, v_d)  # 损伤残差
    J_d = ufl.derivative(F_d, d, d_trial)  # 损伤雅可比


    #线性化形式（实际代码用非线性）
    a_d = (
                  2.0 * H * d_trial * v_d
                  + 2.0 * Gc_val * ell_val * ufl.inner(ufl.grad(d_trial), ufl.grad(v_d))
          ) * dx

    L_d = ((2.0 * H - Gc_val / ell_val) * v_d) * dx

    # -------------------------
    # 6. 边界与求解器
    # -------------------------
    # 底部边界：完全固定 (u_x = u_y = 0)
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)

    dofs_bot = fem.locate_dofs_topological(Vu, fdim, mesh.locate_entities_boundary(msh, fdim, bottom_boundary))
    dofs_top = fem.locate_dofs_topological(Vu, fdim, top_facets)

    bc_bot = fem.dirichletbc(np.array((0.0, 0.0), dtype=ScalarType), dofs_bot, Vu)

    # 顶部边界：施加位移（通过函数控制，每步更新）
    top_disp = fem.Function(Vu)
    bc_top = fem.dirichletbc(top_disp, dofs_top)
    bcs_u = [bc_bot, bc_top]

    # =============================================
    # 位移场 Newton 求解器配置
    # =============================================

    problem_u = NewtonSolverNonlinearProblem(F_u, u, bcs=bcs_u, J=J_u)
    solver_u = NewtonSolver(MPI.COMM_WORLD, problem_u)
    solver_u.convergence_criterion = "incremental"  # 基于增量的收敛判据
    solver_u.rtol = 1e-6  # 相对容差
    solver_u.atol = 1e-8  # 绝对容差
    solver_u.max_it = 50  # 最大迭代次数
    solver_u.relaxation_parameter = 0.8  # 松弛因子（阻尼，防止震荡）

    # 强制直接求解器 (LU/MUMPS)
    ksp = solver_u.krylov_solver
    ksp.setType("preonly")  # 不使用 Krylov 迭代
    ksp.getPC().setType("lu")  # LU 分解
    try:
        ksp.getPC().setFactorSolverType("mumps") # 尝试使用 MUMPS 并行直接求解器
    except:
        pass

    # # 线性求解d
    # problem_d = LinearProblem(a_d, L_d, bcs=[], u=d,
    #                           petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    #                           petsc_options_prefix="d_solver")

    # =============================================
    # 损伤场 Newton 求解器配置（类似设置）
    # =============================================
    damage_problem = NewtonSolverNonlinearProblem(F_d, d, bcs=[], J=J_d)
    damage_solver = NewtonSolver(MPI.COMM_WORLD, damage_problem)
    damage_solver.convergence_criterion = "incremental"
    damage_solver.rtol = 1e-6
    damage_solver.atol = 1e-8
    damage_solver.max_it = 50
    damage_solver.relaxation_parameter = 0.8

    ksp_d = damage_solver.krylov_solver
    ksp_d.setType("preonly")
    ksp_d.getPC().setType("lu")
    try:
        ksp_d.getPC().setFactorSolverType("mumps")
    except:
        pass



    # 7. 加载循环（交替迭代求解）
    load_steps_arr = np.linspace(0.0, max_disp, n_steps)  # 位移加载路径
    u_hist, d_hist, reaction_hist = [], [], []  # 存储历史数据

    # 反力计算公式：σ·n 在顶部边界的积分
    def sigma_calc(u):
        return 2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(2)

    traction_form = fem.form(
        ufl.dot(ufl.dot(g_d * sigma_calc(u), ufl.FacetNormal(msh)), ufl.as_vector((0.0, 1.0))) * ds_top)

    for i_step, disp_val in enumerate(load_steps_arr):
        if MPI.COMM_WORLD.rank == 0: print(f"--- Step {i_step + 1}/{n_steps}, Disp: {disp_val * 1000:.4f} mm ---")

        # 更新顶部边界位移（只施加 y 方向位移）
        top_disp.x.array[:] = 0.0
        top_disp.x.array[1::2] = disp_val

        # 交替迭代求解（staggered scheme）
        for sub_it in range(10):
            try:
                # Step A: 固定 d，求解 u
                num_iter, converged = solver_u.solve(u)
            except RuntimeError:
                if MPI.COMM_WORLD.rank == 0:
                    print(f"!!! Solver divergence at step {i_step + 1}, sub_it {sub_it}. Retrying with relaxation...")
                # 遇到困难时，临时减小松弛因子
                old_relax = solver_u.relaxation_parameter
                solver_u.relaxation_parameter = 0.5
                try:
                    solver_u.solve(u)
                except:
                    # 如果还不行，这一步可能崩了，跳出或报错
                    if MPI.COMM_WORLD.rank == 0: print("!!! Retry failed.")
                    raise
                finally:
                    solver_u.relaxation_parameter = old_relax  # 恢复

            # 2) 更新历史场 H = max(H_old, psi_plus(u))   ψ⁺(u)
            #    先把 psi_plus 投影到 Vd，再与 H_old 取最大值
            psi_expr = fem.Expression(psi_plus, Vd.element.interpolation_points)
            psi_fun = fem.Function(Vd)
            psi_fun.interpolate(psi_expr)

            H_array = H.x.array
            H_old_array = H_old.x.array
            psi_array = psi_fun.x.array

            H_array[:] = np.maximum(H_old_array, psi_array)
            H.x.array[:] = H_array
            H_old.x.array[:] = H_array  # 更新历史

            # 求解 D
            # Step C: 固定 u 和 H，求解 d
            num_it_d, converged_d = damage_solver.solve(d)

            # 投影
            d_vec = d.x.array
            d_vec[:] = np.maximum(d_vec, d_old.x.array)  # d 只能增加
            d_vec[:] = np.clip(d_vec, 0.0, 1.0)  # d ∈ [0, 1]
            d.x.array[:] = d_vec

        d_old.x.array[:] = d.x.array # 保存当前步作为下一步的"旧值"

        # 计算反力
        Ry = msh.comm.allreduce(fem.assemble_scalar(traction_form), op=MPI.SUM)
        reaction_hist.append(Ry)

        if MPI.COMM_WORLD.rank == 0:
            # 打印当前步的最大损伤，确保不是全场 1.0
            print(f"   Reaction: {Ry:.4e} N, Max D: {d.x.array.max():.4f}")

        if MPI.COMM_WORLD.rank == 0:
            u_hist.append(u.x.array.copy())
            d_hist.append(d.x.array.copy())

    # 8. 保存
    if MPI.COMM_WORLD.rank == 0:
        # 保存为 NPZ 文件（NumPy 压缩格式）
        np.savez(npz_path, coords_u=coords_d, coords_d=coords_d,
                 u=np.array(u_hist).T, d=np.array(d_hist).T,
                 load_steps=load_steps_arr, reactions=np.array(reaction_hist),
                 L=L_val, H=H_val, notch_length=notch_length, E=E_val, nu=nu_val, G_c=Gc_val, l=ell_val)
        print(f"✅ Saved to {npz_path}")

        # 保存为 XDMF 文件（可用 ParaView 可视化）
        with io.XDMFFile(msh.comm, xdmf_path, "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(u)
            xdmf.write_function(d)


if __name__ == "__main__":
    run_sent_phasefield_vertical(debug=False)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("当前时间戳:", timestamp)

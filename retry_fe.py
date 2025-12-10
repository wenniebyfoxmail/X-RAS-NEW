# sent_phasefield_fe.py

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import numpy as np
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem, NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

# 统一读取 config
from config import create_config, print_config


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

    L_val = float(cfg["L"])
    H_val = float(cfg["H"])
    notch_length = float(cfg["notch_length"])

    E_val = float(cfg["E"])
    nu_val = float(cfg["nu"])
    Gc_val = float(cfg["G_c"])
    ell_val = float(cfg["l"])

    n_steps = int(cfg["n_loading_steps"])
    max_disp = float(cfg["max_displacement"])

    nx_fe = int(cfg.get("nx_fe", 160))
    ny_fe = int(cfg.get("ny_fe", 160))

    # -------------------------
    # 2. 几何 & 网格
    # -------------------------
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        points=((0.0, 0.0), (L_val, H_val)),
        n=(nx_fe, ny_fe),
        cell_type=mesh.CellType.triangle,
    )

    # 定义边界标记，用于反力积分
    def top_boundary(x):
        return np.isclose(x[1], H_val)

    fdim = msh.topology.dim - 1
    top_facets = mesh.locate_entities_boundary(msh, fdim, top_boundary)

    # 标记 ID: 1
    top_tag = 1
    sorted_top = np.argsort(top_facets)
    mt = mesh.meshtags(msh, fdim, top_facets[sorted_top], np.full(len(top_facets), top_tag, dtype=np.int32))

    dx = ufl.Measure("dx", domain=msh)
    ds_top = ufl.Measure("ds", domain=msh, subdomain_data=mt, subdomain_id=top_tag)

    # -------------------------
    # 3. 函数空间
    # -------------------------
    Vu = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    Vd = fem.functionspace(msh, ("Lagrange", 1))

    u = fem.Function(Vu, name="u")
    du = ufl.TestFunction(Vu)

    d = fem.Function(Vd, name="d")
    v_d = ufl.TestFunction(Vd)
    d_trial = ufl.TrialFunction(Vd)

    # 历史场 H（与 d 同一空间，便于实现）
    H = fem.Function(Vd, name="H")
    H_old = fem.Function(Vd, name="H_old")
    H.x.array[:] = 0.0
    H_old.x.array[:] = 0.0

    # -------------------------
    # 4. 初始损伤 (Notch)
    # -------------------------
    coords_d = Vd.tabulate_dof_coordinates()
    x_d = coords_d[:, 0]
    y_d = coords_d[:, 1]

    initial_d = float(cfg["initial_d"])
    seed_radius = float(cfg["notch_seed_radius"])

    d_vec = d.x.array
    d_vec[:] = 0.0

    # 预制裂缝带
    line_mask = (x_d <= notch_length) & (np.abs(y_d - H_val / 2.0) <= seed_radius)
    d_vec[line_mask] = 1.0

    # 裂尖平滑过渡
    r_tip = np.sqrt((x_d - notch_length) ** 2 + (y_d - H_val / 2.0) ** 2)
    tip_mask = (x_d > notch_length - seed_radius) & (r_tip < 3.0 * seed_radius)
    d_gauss = initial_d * np.exp(-(r_tip[tip_mask] / seed_radius) ** 2)
    d_vec[tip_mask] = np.maximum(d_vec[tip_mask], d_gauss)

    d.x.array[:] = d_vec

    d_old = fem.Function(Vd)
    d_old.x.array[:] = d.x.array

    # -------------------------
    # 5. 能量与方程
    # -------------------------
    mu = E_val / (2.0 * (1.0 + nu_val))
    lam = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    kappa = 1e-5

    def eps(v):
        return ufl.sym(ufl.grad(v))

    eps_u = eps(u)
    tr_eps = ufl.tr(eps_u)
    eps_dev = eps_u - tr_eps / 2 * ufl.Identity(2)
    tr_plus = 0.5 * (tr_eps + ufl.sqrt(tr_eps ** 2 + 1e-14))
    tr_minus = tr_eps - tr_plus

    psi_plus = 0.5 * lam * tr_plus ** 2 + mu * ufl.inner(eps_dev, eps_dev)
    psi_minus = 0.5 * lam * tr_minus ** 2 + mu * ufl.inner(eps_dev, eps_dev)

    # U 的能量 (AT1/AT2 混合形式，主要用于求导)
    # 这里用简单的退化函数近似，足以驱动位移场
    g_d = (1.0 - d) ** 2 + kappa
    crack_energy = Gc_val * (d / ell_val + ell_val * ufl.inner(ufl.grad(d), ufl.grad(d)))
    energy = (g_d * psi_plus + psi_minus) * dx + crack_energy * dx

    F_u = ufl.derivative(energy, u, du)
    J_u = ufl.derivative(F_u, u)

    # # === D 的方程 (修正版) ===
    # # 只有左边(LHS)加稳定项，右边(RHS)保持物理原义
    # # 方程: (2*psi_plus + stabilizer)*d - diffusion = 2*psi_plus
    # # 仅在 LHS 加稳定项，防止 u=0 时矩阵奇异；RHS 保持为 0，确保无载荷时 d 不增长
    # psi_safe = ufl.max_value(psi_plus, 1e-5 * E_val)
    # a_d = (2.0 * psi_safe * d_trial * v_d + 2.0 * Gc_val * ell_val * ufl.inner(ufl.grad(d_trial), ufl.grad(v_d))) * dx
    # L_d = (2.0 * psi_plus * v_d) * dx

    # # 系数定义
    # c_damage = Gc_val / ell_val
    # c_diff = Gc_val * ell_val
    #
    # # LHS (左边): (Gc/l + 2*psi)*d - Gc*l*div(grad(d))
    # a_d = ((c_damage + 2.0 * psi_plus) * d_trial * v_d +
    #        c_diff * ufl.inner(ufl.grad(d_trial), ufl.grad(v_d))) * dx
    #
    # # RHS (右边): 2*psi
    # L_d = (2.0 * psi_plus * v_d) * dx

    # # === AT1 损伤方程 (带历史场 H) ===
    # # 强形式: -2(1-d)H + Gc/l - 2Gc l Δd = 0
    # # 弱形式（以 d 为未知）:
    # #   ∫ 2H d_trial v_d + 2Gc l ∇d_trial·∇v_d dx
    # # = ∫ (2H - Gc/l) v_d dx
    #
    # a_d = (
    #               2.0 * H * d_trial * v_d
    #               + 2.0 * Gc_val * ell_val * ufl.inner(ufl.grad(d_trial), ufl.grad(v_d))
    #       ) * dx
    #
    # L_d = ((2.0 * H - Gc_val / ell_val) * v_d) * dx

    # === AT1 Damage PDE ===
    # LHS = ∫ 2H * d_trial * v_d + 2Gc*l ∇d_trial·∇v_d dx
    # RHS = ∫ (2H - Gc/l)*v_d dx

    a_d = (
                  2.0 * H * d_trial * v_d
                  + 2.0 * Gc_val * ell_val * ufl.inner(ufl.grad(d_trial), ufl.grad(v_d))
          ) * dx

    L_d = ((2.0 * H - Gc_val / ell_val) * v_d) * dx

    # -------------------------
    # 6. 边界与求解器
    # -------------------------
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)

    dofs_bot = fem.locate_dofs_topological(Vu, fdim, mesh.locate_entities_boundary(msh, fdim, bottom_boundary))
    dofs_top = fem.locate_dofs_topological(Vu, fdim, top_facets)

    bc_bot = fem.dirichletbc(np.array((0.0, 0.0), dtype=ScalarType), dofs_bot, Vu)
    top_disp = fem.Function(Vu)
    bc_top = fem.dirichletbc(top_disp, dofs_top)
    bcs_u = [bc_bot, bc_top]

    # 【修复3】增强 U 求解器设置
    problem_u = NewtonSolverNonlinearProblem(F_u, u, bcs=bcs_u, J=J_u)
    solver_u = NewtonSolver(MPI.COMM_WORLD, problem_u)
    solver_u.convergence_criterion = "incremental"
    solver_u.rtol = 1e-6  # 稍微放宽一点相对误差
    solver_u.atol = 1e-8
    solver_u.max_it = 50
    solver_u.relaxation_parameter = 0.8  # 阻尼，防止震荡

    # 强制直接求解器 (LU/MUMPS)
    ksp = solver_u.krylov_solver
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    try:
        ksp.getPC().setFactorSolverType("mumps")
    except:
        pass

    # 线性求解d
    problem_d = LinearProblem(a_d, L_d, bcs=[], u=d,
                              petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                              petsc_options_prefix="d_solver")

    # # 非线性
    # damage_problem = fem.petsc.NonlinearProblem(F_d, d, bcs_d)
    # damage_solver = fem.petsc.NewtonSolver(comm, damage_problem)

    # 7. 循环
    load_steps_arr = np.linspace(0.0, max_disp, n_steps)
    u_hist, d_hist, reaction_hist = [], [], []

    # 反力辅助
    def sigma_calc(u):
        return 2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(2)

    traction_form = fem.form(
        ufl.dot(ufl.dot(g_d * sigma_calc(u), ufl.FacetNormal(msh)), ufl.as_vector((0.0, 1.0))) * ds_top)

    for i_step, disp_val in enumerate(load_steps_arr):
        if MPI.COMM_WORLD.rank == 0: print(f"--- Step {i_step + 1}/{n_steps}, Disp: {disp_val * 1000:.4f} mm ---")

        top_disp.x.array[:] = 0.0
        top_disp.x.array[1::2] = disp_val

        # 尝试求解，如果失败则尝试更多松弛
        for sub_it in range(10):
            try:
                # 求解 U
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

            # 2) 更新历史场 H = max(H_old, psi_plus(u))
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
            problem_d.solve()

            # 投影
            d_vec = d.x.array
            d_vec[:] = np.maximum(d_vec, d_old.x.array)
            d_vec[:] = np.clip(d_vec, 0.0, 1.0)
            d.x.array[:] = d_vec

        d_old.x.array[:] = d.x.array

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
        np.savez(npz_path, coords_u=coords_d, coords_d=coords_d,
                 u=np.array(u_hist).T, d=np.array(d_hist).T,
                 load_steps=load_steps_arr, reactions=np.array(reaction_hist),
                 L=L_val, H=H_val, notch_length=notch_length, E=E_val, nu=nu_val, G_c=Gc_val, l=ell_val)
        print(f"✅ Saved to {npz_path}")
        with io.XDMFFile(msh.comm, xdmf_path, "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(u)
            xdmf.write_function(d)


if __name__ == "__main__":
    run_sent_phasefield_vertical(debug=False)
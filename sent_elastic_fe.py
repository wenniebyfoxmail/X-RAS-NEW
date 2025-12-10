# sent_elastic_fe_y.py

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import numpy as np
import ufl

from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem


def sigma(u, mu, lam):
    """应力张量 σ(u)"""
    eps = ufl.sym(ufl.grad(u))
    return 2.0 * mu * eps + lam * ufl.tr(eps) * ufl.Identity(len(u))


def create_sent_mesh(Lx=1.0, Ly=1.0, nx=80, ny=80):
    """创建矩形网格"""
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (Lx, Ly)),
        n=(nx, ny),
        cell_type=mesh.CellType.triangle,
    )
    return msh


def tag_boundaries(msh, Lx, Ly):
    """
    [修改 1] 标记边界:
    - 底部 (y=0) 标记为 1
    - 顶部 (y=Ly) 标记为 2
    """
    tdim = msh.topology.dim
    fdim = tdim - 1

    def bottom(x):
        return np.isclose(x[1], 0.0)  # 检查 y 坐标是否为 0

    def top(x):
        return np.isclose(x[1], Ly)  # 检查 y 坐标是否为 Ly

    facets_bottom = mesh.locate_entities_boundary(msh, fdim, bottom)
    facets_top = mesh.locate_entities_boundary(msh, fdim, top)

    values = np.zeros(len(facets_bottom) + len(facets_top), dtype=np.int32)
    values[: len(facets_bottom)] = 1  # 1 代表底部
    values[len(facets_bottom):] = 2  # 2 代表顶部

    facets = np.hstack([facets_bottom, facets_top])

    # 注意：需确保 facets 索引排序，dolfinx 有时对未排序索引敏感，
    # 这里 mesh.meshtags 内部通常处理得当，但若出错可 argsort facets。
    arg_sort = np.argsort(facets)
    facet_tags = mesh.meshtags(msh, fdim, facets[arg_sort], values[arg_sort])
    return facet_tags


def run_sent_elastic_fe(
        Lx=1.0,
        Ly=1.0,
        nx=80,
        ny=80,
        E=210.0,
        nu=0.3,
        U_max=0.01,
        n_steps=10,
        npz_path="sent_fe_elastic_y.npz",
        xdmf_path="sent_elastic_displacement_y.xdmf",
):
    """
    平面应变 SENT 线弹性 FE 基准（Y 方向加载）：
    - 底部固支 (y=0)，顶部 (y=Ly) y 向位移加载
    """

    comm = MPI.COMM_WORLD

    # 1. 网格 & 有限元空间
    msh = create_sent_mesh(Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))

    dx = ufl.Measure("dx", domain=msh)
    facet_tags = tag_boundaries(msh, Lx, Ly)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    # 材料参数
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f_vec = fem.Constant(msh, np.array((0.0, 0.0), dtype=ScalarType))

    a = ufl.inner(sigma(u, mu, lam), ufl.grad(v)) * dx
    L_form = ufl.inner(f_vec, v) * dx

    # 2. Dirichlet 边界条件 [修改 2]

    # --- 底部固支 (Tag = 1) ---
    bottom_dofs = fem.locate_dofs_topological(
        V, msh.topology.dim - 1, facet_tags.find(1)
    )
    bc_bottom_value = np.array((0.0, 0.0), dtype=ScalarType)
    bc_bottom = fem.dirichletbc(bc_bottom_value, bottom_dofs, V)

    # --- 顶部位移加载 (Tag = 2) ---
    top_dofs = fem.locate_dofs_topological(
        V, msh.topology.dim - 1, facet_tags.find(2)
    )

    # 3. 存储
    displacements = []
    load_levels = []
    reactions = []

    xdmf = io.XDMFFile(comm, xdmf_path, "w")
    xdmf.write_mesh(msh)

    uh = fem.Function(V, name="u")
    n_vec = ufl.FacetNormal(msh)

    # [修改 3] 定义 Y 方向单位向量，用于计算反力
    ey = ufl.as_vector((0.0, 1.0))

    # 5. 加载步循环
    for step in range(1, n_steps + 1):
        U_t = U_max * step / n_steps
        load_levels.append(float(U_t))

        # [修改 4] 更新顶部位移：x=0, y=U_t
        bc_top_value = np.array((0.0, U_t), dtype=ScalarType)
        bc_top = fem.dirichletbc(bc_top_value, top_dofs, V)

        bcs = [bc_bottom, bc_top]

        problem = LinearProblem(
            a,
            L_form,
            bcs=bcs,
            petsc_options_prefix=f"sent_el_{step}_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "ksp_error_if_not_converged": True,
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()

        # [修改 5] 计算顶部反力 Ry = ∫_{Γ_top} (σ(u)n)·e_y ds
        # 注意 ds(2) 对应顶部
        ry_form = fem.form(ufl.dot(ufl.dot(sigma(uh, mu, lam), n_vec), ey) * ds(2))
        Ry = fem.assemble_scalar(ry_form)
        reactions.append(float(Ry))

        if comm.rank == 0:
            print(f"[Step {step}/{n_steps}] U_y = {U_t:.4e}, Ry = {Ry:.4e}")

        xdmf.write_function(uh, float(step))

        if step == n_steps:
            u_vec = uh.x.array.real.copy()
            displacements.append(u_vec)

    xdmf.close()

    coords = V.tabulate_dof_coordinates()

    if comm.rank == 0:
        displacements = np.array(displacements)
        load_levels = np.array(load_levels)
        reactions = np.array(reactions)

        np.savez(
            npz_path,
            coords=coords,
            u_last=displacements[-1],
            loads=load_levels,
            reactions=reactions,  # 这里存的是 Ry
        )
        print("FE SENT elastic (Y-pull) saved to", npz_path)
        print("  Max |u|:",
              float(np.sqrt(displacements[-1][0::2] ** 2 + displacements[-1][1::2] ** 2).max()))


if __name__ == "__main__":
    run_sent_elastic_fe()
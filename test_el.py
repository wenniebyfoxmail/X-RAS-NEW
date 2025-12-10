from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import numpy as np
import ufl

from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem

# ===========================
# 1. 几何与材料参数
# ===========================
Lx = 1.0   # 试样长度
Ly = 0.2   # 试样高度
nx = 80    # 网格划分（x）
ny = 16    # 网格划分（y）

E = 210e9      # 弹性模量
nu = 0.3       # 泊松比
U0 = 1e-3      # 右端 x 方向拉伸位移

# Lamé 参数（平面应变）
mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def sigma(u):
    """应力张量 σ(u)"""
    eps = ufl.sym(ufl.grad(u))
    return 2.0 * mu * eps + lam * ufl.tr(eps) * ufl.Identity(len(u))


# ===========================
# 2. 网格 & 向量位移空间
# ===========================
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (Lx, Ly)),
    n=(nx, ny),
    cell_type=mesh.CellType.triangle,
)

# 显式定义积分测度，绑定到 mesh
dx = ufl.Measure("dx", domain=msh)

# 向量空间：每个节点 2 自由度 (ux, uy)
V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# 体力：用 fem.Constant，明确绑定到 mesh
f_vec = fem.Constant(msh, np.array((0.0, 0.0), dtype=ScalarType))

a = ufl.inner(sigma(u), ufl.grad(v)) * dx
L = ufl.inner(f_vec, v) * dx


# ===========================
# 3. 边界条件：左端固支，右端位移加载
# ===========================
def left_boundary(x):
    return np.isclose(x[0], 0.0)


def right_boundary(x):
    return np.isclose(x[0], Lx)


# 左侧固支
facets_left = mesh.locate_entities_boundary(
    msh, dim=msh.topology.dim - 1, marker=left_boundary
)
dofs_left = fem.locate_dofs_topological(
    V, entity_dim=msh.topology.dim - 1, entities=facets_left
)
bc_left_value = np.array((0.0, 0.0), dtype=ScalarType)
bc_left = fem.dirichletbc(bc_left_value, dofs_left, V)

# 右侧位移加载
facets_right = mesh.locate_entities_boundary(
    msh, dim=msh.topology.dim - 1, marker=right_boundary
)
dofs_right = fem.locate_dofs_topological(
    V, entity_dim=msh.topology.dim - 1, entities=facets_right
)
bc_right_value = np.array((U0, 0.0), dtype=ScalarType)
bc_right = fem.dirichletbc(bc_right_value, dofs_right, V)

bcs = [bc_left, bc_right]


# ===========================
# 4. 线性问题组装 & 求解
# ===========================
problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    petsc_options_prefix="sent_el_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
    },
)

uh = problem.solve()
uh.x.scatter_forward()  # 并行同步（单核也安全）

u_vec = uh.x.array.real
coords = V.tabulate_dof_coordinates()

if msh.comm.rank == 0:
    print("FE solve done.")
    print("Total DOFs:", u_vec.size)
    disp_mag = np.sqrt(u_vec[0::2] ** 2 + u_vec[1::2] ** 2)
    print("Max |u|:", float(disp_mag.max()))

    # 存成 npz：给 X-RAS-PINN 用
    np.savez("sent_fe_baseline.npz", coords=coords, u=u_vec)
    print("Saved FE baseline to sent_fe_baseline.npz")

# ===========================
# 5. 写 XDMF（可选，用 ParaView 看形变）
# ===========================
with io.XDMFFile(msh.comm, "sent_displacement.xdmf", "w") as f:
    f.write_mesh(msh)
    f.write_function(uh)

if msh.comm.rank == 0:
    print("XDMF file written: sent_displacement.xdmf")

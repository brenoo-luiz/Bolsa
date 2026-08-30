import gmsh
import dolfinx
import dolfinx.fem.petsc
import ufl
import basix
import pyvista
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import gmsh as gmshio
import os

os.makedirs("outputs", exist_ok=True)

BG     = "#1e1e2e"
PANEL  = "#2a2a4a"
BORDER = "#4a4a6a"

# Parameters 
SPHERE_CENTER = np.array([0.0, 0.0, 0.0])
SPHERE_RADIUS = 0.35
GAMMA_IN      = 2.0
GAMMA_OUT     = 1.0

TOP_TAG     = 1
BOTTOM_TAG  = 2
LATERAL_TAG = 3
VOLUME_TAG  = 10

# Current patterns I_all: list of (g_top, g_bottom)
# integral(g) = Area_top * g_top + Area_bot * g_bot = pi - pi = 0 
I_all = [
    ( 1.0, -1.0),   # Pattern 1: inject top, extract bottom
]

# Mesh 
print("Generating mesh...")
gmsh.finalize()
gmsh.initialize()

cylinder = gmsh.model.occ.addCylinder(0, 0, -1, 0, 0, 2, 1)
gmsh.model.occ.synchronize()

top_surfs, bot_surfs, lat_surfs = [], [], []
for s in gmsh.model.getEntities(dim=2):
    bb    = gmsh.model.getBoundingBox(s[0], s[1])
    z_min, z_max = bb[2], bb[5]
    if z_min > 0.9:
        top_surfs.append(s[1])
    elif z_max < -0.9:
        bot_surfs.append(s[1])
    else:
        lat_surfs.append(s[1])

gmsh.model.addPhysicalGroup(2, top_surfs,  tag=TOP_TAG)
gmsh.model.addPhysicalGroup(2, bot_surfs,  tag=BOTTOM_TAG)
gmsh.model.addPhysicalGroup(2, lat_surfs,  tag=LATERAL_TAG)
gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(dim=3)],
                            tag=VOLUME_TAG)

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.02)
gmsh.option.setNumber("Mesh.Algorithm",      6)
gmsh.option.setNumber("Mesh.Algorithm3D",    1)
gmsh.option.setNumber("Mesh.Optimize",       1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize("Netgen")
gmsh.model.mesh.optimize("Relocate3D")

mesh_comm = MPI.COMM_WORLD
mesh_data  = gmshio.model_to_mesh(gmsh.model, mesh_comm, 0, gdim=3)
mesh       = mesh_data.mesh
facet_tags = mesh_data.facet_tags
gmsh.finalize()

n_cells = mesh.topology.index_map(3).size_global
n_verts = mesh.topology.index_map(0).size_global
print(f"  Cells: {n_cells}  Vertices: {n_verts}")

# Gamma (DG0) 
V0    = dolfinx.fem.functionspace(mesh, ("DG", 0))
gamma = dolfinx.fem.Function(V0)

tdim            = mesh.topology.dim
num_cells_local = mesh.topology.index_map(tdim).size_local
cells           = np.arange(num_cells_local, dtype=np.int32)
midpoints       = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)

gamma.x.array[:] = GAMMA_OUT
dist_sq = np.sum((midpoints - SPHERE_CENTER)**2, axis=1)
inside  = dist_sq < SPHERE_RADIUS**2
gamma.x.array[inside] = GAMMA_IN

n_inside = int(inside.sum())
print(f"  Sphere: {n_inside} cells inside (γ={GAMMA_IN}), "
        f"{num_cells_local - n_inside} outside (γ={GAMMA_OUT})")

# Function space P2 
Ve     = basix.ufl.element('Lagrange', 'tetrahedron', degree=2, shape=())
V      = dolfinx.fem.functionspace(mesh, Ve)
n_dofs = V.dofmap.index_map.size_global
print(f"  DOFs: {n_dofs}")

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

# Bilinear form (same for all patterns — assembled once)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(gamma * ufl.grad(u), ufl.grad(v)) * ufl.dx

# Assemble matrix once 
print("Assembling system matrix (once for all patterns)...")
a_form = dolfinx.fem.form(a)
A      = dolfinx.fem.petsc.assemble_matrix(a_form)
A.assemble()

# Null space — uniqueness (mathematically equiv. to Lagrange multiplier)
ns_vec = A.createVecLeft()
ns_vec.set(1.0)
ns_vec.normalize()
ns = PETSc.NullSpace().create(vectors=[ns_vec], comm=mesh_comm)
A.setNullSpace(ns)
A.setTransposeNullSpace(ns)

# Solver (CG + HYPRE) 
solver = PETSc.KSP().create(mesh_comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)
solver.getPC().setType(PETSc.PC.Type.HYPRE)
solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
solver.setFromOptions()

# RHS vector created once and reused (no reallocation per pattern)
b = A.createVecRight()

# Solve for each current pattern (I_all — Marcelo's structure)
print(f"Solving {len(I_all)} pattern(s)...")
solutions = []

for i, (g_top, g_bot) in enumerate(I_all):
    # RHS: g_top on top, g_bot on bottom, 0 on lateral
    L      = g_top * v * ds(TOP_TAG) + g_bot * v * ds(BOTTOM_TAG)
    L_form = dolfinx.fem.form(L)

    with b.localForm() as loc_b:
        loc_b.set(0)
    dolfinx.fem.petsc.assemble_vector(b, L_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    ns.remove(b)

    u_h = dolfinx.fem.Function(V)
    solver.solve(b, u_h.x.petsc_vec)
    u_h.x.scatter_forward()

    u_min = float(u_h.x.array.min())
    u_max = float(u_h.x.array.max())
    print(f"  Pattern {i+1}  (g_top={g_top:+.1f}, g_bot={g_bot:+.1f}): "
            f"converged in {solver.getIterationNumber()} iters  "
            f"u=[{u_min:.4f}, {u_max:.4f}]")
    solutions.append(u_h)

# Use first solution for visualization
u_h    = solutions[0]
u_min  = float(u_h.x.array.min())
u_max  = float(u_h.x.array.max())
clim_u = [u_min, u_max]


# Visualizacao interativa 
topo_u, ct_u, geo_u = dolfinx.plot.vtk_mesh(V)
grid_u  = pyvista.UnstructuredGrid(topo_u, ct_u, geo_u)
grid_u["u"] = u_h.x.array.real
clim_u  = [float(u_h.x.array.min()), float(u_h.x.array.max())]
surface = grid_u.extract_surface(algorithm="dataset_surface")

clip_u = grid_u.clip(normal="x", origin=(0, 0, 0))
theta  = __import__("numpy").linspace(0, 2*3.14159, 200)
circle_pts = __import__("numpy").column_stack([
    __import__("numpy").zeros(200),
    SPHERE_RADIUS * __import__("numpy").cos(theta),
    SPHERE_RADIUS * __import__("numpy").sin(theta),
])
circle_pd = pyvista.Spline(circle_pts, n_points=200)

sphere_mesh = pyvista.Sphere(radius=SPHERE_RADIUS, center=SPHERE_CENTER.tolist(),
                            theta_resolution=60, phi_resolution=60)
cyl_surf    = pyvista.Cylinder(center=(0,0,0), direction=(0,0,1),
                                radius=1.0, height=2.0,
                                resolution=100, capping=True).extract_surface()

pl = pyvista.Plotter(shape=(1, 3), window_size=(1800, 700))

pl.subplot(0, 0)
pl.add_text("Conductivity gamma", font_size=12, color="white")
pl.add_mesh(cyl_surf, color="lightblue", opacity=0.3,
            show_edges=False, lighting=True)
pl.add_mesh(sphere_mesh, color="red", opacity=0.95,
            show_edges=False, lighting=True, smooth_shading=True)
pl.set_background("#1e1e2e")
pl.add_axes()

pl.subplot(0, 1)
pl.add_text("Solution u", font_size=12, color="white")
pl.add_mesh(surface, scalars="u", cmap="turbo", clim=clim_u,
            show_scalar_bar=True, show_edges=False,
            lighting=True, smooth_shading=True)
pl.set_background("#1e1e2e")
pl.add_axes()

pl.subplot(0, 2)
pl.add_text("Cross-section at x=0", font_size=12, color="white")
pl.add_mesh(clip_u, scalars="u", cmap="turbo", clim=clim_u,
            show_scalar_bar=False, show_edges=False,
            lighting=True, smooth_shading=True)
pl.add_mesh(circle_pd, color="white", line_width=3)
pl.set_background("#1e1e2e")
pl.add_axes()
pl.camera_position = [(5,0,0),(0,0,0),(0,0,1)]

pl.link_views()
pl.show()

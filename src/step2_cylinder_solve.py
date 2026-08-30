"""
Step 2 — Problema Direto de EIT 3D

Equação diferencial parcial (EDP):
    div(γ ∇u) = 0    em Ω (cilindro)

Condição de contorno de Neumann (corrente elétrica física):
    γ ∂u/∂n = g    em ∂Ω

    g = +1   base superior  (z = +1)
    g = -1   base inferior  (z = -1)
    g =  0   superfície lateral

    → ∫_{∂Ω} g ds = π - π = 0  (condição de existência)

Condutividade não-homogênea:
    γ = 2   dentro da esfera (inclusão)
    γ = 1   fora da esfera   (fundo)
    0.1 ≤ γ ≤ 20  (condições de Lax-Milgram)

Unicidade da solução:
    Remoção do espaço nulo (null space) via PETSc
    → impõe ∫_{∂Ω} u ds = 0

Forma variacional:
    a(u, v) = ∫_Ω γ ∇u · ∇v dx
    L(v)    = ∫_{topo} v ds  −  ∫_{base} v ds

Estrutura I_all:
    Lista de padrões de corrente
    A matriz A é montada uma única vez e reutilizada para todos os padrões,
"""

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

# Cores para visualização
BG     = "#1e1e2e"   
PANEL  = "#2a2a4a"   
BORDER = "#4a4a6a"   

# Parâmetros do problema 
SPHERE_CENTER = np.array([0.0, 0.0, 0.0])  # centro da esfera (origem)
SPHERE_RADIUS = 0.35                         # raio da esfera
GAMMA_IN      = 2.0    # condutividade dentro da esfera (inclusão)
GAMMA_OUT     = 1.0    # condutividade fora da esfera (fundo)

# Tags das superfícies físicas no Gmsh
TOP_TAG     = 1   # base superior (z ≈ +1)
BOTTOM_TAG  = 2   # base inferior (z ≈ -1)
LATERAL_TAG = 3   # superfície lateral curva
VOLUME_TAG  = 10  # volume interior

# Padrões de corrente
# Cada entrada é (g_topo, g_base):
#   g_topo   = corrente na base superior
#   g_base   = corrente na base inferior
# Condição de existência: ∫g ds = Área_topo * g_topo + Área_base * g_base
#                               = π * 1 + π * (-1) = 0  ✓
I_all = [
    ( 1.0, -1.0),   # Padrão 1: injeta +1 no topo, extrai -1 na base
]

# GERAÇÃO DA MALHA
# O cilindro tem raio 1, altura 2, centrado na origem (z ∈ [-1, +1]).
# As superfícies são marcadas com tags físicas para que a integração de
# contorno ds(TOP_TAG), ds(BOTTOM_TAG) funcione corretamente.
print("Gerando malha...")
gmsh.finalize()    
gmsh.initialize()

# Cria o cilindro: addCylinder(x, y, z_base, dx, dy, altura, raio)
cylinder = gmsh.model.occ.addCylinder(0, 0, -1, 0, 0, 2, 1)
gmsh.model.occ.synchronize()   # sincroniza geometria com o modelo

# Identifica cada superfície pelo bounding box (caixa delimitadora)
# e classifica em: topo (z≈+1), base (z≈-1) ou lateral
top_surfs, bot_surfs, lat_surfs = [], [], []
for s in gmsh.model.getEntities(dim=2):
    bb    = gmsh.model.getBoundingBox(s[0], s[1])
    z_min, z_max = bb[2], bb[5]
    if z_min > 0.9:            # superfície cujos pontos estão todos em z ≈ +1
        top_surfs.append(s[1])
    elif z_max < -0.9:         # superfície cujos pontos estão todos em z ≈ -1
        bot_surfs.append(s[1])
    else:                      # superfície lateral (z varia de -1 a +1)
        lat_surfs.append(s[1])

# Registra grupos físicos
gmsh.model.addPhysicalGroup(2, top_surfs,  tag=TOP_TAG)
gmsh.model.addPhysicalGroup(2, bot_surfs,  tag=BOTTOM_TAG)
gmsh.model.addPhysicalGroup(2, lat_surfs,  tag=LATERAL_TAG)
gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(dim=3)],
                            tag=VOLUME_TAG)

# Parâmetros de tamanho dos elementos da malha
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.02)

# Algoritmo 2D: 6 = Frontal-Delaunay
# Algoritmo 3D: 1 = Delaunay
gmsh.option.setNumber("Mesh.Algorithm",      6)
gmsh.option.setNumber("Mesh.Algorithm3D",    1)

# Otimizações de qualidade da malha
gmsh.option.setNumber("Mesh.Optimize",       1)   
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)   

# Gera a malha 3D e aplica dois passes de otimização
gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize("Netgen")       # melhora ângulos dos tetraedros
gmsh.model.mesh.optimize("Relocate3D")  # reposiciona nós para melhor qualidade

# Converte a malha do Gmsh para o formato DOLFINx
# gdim=3: problema em 3 dimensões
# facet_tags: contém as tags TOP_TAG, BOTTOM_TAG, LATERAL_TAG para cada face
mesh_comm = MPI.COMM_WORLD
mesh_data  = gmshio.model_to_mesh(gmsh.model, mesh_comm, 0, gdim=3)
mesh       = mesh_data.mesh         
facet_tags = mesh_data.facet_tags   # marcadores de superfície
gmsh.finalize()

n_cells = mesh.topology.index_map(3).size_global
n_verts = mesh.topology.index_map(0).size_global
print(f"  Células (tetraedros): {n_cells}  Vértices: {n_verts}")

# DEFINIÇÃO DA CONDUTIVIDADE γ
# DG0 = Galerkin descontínuo de grau 0 = constante por célula
# Ideal para γ não-homogênea que é constante por partes (γ=2 na esfera, γ=1 fora)
#
# Processo:
#   1. Calcula o centroide (midpoint) de cada tetraedro
#   2. Se o centroide está dentro da esfera → γ = 2
#   3. Se está fora → γ = 1
V0    = dolfinx.fem.functionspace(mesh, ("DG", 0))
gamma = dolfinx.fem.Function(V0)

tdim            = mesh.topology.dim                          # dimensão = 3
num_cells_local = mesh.topology.index_map(tdim).size_local   
cells           = np.arange(num_cells_local, dtype=np.int32)

# Calcula os centroides de todos os tetraedros
midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)

# Inicializa toda a condutividade com o valor de fundo
gamma.x.array[:] = GAMMA_OUT

# Identifica células cujo centroide está dentro da esfera: ||x - centro||² < r²
dist_sq = np.sum((midpoints - SPHERE_CENTER)**2, axis=1)
inside  = dist_sq < SPHERE_RADIUS**2
gamma.x.array[inside] = GAMMA_IN  # atribui γ=2 às células dentro da esfera

n_inside = int(inside.sum())
print(f"  Esfera: {n_inside} células dentro (γ={GAMMA_IN}), "
        f"{num_cells_local - n_inside} fora (γ={GAMMA_OUT})")

# ESPAÇO DE FUNÇÕES P2 (Lagrange de grau 2)
# P2 = polinômios de grau 2 por tetraedro
# Mais preciso que P1 (grau 1): captura melhor gradientes suaves da solução u
# Grau 2 significa: nó em cada vértice + nó no meio de cada aresta do tetraedro
# Isso aumenta o número de DOFs (~284k) mas melhora significativamente a precisão
Ve     = basix.ufl.element('Lagrange', 'tetrahedron', degree=2, shape=())
V      = dolfinx.fem.functionspace(mesh, Ve)
n_dofs = V.dofmap.index_map.size_global
print(f"  Graus de liberdade (DOFs): {n_dofs}")

# MEDIDAS DE INTEGRAÇÃO DE CONTORNO
# ds(TOP_TAG)    → integra apenas sobre a base superior
# ds(BOTTOM_TAG) → integra apenas sobre a base inferior
# ds(LATERAL_TAG)→ integra apenas sobre a lateral (não usado no RHS pois g=0)
# Isso permite implementar g = +1 no topo, -1 na base, 0 na lateral
# sem precisar de expressões condicionais complexas
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

# FORMA VARIACIONAL 
# A forma fraca é obtida multiplicando a EDP por uma função teste v ∈ H¹(Ω)
# e integrando por partes (identidade de Green):
#
#   a(u, v) = ∫_Ω γ ∇u · ∇v dx          (forma bilinear)
#   L(v)    = ∫_{topo} v ds - ∫_{base} v ds  (funcional linear = RHS)
#
# Esta forma é válida para qualquer padrão de corrente g_top, g_bot.
# A matriz A (montada a partir de a) não depende de g → montada UMA VEZ.
# O vetor b (montado a partir de L) depende de g → montado POR PADRÃO.
u = ufl.TrialFunction(V)   # função incógnita (u que queremos encontrar)
v = ufl.TestFunction(V)    # função teste (usada para "testar" a equação)

# Forma bilinear: ∫_Ω γ ∇u · ∇v dx
# Nota: quando γ=1 em todo lugar, isso vira o laplaciano ∫ ∇u·∇v dx
a = ufl.inner(gamma * ufl.grad(u), ufl.grad(v)) * ufl.dx

# MONTAGEM DA MATRIZ DO SISTEMA
# A matriz A representa a discretização da forma bilinear a(u,v).
# Como A não depende de g (corrente), ela é montada UMA VEZ e
# reutilizada para todos os padrões de corrente em I_all.
print("Montando matriz do sistema...")
a_form = dolfinx.fem.form(a)              # compila a forma simbólica
A      = dolfinx.fem.petsc.assemble_matrix(a_form)  # monta a matriz esparsa
A.assemble()                               # finaliza a montagem 

# UNICIDADE DA SOLUÇÃO — Remoção do Espaço Nulo (Null Space)
# O problema de Neumann puro (∫g ds = 0) tem infinitas soluções:
# se u é solução, então u + C também é solução para qualquer constante C.
# Para garantir unicidade, impõe-se que ∫_{∂Ω} u ds = 0.
#
# Aqui isso é feito via remoção do espaço nulo do PETSc:
#   - O espaço nulo de A é span{1} (funções constantes)
#   - Ao informar esse espaço nulo ao solver, ele garante
#     que a solução é ortogonal a constantes → integral zero
#
# Isso é MATEMATICAMENTE EQUIVALENTE ao multiplicador de Lagrange do Marcelo
# (que adiciona uma variável extra c ao sistema para impor a mesma condição).
# A diferença é só de implementação: o null space é mais simples no DOLFINx.
ns_vec = A.createVecLeft()    # cria vetor compatível com A
ns_vec.set(1.0)               # preenche com 1 (representa a função constante)
ns_vec.normalize()            # normaliza para ter norma unitária
ns = PETSc.NullSpace().create(vectors=[ns_vec], comm=mesh_comm)
A.setNullSpace(ns)            # informa ao PETSc o espaço nulo de A
A.setTransposeNullSpace(ns)   # necessário para sistemas simétricos

# CONFIGURAÇÃO DO SOLVER LINEAR (CG + HYPRE)
# CG (Gradiente Conjugado): solver iterativo eficiente para sistemas
# simétricos positivos definidos
# HYPRE: pré-condicionador algébrico multigrid (AMG)
#
# Nota: o LU direto (como o Marcelo usa) seria mais robusto mas impraticável
# para 284k DOFs — a fatoração LU de uma matriz esparsa 3D nesse tamanho
# demora muito e consome grande quantidade de memória.
solver = PETSc.KSP().create(mesh_comm)
solver.setOperators(A)                           # define A como matriz do sistema
solver.setType(PETSc.KSP.Type.CG)               # solver: Gradiente Conjugado
solver.getPC().setType(PETSc.PC.Type.HYPRE)     # pré-condicionador: HYPRE AMG
solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)  # critérios de parada
solver.setFromOptions()                          # permite override via linha de cmd

# Vetor do lado direito (b) criado UMA VEZ e reutilizado para cada padrão
# Evita realocação de memória a cada iteração do loop I_all
b = A.createVecRight()

# LOOP SOBRE OS PADRÕES DE CORRENTE (estrutura I_all do Marcelo)
# Para cada padrão de corrente (g_top, g_bot) em I_all:
#   1. Monta o vetor b (lado direito) com a corrente específica
#   2. Remove a componente no espaço nulo de b (garante compatibilidade)
#   3. Resolve A·u = b
#   4. Armazena a solução u_h
#
# A matriz A é REUTILIZADA em todos os padrões
# Apenas b é remontado para cada padrão diferente.
print(f"Resolvendo {len(I_all)} padrão(ões) de corrente...")
solutions = []   # lista para armazenar as soluções de cada padrão

for i, (g_top, g_bot) in enumerate(I_all):

    # Funcional linear L(v): lado direito do sistema variacional
    # Implementa g = g_top no topo, g_bot na base, 0 na lateral
    L      = g_top * v * ds(TOP_TAG) + g_bot * v * ds(BOTTOM_TAG)
    L_form = dolfinx.fem.form(L)

    # Zera o vetor b e remonta para o padrão atual (sem realocar memória)
    with b.localForm() as loc_b:
        loc_b.set(0)
    dolfinx.fem.petsc.assemble_vector(b, L_form)

    # Sincronização MPI: soma contribuições de processos vizinhos nas interfaces
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Remove componente do espaço nulo de b (condição de compatibilidade)
    # Garante que o sistema Ax=b tem solução (b deve ser ortogonal ao null space)
    ns.remove(b)

    # Resolve o sistema linear A·u = b
    u_h = dolfinx.fem.Function(V)
    solver.solve(b, u_h.x.petsc_vec)
    u_h.x.scatter_forward()   # sincroniza valores nos processos MPI

    u_min = float(u_h.x.array.min())
    u_max = float(u_h.x.array.max())
    print(f"  Padrão {i+1}  (g_topo={g_top:+.1f}, g_base={g_bot:+.1f}): "
            f"convergiu em {solver.getIterationNumber()} iterações  "
            f"u=[{u_min:.4f}, {u_max:.4f}]")
    solutions.append(u_h)

# Usa o primeiro padrão para visualização
u_h    = solutions[0]
u_min  = float(u_h.x.array.min())
u_max  = float(u_h.x.array.max())
clim_u = [u_min, u_max]   # intervalo de cores para os plots

# VISUALIZAÇÃO
topo_u, ct_u, geo_u = dolfinx.plot.vtk_mesh(V)
grid_u  = pyvista.UnstructuredGrid(topo_u, ct_u, geo_u)
grid_u["u"] = u_h.x.array.real   # associa os valores de u a cada nó

# Extrai apenas a superfície exterior (não mostra interior)
surface = grid_u.extract_surface(algorithm="dataset_surface")

# Configuração da barra de cores
sargs_u = dict(
    title="u", title_font_size=20, label_font_size=16,
    color="white", position_x=0.03, position_y=0.03,
    width=0.38, height=0.05,
)

# RENDER: Condutividade γ (cilindro transparente + esfera)
# Mostra a geometria do problema: cilindro semi-transparente com a esfera
# (inclusão de γ=2) visível no interior.
# A esfera é renderizada geometricamente (pyvista.Sphere) para aparecer
print("Renderizando condutividade γ...")

sphere_mesh = pyvista.Sphere(radius=SPHERE_RADIUS,
                            center=SPHERE_CENTER.tolist(),
                            theta_resolution=60, phi_resolution=60)

# Cilindro geométrico semi-transparente
cyl_surf    = pyvista.Cylinder(center=(0,0,0), direction=(0,0,1),
                                radius=1.0, height=2.0,
                                resolution=100, capping=True).extract_surface()

p1 = pyvista.Plotter(off_screen=True, window_size=(900, 900))
p1.add_mesh(cyl_surf, color="#1a3a6a", opacity=0.45,
            show_edges=False, lighting=True, smooth_shading=True)
p1.add_mesh(sphere_mesh, color="#cc3333", opacity=0.95,
            show_edges=False, lighting=True, smooth_shading=True)
p1.add_text("γ = 2 (esfera)  /  γ = 1 (fora)",
            position="lower_left", font_size=12, color="white")
p1.set_background(BG)
p1.view_isometric()
p1.camera.zoom(1.1)
p1.screenshot("outputs/_tmp_gamma.png")
p1.close()

# RENDER: Solução u na superfície do cilindro
# Exibe o potencial elétrico u na superfície externa do cilindro.
# Colormap "turbo": azul = baixo potencial, vermelho = alto potencial.
# O gradiente vertical reflete a corrente: +1 no topo → alto potencial,
# -1 na base → baixo potencial.
print("Renderizando solução u...")
p2 = pyvista.Plotter(off_screen=True, window_size=(900, 900))
p2.add_mesh(surface, scalars="u", cmap="turbo", clim=clim_u,
            show_edges=False, lighting=True, smooth_shading=True,
            show_scalar_bar=True, scalar_bar_args=sargs_u)
p2.set_background(BG)
p2.view_isometric()
p2.screenshot("outputs/_tmp_u.png")
p2.close()

# 13. RENDER: Seção transversal em x=0 com contorno da esfera
# Corta o cilindro no plano x=0 para visualizar a distribuição interna de u.
# O círculo branco marca o contorno da esfera (inclusão de γ=2).
# A perturbação visível no campo u ao redor da esfera é a "assinatura" da
# inclusão — exatamente o sinal que o EIT tenta detectar nas medições.
print("Renderizando seção transversal...")

# Corte no plano x=0 (normal = eixo X, passando pela origem)
clip_u    = grid_u.clip(normal="x", origin=(0, 0, 0))

# Círculo no plano YZ (x=0) marcando o contorno da esfera
theta     = np.linspace(0, 2*np.pi, 200)
circle_pts = np.column_stack([
    np.zeros(200),                       # x = 0 (no plano do corte)
    SPHERE_RADIUS * np.cos(theta),       # y = r·cos(θ)
    SPHERE_RADIUS * np.sin(theta),       # z = r·sin(θ)
])
circle_pd = pyvista.Spline(circle_pts, n_points=200)

p3 = pyvista.Plotter(off_screen=True, window_size=(900, 900))
p3.add_mesh(clip_u, scalars="u", cmap="turbo", clim=clim_u,
            show_edges=False, lighting=True, smooth_shading=True,
            show_scalar_bar=False)
p3.add_mesh(circle_pd, color="white", line_width=3, opacity=0.9)
p3.set_background(BG)
# Câmera olhando de frente para o plano YZ (direção +X)
p3.camera_position = [(5, 0, 0), (0, 0, 0), (0, 0, 1)]
p3.camera.zoom(1.5)
p3.screenshot("outputs/_tmp_clip.png")
p3.close()

# COMPOSIÇÃO DA IMAGEM FINAL
# Combina os 3 renders do PyVista numa imagem final lado a lado.
# Cada painel recebe um título descritivo.
print("Compondo imagem final...")
img_gamma = np.array(Image.open("outputs/_tmp_gamma.png"))
img_u     = np.array(Image.open("outputs/_tmp_u.png"))
img_clip  = np.array(Image.open("outputs/_tmp_clip.png"))

fig = plt.figure(figsize=(22, 8), facecolor=BG)
gs  = gridspec.GridSpec(1, 3, figure=fig,
                        hspace=0.01, wspace=0.03,
                        left=0.02, right=0.98,
                        top=0.93, bottom=0.01)

for i, (img, title) in enumerate([
    (img_gamma, "Condutividade γ  (inclusão esférica)"),
    (img_u,     "Solução u no cilindro"),
    (img_clip,  "Seção transversal em x=0"),
]):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img)
    ax.axis("off")
    ax.set_facecolor(BG)
    ax.set_title(title, color="white", fontsize=14, pad=8)

plt.savefig("outputs/step2_cylinder_solution.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

# Remove arquivos temporários usados na composição
for p in ["outputs/_tmp_gamma.png", "outputs/_tmp_u.png", "outputs/_tmp_clip.png"]:
    os.remove(p)

# LIMPEZA DE OBJETOS PETSc
# Libera memória dos objetos PETSc explicitamente.
# O Python/PETSc às vezes não libera corretamente em scripts longos.
A.destroy()
b.destroy()
ns_vec.destroy()
solver.destroy()

print("Salvo: outputs/step2_cylinder_solution.png")
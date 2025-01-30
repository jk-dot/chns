import firedrake as fd
from firedrake.utility_meshes import RectangleMesh
from firedrake.output import VTKFile


# Function for barycentric refinement (Alfeld's split)
def refine_bary(coarse_mesh):
    """Return barycentric refinement of given input mesh"""
    coarse_dm = coarse_mesh.topology_dm
    transform = fd.PETSc.DMPlexTransform().create(comm=coarse_dm.getComm())
    transform.setType(fd.PETSc.DMPlexTransformType.REFINEALFELD)
    transform.setDM(coarse_dm)
    transform.setUp()
    fine_dm = transform.apply(coarse_dm)
    fine_mesh = fd.Mesh(fine_dm)
    return fine_mesh

# Load the mesh (replace "torus.msh" with your mesh file)
Lx, Ly = 1.0, 3.0
nx, ny = 10, 30
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)

# Check the number of points and cells in the original mesh
print("Original Mesh:")
print(f"Number of vertices: {mesh.num_vertices()}")
print(f"Number of cells: {mesh.num_cells()}")
# Save the original mesh to a .pvd file
original_mesh_file = VTKFile("output/original_mesh.pvd")
original_mesh_file.write(mesh)

# Refine the mesh using Alfeld's split
mesh = refine_bary(mesh)

# Check the number of points and cells in the refined mesh
print("\nRefined Mesh:")
print(f"Number of vertices: {mesh.num_vertices()}")
print(f"Number of cells: {mesh.num_cells()}")

# Save the refined mesh to a .pvd file
refined_mesh_file = VTKFile("output/refined_mesh.pvd")
refined_mesh_file.write(mesh)

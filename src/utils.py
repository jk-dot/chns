import firedrake as fd


def refine_bary(coarse_mesh):
    """Return barycentric refinement of given input mesh."""
    coarse_dm = coarse_mesh.topology_dm
    transform = fd.PETSc.DMPlexTransform().create(comm=coarse_dm.getComm())
    transform.setType(fd.PETSc.DMPlexTransformType.REFINEALFELD)
    transform.setDM(coarse_dm)
    transform.setUp()
    fine_dm = transform.apply(coarse_dm)
    return fd.Mesh(fine_dm)

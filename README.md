
# Self-propelled droplet

Possibly a simulation of a self-propelling droplet governed by the Cahn-Hilliard-Navier-Stokes equations on a torus, i.e. a two-dimensional structure. To conform with the CHNS equations, we shall use the Scott-Vogelius pair, with some adaptive mesh refinement. The droplet will be self-propelled by a gradient in the chemical potential, which will be implemented as a source term in the Navier-Stokes equations. The droplet will be confined to a torus, which will be implemented either as a periodic boundary condition or as a three-dimensional object.


TODO:
    **report**
        - [ ] literature + lit into readme?


    **src**
        - [ ] error estimation and mesh adaptivity

        - [x] pressure-robust methods
        - [ ] possible comparison with some normal method like Taylor-Hood

        - [ ] mesh could be a torus, however start with a rectangle,

        - [ ] computation usign PETSc parallelization on GPU?
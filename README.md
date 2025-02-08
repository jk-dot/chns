
# Self-propelled droplet

Possibly a simulation of a self-propelling droplet governed by the Cahn-Hilliard-Navier-Stokes equations on a torus, i.e. a two-dimensional structure. To conform with the CHNS equations, we shall use the Scott-Vogelius pair, with some adaptive mesh refinement. The droplet will be self-propelled by a gradient in the chemical potential, which will be implemented as a source term in the Navier-Stokes equations. The droplet will be confined to a torus, which will be implemented either as a periodic boundary condition or as a three-dimensional object.

<p align="center">
  <video width="45%" controls>
    <source src="./report/graphics/bublina.mp4" type="video/mp4">
  </video>
  <video width="45%" controls>
    <source src="./report/graphics/bubliny.mp4" type="video/mp4">
  </video>
</p>




# TODO
- [ ] error estimation and mesh adaptivity
- [x] pressure-robust methods

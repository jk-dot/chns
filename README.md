
# Self-propelled droplet

Possibly a simulation of a self-propelling droplet governed by the Cahn-Hilliard-Navier-Stokes equations on a torus, i.e. a two-dimensional structure. To conform with the CHNS equations, we shall use the Scott-Vogelius pair, with some adaptive mesh refinement. The droplet will be self-propelled by a gradient in the chemical potential, which will be implemented as a source term in the Navier-Stokes equations. The droplet will be confined to a torus, which will be implemented either as a periodic boundary condition or as a three-dimensional object.

<p align="center">
  <img width="45%" src="https://github.com/jk-dot/chns/blob/main/report/graphics/bublina.gif" alt="bublina">
  <img width="45%" src="https://github.com/jk-dot/chns/blob/main/report/graphics/bubliny.gif" alt="bubliny">
</p>


# TODO
- [ ] error estimation and mesh adaptivity
- [x] pressure-robust methods

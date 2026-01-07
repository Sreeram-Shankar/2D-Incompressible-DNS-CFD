# Julia Incompressible DNS (2D Cylinder Flow)

This repository contains a **from-scratch implementation of a 2-D incompressible Navier–Stokes solver** written entirely in **Julia**, including both the numerical backend and a native Julia-based frontend.

The project was built primarily as a **learning and experimentation platform** for numerical methods in CFD: projection methods, multigrid, Krylov solvers, and time integration schemes. While it produces physically plausible results for moderate Reynolds numbers, it is **not intended to be a production CFD code**.

---

## Overview

The solver computes incompressible flow on a Cartesian grid using a **projection method**:

1. Explicit time integration of the momentum equations  
2. Solution of a pressure Poisson equation  
3. Projection of the velocity field to enforce incompressibility  

A circular cylinder is represented using a simple masked immersed-boundary approach, and the code supports continuous inflow with an open outlet.

The emphasis of the project is **numerical stability, solver correctness, and architectural clarity**, rather than ultimate physical fidelity or performance tuning.

---

## Features

### Fluid Solver
- 2-D incompressible Navier–Stokes equations
- Projection method with pressure correction
- Cartesian grid with ghost cells
- Masked immersed boundary for circular obstacles
- Continuous inflow and open outlet
- Stable for moderate Reynolds numbers (e.g. Re ≲ O(10³) on coarse grids)

### Pressure Solvers
- Standalone Weighted Jacobi (fully parralel)
- Standalone Red Black Gauss-Seidel (semi-parralel)
- Standalone Chebyshev (fully parralel)
- Standalone Multigrid (W-cycle using smoother choice)
- Preconditioned Conjugate Gradient (PCG)
- Flexible Generalized Minimal Residual (FGMRES)
- Preconditioned Biconjugate Gradient Stabilized (PBiCGSTAB)
- Presents options between raw relaxation, raw multigrid, and multigrid-preconditioned krylov
- Pressure operator implemented consistently between all solvers

### Time Integration
- Explicit Runge–Kutta methods
- SSPRK schemes
- SDIRK and IRK methods (fixed-point iteration, no LU factorization)
- Fixed time step (no adaptive stepping yet)

### Frontend & Visualization
- Native **Julia / GLMakie** frontend
- Interactive parameter selection
- Real-time visualization of velocity and derived fields
- ParaView-compatible output for offline visualization

---

## Repository Structure

```
.
├── project.toml       # Project Packages
├── dns_backend.jl     # Core DNS solver 
├── multigrid.jl       # Multigrid hierarchy and smoothers
├── cg.jl              # Preconditioned Conjugate Gradient solver
├── fgmres.jl          # Flexible GMRES solver
├── bicgstab.jl        # Preconditioned BiCGSTAB solver
├── rk.jl              # Explicit Runge–Kutta methods
├── ssprk.jl           # SSPRK time integrators
├── sdirk.jl           # SDIRK fixed point methods 
├── irk.jl             # Implicit Runge–Kutta fixed point methods
└── dns_frontend.jl    # GLMakie-based frontend

```

---

## Example Problem

A typical test case is **flow past a circular cylinder**:

- Uniform inflow on the left boundary  
- No-slip walls (top/bottom)  
- Open outlet on the right  
- Cylinder represented via velocity masking  

At Reynolds numbers around **Re ≈ 100–300**, the solver produces:
- Acceleration over the cylinder
- Boundary-layer separation
- Wake formation
- Onset of unsteady behavior in 2-D

Results are intended to be **qualitatively correct**, not quantitatively validated against benchmark data.

---

## Known Limitations (Important)

This project makes several **intentional simplifications**:

- Pressure is currently imposed with **Dirichlet boundary conditions everywhere**  
  - This improves numerical stability and solver robustness
  - It slightly constrains pressure recovery near boundaries
- The immersed boundary treatment is simple masking (no forcing or interpolation)
- No adaptive mesh refinement
- No turbulence modeling (DNS only, 2-D)
- No adaptive time stepping
- Not optimized for large-scale or high-Reynolds-number simulations
- 2-D incompressible flow only

These choices were made to prioritize **clarity, stability, and learning value**.

---

## Goals of the Project

This code was written to:
- Explore projection methods in incompressible flow
- Understand multigrid and Krylov solvers in practice
- Study boundary-condition sensitivity and solver stability
- Build a complete solver + visualization pipeline in Julia

It is **not** meant to compete with established CFD packages.

---

## Running the Code

This repository assumes a working Julia installation.

Typical workflow:
1. Configure parameters in `test_dns.jl` or via the frontend
2. Run the solver
3. Visualize results live (GLMakie) or offline (ParaView)

Exact usage depends on the chosen frontend configuration.

---

## Acknowledgments

This project was developed independently as a personal learning and exploration effort in numerical methods and CFD.

Any mistakes, simplifications, or inaccuracies are entirely my own.

---

```

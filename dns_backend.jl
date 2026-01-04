module DNSBackend
using LinearAlgebra
using Statistics
using Base: @propagate_inbounds
import Main.MultigridBackend: MultigridHierarchy, solve_poisson_mg, build_hierarchy, PressureBC, default_pressure_bc, apply_pressure_operator, apply_pressure_bc!
include("rk.jl")
include("ssprk.jl")
include("irk.jl")
include("sdirk.jl")
include("cg.jl")

#defines the structure for the grid
struct GridParams
    n::Int
    L::Float64
    dx::Float64
end
GridParams(n::Int; L::Float64 = 1.0) = GridParams(n, L, L / (n - 1))

#defines the structure for the cylinder
struct CylinderParams
    center::Tuple{Float64,Float64}
    radius::Float64
end

#defines the structure for the fluid
struct FluidParams
    rho::Float64
    nu::Float64
end

#defines the structure for the time
struct TimeParams
    T::Float64
    steps::Int
    dt::Float64
end
TimeParams(T::Float64, steps::Int) = TimeParams(T, steps, T/steps)

#defines the structure for the multigrid parameters
struct MGParams
    cycle_type::Symbol
    smoother::Symbol
    sweeps::Int
    ω::Float64
    max_cycles::Int
    tolerance::Float64
    initial_guess::Symbol
    bc::PressureBC
end
function MGParams(; cycle_type=:v_cycle, smoother=:rbgs, sweeps::Int=3, ω::Float64=0.8, max_cycles::Int=50, tolerance::Float64=1e-6, initial_guess::Symbol=:previous, bc::PressureBC=default_pressure_bc())
    return MGParams(cycle_type, smoother, sweeps, ω, max_cycles, tolerance, initial_guess, bc)
end

#defines the structure for the ODE parameters
struct ODEParams
    method::Symbol
end
ODEParams(; method::Symbol = :rk4) = ODEParams(method)

#function to create the fluid parameters from either nu or mu and rho
fluid_from(; rho::Float64 = 1.0, nu::Union{Nothing,Float64}=nothing, mu::Union{Nothing,Float64}=nothing) = begin
    if nu === nothing
        @assert mu !== nothing "Provide either ν or (μ and ρ)"
        nu = mu / rho
    end
    FluidParams(rho, nu)
end

#function to select the RK method
function select_stepper(method::Symbol)
    if method == :rk1
        return step_rk1
    elseif method == :rk2
        return step_rk2
    elseif method == :rk3
        return step_rk3
    elseif method == :rk4
        return step_rk4
    elseif method == :rk5
        return step_rk5
    elseif method == :ssprk2
        return step_ssprk2
    elseif method == :ssprk3
        return step_ssprk3
    elseif method == :gauss2
        return (f,t,y,h) -> IRK.step_irk_gauss2(f,t,y,h)
    elseif method == :gauss3
        return (f,t,y,h) -> IRK.step_irk_gauss3(f,t,y,h)
    elseif method == :gauss4
        return (f,t,y,h) -> IRK.step_irk_gauss4(f,t,y,h)
    elseif method == :gauss5
        return (f,t,y,h) -> IRK.step_irk_gauss5(f,t,y,h)
    elseif method == :radau2
        return (f,t,y,h) -> IRK.step_irk_radau2(f,t,y,h)
    elseif method == :radau3
        return (f,t,y,h) -> IRK.step_irk_radau3(f,t,y,h)
    elseif method == :radau4
        return (f,t,y,h) -> IRK.step_irk_radau4(f,t,y,h)
    elseif method == :radau5
        return (f,t,y,h) -> IRK.step_irk_radau5(f,t,y,h)
    elseif method == :lobatto2
        return (f,t,y,h) -> IRK.step_irk_lobatto2(f,t,y,h)
    elseif method == :lobatto3
        return (f,t,y,h) -> IRK.step_irk_lobatto3(f,t,y,h)
    elseif method == :lobatto4
        return (f,t,y,h) -> IRK.step_irk_lobatto4(f,t,y,h)
    elseif method == :lobatto5
        return (f,t,y,h) -> IRK.step_irk_lobatto5(f,t,y,h)
    elseif method == :sdirk2
        return (f,t,y,h) -> step_sdirk2_method(f,t,y,h)
    elseif method == :sdirk3
        return (f,t,y,h) -> step_sdirk3_method(f,t,y,h)
    elseif method == :sdirk4
        return (f,t,y,h) -> step_sdirk4_method(f,t,y,h)
    else
        error("Unsupported RK method: $method")
    end
end

#function to make the cylinder mask
function make_cylinder_mask(grid::GridParams, cyl::CylinderParams)
    n = grid.n
    dx = grid.dx
    mask = falses(n+2, n+2)
    xc, yc = cyl.center
    r2 = cyl.radius^2
    @inbounds for j in 2:n+1
        y = (j-1) * dx
        for i in 2:n+1
            x = (i-1) * dx
            if (x - xc)^2 + (y - yc)^2 <= r2
                mask[i,j] = true
            end
        end
    end
    return mask
end

#defines the rhs for the momentum ode
function default_momentum_rhs!(t, vel, fluid, grid, mask)
    u = vel.u; v = vel.v
    n = grid.n
    dx = grid.dx
    invdx = 1.0 / dx
    invdx2 = 1.0 / (dx * dx)
    du = zeros(size(u)); dv = zeros(size(v))
    @inbounds for j in 2:n+1
        for i in 2:n+1
            if mask[i,j]
                du[i,j] = 0.0; dv[i,j] = 0.0; continue
            end
            du_dx = u[i,j] ≥ 0 ? (u[i,j] - u[i-1,j]) * invdx : (u[i+1,j] - u[i,j]) * invdx
            du_dy = v[i,j] ≥ 0 ? (u[i,j] - u[i,j-1]) * invdx : (u[i,j+1] - u[i,j]) * invdx
            dv_dx = u[i,j] ≥ 0 ? (v[i,j] - v[i-1,j]) * invdx : (v[i+1,j] - v[i,j]) * invdx
            dv_dy = v[i,j] ≥ 0 ? (v[i,j] - v[i,j-1]) * invdx : (v[i,j+1] - v[i,j]) * invdx
            lap_u = (u[i+1,j] - 2u[i,j] + u[i-1,j]) * invdx2 + (u[i,j+1] - 2u[i,j] + u[i,j-1]) * invdx2
            lap_v = (v[i+1,j] - 2v[i,j] + v[i-1,j]) * invdx2 + (v[i,j+1] - 2v[i,j] + v[i,j-1]) * invdx2
            du[i,j] = -(u[i,j] * du_dx + v[i,j] * du_dy) + fluid.nu * lap_u
            dv[i,j] = -(u[i,j] * dv_dx + v[i,j] * dv_dy) + fluid.nu * lap_v
        end
    end
    return (u=du, v=dv)
end

#default velocity BC with optional inflow seeding
function make_velocity_bc(Uin::Real=0.1)
    return (u, v, mask) -> begin
        n = size(u, 1) - 2
        #applies no slip boundary conditions on the top and bottom
        u[:, 1] .= 0.0; u[:, n+2] .= 0.0
        v[:, 1] .= 0.0; v[:, n+2] .= 0.0

        #applies the dirichlet inflow boundary condition on the left
        u[1, 2:n+1] .= Uin
        u[2, 2:n+1] .= Uin
        v[1, 2:n+1] .= 0.0
        v[2, 2:n+1] .= 0.0

        #applies neumann outlet boundary conditions on the right with backflow guard
        u[n+2, 2:n+1] .= max.(u[n+1, 2:n+1], 0.0)
        v[n+2, 2:n+1] .= v[n+1, 2:n+1]

        u[mask] .= 0.0
        v[mask] .= 0.0
    end
end

#function to apply the velocity boundary conditions
@propagate_inbounds function apply_velocity_bc!(u::AbstractArray, v::AbstractArray, mask::AbstractArray)
    n = size(u, 1) - 2
    u[1, :] .= 0.0; u[n+2, :] .= 0.0; u[:, 1] .= 0.0; u[:, n+2] .= 0.0
    v[1, :] .= 0.0; v[n+2, :] .= 0.0; v[:, 1] .= 0.0; v[:, n+2] .= 0.0
    u[mask] .= 0.0
    v[mask] .= 0.0
end

#function to compute the divergence
@propagate_inbounds function divergence(u::AbstractArray, v::AbstractArray, dx::Float64)
    n = size(u, 1) - 2
    div = zeros(size(u))
    inv2dx = 1.0 / (2*dx)
    @inbounds for j in 2:n+1
        for i in 2:n+1
            div[i,j] = (u[i+1,j] - u[i-1,j] + v[i,j+1] - v[i,j-1]) * inv2dx
        end
    end
    return div
end

#function to compute the gradient
@propagate_inbounds function gradient(p::AbstractArray, dx::Float64)
    n = size(p, 1) - 2
    gx = zeros(size(p))
    gy = zeros(size(p))
    inv2dx = 1.0 / (2*dx)
    @inbounds for j in 2:n+1
        for i in 2:n+1
            gx[i,j] = (p[i+1,j] - p[i-1,j]) * inv2dx
            gy[i,j] = (p[i,j+1] - p[i,j-1]) * inv2dx
        end
    end
    return gx, gy
end

#function to project the velocity
function project_velocity(u_star, v_star, p, dt, rho, dx)
    gx, gy = gradient(p, dx)
    u = u_star .- (dt / rho) .* gx
    v = v_star .- (dt / rho) .* gy
    return u, v
end

#main solver function for DNS
function run_dns(grid::GridParams, cylinder::CylinderParams, fluid::FluidParams, time_params::TimeParams, mg_params::MGParams, ode_params::ODEParams; momentum_rhs! = default_momentum_rhs!, velocity_bc! = make_velocity_bc(0.1), subtract_mean_rhs::Bool = true, pressure_solver::Symbol = :mg, pcg_tol::Float64 = 1e-6, pcg_max_iters::Int = 50, verbose::Bool = false, save_callback = nothing)
    n = grid.n
    dt = time_params.dt
    dx = grid.dx
    mask = make_cylinder_mask(grid, cylinder)

    #initializes the velocity and pressure fields
    u = zeros(n+2, n+2)
    v = zeros(n+2, n+2)
    p = zeros(n+2, n+2)
    velocity_bc!(u, v, mask)

    #builds the multigrid hierarchy for pressure with chosen BCs
    mg = build_hierarchy(n; bc=mg_params.bc)

    #selects the RK method
    rk_step = select_stepper(ode_params.method)

    #initializes the initial state and residual history
    initial_state = nothing
    residual_history = Vector{Float64}[]
    div_history = Float64[]

    #loops across time steps
    for step in 1:time_params.steps
        t = (step-1) * dt
        if verbose
            println("Step $step: t = $t")
        end

        #advances the momentum with the ode solver
        vel = (u=u, v=v)
        rhs_fun = (τ, state) -> momentum_rhs!(τ, state, fluid, grid, mask)
        vel_star = rk_step(rhs_fun, t, vel, dt)
        u_star, v_star = vel_star.u, vel_star.v
        velocity_bc!(u_star, v_star, mask)

        #computes pressure using the multigrid solver
        rhs_p = -(fluid.rho / dt) .* divergence(u_star, v_star, dx)
        if subtract_mean_rhs
            rhs_p .-= mean(rhs_p[2:end-1, 2:end-1])
        end
        if pressure_solver == :mg
            p, res = solve_poisson_mg(mg, rhs_p; cycle_type=mg_params.cycle_type, smoother=mg_params.smoother, sweeps=mg_params.sweeps, max_cycles=mg_params.max_cycles, ω=mg_params.ω, tolerance=mg_params.tolerance, initial_guess=mg_params.initial_guess, initial_state=initial_state, verbose=verbose)
            push!(residual_history, res)
        elseif pressure_solver == :pcg
            p, res = solve_pressure_pcg(mg, rhs_p, mg_params; tol=pcg_tol, max_iters=pcg_max_iters, verbose=verbose)
            push!(residual_history, res)
        else
            error("Unsupported pressure solver: $pressure_solver")
        end

        #projects the velocity and measure divergence
        u_proj, v_proj = project_velocity(u_star, v_star, p, dt, fluid.rho, dx)
        div_field = divergence(u_proj, v_proj, dx)
        push!(div_history, maximum(abs.(div_field[2:end-1, 2:end-1])))
        velocity_bc!(u_proj, v_proj, mask)
        u, v = u_proj, v_proj

        if save_callback !== nothing
            save_callback(step, u, v, p, grid)
        end

        #stores the hierarchy state for warm-starting
        if mg_params.initial_guess == :previous
            initial_state = [copy(lev.u) for lev in mg.levels]
        end
    end
    return (u=u, v=v, p=p, mask=mask, residual_history=residual_history, div_history=div_history)
end
export GridParams, CylinderParams, FluidParams, TimeParams, MGParams, ODEParams, PressureBC, default_pressure_bc, fluid_from, run_dns, apply_velocity_bc!, divergence, project_velocity, make_cylinder_mask, select_stepper, make_velocity_bc, default_momentum_rhs!
end
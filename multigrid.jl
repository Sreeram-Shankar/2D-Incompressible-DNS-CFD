module MultigridBackend

using LinearAlgebra
using Random
using Base.Threads

#pressure boundary condition container
struct PressureBC
    left::Symbol
    right::Symbol
    bottom::Symbol
    top::Symbol
    left_value::Float64
    right_value::Float64
    bottom_value::Float64
    top_value::Float64
end

#enforces the dirichlet boundary conditions on all boundaries
default_pressure_bc() = PressureBC(:dirichlet, :dirichlet, :dirichlet, :dirichlet, 0.0, 0.0, 0.0, 0.0)

#defines the structure for each multigrid level
struct MGLevel
    n::Int
    h::Float64
    u::Array{Float64}
    f::Array{Float64}
    res::Array{Float64}
    bc::PressureBC
end

#defines the structure for the multigrid hierarchy
struct MultigridHierarchy
    levels::Vector{MGLevel}
end

#applies pressure boundary conditions (Dirichlet or Neumann) to ghost cells
function apply_pressure_bc!(u::AbstractMatrix, h::Float64, bc::PressureBC)
    n = size(u, 1) - 2
    @inbounds begin
        #enforces the left boundary
        if bc.left == :dirichlet
            u[1, :] .= bc.left_value
        elseif bc.left == :neumann
            u[1, :] .= u[2, :] .+ h * bc.left_value
        else
            error("Unsupported BC type on left boundary: $(bc.left)")
        end

        #enforces the right boundary
        if bc.right == :dirichlet
            u[n+2, :] .= bc.right_value
        elseif bc.right == :neumann
            u[n+2, :] .= u[n+1, :] .- h * bc.right_value
        else
            error("Unsupported BC type on right boundary: $(bc.right)")
        end

        #enforces the bottom boundary
        if bc.bottom == :dirichlet
            u[:, 1] .= bc.bottom_value
        elseif bc.bottom == :neumann
            u[:, 1] .= u[:, 2] .+ h * bc.bottom_value
        else
            error("Unsupported BC type on bottom boundary: $(bc.bottom)")
        end

        #enforces the top boundary
        if bc.top == :dirichlet
            u[:, n+2] .= bc.top_value
        elseif bc.top == :neumann
            u[:, n+2] .= u[:, n+1] .- h * bc.top_value
        else
            error("Unsupported BC type on top boundary: $(bc.top)")
        end
    end
end

#convenience wrapper to keep existing call sites
apply_pressure_bc!(lev::MGLevel) = apply_pressure_bc!(lev.u, lev.h, lev.bc)

#applies the pressure operator to the velocity field
function apply_pressure_operator(u::AbstractMatrix, h::Float64, bc::PressureBC)
    work = copy(u)
    apply_pressure_bc!(work, h, bc)
    n = size(work, 1) - 2
    r = zeros(size(work))
    h2inv = 1.0 / (h^2)

    #zeroes out the boundaries
    @inbounds begin
        r[1, :] .= 0.0
        r[n+2, :] .= 0.0
        r[:, 1] .= 0.0
        r[:, n+2] .= 0.0
    end
    @threads for j in 2:n+1
        @inbounds for i in 2:n+1
            r[i, j] = h2inv * (4 * work[i, j] - work[i+1, j] - work[i-1, j] - work[i, j+1] - work[i, j-1])
        end
    end
    return r
end

#sets all the levels to zero
function zero_hierarchy!(levels::Vector{MGLevel})
    for lev in levels
        fill!(lev.u, 0.0)
        apply_pressure_bc!(lev)
    end
end

#sets all the levels to random values
function randomize_hierarchy!(levels::Vector{MGLevel})
    for lev in levels
        fill!(lev.u, 0.0)
        interior = @view lev.u[2:lev.n+1, 2:lev.n+1]
        rand!(interior)
        interior .-= 0.5
        apply_pressure_bc!(lev)
    end
end

#applies the initial guess to the hierarchy
function apply_initial_guess!(levels::Vector{MGLevel}, initial_guess::Symbol, previous_states)
    if initial_guess == :previous
        if previous_states === nothing
            zero_hierarchy!(levels)
        else
            for (lev, state) in zip(levels, previous_states)
                lev.u .= state
            end
        end
    elseif initial_guess == :zero
        zero_hierarchy!(levels)
    elseif initial_guess == :random
        randomize_hierarchy!(levels)
    else
        error("Unsupported initial guess type: $initial_guess")
    end
    for lev in levels
        apply_pressure_bc!(lev)
    end
end

#loads a static RHS onto the finest grid
function set_rhs!(finest::MGLevel, rhs)
    if rhs isa Function
        n = finest.n
        h = finest.h
        @inbounds for j in 2:n+1
            y = (j-1) * h
            for i in 2:n+1
                x = (i-1) * h
                finest.f[i, j] = rhs(x, y)
            end
        end
    elseif rhs isa AbstractArray
        @assert size(rhs) == size(finest.f) "RHS array must match grid size $(size(finest.f))"
        finest.f .= rhs
    else
        error("rhs must be either a function f(x, y) or an array")
    end
end

#builds the hierarchy from finest to coarsest
function build_hierarchy(N::Int; bc::PressureBC = default_pressure_bc())
    levels = MGLevel[]
    n = N
    while n ≥ 4
        h = 1.0/(n + 1)
        #creates the variables for the current level
        push!(levels, MGLevel(n, h, zeros(n+2, n+2), zeros(n+2, n+2), zeros(n+2, n+2), bc))
        n ÷= 2
    end
    return MultigridHierarchy(levels)
end

#computes the residual using the 5 point stencil
function compute_residual!(lev::MGLevel)
    n = lev.n
    u = lev.u; f = lev.f; r = lev.res
    h2inv = 1.0/(lev.h^2)

    #enforces zero residual on boundaries
    @inbounds begin
        r[1, :] .= 0.0
        r[n+2, :] .= 0.0
        r[:, 1] .= 0.0
        r[:, n+2] .= 0.0
    end
    @inbounds for j in 2:n+1
        for i in 2:n+1
            r[i,j] = f[i,j] - h2inv * (4*u[i,j] - u[i+1,j] - u[i-1,j] - u[i,j+1] - u[i,j-1])
        end
    end
end

#performs extra smoothing on the coarsest grid
function coarse_exact_solve!(lev::MGLevel)
    apply_smoother!(lev, :weighted_jacobi, 20, 0.8)
end

#performs the weighting restriction
function restrict!(coarse::MGLevel, fine::MGLevel)
    nc = coarse.n
    @inbounds for jc in 2:nc+1
        jf = 2*(jc-1)
        for ic in 2:nc+1
            i_f = 2*(ic-1)
            coarse.f[ic,jc] =
                ( fine.res[i_f,   jf]   * 1 +
                  fine.res[i_f+1, jf]   * 2 +
                  fine.res[i_f-1, jf]   * 2 +
                  fine.res[i_f,   jf+1] * 2 +
                  fine.res[i_f,   jf-1] * 2 +
                  fine.res[i_f+1, jf+1] * 4 +
                  fine.res[i_f-1, jf+1] * 4 +
                  fine.res[i_f+1, jf-1] * 4 +
                  fine.res[i_f-1, jf-1] * 4 ) / 16
        end
    end
end

#prolongates and corrects the fine grid
function prolongate_and_correct!(fine::MGLevel, coarse::MGLevel)
    nc = coarse.n
    uC = coarse.u
    uF = fine.u
    #avoids coarse indexing overflow
    val(ic, jc) = uC[clamp(ic, 1, nc+2), clamp(jc, 1, nc+2)]
    @inbounds for jc in 2:nc+1
        jf = 2*(jc-1)
        for ic in 2:nc+1
            i_f = 2*(ic-1)

            #gets the coarse point
            uF[i_f, jf] += val(ic, jc)

            #calculates the edge interpolations
            uF[i_f+1, jf]   += 0.5*(val(ic, jc) + val(ic+1, jc))
            uF[i_f, jf+1]   += 0.5*(val(ic, jc) + val(ic, jc+1))

            #calculates the center interpolation
            uF[i_f+1, jf+1] += 0.25*(val(ic, jc) + val(ic+1, jc) + val(ic, jc+1) + val(ic+1, jc+1))
        end
    end
end

#defines the Weighted Jacobi smoother
function smooth_weighted_jacobi!(lev::MGLevel, sweeps::Int, ω)
    @assert 0 < ω < 2 "Weighted Jacobi relaxation ω should be in (0, 2) for stability"
    n = lev.n
    u = lev.u; f = lev.f
    h2 = lev.h^2
    u_new = similar(u)
    for _ in 1:sweeps
        copy!(u_new, u)
        @threads for j in 2:n+1
            @inbounds for i in 2:n+1
                gs = 0.25*(u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] + h2*f[i,j])
                u_new[i,j] = u[i,j] + ω*(gs - u[i,j])
            end
        end
        copy!(u, u_new)
    end
end

#defines the Red-Black Gauss-Seidel smoother
function smooth_rbgs!(lev::MGLevel, sweeps::Int)
    n = lev.n
    u = lev.u; f = lev.f
    h2 = lev.h^2
    for _ in 1:sweeps
        #performs the red sweep
        @threads for j in 2:n+1
            @inbounds for i in 2:n+1
                if (i+j) % 2 == 0
                    u[i,j] = 0.25*(u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] + h2*f[i,j])
                end
            end
        end

        #performs the black sweep
        @threads for j in 2:n+1
            @inbounds for i in 2:n+1
                if (i+j) % 2 == 1
                    u[i,j] = 0.25*(u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] + h2*f[i,j])
                end
            end
        end
    end
end

#applies the smoother based on selection
function apply_smoother!(lev::MGLevel, smoother, sweeps, ω)
    if smoother == :weighted_jacobi
        smooth_weighted_jacobi!(lev, sweeps, ω)
    elseif smoother == :rbgs
        smooth_rbgs!(lev, sweeps)
    else
        error("Unsupported smoother: $smoother")
    end
    apply_pressure_bc!(lev)
end

#defines the V-cycle recursion
function vcycle!(mg::MultigridHierarchy, ℓ::Int, smoother, sweeps, ω)
    lev = mg.levels[ℓ]

    #performs the pre-smoothing
    apply_smoother!(lev, smoother, sweeps, ω)

    #extra smoothing on coarsest grid
    if ℓ == length(mg.levels)
        coarse_exact_solve!(lev)
        apply_pressure_bc!(lev)
        return
    end

    #computes the residual
    compute_residual!(lev)

    #restricts to the coarse grid
    coarse = mg.levels[ℓ+1]
    fill!(coarse.f, 0.0)
    fill!(coarse.res, 0.0)
    restrict!(coarse, lev)
    fill!(coarse.u, 0.0)
    apply_pressure_bc!(coarse)

    #recursives the V-cycle on a coarser grid
    vcycle!(mg, ℓ+1, smoother, sweeps, ω)

    #prolongates and corrects
    prolongate_and_correct!(lev, coarse)
    apply_pressure_bc!(lev)

    #performs the post-smoothing
    apply_smoother!(lev, smoother, sweeps, ω)
end

#defines the W-cycle recursion
function wcycle!(mg, ℓ, smoother, sweeps, ω)
    lev = mg.levels[ℓ]

    #performs the pre-smoothing
    apply_smoother!(lev, smoother, sweeps, ω)

    #extra smoothing on coarsest grid
    if ℓ == length(mg.levels)
        coarse_exact_solve!(lev)
        apply_pressure_bc!(lev)
        return
    end

    #computes the residual
    compute_residual!(lev)

    #restricts to the coarse grid
    coarse = mg.levels[ℓ+1]
    fill!(coarse.f, 0.0)
    fill!(coarse.res, 0.0)
    restrict!(coarse, lev)
    fill!(coarse.u, 0.0)
    apply_pressure_bc!(coarse)

    #recursives the W-cycle on a coarser grid
    wcycle!(mg, ℓ+1, smoother, sweeps, ω)
    wcycle!(mg, ℓ+1, smoother, sweeps, ω)

    #prolongates and corrects
    prolongate_and_correct!(lev, coarse)
    apply_pressure_bc!(lev)

    #performs the post-smoothing
    apply_smoother!(lev, smoother, sweeps, ω)
end


#defines the F-cycle recursion
function fcycle!(mg, ℓ, smoother, sweeps, ω)
    lev = mg.levels[ℓ]

    #performs the pre-smoothing
    apply_smoother!(lev, smoother, sweeps, ω)

    #extra smoothing on coarsest grid
    if ℓ == length(mg.levels)
        coarse_exact_solve!(lev)
        apply_pressure_bc!(lev)
        return
    end

    #computes the residual
    compute_residual!(lev)

    #restricts to the coarse grid
    coarse = mg.levels[ℓ+1]
    fill!(coarse.f, 0.0)
    fill!(coarse.res, 0.0)
    restrict!(coarse, lev)
    fill!(coarse.u, 0.0)
    apply_pressure_bc!(coarse)

    #recursives the W-cycle on a coarser grid
    wcycle!(mg, ℓ+1, smoother, sweeps, ω)

    #recursives the V-cycle on a coarser grid
    vcycle!(mg, ℓ+1, smoother, sweeps, ω)

    #prolongates and corrects
    prolongate_and_correct!(lev, coarse)
    apply_pressure_bc!(lev)

    #performs the post-smoothing
    apply_smoother!(lev, smoother, sweeps, ω)
end

#solves Poisson problem for a given RHS and returns the solution
function solve_poisson_mg(mg::MultigridHierarchy, rhs; cycle_type, smoother, sweeps, max_cycles, ω, tolerance, initial_guess::Symbol = :zero, initial_state = nothing, verbose::Bool = false)
    levels = mg.levels
    finest = levels[1]

    #loads the RHS and applies the initial guess
    set_rhs!(finest, rhs)
    apply_initial_guess!(levels, initial_guess, initial_state)
    residual_history = Float64[]

    #runs the multigrid cycles
    for cyc in 1:max_cycles
        if cycle_type == :v_cycle
            vcycle!(mg, 1, smoother, sweeps, ω)
        elseif cycle_type == :w_cycle
            wcycle!(mg, 1, smoother, sweeps, ω)
        elseif cycle_type == :f_cycle
            fcycle!(mg, 1, smoother, sweeps, ω)
        else
            error("Unsupported cycle type: $cycle_type")
        end
        compute_residual!(finest)
        resnorm = norm(finest.res)
        push!(residual_history, resnorm)
        if verbose
            println("Cycle $cyc: residual = $resnorm")
        end

        #uses the user defined tolerance
        if resnorm < tolerance
            break
        end
    end
    return copy(finest.u), residual_history
end
export solve_poisson_mg, build_hierarchy, PressureBC, default_pressure_bc, apply_pressure_bc!, apply_pressure_operator
end
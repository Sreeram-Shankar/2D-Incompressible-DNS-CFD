using LinearAlgebra
using Base.Threads
import Main.MultigridBackend: apply_pressure_operator

#implements the preconditioned CG solver
function solve_pressure_pcg(mg, rhs, mg_params; tol=1e-6, max_iters=100, verbose::Bool=false)
    levels = mg.levels
    finest = levels[1]
    n = finest.n
    h = finest.h
    nt = Threads.nthreads()
    dotbuf = zeros(Float64, nt)
    normbuf = zeros(Float64, nt)

    #parallel dot product and norm
    dot_threaded!(buf, a, b) = begin
        fill!(buf, 0.0)
        Threads.@threads for idx in eachindex(a, b)
            buf[threadid()] += a[idx] * b[idx]
        end
        return sum(buf)
    end
    norm_threaded!(buf, a) = begin
        fill!(buf, 0.0)
        Threads.@threads for idx in eachindex(a)
            buf[threadid()] += a[idx] * a[idx]
        end
        return sqrt(sum(buf))
    end

    #appplies an initial geuss of 0
    x = zeros(size(rhs))
    r = copy(rhs)
    rhs_norm = norm(r)
    if rhs_norm == 0.0
        return x, [0.0]
    end

    #uses a linear preconditioner of one w cycle
    precond = function(res)
        psol, _ = solve_poisson_mg(mg, res; cycle_type=:w_cycle, smoother=mg_params.smoother, sweeps=mg_params.sweeps, max_cycles=1, ω=mg_params.ω, tolerance=0.0, initial_guess=:zero, initial_state=nothing, verbose=false)
        return psol
    end
    z = precond(r)
    p = copy(z)
    rz_old = dot_threaded!(dotbuf, r, z)
    reshist = Float64[]

    #performs the conjugate gradient iterations with stagnation detection
    stall_window = 3
    for k in 1:max_iters
        #applies the pressure operator to the field
        q = apply_pressure_operator(p, h, mg.levels[1].bc)
        pq = dot_threaded!(dotbuf, p, q)
        if pq == 0
            break
        end

        #computes the alpha parameter
        alpha = rz_old / pq
        x .= x .+ alpha .* p
        r .= r .- alpha .* q
        resnorm = norm_threaded!(normbuf, r)
        push!(reshist, resnorm)
        if verbose
            println("PCG iter $k: residual = $resnorm")
        end

        #checks if the residual is below the tolerance
        if resnorm < tol || resnorm / rhs_norm < tol
            return x, reshist
        end

        #checks if the residual is stagnating
        if k > stall_window
            recent = reshist[end-stall_window+1:end]
            if maximum(recent) - minimum(recent) < 1e-3 * maximum(recent)
                if verbose
                    println("PCG stagnation detected at iter $k with residual $resnorm")
                end
                return x, reshist
            end
        end

        #applies the preconditioner to the residual
        z = precond(r)
        rz_new = dot_threaded!(dotbuf, r, z)
        beta = rz_new / rz_old
        p .= z .+ beta .* p
        rz_old = rz_new
    end
    return x, reshist
end
using LinearAlgebra
using Base.Threads
import Main.MultigridBackend: apply_pressure_operator, solve_poisson_mg

#implements the preconditioned BiCGSTAB solver with stagnation detection
function solve_pressure_bicgstab(mg, rhs, mg_params; tol=1e-6, max_iters=100, verbose::Bool=false)
    finest = mg.levels[1]
    h = finest.h
    bc = finest.bc

    #parallel dot product and norm
    nt = Threads.nthreads()
    dotbuf = zeros(Float64, nt)
    normbuf = zeros(Float64, nt)
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

    #uses a linear preconditioner of one w cycle
    precond = function(res)
        psol, _ = solve_poisson_mg(mg, res; cycle_type=:w_cycle, smoother=mg_params.smoother, sweeps=mg_params.sweeps, max_cycles=1, ω=mg_params.ω, tolerance=0.0, initial_guess=:zero, initial_state=nothing, verbose=false)
        return psol
    end

    #applies an initial guess of 0
    x = zeros(size(rhs))
    r = copy(rhs)
    rhs_norm = norm_threaded!(normbuf, rhs)
    if rhs_norm == 0.0
        return x, [0.0]
    end
    r_hat = copy(r)
    rho_old = 1.0
    alpha = 1.0
    omega = 1.0
    v = zeros(size(rhs))
    p = zeros(size(rhs))
    reshist = Float64[]
    x_prev = copy(x)
    prev_res = rhs_norm

    #performs the BiCGSTAB iterations
    stall_window = 3
    for k in 1:max_iters
        rho_new = dot_threaded!(dotbuf, r_hat, r)
        if rho_new == 0.0 || !isfinite(rho_new)
            push!(reshist, norm_threaded!(normbuf, r))
            return x_prev, reshist
        end

        if k == 1
            p .= r
        else
            beta = (rho_new / rho_old) * (alpha / omega)
            p .= r .+ beta .* (p .- omega .* v)
        end

        #performs right preconditioning
        phat = precond(p)
        v .= apply_pressure_operator(phat, h, bc)
         denom_alpha = dot_threaded!(dotbuf, r_hat, v)
         if denom_alpha == 0.0 || !isfinite(denom_alpha)
             push!(reshist, norm_threaded!(normbuf, r))
             return x_prev, reshist
         end
         alpha = rho_new / denom_alpha
        s = r .- alpha .* v

        s_norm = norm_threaded!(normbuf, s)
        push!(reshist, s_norm)
        if verbose
            println("BiCGSTAB iter $k: residual = $s_norm")
        end
         if s_norm < tol || s_norm / rhs_norm < tol
             x .+= alpha .* phat
             return x, reshist
         end

        #applies the preconditioner to the residual
        shat = precond(s)
        t = apply_pressure_operator(shat, h, bc)
         omega_num = dot_threaded!(dotbuf, t, s)
         omega_den = dot_threaded!(dotbuf, t, t)
         if omega_den == 0.0 || !isfinite(omega_den)
             omega = 0.0
         else
             omega = omega_num / omega_den
         end
        x .+= alpha .* phat .+ omega .* shat
        r .= s .- omega .* t
         resnorm = norm_threaded!(normbuf, r)
         push!(reshist, resnorm)
         if verbose
             println("BiCGSTAB iter $k: residual = $resnorm")
         end

        #checks if the residual is below the tolerance
        converged = resnorm < tol || resnorm / rhs_norm < tol

        #checks if the residual is stagnating
         stagnating = false
         if !converged && length(reshist) >= stall_window
             recent = reshist[end-stall_window+1:end]
             rng = maximum(recent) - minimum(recent)
             stagnating = rng < 1e-3 * maximum(recent)
             if stagnating && verbose
                 println("BiCGSTAB stagnation detected at iter $k (window Δ = $rng)")
             end
         end

         #checks if the residual is monotonically increasing
         if k > 1 && resnorm > prev_res
             return x_prev, reshist
         end

         if converged || stagnating
             return x, reshist
         end
         if omega == 0.0 || !isfinite(omega)
             return x, reshist
         end

         #accepts  iteration
         x_prev .= x
         prev_res = resnorm
         rho_old = rho_new
    end
    return x, reshist
end

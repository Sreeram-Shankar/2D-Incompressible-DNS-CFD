using LinearAlgebra
import Main.MultigridBackend: apply_pressure_operator

#implements the preconditioned FGMRES solver
function solve_pressure_fgmres(mg, rhs, mg_params; tol=1e-6, max_iters=100, restart=30, verbose::Bool=false)
    finest = mg.levels[1]
    h = finest.h
    bc = finest.bc
    x = zeros(size(rhs))
    rhs_norm = norm(rhs)
    if rhs_norm == 0.0
        return x, [0.0]
    end

    #uses a preconditioner of one w cycle
    precond = function(res)
        psol, _ = solve_poisson_mg(mg, res; cycle_type=:w_cycle, smoother=mg_params.smoother, sweeps=mg_params.sweeps, max_cycles=1, ω=mg_params.ω, tolerance=0.0, initial_guess=:zero, initial_state=nothing, verbose=false)
        return psol
    end
    restart = max(1, min(restart, max_iters))
    reshist = Float64[]
    total_iter = 0

    #performs the GMRES iterations
    stall_window = 3
    for k in 1:max_iters
        r = rhs .- apply_pressure_operator(x, h, bc)
        beta = norm(r)
        push!(reshist, beta)
        if verbose
            println("FGMRES iter $total_iter: residual = $beta")
        end
        if beta < tol || beta / rhs_norm < tol
            break
        end

        #initializes the vectors for the GMRES iteration
        V = Vector{Array{Float64}}(undef, restart + 1)
        Z = Vector{Array{Float64}}(undef, restart)
        H = zeros(Float64, restart + 1, restart)
        cs = zeros(Float64, restart)
        sn = zeros(Float64, restart)
        g = zeros(Float64, restart + 1)
        V[1] = r / beta
        g[1] = beta
        iters_this_cycle = 0
        for j in 1:restart
            iters_this_cycle += 1
            total_iter += 1
            Z[j] = precond(V[j])
            w = apply_pressure_operator(Z[j], h, bc)

            #performs the Arnoldi iteration
            for i in 1:j
                H[i, j] = dot(w, V[i])
                w .-= H[i, j] .* V[i]
            end
            H[j + 1, j] = norm(w)
            if H[j + 1, j] != 0.0
                V[j + 1] = w ./ H[j + 1, j]
            else
                V[j + 1] = zeros(size(w))
            end

            #applies the existing Givens rotations
            for i in 1:j-1
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp
            end

            #computes and applies the new Givens rotation
            denom = sqrt(H[j, j]^2 + H[j + 1, j]^2)
            if denom != 0.0
                cs[j] = H[j, j] / denom
                sn[j] = H[j + 1, j] / denom
                H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
                H[j + 1, j] = 0.0
                g[j + 1] = -sn[j] * g[j]
                g[j] = cs[j] * g[j]
            end
            resnorm = abs(g[j + 1])
            push!(reshist, resnorm)
            if verbose
                println("FGMRES iter $total_iter: residual = $resnorm")
            end

            #checks if the residual is below the tolerance  
            converged = resnorm < tol || resnorm / rhs_norm < tol || total_iter >= max_iters

            #checks if the residual is stagnating
            stagnating = false
            if !converged && length(reshist) >= stall_window
                recent = reshist[end-stall_window+1:end]
                rng = maximum(recent) - minimum(recent)
                stagnating = rng < 1e-3 * maximum(recent)
                if stagnating && verbose
                    println("FGMRES stagnation detected at iter $total_iter (window Δ = $rng)")
                end
            end
            if converged || stagnating
                y = H[1:j, 1:j] \ g[1:j]
                for i in 1:j
                    x .+= y[i] .* Z[i]
                end
                return x, reshist
            end
        end

        #updates x with the last cycle solution
        j = iters_this_cycle
        y = H[1:j, 1:j] \ g[1:j]
        for i in 1:j
            x .+= y[i] .* Z[i]
        end
    end
    return x, reshist
end



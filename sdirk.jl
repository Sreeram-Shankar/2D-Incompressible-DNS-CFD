#defines detectors for the state
_is_tuple_state(y) = y isa NamedTuple && hasproperty(y, :u) && hasproperty(y, :v)
_copy_state(y) = _is_tuple_state(y) ? (u = copy(y.u), v = copy(y.v)) : copy(y)
_zero_state(y) = _is_tuple_state(y) ? (u=zeros(size(y.u)), v=zeros(size(y.v))) : zeros(size(y))
_scale_state(y, a) = _is_tuple_state(y) ? (u = a .* y.u, v = a .* y.v) : a .* y
_add_state(y1, y2) = _is_tuple_state(y1) ? (u = y1.u .+ y2.u, v = y1.v .+ y2.v) : y1 .+ y2
_add_scaled_state(y, a, x) = _is_tuple_state(y) ? (u = y.u .+ a .* x.u, v = y.v .+ a .* x.v) : y .+ a .* x
_norm_state(y) = _is_tuple_state(y) ? sqrt(sum(abs2, y.u) + sum(abs2, y.v)) : norm(y)

#defines the SDIRK step with Gauss-Seidel relaxation (tuple-aware)
function step_sdirk(f, t, y, h, A, b, c; sweeps=25, tol=1e-10)
    s = length(b)

    #defines the initial guesses for all stages
    Y = [_copy_state(y) for _ in 1:s]

    #implements Gauss-Seidel relaxation
    for _ in 1:sweeps
        Y_old = [_copy_state(Yi) for Yi in Y]
        for i in 1:s
            rhs = _zero_state(y)
            for j in 1:s
                fj = f(t + c[j]*h, Y[j])
                rhs = _add_scaled_state(rhs, A[i,j], fj)
            end
            Y[i] = _add_scaled_state(y, h, rhs)
        end
        diff_norm = sqrt(sum(_norm_state(_add_state(Y[i], _scale_state(Y_old[i], -1)))^2 for i in 1:s))
        if diff_norm < tol
            break
        end
    end

    #computes the final state update
    K = [f(t + c[i]*h, Y[i]) for i in 1:s]
    accum = _zero_state(y)
    for i in 1:s
        accum = _add_scaled_state(accum, b[i], K[i])
    end
    y_next = _add_scaled_state(y, h, accum)
    return y_next
end


#solves the nonlinear system of equations with a Gauss-Seidel relaxation SDIRK2
function solve_sdirk2(f, t_span, y0, h; sweeps=25, tol=1e-10)
    gamma = 1.0 - 1.0/sqrt(2.0)
    A = [
        gamma  0.0;
        1.0 - gamma  gamma
    ]
    b = [1.0 - gamma, gamma]
    c = [gamma, 1.0]

    t0, tf = t_span
    N = ceil(Int, (tf - t0)/h)
    t_grid = range(t0, length=N+1, step=h)
    Y = zeros(length(t_grid), length(y0))
    Y[1,:] = y0

    for n in 1:N
        Y[n+1,:] = step_sdirk(f, t_grid[n], Y[n,:], h, A, b, c; sweeps=sweeps, tol=tol)
    end
    return t_grid, Y
end

#single-step helpers for DNS
function step_sdirk2_method(f, t, y, h; sweeps=25, tol=1e-10)
    gamma = 1.0 - 1.0/sqrt(2.0)
    A = [gamma 0.0; 1.0 - gamma gamma]
    b = [1.0 - gamma, gamma]
    c = [gamma, 1.0]
    return step_sdirk(f, t, y, h, A, b, c; sweeps=sweeps, tol=tol)
end

function step_sdirk3_method(f, t, y, h; sweeps=25, tol=1e-10)
    gamma = 0.435866521508459
    A = [gamma 0.0 0.0;
         0.2820667395 gamma 0.0;
         1.208496649 -0.644363171 gamma]
    b = [1.208496649, -0.644363171, gamma]
    c = [gamma, 0.7179332605, 1.0]
    return step_sdirk(f, t, y, h, A, b, c; sweeps=sweeps, tol=tol)
end

function step_sdirk4_method(f, t, y, h; sweeps=25, tol=1e-10)
    gamma = 0.572816062482135
    A = [gamma 0.0 0.0 0.0;
         -0.6557110092 gamma 0.0 0.0;
         0.757184241 0.237758128 gamma 0.0;
         0.155416858 0.701913790 0.142669351 gamma]
    b = [0.155416858, 0.701913790, 0.142669351, gamma]
    c = [gamma, 0.344, 0.995, 1.0]
    return step_sdirk(f, t, y, h, A, b, c; sweeps=sweeps, tol=tol)
end


#solves the nonlinear system of equations with a Gauss-Seidel relaxation SDIRK3
function solve_sdirk3(f, t_span, y0, h; sweeps=25, tol=1e-10)
    gamma = 0.435866521508459
    A = [
        gamma           0.0               0.0;
        0.2820667395    gamma             0.0;
        1.208496649    -0.644363171       gamma
    ]
    b = [1.208496649, -0.644363171, gamma]
    c = [gamma, 0.7179332605, 1.0]

    t0, tf = t_span
    N = ceil(Int, (tf - t0)/h)
    t_grid = range(t0, length=N+1, step=h)
    Y = zeros(length(t_grid), length(y0))
    Y[1,:] = y0

    for n in 1:N
        Y[n+1,:] = step_sdirk(f, t_grid[n], Y[n,:], h, A, b, c; sweeps=sweeps, tol=tol)
    end
    return t_grid, Y
end


#solves the nonlinear system of equations with a Gauss-Seidel relaxation SDIRK4
function solve_sdirk4(f, t_span, y0, h; sweeps=25, tol=1e-10)
    gamma = 0.572816062482135
    A = [
        gamma           0.0           0.0             0.0;
       -0.6557110092    gamma         0.0             0.0;
        0.757184241     0.237758128   gamma           0.0;
        0.155416858     0.701913790   0.142669351     gamma
    ]
    b = [0.155416858, 0.701913790, 0.142669351, gamma]
    c = [gamma, 0.344, 0.995, 1.0]

    t0, tf = t_span
    N = ceil(Int, (tf - t0)/h)
    t_grid = range(t0, length=N+1, step=h)
    Y = zeros(length(t_grid), length(y0))
    Y[1,:] = y0

    for n in 1:N
        Y[n+1,:] = step_sdirk(f, t_grid[n], Y[n,:], h, A, b, c; sweeps=sweeps, tol=tol)
    end
    return t_grid, Y
end
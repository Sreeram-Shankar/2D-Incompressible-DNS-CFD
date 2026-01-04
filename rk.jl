#defines the structure for the state
_is_tuple_state(y) = y isa NamedTuple && hasproperty(y, :u) && hasproperty(y, :v)

#defines the function to add a scaled vector to the state
_add_scaled(y, k, a) = _is_tuple_state(y) ?
    (u = y.u .+ a .* k.u, v = y.v .+ a .* k.v) :
    y .+ a .* k

#defines the function to combine two vectors with weights
_combine(y, k1, a1, k2, a2) = _is_tuple_state(y) ?
    (u = y.u .+ a1 .* k1.u .+ a2 .* k2.u,
     v = y.v .+ a1 .* k1.v .+ a2 .* k2.v) :
    y .+ a1 .* k1 .+ a2 .* k2

#defines the function to combine three vectors with weights
_combine3(y, k1, a1, k2, a2, k3, a3) = _is_tuple_state(y) ?
    (u = y.u .+ a1 .* k1.u .+ a2 .* k2.u .+ a3 .* k3.u,
     v = y.v .+ a1 .* k1.v .+ a2 .* k2.v .+ a3 .* k3.v) :
    y .+ a1 .* k1 .+ a2 .* k2 .+ a3 .* k3

#defines the function to combine four vectors with weights
_combine4(y, k1, a1, k2, a2, k3, a3, k4, a4) = _is_tuple_state(y) ?
    (u = y.u .+ a1 .* k1.u .+ a2 .* k2.u .+ a3 .* k3.u .+ a4 .* k4.u,
     v = y.v .+ a1 .* k1.v .+ a2 .* k2.v .+ a3 .* k3.v .+ a4 .* k4.v) :
    y .+ a1 .* k1 .+ a2 .* k2 .+ a3 .* k3 .+ a4 .* k4

#defines the function to take a step for RK1
function step_rk1(f, t, y, h)
    k1 = f(t, y)
    return _add_scaled(y, k1, h)
end

#defines the main solver for RK1
function solve_rk1(f, t_span, y0, h)
    t0, tf = t_span
    N = ceil(Int, (tf - t0) / h)
    t_grid = range(t0, length=N+1, step=h)
    Y = zeros(length(t_grid), length(y0))
    Y[1,:] = y0
    for n in 1:N
        Y[n+1,:] = step_rk1(f, t_grid[n], Y[n,:], h)
    end
    return t_grid, Y
end

#defines the function to take a step for RK2
function step_rk2(f, t, y, h)
    k1 = f(t, y)
    y2 = _add_scaled(y, k1, h)
    k2 = f(t + h, y2)
    return _combine(y, k1, h/2, k2, h/2)
end

#defines the main solver for RK2
function solve_rk2(f, t_span, y0, h)
    t0, tf = t_span
    N = ceil(Int, (tf - t0) / h)
    t_grid = range(t0, length=N+1, step=h)
    Y = zeros(length(t_grid), length(y0))
    Y[1,:] = y0
    for n in 1:N
        Y[n+1,:] = step_rk2(f, t_grid[n], Y[n,:], h)
    end
    return t_grid, Y
end

#defines the function to take a step for RK3
function step_rk3(f, t, y, h)
    k1 = f(t, y)
    y2 = _add_scaled(y, k1, h/2)
    k2 = f(t + h/2, y2)
    y3 = _combine(y, k1, -h, k2, 2h)
    k3 = f(t + h, y3)
    return _combine3(y, k1, h/6, k2, 4h/6, k3, h/6)
end

#defines the main solver for RK3
function solve_rk3(f, t_span, y0, h)
    t0, tf = t_span
    N = ceil(Int, (tf - t0) / h)
    t_grid = range(t0, length=N+1, step=h)
    Y = zeros(length(t_grid), length(y0))
    Y[1,:] = y0
    for n in 1:N
        Y[n+1,:] = step_rk3(f, t_grid[n], Y[n,:], h)
    end
    return t_grid, Y
end

#defines the function to take a step for RK4
function step_rk4(f, t, y, h)
    k1 = f(t, y)
    y2 = _add_scaled(y, k1, h/2)
    k2 = f(t + h/2, y2)
    y3 = _add_scaled(y, k2, h/2)
    k3 = f(t + h/2, y3)
    y4 = _add_scaled(y, k3, h)
    k4 = f(t + h, y4)
    return _combine4(y, k1, h/6, k2, h/3, k3, h/3, k4, h/6)
end

#defines the main solver for RK4
function solve_rk4(f, t_span, y0, h)
    t0, tf = t_span
    N = ceil(Int, (tf - t0) / h)
    t_grid = range(t0, length=N+1, step=h)
    Y = zeros(length(t_grid), length(y0))
    Y[1,:] = y0
    for n in 1:N
        Y[n+1,:] = step_rk4(f, t_grid[n], Y[n,:], h)
    end
    return t_grid, Y
end

#defines the function to take a step for RK5
function step_rk5(f, t, y, h)
    k1 = f(t, y)
    y2 = _add_scaled(y, k1, h/4)
    k2 = f(t + h/4, y2)
    y3 = _combine(y, k1, h/8, k2, h/8)
    k3 = f(t + h/4, y3)
    y4 = _add_scaled(y, k3, h/2)
    k4 = f(t + h/2, y4)
    y5 = _combine(y, k1, 3h/16, k4, 9h/16)
    k5 = f(t + 3h/4, y5)
    y6 = _combine4(y, k1, 2h/7, k2, 3h/7, k4, 4h/7, k3, -12h/7)
    k6 = f(t + h, y6)
    return _combine4(y, k1, 7h/90, k3, 32h/90, k4, 12h/90, k5, 32h/90) |> x -> _add_scaled(x, k6, 7h/90)
end

#defines the main solver for RK5
function solve_rk5(f, t_span, y0, h)
    t0, tf = t_span
    N = ceil(Int, (tf - t0) / h)
    t_grid = range(t0, length=N+1, step=h)
    Y = zeros(length(t_grid), length(y0))
    Y[1,:] = y0
    for n in 1:N
        Y[n+1,:] = step_rk5(f, t_grid[n], Y[n,:], h)
    end
    return t_grid, Y
end
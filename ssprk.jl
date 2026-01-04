#defines the function to scale the state
_scale_state(y, a) = _is_tuple_state(y) ? (u = a .* y.u, v = a .* y.v) : a .* y
_add_state(y1, y2) = _is_tuple_state(y1) ? (u = y1.u .+ y2.u, v = y1.v .+ y2.v) : y1 .+ y2

#defines the function to take a step for ssprk2
function step_ssprk2(f, t, y, h)
    k1 = f(t, y)
    y1 = _add_scaled(y, k1, h)
    k2 = f(t + h, y1)
    y2 = _add_scaled(y1, k2, h)
    return _add_state(_scale_state(y, 0.5), _scale_state(y2, 0.5))
end

#defines the function to take a step for ssprk3
function step_ssprk3(f, t, y, h)
    k1 = f(t, y)
    y1 = _add_scaled(y, k1, h)

    k2 = f(t + h, y1)
    y2 = _add_state(_scale_state(y, 3/4), _scale_state(_add_scaled(y1, k2, h), 1/4))

    k3 = f(t + h/2, y2) # c3 = 1/2 in the Butcher table
    y3 = _add_state(_scale_state(y, 1/3), _scale_state(_add_scaled(y2, k3, h), 2/3))
    return y3
end
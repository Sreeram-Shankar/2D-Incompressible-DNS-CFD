module IRK

#defines detectors for the state
_is_tuple_state(y) = y isa NamedTuple && hasproperty(y, :u) && hasproperty(y, :v)
_copy_state(y) = _is_tuple_state(y) ? (u = copy(y.u), v = copy(y.v)) : copy(y)
_zero_state(y) = _is_tuple_state(y) ? (u=zeros(size(y.u)), v=zeros(size(y.v))) : zeros(size(y))
_scale_state(y, a) = _is_tuple_state(y) ? (u = a .* y.u, v = a .* y.v) : a .* y
_add_state(y1, y2) = _is_tuple_state(y1) ? (u = y1.u .+ y2.u, v = y1.v .+ y2.v) : y1 .+ y2
_add_scaled_state(y, a, x) = _is_tuple_state(y) ? (u = y.u .+ a .* x.u, v = y.v .+ a .* x.v) : y .+ a .* x
_norm_state(y) = _is_tuple_state(y) ? sqrt(sum(abs2, y.u) + sum(abs2, y.v)) : norm(y)

#defines the Butcher tableau for Gauss-Legendre s = 1
const A_gl_1 = [0.5]
const b_gl_1 = [1.0]
const c_gl_1 = [0.5]

#defines the Butcher tableau for Gauss-Legendre s = 2
const A_gl_2 = [
    0.25                     0.5386751345948129
   -0.03867513459481288     0.25
]
const b_gl_2 = [0.5, 0.5]
const c_gl_2 = [0.7886751345948129, 0.21132486540518712]

#defines the Butcher tableau for Gauss-Legendre s = 3
const A_gl_3 = [
    0.1388888888888889     0.48042111196938335     0.26798833376246945
   -0.022485417203086815   0.2222222222222222      0.3002631949808646
    0.009789444015308326  -0.03597666752493890     0.1388888888888889
]
const b_gl_3 = [0.2777777777777778, 0.4444444444444444, 0.2777777777777778]
const c_gl_3 = [0.8872983346207417, 0.5, 0.1127016653792583]

#defines the Butcher tableau for Gauss-Legendre s = 4
const A_gl_4 = [
    0.08696371128436346    0.35267675751627186    0.31344511474186835     0.1774825722545226
   -0.014190694931141143   0.16303628871563654    0.3539530060337440      0.16719192197418877
    0.006735500594538155  -0.027880428602470895   0.16303628871563654     0.18811811749986807
   -0.003555149685795683   0.012627462689404725  -0.026604180084998793    0.08696371128436346
]
const b_gl_4 = [0.17392742256872693, 0.32607257743127307, 0.32607257743127307, 0.17392742256872693]
const c_gl_4 = [0.9305681557970263, 0.6699905217924281, 0.33000947820757187, 0.06943184420297371]

#defines the Butcher tableau for Gauss-Legendre s = 5
const A_gl_5 = [
    0.05923172126404727    0.25888469960875927   0.2731900436258015    0.24490812891049542   0.11687532956022855
   -0.00968756314195074    0.11965716762484162   0.30903655906408665   0.22899605457899988   0.12123243692686415
    0.004687154523869941  -0.020690316430958285  0.14222222222222222   0.2600046516806415    0.11377628800422460
   -0.002768994398769603   0.010318280670683357 -0.024592114619642200  0.11965716762484162   0.12815100567004528
    0.001588112967865998  -0.005593793660812185  0.011254400818642956 -0.019570364359076037  0.05923172126404727
]
const b_gl_5 = [
    0.11846344252809454, 0.23931433524968323, 0.28444444444444444,
    0.23931433524968323, 0.11846344252809454
]
const c_gl_5 = [
    0.9530899229693320, 0.7692346550528416, 0.5,
    0.23076534494715845, 0.04691007703066800
]

#defines the Butcher tableau for Radau IIA s = 2
const A_radau_2 = [
    0.41666666666666667   -0.08333333333333333
    0.75                   0.25
]
const b_radau_2 = [0.75, 0.25]
const c_radau_2 = [0.3333333333333333, 1.0]

#defines the Butcher tableau for Radau IIA s = 3
const A_radau_3 = [
    0.19681547722366043   -0.06553542585019839    0.023770974348220152
    0.39442431473908727    0.29207341166522846   -0.04154875212599793
    0.37640306270046727    0.5124858261884216     0.11111111111111111
]
const b_radau_3 = [0.37640306270046727, 0.5124858261884216, 0.11111111111111111]
const c_radau_3 = [0.15505102572168219, 0.6449489742783178, 1.0]

#defines the Butcher tableau for Radau IIA s = 4
const A_radau_4 = [
    0.20689257393535890    0.23438399574740026   -0.04785712804854072    0.01604742280651627
   -0.04030922072352221    0.11299947932315619    0.02580237742033639   -0.009904676507266424
    0.4061232638673733     0.21668178462325034    0.18903651817005634   -0.02418210489983294
    0.38819346884317188    0.22046221117676838    0.32884431998005974    0.06250000000000000
]
const b_radau_4 = [
    0.38819346884317188, 0.22046221117676838, 0.32884431998005974, 0.0625
]
const c_radau_4 = [
    0.4094668644407347, 0.08858795951270395, 0.7876594617608471, 1.0
]

#defines the Butcher tableau for Radau IIA s = 5
const A_radau_5 = [
    0.14621486784749350   0.15377523147918247   -0.03644456890512809   0.02123306311930472   -0.007935579902728778
   -0.02673533110794557   0.07299886431790332    0.01867692976398435  -0.01287910609330644    0.005042839233882015
    0.29896712949128348   0.14006304568480987    0.16758507013524896  -0.03396910168661775    0.010944288744192252
    0.27650006876015923   0.14489430810953476    0.32579792291042103   0.12875675325490976   -0.015708917378805328
    0.28135601514946206   0.14371356079122594    0.31182652297574125   0.22310390108357074    0.04000000000000000
]
const b_radau_5 = [
    0.28135601514946206, 0.14371356079122594, 0.31182652297574125,
    0.22310390108357074, 0.04
]
const c_radau_5 = [
    0.27684301363812383, 0.05710419611451768,
    0.5835904323689168, 0.8602401356562194, 1.0
]


#defines the Butcher tableau for Lobatto IIIC s = 2
const A_lob_2 = [
    0.0   0.0
    0.5   0.5
]
const b_lob_2 = [0.5, 0.5]
const c_lob_2 = [0.0, 1.0]

#defines the Butcher tableau for Lobatto IIIC s = 3
const A_lob_3 = [
    0.0      0.0      0.0
    0.20833333333333333  0.3333333333333333  -0.041666666666666664
    0.16666666666666667  0.6666666666666666   0.16666666666666666
]
const b_lob_3 = [
    0.16666666666666666, 0.6666666666666666, 0.16666666666666666
]
const c_lob_3 = [
    0.0, 0.5, 1.0
]

#defines the Butcher tableau for Lobatto IIIC s = 4
const A_lob_4 = [
    0.0     0.0     0.0     0.0
    0.07303276685416842  0.22696723314583158   0.45057403089581055  -0.02696723314583158
    0.11030056647916491 -0.03390736422914388   0.18969943352083508   0.01030056647916491
    0.08333333333333333  0.4166666666666667    0.4166666666666667    0.08333333333333333
]
const b_lob_4 = [0.08333333333333333, 0.4166666666666667, 0.4166666666666667, 0.08333333333333333]
const c_lob_4 = [
    0.0,
    0.7236067977499790,
    0.2763932022500210,
    1.0
]

#defines the Butcher tableau for Lobatto IIIC s = 5
const A_lob_5 = [
    0.0        0.0        0.0        0.0        0.0
    0.05370013924241453  0.15247745287881054  0.37729127742211367  0.26158639799680673 -0.017728432186156897
    0.040625  -0.030961961100820556  0.17777777777777778  0.30318418332304278  0.009375
    0.06772843218615690  0.01063582422541549 -0.021735721866558114 0.11974476934341168 -0.003700139242414531
    0.05      0.2722222222222222      0.35555555555555557  0.2722222222222222  0.05
]
const b_lob_5 = [
    0.05, 0.2722222222222222, 0.35555555555555557,
    0.2722222222222222, 0.05
]
const c_lob_5 = [
    0.0,
    0.8273268353539886,
    0.5,
    0.1726731646460114,
    1.0
]


#loads the Butcher tableaus for the given family and stage count
function get_tableau(family::String, s::Int)
    family = lowercase(family)

    if family == "gauss"
        if     s == 1; return A_gl_1, b_gl_1, c_gl_1
        elseif s == 2; return A_gl_2, b_gl_2, c_gl_2
        elseif s == 3; return A_gl_3, b_gl_3, c_gl_3
        elseif s == 4; return A_gl_4, b_gl_4, c_gl_4
        elseif s == 5; return A_gl_5, b_gl_5, c_gl_5
        else; error("Gauss-Legendre only implemented for s = 1…5")
        end

    elseif family == "radau"
        if     s == 2; return A_radau_2, b_radau_2, c_radau_2
        elseif s == 3; return A_radau_3, b_radau_3, c_radau_3
        elseif s == 4; return A_radau_4, b_radau_4, c_radau_4
        elseif s == 5; return A_radau_5, b_radau_5, c_radau_5
        else; error("Radau IIA only implemented for s = 2…5")
        end

    elseif family == "lobatto"
        if     s == 2; return A_lob_2, b_lob_2, c_lob_2
        elseif s == 3; return A_lob_3, b_lob_3, c_lob_3
        elseif s == 4; return A_lob_4, b_lob_4, c_lob_4
        elseif s == 5; return A_lob_5, b_lob_5, c_lob_5
        else; error("Lobatto IIIC only implemented for s = 2…5")
        end

    else
        error("Unknown IRK family '$family'. Must be 'gauss', 'radau', or 'lobatto'.")
    end
end

#handles the single steps
function step_irk_gauss2(f, t, y, h; sweeps=25, tol=1e-10)
    A, b, c = get_tableau("gauss", 2)
    return step_collocation(f, t, y, h, A, b, c; sweeps=sweeps, tol=tol)
end

function step_irk_gauss3(f, t, y, h; sweeps=25, tol=1e-10)
    A, b, c = get_tableau("gauss", 3)
    return step_collocation(f, t, y, h, A, b, c; sweeps=sweeps, tol=tol)
end

function step_irk_radau2(f, t, y, h; sweeps=25, tol=1e-10)
    A, b, c = get_tableau("radau", 2)
    return step_collocation(f, t, y, h, A, b, c; sweeps=sweeps, tol=tol)
end

function step_irk_radau3(f, t, y, h; sweeps=25, tol=1e-10)
    A, b, c = get_tableau("radau", 3)
    return step_collocation(f, t, y, h, A, b, c; sweeps=sweeps, tol=tol)
end

function step_irk_lobatto2(f, t, y, h; sweeps=25, tol=1e-10)
    A, b, c = get_tableau("lobatto", 2)
    return step_collocation(f, t, y, h, A, b, c; sweeps=sweeps, tol=tol)
end

function step_irk_lobatto3(f, t, y, h; sweeps=25, tol=1e-10)
    A, b, c = get_tableau("lobatto", 3)
    return step_collocation(f, t, y, h, A, b, c; sweeps=sweeps, tol=tol)
end

#defines the IRK step with Gauss–Seidel relaxation
function step_collocation(f, t, y, h, A, b, c; sweeps=25, tol=1e-10)
    s = length(b)

    #initial stage guesses
    Y = [_copy_state(y) for _ in 1:s]

    #implements Gauss–Seidel relaxation
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

#main solver for any collocation IRK method using Gauss–Seidel relaxation
function solve_collocation(f, tspan, y0, h; family="gauss", s=3, sweeps=25, tol=1e-10)
    A, b, c = get_tableau(family, s)
    t0, tf = tspan
    N = ceil(Int, (tf - t0)/h)
    tgrid = range(t0, length=N+1, step=h)

    Y = zeros(N+1, length(y0))
    Y[1,:] = y0

    for n in 1:N
        Y[n+1,:] = step_collocation(f, tgrid[n], Y[n,:], h, A, b, c; sweeps=sweeps, tol=tol)
    end

    return tgrid, Y
end
end
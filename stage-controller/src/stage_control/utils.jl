using LibSerialPort
using LinearAlgebra
using Serialization
using Distributions
using LinearAlgebra
using ForwardDiff
using Flux
using ControlSystemsBase
using LsqFit
using RobustModels

function send_motor_cmd(sp, vel)
    if isnothing(sp)
        return
    end
    write(sp, "G91\n")
    while bytesavailable(sp) < 1
    end
    readline(sp, keep=false)
    vel = clamp.(vel, -0.1, 0.1)
    write(sp, "G0 X$(vel[1]) Y$(vel[2]) Z$(vel[3])\n")
    while bytesavailable(sp) < 1
    end
    readline(sp, keep=false)
end
function recieve_inductance(sp, x0)
    if isnothing(sp)
        return zeros(size(x0))
    end
    data = zeros(size(x0))
    write(sp, "x\n")
    while bytesavailable(sp) < 1
    end
    try
        sp_data = String(readline(sp, keep=false))
        sp_reading = split(sp_data, ',')
        data[1] = time_ns() * 1.0e-9
        data[2] = ((parse(Int, sp_reading[1])) - x0[1]) * 1.0e-7
        data[3] = ((parse(Int, sp_reading[2])) - x0[2]) * 1.0e-7
        data[4] = 0.0
    catch
    end
    return data
end

function bootstrap_stepper_u0(t, scale)
    # if t >= 0.0 && t < 1.0
    #     return scale * [sin((t - 0) * (3 * 2 * pi)), 0.0, 0.0]
    # elseif t >= 1.0 && t < 2.0
    #     return scale * [0.0, sin((t - 1) * (3 * 2 * pi)), 0.0]
    # elseif t >= 2.0 && t < 3.0
    #     return scale * [0.0, 0.0, 10 * sin((t - 2) * (3 * 2 * pi))]
    # else
    #     return scale * [sin((t - 0) * (3 * 2 * pi)), cos((t - 0) * (3 * 2 * pi)), sin((t - 0) * (3 * 2 * pi))]
    # end
    if t >= 0.0 && t < 20.0
        return scale .* [0.0, cos((t - 0) * (0.1 * 2 * pi)), 0.0]
    elseif t >= 20.0 && t < 40.0
        return scale .* [sin((t - 5) * (0.1 * 2 * pi)), 0.0, 0.0]
    elseif t >= 40.0 && t < 70.0
        return scale .* [sin((t - 5) * (0.1 * 2 * pi)), 0.0, 0.0]
    elseif t >= 70.0 && t < 100.0
        return scale .* [0.0, cos((t - 0) * (0.1 * 2 * pi)), 0.0]
    else
        return [0.0, 0.0, 0.0]
    end
end

function bootstrap_stepper_u(t)
    if t >= 0.0 && t < 1.0
        return [sin((t - 0) * (5 * 2 * pi)), 0.0, 0.0]
    elseif t >= 1.0 && t < 2.0
        return [0.0, sin((t - 1) * (5 * 2 * pi)), 0.0]
    elseif t >= 2.0 && t < 3.0
        return [0.0, 0.0, 10 * sin((t - 2) * (5 * 2 * pi))]
    else
        return [sin((t - 0) * (5 * 2 * pi)), cos((t - 0) * (5 * 2 * pi)), sin((t - 0) * (5 * 2 * pi))]
    end
end

function chipFK(state)::Vector
    # x of base in pixel frame
    # y of base in pixel frame
    # theta of rotation stage in pixel frame
    # cx of rotation stage in pixel frame
    # cy of rotation stage in pixel frame
    # ctheta of rotation stage in pixel frame (object orientation)
    # cr of rotation stage in pixel frame
    x, y, theta, cx, cy, ctheta, cr = state
    return [
        x + cx + cr * cos(theta),
        y + cy + cr * sin(theta),
        theta + ctheta
    ]
end

function chipFKreverse(state)::Vector
    # xend of endeffector in pixel frame
    # yend of endeffector in pixel frame
    # thetaend of endeffector in pixel frame
    # cx of rotation stage in pixel frame
    # cy of rotation stage in pixel frame
    # ctheta of rotation stage in pixel frame (object orientation)
    # cr of rotation stage in pixel frame
    xend, yend, thetaend, cx, cy, ctheta, cr = state
    return [
        xend - cx - cr * cos(thetaend),
        yend - cy - cr * sin(thetaend),
        thetaend - ctheta
    ]
end

function chipFKbatch(x, y, theta, cx, cy, ctheta, cr)::Matrix
    # x of base in pixel frame
    # y of base in pixel frame
    # theta of rotation stage in pixel frame
    # cx of rotation stage in pixel frame
    # cy of rotation stage in pixel frame
    # ctheta of rotation stage in pixel frame (object orientation)
    # cr of rotation stage in pixel frame
    return reduce(hcat,
        [x .+ cx .+ cr .* cos.(theta),
        y .+ cy .+ cr .* sin.(theta),
        theta .+ ctheta])'
end

function estimateB(us, ys)
    dimensions = size(us[1])[1]
    u_ = reduce(vcat, [kron(u, I(dimensions) |> Matrix)' for u in us])
    B_ = reduce(vcat, [y for y in ys])
    # try
    #     m = rlm(u_, B_, MEstimator{TukeyLoss}(); method=:cg, initial_scale=:L1, correct_leverage=true)
    #     return reshape(coef(m), dimensions, dimensions)'
    # catch
    #     m = rlm(u_, B_, MEstimator{L2Loss}(); method=:cg, initial_scale=:L1, correct_leverage=true)
    #     return reshape(coef(m), dimensions, dimensions)'
    # end
    return reshape(u_ \ B_, dimensions, dimensions)'
end

function predictUpdateKF(sys, x, x_prev, u_prev, P, dt)
    # Discretize
    F = Diagonal(ones(size(sys[:A]))) + sys[:A] * dt
    B = sys[:B] * dt
    H = sys[:C]

    # Noise model
    Q = sys[:Q] * dt
    R = sys[:R]

    # Predict
    x_pred = F * x_prev + B * u_prev
    P_pred = F * P * F' + Q

    if isnothing(x)
        return x_pred, 1.0e6 * Matrix(I, size(P_pred)...)
    else
        # Update
        K = P_pred * H' / (H * P_pred * H' + R)
        x_new = x_pred + K * (x - H * x_pred)
        P_new = (I - K * H) * P_pred
        return x_new, P_new
    end
end

function predictUpdateRLS!(rls, x, C)
    Lmatrix = rls[:R] + C * rls[:P0] * C'
    LmatrixInv = inv(Lmatrix)
    gainMatrix = rls[:P0] * C' * LmatrixInv
    error = x - C * rls[:x0]
    estimate = rls[:x0] + gainMatrix * error
    ImKc = I - gainMatrix * C
    if any(isnan, estimate) || any(isnan, gainMatrix) || any(isnan, error) || any(isnan, ImKc)
        return rls
    end
    rls[:P0] = (ImKc * rls[:P0]) / rls[:λ]
    rls[:x0] = estimate
    return rls
end

function estimateB(us, ys, rls)
    dimensions = size(us[1])[1]
    C = reduce(vcat, [kron(u, I(dimensions) |> Matrix)' for u in us])
    x = reduce(vcat, [y for y in ys])
    predictUpdateRLS!(rls, x, C)

    return reshape(rls[:x0], dimensions, dimensions)'
    # return reshape(u_ \ B_, dimensions, dimensions)'
end

function estimateCircle(x, y)
    cA = [
        -2 * x,
        -2 * y,
        ones(size(x))
    ]
    cA = reduce(hcat, cA)
    cb = -x .^ 2 - y .^ 2
    cres = cb \ cA
    radius = sqrt(cres[1]^2 + cres[2]^2 - cres[3])
    return cres[1], cres[2], radius
end

function mrac(sys, mracparams, r, r_prev, x, x_prev, dt)
    # calculate u
    feat = [x; r; 1]
    u = mracparams[:K_x]' * x + mracparams[:K_r]' * r + mracparams[:W]' * sigmoid(mracparams[:V]' * feat)

    # ideal reference state
    x_m_dot = sys[:A_m] * x_prev + sys[:B_m] * r_prev
    x_m = x_prev + x_m_dot * dt

    # error prop
    e = x - x_m
    feat = [x_prev; r_prev; 1]

    K_ẋ = -mracparams[:Γ_x] * x * e' * sys[:B] * mracparams[:Λ] * mracparams[:P]
    K_ṙ = -mracparams[:Γ_r] * r * e' * sys[:B] * mracparams[:Λ] * mracparams[:P]
    Ẇ = mracparams[:Γ_w] * ((sigmoid(mracparams[:V]' * feat) - ForwardDiff.jacobian(sigmoid, mracparams[:V]' * feat) * (mracparams[:V]' * feat)) * e' * mracparams[:P] * sys[:B] * mracparams[:Λ] + mracparams[:Γ_σ] * mracparams[:W])
    V̇ = mracparams[:Γ_v] * (feat * e' * mracparams[:P] * sys[:B] * mracparams[:Λ] * mracparams[:W]' * ForwardDiff.jacobian(sigmoid, mracparams[:V]' * feat) + mracparams[:Γ_σ] * mracparams[:V])


    mracparams[:K_x] += K_ẋ * dt
    mracparams[:K_r] += K_ṙ * dt
    mracparams[:W] += Ẇ * dt
    mracparams[:V] += V̇ * dt
    return u, e
end


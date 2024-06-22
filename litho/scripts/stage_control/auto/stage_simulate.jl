using GLMakie
using ProgressMeter
using Distributions
GLMakie.activate!(inline=false)
include("utils.jl")

# Model
window_size = 5000
dt = 0.01

## Stepper-Inductance System
#### Real System
StepIndSys = Dict(
    :A => [0.0 0 0
        0 0.0 0
        0 0 0.0],
    :B => [1 0 0 # estimated
        0 2 0
        0 0 1],
    :C => [1.0 0 0
        0 1.0 0
        0 0 1.0],
    :D => [0.0 0 0
        0 0 0
        0 0 0],
    :A_m => [0.0 0 0
        0 0.0 0
        0 0 0.0],
    :B_m => [1 0 0
        0 1 0
        0 0 1],
    # Noise Model
    :R => 1.0e-5 * Diagonal([1.0 for i in 1:3]),
    :Q => 0.0 * Diagonal([1.0 for i in 1:3]),
)
StepIndMRAC = Dict(
    :P => lyap((StepIndSys[:A_m] + Diagonal([1.0, 1.0, 1.0]))', -Diagonal([1.0, 1.0, 1.0])),
    :Γ_x => 50.0,
    :Γ_r => 50.0,
    :Γ_w => 10.0,
    :Γ_v => 10.0,
    :Γ_σ => 0.0,
    :Λ => Diagonal([1.0, 1.0, 1.0]),
    :K_x => zeros(3, 3),
    :K_r => Diagonal([1.0, 1.0, 1.0]),
    :W => zeros(10, 3), # (hidden_size, state_size)
    :V => Flux.glorot_uniform(7, 10), # (feature_size, hidden_size)
)
StepIndState = Dict(
    :x => Observable(zeros(3, window_size)),
    :u => Observable(zeros(3, window_size)),
    :r => Observable(zeros(3, window_size)),
    :t => Observable(zeros(window_size)),
    :em => Observable(zeros(3, window_size)), # error of model
    :P => Observable(zeros(3, 3, window_size)),
)
StepIndBRLS = Dict(
    :x0 => zeros(9),
    :P0 => 1.0 * Diagonal([1.0 for i in 1:9]),
    :R => 1.0 * Diagonal([1.0 for i in 1:9]),
    :λ => 0.9,
)
## Inductance-Pixel System
#### Real System
IndPixSys = Dict(
    :A => [0.0 0 0
        0 0.0 0
        0 0 0.0],
    :B => [1 0 0 # estimated
        0 1 0
        0 0 1],
    :C => [1.0 0 0
        0 1.0 0
        0 0 1.0],
    :D => [0.0 0 0
        0 0 0
        0 0 0],
    :A_m => [0.0 0 0
        0 0.0 0
        0 0 0.0],
    :B_m => [1 0 0
        0 1 0
        0 0 1],
    # Noise Model
    :R => 1.0e-5 * Diagonal([1.0 for i in 1:3]),
    :Q => 0.0 * Diagonal([1.0 for i in 1:3]),
)
IndPixMRAC = (
    :P => lyap((StepIndSys[:A_m] + Diagonal([1.0, 1.0, 1.0]))', -Diagonal([1.0, 1.0, 1.0])),
    :Γ_x => 50.0,
    :Γ_r => 50.0,
    :Γ_w => 10.0,
    :Γ_v => 10.0,
    :Γ_σ => 0.0,
    :Λ => Diagonal([1.0, 1.0, 1.0]),
    :K_x => zeros(3, 3),
    :K_r => Diagonal([1.0, 1.0, 1.0]),
    :W => zeros(10, 3), # (hidden_size, state_size)
    :V => Flux.glorot_uniform(7, 10), # (feature_size, hidden_size)
)
IndPixState = Dict(
    :x => Observable(zeros(3, window_size)),
    :u => Observable(zeros(3, window_size)),
    :r => Observable(zeros(3, window_size)),
    :t => Observable(zeros(window_size)),
    :em => Observable(zeros(3, window_size)), # error of model
    :P => Observable(zeros(3, 3, window_size)),
)
IndPixBRLS = Dict(
    :x0 => zeros(9),
    :P0 => 1.0 * Diagonal([1.0 for i in 1:9]),
    :R => 1.0 * Diagonal([1.0 for i in 1:9]),
    :λ => 0.9,
)
## Rot Stage Arm
RotStageState = Dict(
    :x => Observable(zeros(3, window_size)), # x, y, theta
    :u => Observable(zeros(3, window_size)),
    :r => Observable(zeros(3, window_size)),
    :t => Observable(zeros(window_size)),
    :em => Observable(zeros(3, window_size)), # error of model
    :P => Observable(zeros(3, 3, window_size)),
)
CircleState = Dict(
    :x => Observable(zeros(4, window_size)), # x, y, theta, cr
    :t => Observable(zeros(window_size)),
    :em => Observable(zeros(4, window_size)), # error of model
)

StepIndState[:x][] .= 0.0
StepIndState[:u][] .= 0.0
StepIndState[:r][] .= 0.0
StepIndState[:t][] .= 0.0
StepIndState[:em][] .= 0.0
StepIndState[:P][] .= 0.0
IndPixState[:x][] .= 0.0
IndPixState[:u][] .= 0.0
IndPixState[:r][] .= 0.0
IndPixState[:t][] .= 0.0
IndPixState[:em][] .= 0.0
IndPixState[:P][] .= 0.0
RotStageState[:x][] .= 0.0
RotStageState[:u][] .= 0.0
RotStageState[:r][] .= 0.0
RotStageState[:t][] .= 0.0
RotStageState[:em][] .= 0.0
RotStageState[:P][] .= 0.0
CircleState[:x][] .= 0.0
CircleState[:t][] .= 0.0
CircleState[:em][] .= 0.0
ts = -6:dt:50
# realStepIndB = [1.0 0.0 0.0;
#     0.0 1.0 0.0;
#     0.0 0.0 1.0]
# realIndPixB = [1.0 0.0 0.0;
#     0.0 1.0 0.0;
#     0.0 0.0 1.0]
realStepIndB = [1.0 1.0 0.0;
    0.0 2.0 1.0;
    0.0 0.0 1.0]
realIndPixB = [1.0 0.0 0.3;
    0.0 1.0 0.3;
    0.2 0.1 1.0]
realRotStage = [1, 1, 0.0, 1.0] # circle_x, circle_y, circle_theta, circle_radius
# CircleState[:x][][:, 1] = [0.991923529037823,
#     0.9998965952620104,
#     0.00038566758340052273,
#     1.008179885883361]
# CircleState[:x][][:, 1] = realRotStage
@showprogress for (i, t) in enumerate(ts)
    ## StepInd

    # shift all
    StepIndState[:x][] = circshift(StepIndState[:x][], (0, 1))
    StepIndState[:u][] = circshift(StepIndState[:u][], (0, 1))
    StepIndState[:r][] = circshift(StepIndState[:r][], (0, 1))
    StepIndState[:em][] = circshift(StepIndState[:em][], (0, 1))
    StepIndState[:t][] = circshift(StepIndState[:t][], (1))

    # update t
    StepIndState[:t][][1] = t

    # update x
    x_dot = (StepIndSys[:A] * StepIndState[:x][][:, 2] + realStepIndB * StepIndState[:u][][:, 2])
    x = StepIndState[:x][][:, 2] + dt * x_dot + rand(MvNormal(3, 0.4))

    StepIndState[:x][][:, 1], StepIndState[:P][][:, :, 1] = predictUpdateKF(StepIndSys,
        x, StepIndState[:x][][:, 2],
        StepIndState[:u][][:, 2],
        StepIndState[:P][][:, :, 2], dt)

    # update B
    B_window = 100
    if (t > 3.0 || t < 0.0)
        curr_rng = 1:3
        prev_rng = 2:4
        dts = (StepIndState[:t][][curr_rng] .- StepIndState[:t][][prev_rng])
        dts = [dts'; dts'; dts']
        y = eachcol((
            (StepIndState[:x][][:, curr_rng] .- StepIndState[:x][][:, prev_rng]) .-
            dts .* (StepIndSys[:A] * StepIndState[:x][][:, prev_rng])) ./ dts)
        u = eachcol(StepIndState[:u][][:, prev_rng])
        try
            StepIndSys[:B] = estimateB(u, y, StepIndBRLS)
        catch
        end
        StepIndState[:em][][:, 1] = StepIndState[:x][][:, 1] .- (StepIndState[:x][][:, 2] + StepIndSys[:A] * StepIndState[:x][][:, 2] + StepIndSys[:B] * StepIndState[:u][][:, 2])
    end


    ## Rot Stage
    # shift all
    RotStageState[:x][] = circshift(RotStageState[:x][], (0, 1))
    RotStageState[:u][] = circshift(RotStageState[:u][], (0, 1))
    RotStageState[:r][] = circshift(RotStageState[:r][], (0, 1))
    RotStageState[:em][] = circshift(RotStageState[:em][], (0, 1))
    RotStageState[:t][] = circshift(RotStageState[:t][], (1))

    # update t
    RotStageState[:t][][1] = t

    # update x
    u = IndPixState[:u][][:, 1]
    x_dot = (IndPixSys[:A] * IndPixState[:x][][:, 1] + realIndPixB * u)
    IndPixState_x_gt = IndPixState[:x][][:, 1] + dt * x_dot + rand(MvNormal(3, 0.002))
    if t > 2.0 && t < 3.0
        IndPixState_x_gt[1:2, 1] .= 0.0
    end
    RotStageState[:x][][:, 1] = chipFK([IndPixState_x_gt; realRotStage]) #+ clamp.(rand(MvNormal(3, 0.003)), -0.01, 0.01)


    ## IndPix
    # shift all
    IndPixState[:x][] = circshift(IndPixState[:x][], (0, 1))
    IndPixState[:u][] = circshift(IndPixState[:u][], (0, 1))
    IndPixState[:r][] = circshift(IndPixState[:r][], (0, 1))
    IndPixState[:em][] = circshift(IndPixState[:em][], (0, 1))
    IndPixState[:t][] = circshift(IndPixState[:t][], (1))

    # update t
    IndPixState[:t][][1] = t

    # update x
    if t > 3.0
        IndPixState[:x][][:, 1] = chipFKreverse([RotStageState[:x][][:, 1]; CircleState[:x][][:, 1]])
        # IndPixState[:x][][:, 1] = IndPixState_x_gt
    else
        IndPixState[:x][][:, 1] = IndPixState[:x][][:, 2]
    end

    # update B
    B_window = 100
    if (t > 3.0)
        curr_rng = 1:3
        prev_rng = 2:4
        dts = (IndPixState[:t][][curr_rng] .- IndPixState[:t][][prev_rng])
        dts = [dts'; dts'; dts']
        y = eachcol((
            (IndPixState[:x][][:, curr_rng] .- IndPixState[:x][][:, prev_rng]) .-
            dts .* (IndPixSys[:A] * IndPixState[:x][][:, prev_rng])) ./ dts)
        u = eachcol(IndPixState[:u][][:, prev_rng])
        try
            IndPixSys[:B] = estimateB(u, y, IndPixBRLS)
        catch
        end
        IndPixState[:em][][:, 1] = IndPixState[:x][][:, 1] .- (IndPixState[:x][][:, 2] + (IndPixSys[:A] * IndPixState[:x][][:, 2] + IndPixSys[:B] * IndPixState[:u][][:, 2]) .* dt)
    end

    ## Circle State
    # shift all
    CircleState[:x][] = circshift(CircleState[:x][], (0, 1))
    CircleState[:em][] = circshift(CircleState[:em][], (0, 1))
    CircleState[:t][] = circshift(CircleState[:t][], (1))

    # update t
    CircleState[:t][][1] = t

    # update x
    rotfilter = (RotStageState[:t][] .> 2.0) .& (RotStageState[:t][] .< 3.0)
    rotfilter = rotfilter .| ((t .- RotStageState[:t][]) .< 0.5)
    rotfilter[1] = false
    if sum(rotfilter) > 50
        # # Linear Regression
        # circlex, circley, circler = estimateCircle(
        #     RotStageState[:x][][1, rotfilter] .- IndPixState[:x][][1, rotfilter],
        #     RotStageState[:x][][2, rotfilter] .- IndPixState[:x][][2, rotfilter])
        # CircleState[:x][][:, 1] = [circlex, circley, 0.0, circler]

        # Nonlinear Regression
        model = function (t, p)
            x, y, theta = t[1, :], t[2, :], t[3, :]
            cx, cy, ctheta, cr = p
            return vec(chipFKbatch(x, y, theta, cx, cy, ctheta, cr))
        end
        # fit_OLS = curve_fit(model,
        #     IndPixState[:x][][:, rotfilter],
        #     vec(RotStageState[:x][][:, rotfilter]),
        #     CircleState[:x][][:, 1],
        #     lower=[-5, -5, -pi, 0.0],
        #     upper=[5, 5, pi, 2.0],
        #     ; autodiff=:forwarddiff)
        # wt = 1 ./ ((fit_OLS.resid .+ 1.0e-10) .^ 2)
        # fit_WLS = curve_fit(model,
        #     IndPixState[:x][][:, rotfilter],
        #     vec(RotStageState[:x][][:, rotfilter]),
        #     wt,
        #     fit_OLS.param,
        #     lower=[-5, -5, -pi, 0.0],
        #     upper=[5, 5, pi, 2.0],
        #     ; autodiff=:forwarddiff)
        # CircleState[:x][][:, 1] = fit_WLS.param
    else
        CircleState[:x][][:, 1] = CircleState[:x][][:, 2]
    end
    CircleState[:x][][:, 1] = realRotStage

    ## Update Controls
    if t < 5.0
        # Calibration mode
        IndPixState[:r][][:, 1] = IndPixState[:u][][:, 1] = bootstrap_stepper_u(t)

        # StepInd control
        StepIndState[:r][][:, 1] = -10.0 * (IndPixState[:u][][:, 1] * dt)
        StepIndState[:r][][:, 1] = clamp.(StepIndState[:r][][:, 1], -10.0, 10.0)
        StepIndState[:u][][:, 1] = StepIndState[:r][][:, 1]
        StepIndState[:u][][:, 1], StepIndState[:em][][:, 1] = mrac(
            StepIndSys,
            StepIndMRAC,
            StepIndState[:r][][:, 1],
            StepIndState[:r][][:, 2],
            StepIndState[:x][][:, 1],
            StepIndState[:x][][:, 2],
            dt)
    else
        # IndPix control
        target_pose = [0.5, 0.5, 1.57]
        current_pose = RotStageState[:x][][:, 1]

        # tracking error
        RotStageState[:em][][:, 1] = (target_pose - current_pose)

        # # Shorten the target to be close when far away to at most 10 pixels
        # pathlenmax = 0.1
        # if norm(target_pose - current_pose) > pathlenmax
        #     target_pose = current_pose + pathlenmax * (target_pose - current_pose) / norm(target_pose - current_pose)
        # end


        # IndPix control
        try
            # Calculate Inverse Kinematics
            # maxiter = 1000
            # tol = 1.0e-6
            # alpha = 0.1
            # x = IndPixState[:x][][:, 1]
            # for i in 1:maxiter
            #     x = x - alpha * (pinv(ForwardDiff.jacobian(chipFK, [x; realRotStage])) * (chipFK([x; realRotStage]) - target_pose))[1:3]
            #     if norm(chipFK([x; realRotStage]) - target_pose) < tol
            #         break
            #     end
            # end
            # IndPixState[:r][][:, 1] = IndPixState[:u][][:, 1] = clamp.(pinv(IndPixSys[:B]) * (x - IndPixState[:x][][:, 1]), -10.0, 10.0)

            difftottarget = (
                pinv(ForwardDiff.jacobian(
                chipFK,
                [
                    IndPixState[:x][][:, 1];
                    CircleState[:x][][:, 1]
                ]
            ))* (target_pose - current_pose)
            )[1:3]
            IndPixState[:r][][:, 1] = IndPixState[:u][][:, 1] = clamp.(pinv(IndPixSys[:B]) * difftottarget, -10.0, 10.0)
        catch
            println("jacobian error")
            break
        end


        # StepInd control
        StepIndState[:r][][:, 1] = -400.0 * (IndPixState[:u][][:, 1] * dt)
        StepIndState[:r][][:, 1] = clamp.(StepIndState[:r][][:, 1], -10.0, 10.0)
        StepIndState[:u][][:, 1] = StepIndState[:r][][:, 1]
        StepIndState[:u][][:, 1], StepIndState[:em][][:, 1] = mrac(
            StepIndSys,
            StepIndMRAC,
            StepIndState[:r][][:, 1],
            StepIndState[:r][][:, 2],
            StepIndState[:x][][:, 1],
            StepIndState[:x][][:, 2],
            dt)
    end
end

f = Figure()

ax = Axis(f[1, 1], title="StepIndState[:r]")
lines!(ax, StepIndState[:t][], StepIndState[:r][][1, :])
lines!(ax, StepIndState[:t][], StepIndState[:r][][2, :])
lines!(ax, StepIndState[:t][], StepIndState[:r][][3, :])

ax = Axis(f[1, 2], title="StepIndState[:u]")
lines!(ax, StepIndState[:t][], StepIndState[:u][][1, :])
lines!(ax, StepIndState[:t][], StepIndState[:u][][2, :])
lines!(ax, StepIndState[:t][], StepIndState[:u][][3, :])

ax = Axis(f[1, 3], title="StepIndState[:x]")
lines!(ax, StepIndState[:t][], StepIndState[:x][][1, :])
lines!(ax, StepIndState[:t][], StepIndState[:x][][2, :])
lines!(ax, StepIndState[:t][], StepIndState[:x][][3, :])

ax = Axis(f[1, 4], title="StepIndState[:em]")
lines!(ax, StepIndState[:t][], StepIndState[:em][][1, :])
lines!(ax, StepIndState[:t][], StepIndState[:em][][2, :])
lines!(ax, StepIndState[:t][], StepIndState[:em][][3, :])

ax = Axis(f[2, 1], title="IndPixState[:r]")
lines!(ax, IndPixState[:t][], IndPixState[:r][][1, :])
lines!(ax, IndPixState[:t][], IndPixState[:r][][2, :])
lines!(ax, IndPixState[:t][], IndPixState[:r][][3, :])

ax = Axis(f[2, 2], title="IndPixState[:u]")
lines!(ax, IndPixState[:t][], IndPixState[:u][][1, :])
lines!(ax, IndPixState[:t][], IndPixState[:u][][2, :])
lines!(ax, IndPixState[:t][], IndPixState[:u][][3, :])

ax = Axis(f[2, 3], title="IndPixState[:x]")
lines!(ax, IndPixState[:t][], IndPixState[:x][][1, :])
lines!(ax, IndPixState[:t][], IndPixState[:x][][2, :])
lines!(ax, IndPixState[:t][], IndPixState[:x][][3, :])

ax = Axis(f[2, 4], title="IndPixState[:em]")
lines!(ax, IndPixState[:t][], IndPixState[:em][][1, :])
lines!(ax, IndPixState[:t][], IndPixState[:em][][2, :])
lines!(ax, IndPixState[:t][], IndPixState[:em][][3, :])

ax = Axis(f[3, 1], title="RotStageState[:x]")
lines!(ax, RotStageState[:t][], RotStageState[:x][][1, :], label="x")
lines!(ax, RotStageState[:t][], RotStageState[:x][][2, :], label="y")
lines!(ax, RotStageState[:t][], RotStageState[:x][][3, :], label="theta")

ax = Axis(f[3, 2], title="RotStageState[:em]")
lines!(ax, RotStageState[:t][], RotStageState[:em][][1, :], label="x")
lines!(ax, RotStageState[:t][], RotStageState[:em][][2, :], label="y")
lines!(ax, RotStageState[:t][], RotStageState[:em][][3, :], label="theta")
axislegend(ax, merge=true, unique=true)

ax = Axis(f[3, 3], title="CircleState[:x]")
lines!(ax, CircleState[:t][], CircleState[:x][][1, :], label="x")
lines!(ax, CircleState[:t][], CircleState[:x][][2, :], label="y")
lines!(ax, CircleState[:t][], CircleState[:x][][3, :], label="r")
axislegend(ax, merge=true, unique=true)

ax = Axis(f[3, 4], aspect=DataAspect())
# color rainbow
scatter!(ax, RotStageState[:x][][1, :], RotStageState[:x][][2, :], color=RotStageState[:t][], colormap=:rainbow)
arrows!(ax, RotStageState[:x][][1, :], RotStageState[:x][][2, :],
    0.01 .* cos.(RotStageState[:x][][3, :]),
    0.01 .* sin.(RotStageState[:x][][3, :]), color=RotStageState[:t][], colormap=:rainbow)
display(f)


using ControlSystemIdentification
using ControlSystemsBase

valid_rng = 1:2000
idinput = reverse(IndPixState[:u][][1:2, valid_rng], dims=2)
idoutput = detrend(reverse(IndPixState[:x][][1:2, valid_rng], dims=2))
ts = reverse(IndPixState[:t][][valid_rng], dims=1)

myiddata = iddata(
    idoutput,
    idinput,
    sum(ts[2:end] - ts[1:end-1]) / length(ts),
)
ssid = subspaceid(myiddata, 2)


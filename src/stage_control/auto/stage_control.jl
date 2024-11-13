using LibSerialPort
using LinearAlgebra
using ProgressMeter
using Serialization
using Distributions
using LinearAlgebra
using Flux
using ForwardDiff
using ZMQ
using ZeroMQ_jll
using MsgPack
using ControlSystemIdentification
using ControlSystemsBase
include("utils.jl")

portname1 = "COM6"
baudrate = 115200

## Data
window_size = 5000

# Vision System
VisionErrSys = Dict(
    :A => [0.0 0.0 0.0
        0.0 0.0 0.0
        0.0 0.0 0.0],
    :B => [1.0 0.0 0.0 # estimated
        0.0 1.0 0.0
        0.0 0.0 1.0],
    :C => [1.0 0.0 0.0 # estimated
        0.0 1.0 0.0
        0.0 0.0 1.0],
    :D => [0.0 0.0 0.0
        0.0 0.0 0.0
        0.0 0.0 0.0],
    :A_m => [0.0 0.0 0.0
        0.0 0.0 0.0
        0.0 0.0 0.0],
    :B_m => [1.0 0.0 0.0 # estimated
        0.0 1.0 0.0
        0.0 0.0 1.0],
    # Noise Model
    :R => 100.0 * Diagonal([1.0 for i in 1:3]),
    :Q => 100.0 * Diagonal([1.0 for i in 1:3]),
    :lock => ReentrantLock()
)

# using ControlSystemsBase
# lqr(ControlSystemsBase.Discrete, VisionErrSys[:A] + I, VisionErrSys[:B], I, I)


VisionErrBRLS = Dict(
    :x0 => zeros(9),
    :P0 => 100.0 * Diagonal([1.0 for i in 1:9]),
    :R => 100.0 * Diagonal([1.0 for i in 1:9]),
    :λ => 0.9999
)

VisionErrMRAC = Dict(
    :P => lyap((VisionErrSys[:A_m] + Diagonal([1.0, 1.0, 1.0]))', -Diagonal([1.0, 1.0, 1.0])),
    :Γ_x => 100.0,
    :Γ_r => 100.0,
    :Γ_w => 50.0,
    :Γ_v => 50.0,
    :Γ_σ => 0.0,
    :Λ => Diagonal([1.0, 1.0, 1.0]),
    :K_x => zeros(3, 3),
    :K_r => Diagonal([1.0, 1.0, 1.0]),
    :W => zeros(10, 3), # (hidden_size, state_size)
    :V => Flux.glorot_uniform(7, 10), # (feature_size, hidden_size)
)

# Vision State
VisionErrState = Dict(
    :x => Observable(zeros(3, window_size)), # x, y, theta
    :xi => Observable(zeros(3, window_size)), # x, y, theta
    :xd => Observable(zeros(3, window_size)), # x, y, theta
    :u => Observable(zeros(3, window_size)),
    :r => Observable(zeros(3, window_size)),
    :t => Observable(zeros(window_size)),
    :em => Observable(zeros(3, window_size)), # error of model
    :P => Observable(stack([Diagonal([10.0 for i in 1:3]) for i in 1:window_size], dims=3)),
    :lock => ReentrantLock()
)

# Vision Thread
function updateVisionError(errCh, sys, state, BRLS, mracparam, t0)
    global running, mouseinit
    initidxend = -1
    LibSerialPort.open(portname1, baudrate) do sp1
        prev_t = time_ns()
        while running
            statedata = take!(errCh)
            # println(statedata)

            lock(state[:lock]) do
                lock(sys[:lock]) do
                    if (time_ns() - t0) * 1.0e-9 > 0.0 && (time_ns() - t0) * 1.0e-9 < 170.0
                        # calibration trajectory
                        # r = bootstrap_stepper_u0((time_ns() - t0) * 1.0e-9, 0.06)
                        r = chirp((time_ns() - t0 - 0) * 1.0e-9, 1.0, 15, 0.03)
                        # if (time_ns() - t0) * 1.0e-9 < 100.0
                        #     r = bootstrap_stepper_u0((time_ns() - t0) * 1.0e-9, 0.06)
                        initidxend += 1
                    else
                        if initidxend != -1
                            open("VisionErrState_chirp.jls", "w") do f
                                serialize(f, Dict(
                                    "x" => state[:x][],
                                    "xi" => state[:xi][],
                                    "xd" => state[:xd][],
                                    "u" => state[:u][],
                                    "r" => state[:r][],
                                    "t" => state[:t][],
                                    "em" => state[:em][],
                                    "P" => state[:P][],
                                ))
                            end
                            valid_rng = 1:initidxend
                            idinput = reverse(state[:u][][1:2, valid_rng], dims=2)
                            idoutput = detrend(reverse(state[:x][][1:2, valid_rng], dims=2))
                            ts = reverse(state[:t][][valid_rng], dims=1)

                            myiddata = iddata(
                                idoutput,
                                idinput,
                                sum(ts[2:end] - ts[1:end-1]) / length(ts),
                            )

                            ssid = subspaceid(myiddata, 2, zeroD=true, verbose=true)
                            sys[:B][1:2, 1:2] = ssid.B
                            sys[:C][1:2, 1:2] = ssid.C
                            initidxend = -1
                        end

                        # minimize reference using simple pid
                        r = (0.008) * pinv(sys[:B]) * ((state[:x][][:, 1])) +
                            (0.00007) * pinv(sys[:B]) * ((state[:xi][][:, 1])) +
                            (0.003) * pinv(sys[:B]) * ((state[:xd][][:, 1]))

                        # r = (0.01) * pinv(sys[:B]) * ((state[:x][][:, 1]))
                        # r = (0.025) * ((state[:x][][:, 1]))
                    end

                    state[:t][] = circshift(state[:t][], (1,))
                    state[:x][] = circshift(state[:x][], (0, 1))
                    state[:xi][] = circshift(state[:xi][], (0, 1))
                    state[:xd][] = circshift(state[:xd][], (0, 1))
                    state[:r][][:, end] .= 0.0
                    state[:r][] = circshift(state[:r][], (0, 1))
                    state[:u][][:, end] .= 0.0
                    state[:u][] = circshift(state[:u][], (0, 1))
                    state[:em][] = circshift(state[:em][], (0, 1))
                    state[:P][] = circshift(state[:P][], (0, 0, 1))

                    # Default
                    state[:t][][1] = statedata[1] / 1e9
                    state[:x][][:, 1] = statedata[2:4]
                    state[:xi][][:, 1] = state[:xi][][:, 2]
                    state[:xd][][:, 1] = state[:xd][][:, 2]
                    state[:r][][:, 1] = r
                    state[:u][][:, 1] = r
                    state[:em][][:, 1] = state[:em][][:, 2]
                    state[:P][][:, :, 1] = state[:P][][:, :, 2]

                    # Update
                    if (time_ns() - t0) * 1.0e-9 > 0
                        # KF only after calibration
                        state[:x][][:, 1], state[:P][][:, :, 1] = predictUpdateKF(
                            sys,
                            state[:x][][:, 1],
                            state[:x][][:, 2],
                            state[:u][][:, 2],
                            state[:P][][:, :, 2],
                            state[:t][][1] - state[:t][][2])
                    end
                    state[:xi][][:, 1] = state[:x][][:, 1] + state[:xi][][:, 2]
                    state[:xd][][:, 1] = (state[:x][][:, 1] - state[:x][][:, 2])
                    state[:em][][:, 1] = (state[:x][][:, 1] - state[:x][][:, 2]) .-
                                         (state[:t][][1] - state[:t][][2]) .*
                                         (sys[:A] * state[:x][][:, 2] .+ sys[:B] * state[:u][][:, 2])


                    # # Update B
                    # if state[:t][][1] - t0 < 0 # disabled
                    #     curr_rng = 1:3
                    #     prev_rng = 2:4
                    #     dts = (state[:t][][curr_rng] .- state[:t][][prev_rng])
                    #     dts = [dts'; dts'; dts']
                    #     y = eachcol((
                    #         (state[:x][][:, curr_rng] .- state[:x][][:, prev_rng]) .-
                    #         dts .* (sys[:A] * state[:x][][:, prev_rng])) ./ dts)
                    #     u = eachcol(state[:u][][:, prev_rng])
                    #     sys[:B] = estimateB(u, y, BRLS)
                    # # end

                    if state[:t][][1] - t0 > 1
                        state[:u][][:, 1], state[:em][][:, 1] = mrac(
                            sys,
                            mracparam,
                            state[:r][][:, 1],
                            state[:r][][:, 2],
                            state[:x][][:, 1],
                            state[:x][][:, 2],
                            state[:t][][1] - state[:t][][2]
                        )
                    end

                    # temporary zero out theta
                    state[:u][][:, 1] = clamp.(state[:u][][:, 1], -0.1, 0.1)
                    state[:u][][3, 1] = 0.0

                end
            end
            yield()
            if mouseinit[]
                send_motor_cmd(sp1, state[:u][][:, 1])
            end
        end
    end
end
Threads.@spawn updateVisionError(visionCh, VisionErrSys, VisionErrState, VisionErrBRLS, VisionErrMRAC, time_ns())
# updateVisionError(visionCh, VisionErrSys, VisionErrState, VisionErrBRLS, VisionErrMRAC, time_ns())

# VisionErrState[:u][][:, 1]
# (-0.3) * pinv(VisionErrSys[:B]) * VisionErrState[:x][][:, 1] #+ (-0.01) * pinv(VisionErrSys[:B]) * VisionErrState[:xi][][:, 1]
# VisionErrSys[:B]
# running = true
# (-0.0001) * pinv(VisionErrSys[:B]) * (pinv(VisionErrSys[:C]) * (VisionErrState[:x][][:, 1] - VisionErrSys[:D] * VisionErrState[:u][][:, 2]))
# VisionErrState[:u][][:, 4]
# VisionErrSys[:B]
# VisionErrState[:u][][:, :]


# LibSerialPort.open(portname1, baudrate) do sp1
#     for i in 1:100
#         send_motor_cmd(sp1, [-0.01, 0.0, 0.0])
#     end
# end

# 1.0 * pinv(VisionErrSys[:B]) * VisionErrState[:x][][:, 1]

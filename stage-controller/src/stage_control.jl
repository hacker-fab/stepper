using LibSerialPort
using GLMakie
using LinearAlgebra
using ProgressMeter
using Serialization
using Distributions
using LowLevelParticleFilters
using LinearAlgebra
using Flux
using ForwardDiff
using ZMQ
using ZeroMQ_jll
using MsgPack
GLMakie.activate!(inline=false)
include("stage_control/utils.jl")

portname1 = "/dev/ttyACM0"
portname2 = "/dev/ttyACM1"
baudrate = 115200

# ZMQ
ctx = Context()
pixelerr = Socket(ctx, SUB)
# Set Conflate == 1
rc = ccall((:zmq_setsockopt, libzmq), Cint, (Ptr{Cvoid}, Cint, Ref{Cint}, Csize_t), pixelerr, 54, 1, sizeof(Cint))
ZMQ.subscribe(pixelerr, "")
connect(pixelerr, "tcp://10.193.10.1:5556")

## Data
window_size = 12000

# Vision System
VisionErrSys = Dict(
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
    :R => 100.0 * Diagonal([1.0 for i in 1:3]),
    :Q => 100.0 * Diagonal([1.0 for i in 1:3]),
    :lock => ReentrantLock()
)

VisionErrBRLS = Dict(
    :x0 => zeros(9),
    :P0 => 100.0 * Diagonal([1.0 for i in 1:9]),
    :R => 100.0 * Diagonal([1.0 for i in 1:9]),
    :λ => 0.9999
)

VisionErrMRAC = Dict(
    :P => lyap((VisionErrSys[:A_m] + Diagonal([1.0, 1.0, 1.0]))', -Diagonal([1.0, 1.0, 1.0])),
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

# Vision State
VisionErrState = Dict(
    :x => Observable(zeros(3, window_size)), # x, y, theta
    :u => Observable(zeros(3, window_size)),
    :r => Observable(zeros(3, window_size)),
    :t => Observable(zeros(window_size)),
    :em => Observable(zeros(3, window_size)), # error of model
    :P => Observable(stack([Diagonal([10.0 for i in 1:3]) for i in 1:window_size], dims=3)),
    :lock => ReentrantLock()
)

## Visualize
f = Figure()

ax = Axis(f[1, 1])
lines!(ax, @lift($(VisionErrState[:x])[1, :]), color=:red)
lines!(ax, @lift($(VisionErrState[:x])[2, :]), color=:black)
# lines!(ax, @lift($(VisionErrState[:x])[3, :]), color=:blue)

ax = Axis(f[2, 1])
lines!(ax, @lift($(VisionErrState[:u])[1, :]), color=:red)
lines!(ax, @lift($(VisionErrState[:u])[2, :]), color=:black)
lines!(ax, @lift($(VisionErrState[:u])[3, :]), color=:blue)

ax = Axis(f[3, 1])
lines!(ax, @lift($(VisionErrState[:em])[1, :]), color=:red)
lines!(ax, @lift($(VisionErrState[:em])[2, :]), color=:black)
lines!(ax, @lift($(VisionErrState[:em])[3, :]), color=:blue)

display(f)
# Global Status
running = true

# Vision Thread
function updateVisionError(zmqsocket, sys, state, BRLS, mracparam, t0)
    global running

    LibSerialPort.open(portname1, baudrate) do sp1
        prev_t = time_ns()
        while running
            data = ZMQ.recv(zmqsocket)
            statedata = Float64.(MsgPack.unpack(data)[1:4])
            
            lock(state[:lock]) do
                lock(sys[:lock]) do
                    if (time_ns() - t0) * 1.0e-9 < 40.0
                        # calibration trajectory
                        r = bootstrap_stepper_u0((time_ns() - t0) * 1.0e-9, 0.01)
                    elseif  (time_ns() - t0) * 1.0e-9 < 100.0
                        r = bootstrap_stepper_u0((time_ns() - t0) * 1.0e-9, 0.01)
                    else
                        # minimize reference using simple pid
                        r = -0.05 * pinv(sys[:B]) * state[:x][][:, 1]
                    end

                    state[:t][] = circshift(state[:t][], (1,))
                    state[:x][] = circshift(state[:x][], (0, 1))
                    state[:r][][:, end] .= 0.0
                    state[:r][] = circshift(state[:r][], (0, 1))
                    state[:u][][:, end] .= 0.0
                    state[:u][] = circshift(state[:u][], (0, 1))
                    state[:em][] = circshift(state[:em][], (0, 1))
                    state[:P][] = circshift(state[:P][], (0, 0, 1))

                    # Default
                    state[:t][][1] = statedata[1] / 1e9
                    state[:x][][:, 1] = statedata[2:4]
                    state[:r][][:, 1] = r
                    state[:u][][:, 1] = r
                    state[:em][][:, 1] = state[:em][][:, 2]
                    state[:P][][:, :, 1] = state[:P][][:, :, 2]

                    # Update
                    state[:x][][:, 1], state[:P][][:, :, 1] = predictUpdateKF(
                        sys,
                        state[:x][][:, 1],
                        state[:x][][:, 2],
                        state[:u][][:, 2],
                        state[:P][][:, :, 2],
                        state[:t][][1] - state[:t][][2])
                    state[:em][][:, 1] = (state[:x][][:, 1] - state[:x][][:, 2]) .-
                                         (state[:t][][1] - state[:t][][2]) .*
                                         (sys[:A] * state[:x][][:, 2] .+ sys[:B] * state[:u][][:, 2])

                    # Update B
                    if state[:t][][1] - t0 < 40.0
                        curr_rng = 1:3
                        prev_rng = 2:4
                        dts = (state[:t][][curr_rng] .- state[:t][][prev_rng])
                        dts = [dts'; dts'; dts']
                        y = eachcol((
                            (state[:x][][:, curr_rng] .- state[:x][][:, prev_rng]) .-
                            dts .* (sys[:A] * state[:x][][:, prev_rng])) ./ dts)
                        u = eachcol(state[:u][][:, prev_rng])
                        sys[:B] = estimateB(u, y, BRLS)
                    end

                    if state[:t][][1] - t0 > 40.0
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
            send_motor_cmd(sp1, state[:u][][:, 1])
        end
    end
end
Threads.@spawn updateVisionError(pixelerr, VisionErrSys, VisionErrState, VisionErrBRLS, VisionErrMRAC, time_ns()) 

while true
    # lock(VisionErrState[:lock]) do
    notify(VisionErrState[:x])
    notify(VisionErrState[:t])
    notify(VisionErrState[:u])
    notify(VisionErrState[:em])
    #    println(VisionErrState[:t][][1])
    # end
    yield()
end
running = false

VisionErrSys[:B]
VisionErrState[:u][][:, :]


LibSerialPort.open(portname1, baudrate) do sp1
    send_motor_cmd(sp1, [0.0, 
    -0.16, -0.0])
end

1.0 * pinv(VisionErrSys[:B]) * VisionErrState[:x][][:, 1]
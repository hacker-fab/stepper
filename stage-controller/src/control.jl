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
GLMakie.activate!(inline=false)

portname1 = "/dev/ttyACM0"
portname2 = "/dev/ttyACM1"
baudrate = 115200

function joint_vel_pwm(joint_vel, dt)
    # convert number of steps per second to number of cycles per steps
    # given cps, cycles per second
    # 1 cycle = 1 step
    # the greater the number of cycles per steps, the slower the motor
    # the greater the joint_vel, the smaller the number of cycles per steps, the faster the motor
    joint_vel = clamp(joint_vel, -5, 5)
    cps = 1 / dt
    if joint_vel == 0
        return 10000000000000
    end
    return round(Int, cps / joint_vel)
end

function send_recieve_serial!(data, sp1, sp2, vel, x0, dt)
    write(sp1, "x\n")
    write(sp2, "mr $(joint_vel_pwm(vel[1], dt)) $(joint_vel_pwm(vel[2], dt)) $(joint_vel_pwm(vel[3], dt))\n")
    while bytesavailable(sp1) < 1
    end
    try
        sp1_data = String(readline(sp1, keep=false))
        sp1_reading = split(sp1_data, ',')
        data[1] = ((parse(Int, sp1_reading[1])) - x0[1]) * 1.0e-7
        data[3] = ((parse(Int, sp1_reading[2])) - x0[3]) * 1.0e-7
    catch
    end
    while bytesavailable(sp2) < 1
    end
    try
        sp2_reading = ((parse(Int, String(readline(sp2, keep=false)))) - x0[5]) * 1.0e-7
        data[5] = sp2_reading
    catch
    end
    return data
end

dt = -1
x0 = zeros(6)
LibSerialPort.open(portname1, baudrate) do sp1
    LibSerialPort.open(portname2, baudrate) do sp2
        start_t = time_ns()
        samples = 100
        global x0
        global dt
        for i in 1:samples
            send_recieve_serial!(x0, sp1, sp2, [0, 0, 0], zeros(6), dt)
        end
        dt = (time_ns() - start_t) / samples / 1e9
    end
end

window_size = 6000
# state, [x, ẋ, y, ẏ, z, ż]
is = Observable(zeros(window_size, 6))
xs = @lift($is[:, 1])
ẋs = @lift($is[:, 2])
ys = @lift($is[:, 3])
ẏs = @lift($is[:, 4])
zs = @lift($is[:, 5])
żs = @lift($is[:, 6])

# error p, i, d, [xd, xp, xi, yd, yp, yi, zd, zp, zi]
es = Observable(zeros(window_size, 9))
exds = @lift($es[:, 1])
exps = @lift($es[:, 2])
exis = @lift($es[:, 3])
eyds = @lift($es[:, 4])
eyps = @lift($es[:, 5])
eyis = @lift($es[:, 6])
ezds = @lift($es[:, 7])
ezps = @lift($es[:, 8])
ezis = @lift($es[:, 9])

# reference control input, [ẋ, ẏ, ż]
us = Observable(zeros(window_size, 3))
ẋs = @lift($us[:, 1])
ẏs = @lift($us[:, 2])
żs = @lift($us[:, 3])

# stepper motor control input, [uA, uB, uC]
ps = Observable(zeros(window_size, 3))
uA = @lift($ps[:, 1])
uB = @lift($ps[:, 2])
uC = @lift($ps[:, 3])

# RMAC stats
x_n = Observable(zeros(window_size, 3))
x_n_x = @lift($x_n[:, 1])
x_n_y = @lift($x_n[:, 2])
x_n_z = @lift($x_n[:, 3])

u_ad = Observable(zeros(window_size, 3))
u_ad_x = @lift($u_ad[:, 1])
u_ad_y = @lift($u_ad[:, 2])
u_ad_z = @lift($u_ad[:, 3])

e_ad = Observable(zeros(window_size, 3))
e_ad_x = @lift($e_ad[:, 1])
e_ad_y = @lift($e_ad[:, 2])
e_ad_z = @lift($e_ad[:, 3])

f = Figure()
axx = Axis(f[1, 1], xlabel="Stage Pos")
lines!(axx, 1:window_size, xs, color=:blue)
lines!(axx, 1:window_size, ys, color=:red)
lines!(axx, 1:window_size, zs, color=:black)
axx = Axis(f[1, 2], xlabel="Stage Vel")
lines!(axx, 1:window_size, ẋs, color=:blue)
lines!(axx, 1:window_size, ẏs, color=:red)
lines!(axx, 1:window_size, żs, color=:black)
axu = Axis(f[2, 1], xlabel="Ref u")
lines!(axu, 1:window_size, ẋs, color=:blue)
lines!(axu, 1:window_size, ẏs, color=:red)
lines!(axu, 1:window_size, żs, color=:black)
axu̇ = Axis(f[2, 2], xlabel="Stepper u")
lines!(axu̇, 1:window_size, uA, color=:blue)
lines!(axu̇, 1:window_size, uB, color=:red)
lines!(axu̇, 1:window_size, uC, color=:black)
axadapt_x = Axis(f[3, 1], xlabel="RMAC x_n")
lines!(axadapt_x, 1:window_size, x_n_x, color=:blue)
lines!(axadapt_x, 1:window_size, x_n_y, color=:red)
lines!(axadapt_x, 1:window_size, x_n_z, color=:black)
axadapt_x = Axis(f[3, 2], xlabel="RMAC u_ad")
lines!(axadapt_x, 1:window_size, u_ad_x, color=:blue)
lines!(axadapt_x, 1:window_size, u_ad_y, color=:red)
lines!(axadapt_x, 1:window_size, u_ad_z, color=:black)
axadapt_e = Axis(f[4, 1], xlabel="e")
lines!(axadapt_e, 1:window_size, exps, color=:blue)
lines!(axadapt_e, 1:window_size, eyps, color=:red)
lines!(axadapt_e, 1:window_size, ezps, color=:black)
axadapt_nn = Axis(f[4, 2], xlabel="adaptation_loss")
lines!(axadapt_nn, 1:window_size, e_ad_x, color=:blue)
lines!(axadapt_nn, 1:window_size, e_ad_y, color=:red)
lines!(axadapt_nn, 1:window_size, e_ad_z, color=:black)
display(f)

# Stage state transition
A_s = [1 dt 0 0 0 0
    0 1 0 0 0 0
    0 0 1 dt 0 0
    0 0 0 1 0 0
    0 0 0 0 1 dt
    0 0 0 0 0 1]
# Stage control input
B_s = [0.5*dt^2 0 0
    dt 0 0
    0 0.5*dt^2 0
    0 dt 0
    0 0 0.5*dt^2
    0 0 dt]
# Stage Observer state transition
C_s = [1.0 0 0 0 0 0
    0 0 1.0 0 0 0
    0 0 0 0 1.0 0]
# Stage Observer control input
D_s = [0 0 0
    0 0 0
    0 0 0]

dw = MvNormal(6, 1.0)          # Dynamics noise Distribution
de = MvNormal(3, 1.0)          # Measurement noise Distribution
d0 = MvNormal(6, 1.0)   # Initial state Distribution
kf = KalmanFilter(A_s, B_s, C_s, D_s, cov(dw), cov(de), d0, α=1.5)

# Refernce state transition
A_m = [1.0 0 0
    0 1.0 0
    0 0 1.0]
# Refernce control input
B_m = [dt 0 0
    0 dt 0
    0 0 dt]
# Refernce Observer state transition
C_m = [1.0 0 0
    0 1.0 0
    0 0 1.0]
# Refernce Observer control input
D_m = [0.0 0 0
    0 0 0
    0 0 0]

# Stepper state transition
A_n = [1.0 0 0
    0 1.0 0
    0 0 1.0]
# Stepper control input
B_n = [dt 0 0
    0 dt 0
    0 0 dt]

LibSerialPort.open(portname1, baudrate) do sp1
    LibSerialPort.open(portname2, baudrate) do sp2
        global W, V, K_x, K_r, x_n, u_n, x_m, Λ, A_n, B_n, dt, Γ_x, Γ_r, Γ_w, Γ_V, Γ_σ, P, A_m, B_m, x0, is, us, x_n_log, u_n_log, e_log, Θ, ϕ, opt_state, x_m_track, u_m, kf, ekf, es, ps, feat

        # Implements Model-Refernce Adaptive Control (MRAC) using 
        # a linear reference model in the joint space
        # and single-layer neural network for Disturbance and Uncertainty Model
        # see http://liberzon.csl.illinois.edu/teaching/ece517notes-post.pdf
        # see https://www.mathworks.com/help/slcontrol/ug/model-reference-adaptive-control.html
        # p127** https://www.cds.caltech.edu/archive/help/uploads/wiki/files/140/IEEE_WorkShop_Slides_Lavretsky.pdf

        # Refernce model in joint space theta (ẋ_m = Am xm + Bm um)
        # this is from the inductance sensor output
        x_m = zeros(3) # current state, [position, velocity, position, velocity, position, velocity]

        # define lyapunov function as V = 1/2 * x^T * P * x
        # state cost for lyapunov function
        Q = Diagonal([0.1, 0.1, 0.1]) # position cost is 100 times velocity cost
        # Solve the P matrix, which is the solution to the Lyapunov equation:
        # A^T * P + P * A = -Q
        P = lyap(A_m', -Q)

        # learning rate for weight update
        Γ_x = 1
        Γ_r = 1
        Γ_w = 1
        Γ_V = 1
        Γ_σ = 0.0 # sigma modification to add damping

        # Adaptive parameters
        ## Nominal stepper linear model ẋ = A_n x_n + B_n Λ (u_n - f(x))
        Λ = Diagonal([1, 1, 1]) # signs, check p80

        ## UNKNOWN gains of nominal u_n
        ## u_n = K_x' * x_n + K_r' * r_n + W' * ϕ(x_n)
        ## where ϕ(x) = σ(V^T * x) is a single-layer neural network
        feature_size = 16
        K_x = zeros(3, 3)
        K_r = zeros(3, 3)
        W = Flux.glorot_uniform(feature_size, 3)
        V = Flux.glorot_uniform(10, feature_size)

        # target joint state and control
        x_m_track = copy(x0[1:2:end])
        x_m_track[1:end] .= 0.0

        for i = 1:100
            t_prev = Int64(time_ns())
            t_start = Int64(time_ns())
            pos, vel, acc = traj(time_ns(), t_start, i % 3 + 1)
            loopcnt = 0
            while true
                loopcnt += 1

                # send command + new state
                is[] = circshift(is[], (1, 0))
                is[][1, :] = send_recieve_serial!(is[][2, :], sp1, sp2, ps[][1, :], x0, dt)
                kf(us[][1, :], is[][1, 1:2:end])
                is[][1, :] = state(kf)

                # x_n
                feat = [is[][2, :]; ps[][2, :]; 1] # use previous state because the adaptation trains on the previous state
                x_n[] = circshift(x_n[], (1, 0))
                x_n[][1, :] = A_n * is[][2, 1:2:end] + B_n * Λ * (ps[][2, :] - W' * sigmoid(V' * feat))

                # adaptation
                # using previous us because they were update above
                # using current is, x_n because they were calculated from previous is, ps
                e = x_n[][1, :] - is[][1, 1:2:end]
                K̇_x = -Γ_x * (is[][1, 1:2:end] * e' * P * B_n * Λ+ Γ_σ * K_x)
                K̇_r = -Γ_r * (us[][2, :] * e' * P * B_n * Λ + Γ_σ * K_r)
                Ẇ = Γ_w * ((sigmoid(V' * feat) - ForwardDiff.jacobian(sigmoid, V' * feat) * (V' * feat)) * e' * P * B_n * Λ + Γ_σ * W)
                V̇ = Γ_V * (feat * e' * P * B_n * Λ * W' * ForwardDiff.jacobian(sigmoid, V' * feat) + Γ_σ * V)

                K_x += K̇_x
                K_r += K̇_r
                W += Ẇ
                V += V̇

                # adaptive stats
                u_ad[] = circshift(u_ad[], (1, 0))
                u_ad[][1, :] = W' * sigmoid(V' * feat)

                x_n_ad = A_n * is[][2, 1:2:end] + B_n * Λ * (ps[][2, :] - W' * sigmoid(V' * feat))
                e_ad[] = circshift(e_ad[], (1, 0))
                e_ad[][1, :] = e

                # tracking error
                es[] = circshift(es[], (1, 0))
                # Proporional tracking error
                es[][1, 2:3:end] = is[][1, 1:2:end] .- x_m_track
                # Integral tracking error
                es[][1, 3:3:end] = es[][2, 3:3:end] .+ es[][1, 2:3:end] * dt
                # Derivative tracking error
                es[][1, 1:3:end] = (es[][1, 2:3:end] .- es[][2, 2:3:end]) / dt

                # reference control (pi controller)
                kp, ki, kd = 1000, 0, 0.0
                us[] = circshift(us[], (1, 0))
                us[][1, :] = kp * es[][1, 2:3:end] + ki * es[][1, 3:3:end] + kd * es[][1, 1:3:end]

                # override x_n with observation
                x_n[][1, :] = is[][1, 1:2:end]

                # next iter params
                if i < 5
                    # track trajectory
                    pos, vel, acc = traj(time_ns(), t_start, i % 3 + 1)
                    if any(isnothing.(vel))
                        break
                    end

                    ps[] = circshift(ps[], (1, 0))
                    ps[][1, :] .= vel

                    x_m_track = is[][1, 1:2:end]
                    x_m_track[1] += 0.008
                    
                    # es[][:, 1:3:end] .= 0.0 # reset integral error
                # x_m_track = [0.12, -0.01, 0.04]
                else
                    # track point
                    Γ_w = 0.001
                    Γ_V = 0.001

                    ps[] = circshift(ps[], (1, 0))
                    ps[][1, :] = K_x' * x_n[][1, :] + K_r' * us[][2, :] + W' * sigmoid(V' * feat)
                    # ps[][1, :] = us[][1, :]
                end

                # update makie
                notify(is)
                notify(es)
                notify(us)
                notify(ps)
                notify(x_n)
                notify(u_ad)
                notify(e_ad)
                yield()
            end
            dt = (Int64(time_ns()) - t_prev) / 1e9 / loopcnt
            write(sp2, "mr 10000000000000 10000000000000 10000000000000\n")
        end
    end
end

# stop
LibSerialPort.open(portname2, baudrate) do sp2
    write(sp2, "mr 100000000000000 100000000000000 100000000000000\n")
end
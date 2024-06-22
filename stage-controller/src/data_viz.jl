# using Pkg
# ENV["PYTHON"] = abspath("venv/bin/python")
# Pkg.build("PyCall")
using StatsBase
using ZMQ
using ZeroMQ_jll
using MsgPack
using Statistics
using PyCall
using GLMakie
using LinearAlgebra
using ProgressMeter
using Serialization
pushfirst!(pyimport("sys")."path", "src")
GLMakie.activate!(inline=false)


py"""
from stage.sanga import SangaStage, SangaDeltaStage
ss = SangaDeltaStage(port = "/dev/ttyACM0")
"""

py"""
import serial
serialPort = serial.Serial(port="/dev/ttyACM0", baudrate=115200,
                                bytesize=8, timeout=1, stopbits=serial.STOPBITS_ONE)
"""


py"""serialPort.close()"""
py"""serialPort.readline()[:-1]"""

window_size = 5000
is = Observable(zeros(window_size, 2))
xs = @lift($is[:, 1])
ys = @lift($is[:, 2])

f = Figure()
ax = Axis(f[1, 1])
lines!(ax, 1:window_size, xs, color = :blue)
lines!(ax, 1:window_size, ys, color = :red)
display(f)


(parse(Int, py"""serialPort.readline()"""))
parse(Int, py"""ss.board.query("i0?")""")

inductances_x = []
inductances_y = []
positions = []
while true
    is[] = circshift(is[], (1, 0))
    # py"""serialPort.write(bytes([ord('4')]))"""
    try
        is[][1, 1] = (parse(Int, py"""serialPort.readline()[:-1]"""))
    catch
        is[][1, 1] = is[][2, 1]
    end

    # push!(inductances_x, is[][1, 1])
    # notify(is)
    # println(is[][1, 1])
    # sleep(0.01)
    yield()
end






py"""ss.move_rel_delta([3000, 3000, 3000])"""
py"""ss.move_rel_delta([-20000, -20000,-20000])"""
curr_position = py"""ss.position"""
x_sweep = curr_position[1] - 500:10:curr_position[1] + 1500
y_sweep = curr_position[2] - 750:10:curr_position[2] + 750

# traverse in z shape, reversing order in y
poses = []
for y in y_sweep
    for x in x_sweep
        push!(poses, [x, y, curr_position[3]])
    end
    x_sweep = reverse(x_sweep)
end
poses = hcat(poses...)
diffs = diff(poses, dims = 2)

inductances_x = []
inductances_y = []
positions = []
@showprogress for pose in eachcol(poses)
    # move
    py"""ss.move_abs($pose)"""

    for i in 1:10
        is[] = circshift(is[], (1, 0))
        is[][1, 1] = parse(Int, py"""ss.board.query("i0?")""")
        is[][1, 2] = parse(Int, py"""ss.board.query("i1?")""")
        position = py"""ss.position"""
        push!(inductances_x, is[][1, 1])
        push!(inductances_y, is[][1, 2])
        push!(positions, position)
        notify(is)
        # println(position)
        yield()
    end
end


expfname = "zigzag3.jld2"
open(f -> serialize(f, Dict(
    "x_sweep" => x_sweep,
    "y_sweep" => y_sweep,
    "poses" => poses,
    "inductances_x" => inductances_x, 
    "inductances_y" => inductances_y, 
    "positions" => positions
    )), expfname, "w")


    

positions_arr = hcat(positions...)

# scatter plot all positions
f = Figure()
ax = Axis3(f[1, 1], aspect = :data)
scatterlines!(ax, positions_arr[1, :], positions_arr[2, :], positions_arr[3, :], color = :blue)
display(f)

iposition_arr = hcat(inductances_x, inductances_y)'
f = Figure()
ax = Axis(f[1, 1])
# lines!(ax, iposition_arr[1, :], color = :blue)
# lines!(ax, iposition_arr[2, :], color = :red)
scatterlines!(ax, iposition_arr[1, :], iposition_arr[2, :], color = :blue)
display(f)



curr_position = py"""ss.position"""
py"""ss.move_abs([88510, 4243, -76239])"""

while true
    is[] = circshift(is[], (1, 0))
    is[][1, 1] = parse(Int, py"""ss.board.query("i0?")""")
    is[][1, 2] = parse(Int, py"""ss.board.query("i1?")""")
    notify(is)
    println(is[][1, 1])
end




# # %% start
# from stage.sanga import SangaStage, SangaDeltaStage
# ss = SangaDeltaStage(port = "/dev/ttyACM0")

# # %% move
# ss.move_rel(-200, axis = "z")

# # %%
# ss.board.query("i?")
# # %%

# num = 0
# import time
# start_t  = time.time()
# for i in range(100):
#     ss.board.query("i?")
#     ss.move_rel(0, axis = "z")
#     num += 1
# end_t = time.time()

# fps = num / (end_t - start_t)
# print(fps)
# # %%

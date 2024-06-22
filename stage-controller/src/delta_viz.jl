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
using Serialization
pushfirst!(pyimport("sys")."path", "src")
GLMakie.activate!(inline=false)


py"""
from stage.sanga import SangaStage, SangaDeltaStage
ss = SangaDeltaStage(port = "/dev/ttyACM0")
"""
tdv = py"""ss.Tdv"""

window_size = 5000
is = Observable(zeros(window_size, 2))
xs = @lift($is[:, 1])
ys = @lift($is[:, 2])

f = Figure()
ax = Axis(f[1, 1])
lines!(ax, 1:window_size, xs, color = :blue)
lines!(ax, 1:window_size, ys, color = :red)
display(f)

while true
    is[] = circshift(is[], (1, 0))
    is[][1, 1] = parse(Int, py"""ss.board.query("i0?")""")
    is[][1, 2] = parse(Int, py"""ss.board.query("i1?")""")
    notify(is)
    # println(is[][1, 1])
    py"""ss.move_rel_delta([100, 100, 100])"""
    yield()
end


py"""ss.move_rel_delta([-500, 0, 0])"""

parse(Int, py"""ss.board.query("i0?")""")

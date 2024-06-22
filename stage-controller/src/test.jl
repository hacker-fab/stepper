using GLMakie
using ProgressMeter
using Serialization
using NearestNeighbors
using Random
using StatsBase
using MLUtils
GLMakie.activate!()


f = Figure()
ax = Axis(f[1, 1])
lines!(ax, [1, 2, 3], [1, 2, 3])
display(f)
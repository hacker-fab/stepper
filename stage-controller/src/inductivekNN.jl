using GLMakie
using ProgressMeter
using Serialization
using NearestNeighbors
using Random
using StatsBase
using MLUtils
GLMakie.activate!()

# deserialize
data = deserialize("zigzag3.jld2")
positions = data["positions"]
inductances_x = data["inductances_x"]
inductances_y = data["inductances_y"]
positions_arr = hcat(positions...)
# positions_arr = tdv * positions_arr
iposition_arr = hcat(inductances_x, inductances_y)'

train_inds = (1:size(iposition_arr, 2)) .% 10 .!= 0
test_inds = (1:size(iposition_arr, 2)) .% 10 .== 0
inds = 1:size(iposition_arr, 2)

kdtree = KDTree(iposition_arr[1:2, inds[train_inds]]; leafsize=10)

points = iposition_arr[1:2, inds[test_inds]]
knninds, knndists = knn(kdtree, points, 1000)

predpositions_arr = zeros(2, size(points, 2))
@showprogress for pt in 1:size(points, 2)
    predpositions_arr[:, pt] = mean(positions_arr[1:2, knninds[pt]], dims=2, weights(1 ./ (knndists[pt] .+ 1e-6)))
    # predpositions_arr[:, pt] = mean(positions_arr[1:2, knninds[pt]], dims=2)
end

gtpositions_arr = positions_arr[1:2, inds[test_inds]]


positions_arr_viz = Observable(zeros(2, 0))
predpositions_arr_viz = Observable(zeros(2, 0))
window_size = 30

f = Figure()
ax1 = Axis(f[1, 1], title = "Executed By Stage (Cartesian)")
ax2 = Axis(f[1, 2], title = "Sensed By Inductance Sensor (Cartesian)")
scatterlines!(ax1, positions_arr_viz, colormap = :blues, color = window_size:-1:1)
scatterlines!(ax2, predpositions_arr_viz, colormap = :reds, color = window_size:-1:1)
limits!(ax1, minimum(positions_arr[1, :]), maximum(positions_arr[1, :]), minimum(positions_arr[2, :]), maximum(positions_arr[2, :]))
limits!(ax2, minimum(positions_arr[1, :]), maximum(positions_arr[1, :]), minimum(positions_arr[2, :]), maximum(positions_arr[2, :]))
display(f)

for i in 1:size(gtpositions_arr, 2)
    if i > window_size
        positions_arr_viz[] = gtpositions_arr[1:2, i-window_size:i]
        predpositions_arr_viz[] = predpositions_arr[1:2, i-window_size:i]
    else
        positions_arr_viz[] = gtpositions_arr[1:2, 1:i]
        predpositions_arr_viz[] = predpositions_arr[1:2, 1:i]
    end
    notify(positions_arr_viz)
    notify(predpositions_arr_viz)
    # autolimits!(ax)
    # yield()
    sleep(0.0005)
end



f = Figure()
ax = Axis3(f[1, 1], aspect = :data, xlabel = "x", ylabel = "y", zlabel = "z", title = "Planned Path")
scatterlines!(ax, positions_arr[1, :], positions_arr[2, :], positions_arr[3, :], color=:blue, label="Ground truth", markersize=0.5)
# scatter!(ax, predpositions_arr[1, :], predpositions_arr[2, :], color=:red, label="Predicted")
display(f)

f = Figure()
ax1 = Axis(f[1, 1], title = "Planned Path")
limits!(ax1, minimum(positions_arr[1, :]), maximum(positions_arr[1, :]), minimum(positions_arr[2, :]), maximum(positions_arr[2, :]))
ax2 = Axis(f[1, 2], title = "Sensing Path")
limits!(ax2, minimum(positions_arr[1, :]), maximum(positions_arr[1, :]), minimum(positions_arr[2, :]), maximum(positions_arr[2, :]))
scatterlines!(ax1, positions_arr[1, :], positions_arr[2, :], color=:blue, label="Ground truth", markersize=0.5)
scatterlines!(ax2, predpositions_arr[1, :], predpositions_arr[2, :], color=:blue, label="Inductance Sensed", markersize=0.5)
display(f)

f = Figure()
ax1 = Axis(f[1, 1], title = "Sensor 1")
ax2 = Axis(f[1, 2], title = "Sensor 2")
lines!(ax1, iposition_arr[1, :], color = :blue)
lines!(ax2, iposition_arr[2, :], color = :blue)
autolimits!(ax1)
autolimits!(ax2)
display(f)







using DataFrames
using CUDA
using Flux
using MLUtils
using Random
using GLMakie
using ProgressMeter
using Serialization
GLMakie.activate!()
CUDA.set_runtime_version!(v"12.2.0")
CUDA.functional()
# deserialize
data = deserialize("zigzag3.jld2")
positions = data["positions"]
inductances_x = data["inductances_x"]
inductances_y = data["inductances_y"]
positions_arr = hcat(positions...)
iposition_arr = hcat(inductances_x, inductances_y)'

# scatter plot all positions
f = Figure()
ax = Axis3(f[1, 1], aspect = :data)
scatterlines!(ax, positions_arr[1, :], positions_arr[2, :], positions_arr[3, :], color = :blue)
display(f)

f = Figure()
ax = Axis(f[1, 1])
# lines!(ax, iposition_arr[1, :], color = :blue)
# lines!(ax, iposition_arr[2, :], color = :red)
scatterlines!(ax, iposition_arr[1, :], iposition_arr[2, :], color = :blue)
display(f)


X = iposition_arr
y = positions_arr[[1, 2], :]
train_inds, test_inds = splitobs(size(X, 2), at=0.9)
inds = shuffle(1:size(X, 2))
train_loader = Flux.DataLoader((X[:, inds[train_inds]], y[:, inds[train_inds]]) |> gpu, batchsize=32, shuffle=true)
val_loader = Flux.DataLoader((X[:, inds[test_inds]], y[:, inds[test_inds]]) |> gpu, batchsize=length(test_inds))

model = Chain(
    Dense(2, 64, relu; init = Flux.glorot_normal(gain=1)),
    Dense(64, 64, relu; init = Flux.glorot_normal(gain=1)),
    Dense(64, 2; init = Flux.glorot_normal(gain=1)),
) |> gpu

function criterion(result, label)
    loss = Flux.mse(result, label)
    return loss
end

opt_state = Flux.setup(Adam(0.001), model)
epochs = 3000
f = Figure()
ax1 = Makie.Axis(f[1, 1], yscale=log10)
ax2 = Makie.Axis(f[1, 2])

train_losses = Vector{Float32}()
val_losses = Vector{Float32}()
@showprogress for epoch in 1:epochs
    loss_ = 0
    loss_cnt = 0
    for (i, data) in enumerate(train_loader)
        input, label = data

        val, grads = Flux.withgradient(model) do m
            result = m(input)
            loss = criterion(result, label)
        end
        loss_ += val * size(input, 2)
        loss_cnt += size(input, 2)

        # Detect loss of Inf or NaN. Print a warning, and then skip update!
        if !isfinite(val)
            @warn "loss is $val on item $i" epoch
            continue
        end

        Flux.update!(opt_state, model, grads[1])
    end
    push!(train_losses, loss_ / loss_cnt)

    # Validation loss
    input, label = first(val_loader)
    result = model(input)
    loss = criterion(result, label)
    push!(val_losses, loss)

    # clear display
    # IJulia.clear_output(true)
    println("Epoch: $epoch, Train loss: $(train_losses[end]), Val loss: $(val_losses[end])")

    # plot loss
    empty!(ax1)
    lines!(ax1, 1:epoch, train_losses, color=:blue, label="Train loss", linewidth=2)
    lines!(ax1, 1:epoch, val_losses, color=:red, label="Val loss", linewidth=2)
    autolimits!(ax1)
    axislegend(ax1, merge=true, unique=true)

    # plot prediction
    empty!(ax2)
    input, label = first(val_loader)
    input = input[:, 1:100:end]
    label = label[:, 1:100:end]
    result = model(input)
    input = input |> cpu
    label = label |> cpu
    result = result |> cpu
    scatter!(ax2, result[1, :], result[2, :], color=:red, label="Predicted")
    scatter!(ax2, label[1, :], label[2, :], color=:blue, label="Ground truth")
    linesegments!(ax2, [result[1, :] label[1, :]]'[:], [result[2, :] label[2, :]]'[:], color=:black)
    axislegend(ax2, merge=true, unique=true)
    # limits!(ax2, 0.13, 0.5, -0.1, 0.1)
    display(f)

    # save model
    serialize("zigzag3_x_2.jld2", model)
end

x_model = deserialize("zigzag3_x.jld2")
y_model = deserialize("zigzag3_y.jld2")

# predict full data
# all_loader = Flux.DataLoader((X, y) |> gpu, batchsize=length(y))
# all_input, all_label = first(all_loader) |> cpu

# result = model(first(all_data)[1])
input, label = first(val_loader)

# result_x = x_model(input) |> cpu
# result_y = y_model(input) |> cpu
result = vcat(result_x, result_y)

result = model(first(val_loader)[1]) |> cpu
label = label |> cpu
input = input |> cpu

((result .- label) .^ 2 |> sum |> sqrt) / size(result, 2)
maximum(diff)
minimum(diff)
diff = sum((result .- label) .^ 2, dims=1)[1, :] .|> sqrt

# plot prediction
f = Figure()
ax = Makie.Axis(f[1, 1])
# scatter!(ax, result[1, :], result[2, :], color=:red, label="Predicted")
# scatterlines!(ax, label[1, :], label[2, :], color=:blue, label="Ground truth")
# scatter!(ax, label[1, :], label[2, :], color=:blue, label="Ground truth")
# scatterlines!(ax, input[1, :], input[2, :], color=:blue, label="Ground truth")
display(f)


f = Figure()
ax = Makie.Axis(f[1, 1])
val_pts = Observable(zeros(2, 0))
scatter!(ax, y[1, :], y[2, :], color=:blue, label="Ground truth")
scatter!(ax, val_pts, color=:red, label="Predicted")
limits!(ax, minimum(y[1, :]), maximum(y[1, :]), minimum(y[2, :]), maximum(y[2, :]))
display(f)

for i in 1:100:size(y, 2)
    val_pts[] = hcat(val_pts[], y[:, i])
    if size(val_pts[], 2) > 50
        val_pts[] = val_pts[][:, end-50:end]
    end
    notify(val_pts)
    # autolimits!(ax)
    sleep(0.0001)
end

X


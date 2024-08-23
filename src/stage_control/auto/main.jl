using GLMakie
GLMakie.activate!(inline=false)

# Global Status
running = true
# Vision Data
visionCh = Channel{Array{Int64}}(100)
refCh = Channel{Array{Int64}}(100)

include("stage_control.jl")
include("vision.jl")

mouseinit = Observable(true)

## Visualize
f = Figure()
ax = Axis(f[1, 1])
lines!(ax, @lift((VisionErrSys[:C]*$(VisionErrState[:x]))[1, :]), color=:red)
lines!(ax, @lift((VisionErrSys[:C]*$(VisionErrState[:x]))[2, :]), color=:black)
# lines!(ax, @lift($(VisionErrState[:x])[3, :]), color=:blue)

ax = Axis(f[1, 2])
lines!(ax, @lift($(VisionErrState[:u])[1, :]), color=:red)
lines!(ax, @lift($(VisionErrState[:u])[2, :]), color=:black)
lines!(ax, @lift($(VisionErrState[:u])[3, :]), color=:blue)

ax = Axis(f[2:5, 1], aspect=DataAspect())
deregister_interaction!(ax, :rectanglezoom)
image!(ax, @lift($(annoimg)[1:4:end, 1:4:end]))
Label(f[6, 1:2], @lift($mouseinit[] ? "Active" : "Inactive"))

ax2 = Axis(f[2:5, 2], aspect=DataAspect())
deregister_interaction!(ax2, :rectanglezoom)
image!(ax2, mouseimg)

display(f)

# Mouse Interactions
start_position = Float64[0, 0]
register_interaction!(ax2, :my_interaction) do event::Makie.MouseEvent, axis
    global mouseinit
    global start_position
    global liveimgsz
    global refCh

    if event.type === Makie.MouseEventTypes.leftclick && !mouseinit[]
        # deregister_interaction!(ax1, :my_interaction)
        # println("Left click, deregistering interaction")
        mouseinit[] = true
        return
    end
    if (event.type === Makie.MouseEventTypes.leftclick && mouseinit[])
        mouseinit[] = false
        start_position[1] = event.data[1]
        start_position[2] = event.data[2]
        return
    else
        if mouseinit[]
            return
        end
        xoff, yoff = round(Int, event.data[1] - start_position[1]), round(Int, event.data[2] - start_position[2])
        push!(refCh, [xoff, yoff])
    end
    return
end

# deregister_interaction!(ax2, :my_interaction)

display(f)


while running
    # start_t = time()
    vislooponce(visionCh, refCh)

    @async begin
        lock(VisionErrState[:lock]) do
            notify(VisionErrState[:x])
            notify(VisionErrState[:t])
            notify(VisionErrState[:u])
            notify(VisionErrState[:em])
        end
        notify(annoimg)
        notify(mouseimg)
    end
    # println(1 / (time() - start_t))
    yield()
end

running = false
using Serialization
open("VisionErrState_chirp.jls", "w") do f
    serialize(f, Dict(
        "x" => VisionErrState[:x][],
        "xi" => VisionErrState[:xi][],
        "xd" => VisionErrState[:xd][],
        "u" => VisionErrState[:u][],
        "r" => VisionErrState[:r][],
        "t" => VisionErrState[:t][],
        "em" => VisionErrState[:em][],
        "P" => VisionErrState[:P][],
    ))
end

using ControlSystemIdentification
using ControlSystemsBase
using Serialization
open("VisionErrState_chirp.jls") do f
    global VisionErrState_id
    VisionErrState_id = deserialize(f)
end

f = Figure()
ax = Axis(f[1, 1])
lines!(ax, VisionErrState_id["x"][1, :], color=:red)
lines!(ax, VisionErrState_id["x"][2, :], color=:black)
display(f)



valid_rng = 35:2200
idinput = reverse(VisionErrState_id["u"][1:2, valid_rng], dims=2)
idoutput = detrend(reverse(VisionErrState_id["x"][1:2, valid_rng], dims=2))
ts = reverse(VisionErrState_id["t"][valid_rng], dims=1)

myiddata = iddata(
    idoutput,
    idinput,
    sum(ts[2:end] - ts[1:end-1]) / length(ts),
)

ssid = subspaceid(myiddata, 2, zeroD = true, verbose = true)

ssid.C



open("VisionErrState_chirp.jls", "w") do f
    serialize(f, Dict(
        "x" => VisionErrState[:x][],
        "xi" => VisionErrState[:xi][],
        "xd" => VisionErrState[:xd][],
        "u" => VisionErrState[:u][],
        "r" => VisionErrState[:r][],
        "t" => VisionErrState[:t][],
        "em" => VisionErrState[:em][],
        "P" => VisionErrState[:P][],
    ))
end
VisionErrState
initidxend = 2200
valid_rng = 1:initidxend
idinput = reverse(VisionErrState[:u][][1:2, valid_rng], dims=2)
idoutput = detrend(reverse(VisionErrState[:x][][1:2, valid_rng], dims=2))
ts = reverse(VisionErrState[:t][][valid_rng], dims=1)

myiddata = iddata(
    idoutput,
    idinput,
    sum(ts[2:end] - ts[1:end-1]) / length(ts),
)

ssid = subspaceid(myiddata, 2, zeroD = true, verbose = true)
ssid.B
VisionErrSys[:B][1:2, 1:2] .= ssid.B
VisionErrSys[:C][1:2, 1:2] = ssid.C
VisionErrSys[:B][1:2, 1:2]


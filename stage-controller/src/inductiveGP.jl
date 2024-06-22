using GaussianProcesses
using Random
using GLMakie
using ProgressMeter
using Serialization

Random.seed!(20140430)
GLMakie.activate!()
# deserialize
data = deserialize("zigzag3.jld2")
positions = data["positions"]
inductances_x = data["inductances_x"]
inductances_y = data["inductances_y"]
positions_arr = hcat(positions...)
iposition_arr = hcat(inductances_x, inductances_y)'


#Training data
d, n = 2, 50;         #Dimension and number of observations
x = iposition_arr
y = positions_arr[1, :]

mZero = MeanZero()                             # Zero mean function
kern = Matern(5/2,[0.0,0.0],0.0) + SE(0.0,0.0)    # Sum kernel with Matern 5/2 ARD kernel 
                                               # with parameters [log(ℓ₁), log(ℓ₂)] = [0,0] and log(σ) = 0
                                               # and Squared Exponential Iso kernel with
                                               # parameters log(ℓ) = 0 and log(σ) = 0


gp = GP(x,y,mZero,kern,-2.0)          # Fit the GP
optimize!(gp)  


f = Figure()
ax = Axis3(f[1, 1])
# plot gp
# xtest = range(0,stop=2π,length=100)
# ytest = range(0,stop=2π,length=100)
# ztest = predict_f(gp, [xtest ytest]')
surface!(ax, x[1, :], x[2, :], y, color = :blue)
display(f)

x[1, :]

xs = LinRange(0, 1, 50)
ys = LinRange(0, 1, 50)
zs = [cos(x) * sin(y) for x in xs, y in ys]

surface(xs, ys, y, axis=(type=Axis3,))

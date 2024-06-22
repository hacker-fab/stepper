using LowLevelParticleFilters, LinearAlgebra,StaticArrays,Distributions

n = 2   # Dimension of state
m = 1   # Dimension of input
p = 1   # Dimension of measurements
N = 500 # Number of particles

const dg = MvNormal(p,1.0)          # Measurement noise Distribution
const df = MvNormal(n,1.0)          # Dynamics noise Distribution
const d0 = MvNormal(randn(n),2.0)   # Initial state Distribution
A = SA[0.97043   -0.097368
             0.09736    0.970437]
B = SA[0.1; 0;;]
C = SA[0 1.0]

R1 = cov(df)
R2 = cov(dg)
kf = KalmanFilter(A, B, C, 0, R1, R2, d0)
u = [0.0]
y = [0.0]
kf(u,y)

for t = 1:10
    kf(u,y) # Performs both correct and predict!!
    print(state(kf))
end
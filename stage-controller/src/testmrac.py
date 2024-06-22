# %%
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
time_steps = 300
dt = 0.1
# Function to run Kalman Filter
def kalman_update(A, x, P, Q, R, measurement):
    # Prediction
    prediction = A @ x
    P = A @ P @ A.T + Q

    # Update
    S = R + P
    K = P @ np.linalg.inv(S)
    x = x + K @ (measurement - prediction)
    P = (np.eye(len(x)) - K) @ P

    return x, P

class RecursiveLeastSquares(object):
    def __init__(self,x0,P0,R,lmda=0.99):
        self.x0=x0
        self.P0=P0
        self.R=R
        self.lamda=lmda

    def predict(self,measurementValue,C):
        Lmatrix = self.R + C @ self.P0 @ C.T
        LmatrixInv=np.linalg.inv(Lmatrix)
        gainMatrix = self.P0 @ C.T @ LmatrixInv
        error = measurementValue - C @ self.x0  
        estimate = self.x0 + gainMatrix @ error
        ImKc = np.eye(np.size(self.x0),np.size(self.x0))- gainMatrix @ C
        self.P0 = (ImKc @ self.P0) / self.lamda
        self.x0=estimate
        return estimate

# %%
# %%
from scipy import signal
t_vec = np.arange(0, time_steps * dt, dt)
r_vec = np.zeros((2, len(t_vec)))
rect = signal.square(2 * np.pi * 1 * t_vec)
r_vec[0, :] = rect
r_vec[1, :] = -r_vec[0, :]


# System and reference model parameters
A = np.array([[-1.0, 0.0], [0.0, -1.0]])  # Known matrix for plant
B = np.array([[0.3, -0.7], [-0.5, 0.3]])          # Known matrix for plant
A_m = np.array([[-2.0, 0.6], [0.0, -2.0]])                                # Reference model matrix
B_m = np.array([[2, 0], [0, 2]])  # Reference model matrix
n, m = A.shape[0], B.shape[1]

# Basis functions
def basis_functions(x):
    return np.array([np.sin(x[0]), np.cos(x[1])])

# Initial states and parameters
x = np.zeros((n, 1))                    # Initial state of the plant
x_m = np.zeros((n, 1))                  # Initial state of the reference model
x_t = np.zeros((n, 1))                  # Initial state of the target model
Lambda = np.diag(np.random.rand(m))     # Unknown diagonal matrix
Theta = np.random.rand(2, m)            # Unknown constant matrix for nonlinearity
K_x_hat = np.zeros((n, n))          # Estimated controller gain
K_r_hat = np.zeros((n, m))          # Estimated controller gain
# Theta_hat = np.random.rand(2, m)        # Estimated Theta
Theta_hat = np.zeros((2, m))            # Estimated Theta
# K_r_hat = np.array([[0.15, 0.0], [0.0, 0.15]])              # Estimated controller gaingain

# Lyapunov design parameters
Gamma_x = np.eye(n) * 50.0                     # Adaptation rate for K_x_hat
Gamma_r = np.eye(m) * 50.0                    # Adaptation rate for K_r_hat
Gamma_Theta = np.eye(2) * 50.0                 # Adaptation rate for Theta_hat
Q = np.eye(n) * 1                           # Chosen positive definite matrix for Lyapunov equation
P = np.linalg.solve(A_m.T + A_m, -Q)    # Solving Lyapunov equation
P = np.eye(n)                           # Solving Lyapunov equation

# B estimation
b0 = np.eye(2).reshape(-1)[np.newaxis, :].T  # Initial state estimate
B_hat = b0.reshape(2, 2).T
# b0 = B.T.reshape(-1)[np.newaxis, :].T
bQ = np.eye(2 * 2) * 1.0     # Process noise covariance
bR = np.eye(2 * 2) * 0.5     # Measurement noise covariance
bP = np.eye(2 * 2) * 1.0  # Initial error covariance matrix

rls = None

# D estimation, estimates length of first and second link
# l1 is translation offset from current x, l2 is length of the rotating link
dl10 = 1.0
dl20 = 10.0
dl1R = np.eye(1, 1) * 0.001     # Process noise covariance
dl1P = np.eye(1, 1) * 0.001     # Process noise covariance
dl2R = np.eye(3, 3) * 0.001     # Measurement noise covariance
dl2P = np.eye(3, 3) * 0.001  # Initial error covariance matrix
rls_dl1 = RecursiveLeastSquares(np.array([[dl10]]), dl1P, dl1R)
rls_dl2 = RecursiveLeastSquares(np.array([[0.0], [0.0], [dl20]]), dl2P, dl2R)

# Simulation loop
history = {'x': [], 'x_m': [], 'u': [], 'r': [], 'k_x_hat': [], 'k_r_hat': [], 'theta_hat': [], 'x_t': [], 'target_input': [], 'target_output': [], 'x_dot': [], 'origin': [], 'offset': [], 'B_hat_error': [], 'StagePose_error': []}
for step in range(time_steps):
    # Reference model update
    # r = np.array([[np.sin(step * dt)], [np.cos(step * dt)]])
    r = [[1.0], [0.0]]
    r = np.array([[r_vec[0, step]], [r_vec[1, step]]])
    x_m_dot = np.dot(A_m, x_m) + np.dot(B_m, r)
    x_m += x_m_dot * dt

    # Control law
    u = np.dot(K_x_hat.T, x) + np.dot(K_r_hat.T, r)- np.dot(Theta_hat.T, basis_functions(x))

    # Plant update
    f_x = np.dot(Theta.T, basis_functions(x))
    # x_dot = np.dot(A, x) + np.dot(B, Lambda.dot(u + f_x))
    x_dot = np.dot(A, x) + np.dot(B, u) + np.random.normal(0, 0.1, (2, 1))
    x += x_dot * dt

    # Target model update
    x_t = np.array([
        [1.3 * x[0][0] + 5 * (np.cos(x[1][0]))],
        [5 * (np.sin(x[1][0]))]
    ])

    # print(B_hat)

    # Error and adaptive law
    e = x - x_m
    K_x_hat_dot = -np.dot(Gamma_x, np.dot(x, e.T).dot(B_hat).dot(Lambda).dot(P))
    K_r_hat_dot = -np.dot(Gamma_r, np.dot(r, e.T).dot(B_hat).dot(Lambda).dot(P))
    Theta_hat_dot = np.dot(Gamma_Theta, np.dot(basis_functions(x), e.T).dot(B_hat).dot(Lambda).dot(P))
    
    K_x_hat += K_x_hat_dot * dt
    K_r_hat += K_r_hat_dot * dt
    Theta_hat += Theta_hat_dot * dt

    # Record for visualization
    history['x'].append(x.flatten())
    history['x_dot'].append(x_dot.flatten())
    history['x_m'].append(x_m.flatten())
    history['x_t'].append(x_t.flatten())
    history['u'].append(u.flatten())
    history['r'].append(r.flatten())
    history['k_x_hat'].append(K_x_hat)
    history['k_r_hat'].append(K_r_hat)
    history['theta_hat'].append(Theta_hat)

    if len(history['u']) < 3:
        pass
    else:
        # identify B
        # x_dot = A @ x + B @ u
        # B @ u = (x_dot - A @ x)
        # ??? @ vec(B) = ???
        trans_mat0 = np.kron(history['u'][-1].reshape(-1, 1), np.eye(2)).T
        trans_mat1 = np.kron(history['u'][-2].reshape(-1, 1), np.eye(2)).T
        trans_mat = np.vstack([trans_mat0, trans_mat1])
        res_vec0 = history['x_dot'][-1].reshape(-1, 1) - A @ history['x'][-2].reshape(-1, 1)
        res_vec1 = history['x_dot'][-2].reshape(-1, 1) - A @ history['x'][-3].reshape(-1, 1)
        res_vec = np.vstack([res_vec0, res_vec1])

        if rls is None:
            rls = RecursiveLeastSquares(b0, bP, bR)
            
        b0 = rls.predict(res_vec, trans_mat)
        B_hat = b0.reshape(2, 2).T
        # print(np.linalg.inv(trans_mat) @ res_vec)
    history['B_hat_error'].append(np.linalg.norm(B_hat - B))

    # identify d
    ## assume l1 correct, estimate l2
    history['origin'].append([0, 0])
    history['offset'].append([0, 0])
    if len(history['x_t']) >= 3 :
        for i in range(10):
            cA = np.array([[
                -2 * (history['x_t'][j][0] - dl10 * history['x'][j][0]), 
                -2 * (history['x_t'][j][1]), 1
            ] for j in range(max(len(history['x_t'])-20, 0), len(history['x_t']))])
            cb = np.array([[
                - (history['x_t'][j][0] - dl10 * history['x'][j][0]) ** 2 
                - (history['x_t'][j][1]) ** 2
            ] for j in range(max(len(history['x_t'])-20, 0), len(history['x_t']))])
            pred = np.linalg.pinv(cA) @ cb
            dl20 = np.sqrt(pred[0][0] ** 2 + pred[1][0] ** 2 - pred[2][0])
        

            # find out the offset
            offset = np.array([
                history['x_t'][j][0] - (history['origin'][j][0] + dl20 * np.cos(
                    np.arctan2(
                        history['x_t'][j][1] - history['origin'][j][1],
                        history['x_t'][j][0] - dl10 * history['x'][j][0] - history['origin'][j][0])
                ))
            for j in range(max(len(history['x_t'])-20, 0), len(history['x_t']))])

            # # print(offset[0])
            sA = np.array([[
                history['x'][j][0]
            ] for j in range(max(len(history['x_t'])-20, 0), len(history['x_t']))])
            sb = offset.reshape(-1, 1)
            cscale = np.linalg.pinv(sA) @ sb
            # dl10 = np.average(offset / sA)
            dl10 = cscale[0][0]
            
    history['StagePose_error'].append(np.array([dl10 - 1.3, dl20 - 5.0]))


    # print("dl10:", dl10, " dl20:", dl20)

# Convert history to numpy array for plotting
history['x'] = np.array(history['x'])
history['x_m'] = np.array(history['x_m'])
history['x_t'] = np.array(history['x_t'])
history['u'] = np.array(history['u'])
history['r'] = np.array(history['r'])
history['k_x_hat'] = np.array([i[0,0] for i in history['k_x_hat']])
history['k_r_hat'] = np.array([i[0,0] for i in history['k_r_hat']])
history['theta_hat'] = np.array([i[0,0] for i in history['theta_hat']])
history['StagePose_error'] = np.array(history['StagePose_error'])


plt.figure(figsize=(12, 8))
plt.subplot(6, 1, 1)
# Plotting the plant state and reference model state
plt.plot(history['x'][:, 0], label='Plant Output 1')
plt.plot(history['x'][:, 1], label='Plant Output 2')
plt.plot(history['x_m'][:, 0], label='Reference Model Output 1', linestyle='dashed')
plt.plot(history['x_m'][:, 1], label='Reference Model Output 2', linestyle='dashed')
plt.title('Plant and Reference Model States')
plt.xlabel('Time Steps')
plt.ylabel('State')
plt.legend()

# Plotting control input
plt.subplot(6, 1, 2)
plt.plot(history['u'][:, 0], label='Control Input 1')
plt.plot(history['u'][:, 1], label='Control Input 2')
plt.title('Control Inputs')
plt.xlabel('Time Steps')
plt.ylabel('Input')
plt.legend()

# Plotting reference input
plt.subplot(6, 1, 3)
plt.plot(history['r'][:, 0], label='Ref Input 1')
plt.plot(history['r'][:, 1], label='Ref Input 2')
plt.title('Ref Inputs')
plt.xlabel('Time Steps')
plt.ylabel('Input')
plt.legend()

# Plotting error
plt.subplot(6, 1, 4)
plt.plot(history['x'][:, 0] - history['x_m'][:, 0], label='Error in Output 1')
plt.plot(history['x'][:, 1] - history['x_m'][:, 1], label='Error in Output 2')
plt.title('Tracking Error')
plt.xlabel('Time Steps')
plt.ylabel('Error')
plt.legend()

# Plotting B Error
plt.subplot(6, 1, 5)
plt.plot(history['B_hat_error'], label='B Error')
plt.title('B Matrix Error')

# Plotting StagePose Error
plt.subplot(6, 1, 6)
plt.plot(history['StagePose_error'][:, 0], label='Translation')
plt.plot(history['StagePose_error'][:, 1], label='Rotation')
plt.title('StagePose Error')


plt.tight_layout()
plt.show()

# %%
history['x_t']
history['offset']
history['origin']
history['x']

# %%
plt.figure(figsize=(12, 8))
ax = plt.subplot(111)
ax.set_aspect('equal')
ax.scatter(history['x_t'][:, 0] - history['x'][:, 0], history['x_t'][:, 1], label='Plant Output')
plt.show()

# %%
history['x_t'].shape
# %%
history['x_t']
# %%
from scipy.optimize import least_squares
x0 = [0.0, 0.0]
def costfunc(x):
    cost = 0
    for i in range(len(history['x_t'])):
        diff = history['target_output'][i] - np.array([[x[0] * history['target_input'][i][1][0], x[1] * history['target_input'][i][1][1]]])
        cost += np.sum(diff ** 2)
    return cost
res_1 = least_squares(costfunc, x0, jac='3-point')
res_1
# %%
costfunc(res_1.x)

# %%
costfunc([3, 6])

# %%
# %%
history['target_output'][0][0]
# %%
history['target_input'][0]
# %%
history['target_input'][0][1][0]* 3 
# %%
G = np.array([[1, 2], [3, 4]])
f = np.array([[5], [6], [7], [8]])
# k = G @ f
# %%
# k = G @ f now assum G unknown
# ?? @ vec(G) = ??
np.kron(f, np.eye(4)).T
# %%
np.linalg.pinv(np.kron(f, np.eye(2)).T) @ k
# %%
trans_mat @ bP
# %%
trans_mat @ b0.T.reshape(-1)
# %%
kalman_update(trans_mat, b0.T.reshape(-1), bP, bQ, bR, res_vec)
# %%
dd = b0.T.reshape(-1)[np.newaxis, :].T
# %%
dd.T.reshape(2, 2)
# %%
history['x_dot'][-1].reshape(-1, 1)
# %%
trans_mat0 = np.kron(history['u'][-1].reshape(-1, 1), np.eye(2)).T
trans_mat1 = np.kron(history['u'][-2].reshape(-1, 1), np.eye(2)).T
trans_mat = np.vstack([trans_mat0, trans_mat1])
res_vec0 = history['x_dot'][-1].reshape(-1, 1) - A @ history['x'][-2].reshape(-1, 1)
res_vec1 = history['x_dot'][-2].reshape(-1, 1) - A @ history['x'][-3].reshape(-1, 1)
res_vec = np.vstack([res_vec0, res_vec1])
b0_ = b0.T.reshape(-1)[np.newaxis, :].T
b0_, bP = kalman_update(trans_mat, b0_, bP, bQ, bR, res_vec)
b0 = b0_.reshape(2, 2).T
B_hat = b0

# %%
# create 3 points
circle_origin = np.array([[1.0], [0.0]])
circle_radius = 1.0
circle_pts = np.array([
    [circle_origin[0] + circle_radius * np.cos(i),
    circle_origin[1] + circle_radius * np.sin(i) 
     ] for i in np.linspace(0, 2 * np.pi, 100)
])
# %%
circle_pts
# %%
# circle linear regressin
cA = np.array([[
    -2 * circle_pts[i][0].item(0), -2 * circle_pts[i][1].item(0), 1
] for i in range(len(circle_pts))])
cb = np.array([[
    - circle_pts[i][0].item(0) ** 2 - circle_pts[i][1].item(0) ** 2
] for i in range(len(circle_pts))])
np.linalg.pinv(cA) @ cb
# %%
rls = RecursiveLeastSquares(cA[:3, :], cb[:3, :])
for i in range(3, len(circle_pts), 3):
    if i + 3 > len(circle_pts):
        break
    print(rls.addData(cA[i:i+3, :], cb[i:i+3, :]))
# %%
P0=100*np.eye(3,3)
R=0.0001*np.eye(3, 3)
nrls = RecursiveLeastSquares(np.array([[0], [0], [0]]), P0, R)
for i in range(3, len(circle_pts), 3):
    if i + 3 > len(circle_pts):
        break
    pred = (nrls.predict(cb[i:i+3, :], cA[i:i+3, :]))
    print("radius:", np.sqrt(pred[0][0] ** 2 + pred[1][0] ** 2 - pred[2][0]))
# %%
ax = plt.subplot(111)
ax.plot([i[0] for i in nrls.estimates])
# ax.plot([i[0] for i in nrls.errors])
plt.show()

        
# %%
for i in range(0, len(history['x_t']) - 3):
    cA = np.array([[
        -2 * (history['x_t'][i + j][0] - 1.0 * history['x'][i + j][0]), 
        -2 * (history['x_t'][i + j][1]), 1
    ] for j in range(3)])
    cb = np.array([[
        - (history['x_t'][i + j][0] - 1.0 * history['x'][i + j][0]) ** 2 
        - (history['x_t'][i + j][1]) ** 2
    ] for j in range(3)])
    circ = rls_dl2.predict(cb, cA)
    print("radius:", np.sqrt(pred[0][0] ** 2 + pred[1][0] ** 2 - pred[2][0]))



# %%
np.rad2deg(np.arctan2(1, 2))
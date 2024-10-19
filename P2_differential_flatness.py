import math
import typing as T

import numpy as np
from numpy import linalg
from scipy.integrate import cumtrapz  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from utils import save_dict, maybe_makedirs

class State:
    def __init__(self, x: float, y: float, V: float, th: float) -> None:
        self.x = x
        self.y = y
        self.V = V
        self.th = th

    @property
    def xd(self) -> float:
        return self.V*np.cos(self.th)

    @property
    def yd(self) -> float:
        return self.V*np.sin(self.th)


def compute_traj_coeffs(initial_state: State, final_state: State, tf: float) -> np.ndarray:
    """
    Inputs:
        initial_state (State)
        final_state (State)
        tf (float) final time
    Output:
        coeffs (np.array shape [8]), coefficients on the basis functions

    Hint: Use the np.linalg.solve function.
    """
    ########## Code starts here ##########
    # Set up the system of linear equations for x(t) and y(t)
    A = np.array([
        [1, 0, 0, 0],           # x(0) = x0
        [0, 1, 0, 0],           # x'(0) = V0*cos(th0)
        [1, tf, tf**2, tf**3],  # x(tf) = xf
        [0, 1, 2*tf, 3*tf**2],  # x'(tf) = Vf*cos(thf)
    ])
    
    B = np.array([
        [1, 0, 0, 0],           # y(0) = y0
        [0, 1, 0, 0],           # y'(0) = V0*sin(th0)
        [1, tf, tf**2, tf**3],  # y(tf) = yf
        [0, 1, 2*tf, 3*tf**2],  # y'(tf) = Vf*sin(thf)
    ])
    
    # Vector of initial and final conditions
    x_conditions = np.array([initial_state.x, initial_state.V * np.cos(initial_state.th), final_state.x, final_state.V * np.cos(final_state.th)])
    y_conditions = np.array([initial_state.y, initial_state.V * np.sin(initial_state.th), final_state.y, final_state.V * np.sin(final_state.th)])

    # Solve for the coefficients using np.linalg.solve
    a_coeffs = np.linalg.solve(A, x_conditions)
    b_coeffs = np.linalg.solve(B, y_conditions)

    # Return the concatenated coefficients
    coeffs = np.hstack((a_coeffs, b_coeffs))

    ########## Code ends here ##########
    return coeffs

def compute_traj(coeffs: np.ndarray, tf: float, N: int) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
        coeffs (np.array shape [8]), coefficients on the basis functions
        tf (float) final_time
        N (int) number of points
    Output:
        t (np.array shape [N]) evenly spaced time points from 0 to tf
        traj (np.array shape [N,7]), N points along the trajectory, from t=0
            to t=tf, evenly spaced in time
    """
    t = np.linspace(0, tf, N) # generate evenly spaced points from 0 to tf
    traj = np.zeros((N, 7))
    ########## Code starts here ##########
    # Coefficients for x(t) and y(t)
    a = coeffs[:4]  # x(t) coefficients
    b = coeffs[4:]  # y(t) coefficients
    
    # Evaluate x(t), y(t), and their derivatives at each time step
    traj[:, 0] = a[0] + a[1] * t + a[2] * t**2 + a[3] * t**3  # x(t)
    traj[:, 1] = b[0] + b[1] * t + b[2] * t**2 + b[3] * t**3  # y(t)
    traj[:, 3] = a[1] + 2 * a[2] * t + 3 * a[3] * t**3  # x'(t)
    traj[:, 4] = b[1] + 2 * b[2] * t + 3 * b[3] * t**3  # y'(t)
    traj[:, 5] = 2 * a[2] + 6 * a[3] * t  # x''(t)
    traj[:, 6] = 2 * b[2] + 6 * b[3] * t  # y''(t)
    
    # Compute the angle theta from the velocity components
    traj[:, 2] = np.arctan2(traj[:, 4], traj[:, 3])  # theta(t)
    ########## Code ends here ##########

    return t, traj

def compute_controls(traj: np.ndarray) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Input:
        traj (np.array shape [N,7])
    Outputs:
        V (np.array shape [N]) V at each point of traj
        om (np.array shape [N]) om at each point of traj
    """
    ########## Code starts here ##########
    V = np.sqrt(traj[:, 3]**2 + traj[:, 4]**2)
    
    # om = (x''*y' - y''*x') / (x'^2 + y'^2)
    om = (traj[:, 5] * traj[:, 4] - traj[:, 6] * traj[:, 3]) / (traj[:, 3]**2 + traj[:, 4]**2)
    ########## Code ends here ##########

    return V, om

if __name__ == "__main__":
    # Constants
    tf = 25.

    # time
    dt = 0.005
    N = int(tf/dt)+1
    t = dt*np.array(range(N))

    # Initial conditions
    s_0 = State(x=0, y=0, V=0.5, th=-np.pi/2)

    # Final conditions
    s_f = State(x=5, y=5, V=0.5, th=-np.pi/2)

    coeffs = compute_traj_coeffs(initial_state=s_0, final_state=s_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V,om = compute_controls(traj=traj)

    maybe_makedirs('plots')

    # Plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(traj[:,0], traj[:,1], 'k-',linewidth=2)
    plt.grid(True)
    plt.plot(s_0.x, s_0.y, 'go', markerfacecolor='green', markersize=15)
    plt.plot(s_f.x, s_f.y, 'ro', markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title("Path (position)")
    plt.axis([-1, 6, -1, 6])

    ax = plt.subplot(1, 2, 2)
    plt.plot(t, V, linewidth=2)
    plt.plot(t, om, linewidth=2)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
    plt.title('Original Control Input')
    plt.tight_layout()

    plt.savefig("plots/differential_flatness.png")
    plt.show()

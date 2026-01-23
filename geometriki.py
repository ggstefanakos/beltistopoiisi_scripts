import numpy as np

def f(x):
    f = 2*x[0]**2 + 10*x[1]**2 + 3*np.sin(x[0]) + 8*np.cos(x[1]) -10

    return f

def starting_point_for_2_dimensions(a):
    alpha = np.array([0,0])
    beta = np.array([a*np.cos(np.deg2rad(15)),a*np.sin(np.deg2rad(15))])
    gama = np.array([beta[1],beta[0]])

    return np.array([alpha,beta,gama])

n = 2
a = 0.3
M_limit = 0.05*n**2 + 1.65*n

# start = starting_point_for_2_dimensions(a)
start = np.array([[0.0,0.0],[0.28971,0.07761],[0.07761,0.28971]])

current = start
M = 1
best_prev = None
R_prev = None
i = 3

while M >= M_limit:

    z = [f(x) for x in current]

    # best = np.min(z)
    best = current[np.where(z == np.min(z))] # gia elaxisto

    # worst = np.max(z)
    worst = current[np.where(z == np.max(z))] # gia elaxisto

    second_worst = current[np.where(z == np.max(np.delete(z,np.where(z == np.max(z)),axis=0)))] # gia elaxisto

    if best == best_prev:
        M += 1
    else:
        M = 1

    R = worst # kanonas 1

    sum_of_all_points = (current.T @ np.ones((n+1,1))).T
    N = (2/n)*sum_of_all_points - 2*R

    # N = (2/n)*np.sum(np.delete(current,R)) - R # oxi akribos sosto

    if N == R_prev: # kanonas 2
        R = current[np.where(z == second_worst)]
        sum_of_all_points = (current.T @ np.ones((n+1,1))).T
        N = (2/n)*sum_of_all_points - 2*R
    
    R_prev = R
    best_prev = best
    current = np.delete(current,R,axis=0)
    i += 1

print(current)

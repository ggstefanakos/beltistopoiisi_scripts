import numpy as np

def f(x):
    '''
    Objective Function

    x : ndarray shape(n,) n = number of dimensions
    Vector of function variables
    '''
    f = 2*x[0]**2 + 10*x[1]**2 + 3*np.sin(x[0]) + 8*np.cos(x[1]) -10

    return f

def starting_shape_for_2_dimensions(a):
    '''
    Generates starting triangle points for 2 dimensional research

    a : float
    Error margin/Polygon vertice length
    '''
    alpha = np.array([0,0])
    beta = np.array([a*np.cos(np.deg2rad(15)),a*np.sin(np.deg2rad(15))])
    gama = np.array([beta[1],beta[0]])

    return np.array([alpha,beta,gama])

def next_int(x):
    '''
    Calculates the next integer from a float

    x : float
    Any float
    '''
    decimals = x - int(x)
    if np.isclose(decimals,0):
        return int(x)
    else:
        return int(x) + 1

n = 2 # Number of dimensions
a = 0.3 # Error margin
M_limit = next_int(0.05*n**2 + 1.65*n) # Max number of iterations with the same best value

start = starting_shape_for_2_dimensions(a)
# start = np.array([[0.0,0.0],[0.28971,0.07761],[0.07761,0.28971]]) # same as line above

current = start
M = 1
best_prev = start[0] + np.array([[2000,2000]]) # random huge values that are for sure diferrent than the ines in start
R_prev = start[0] + np.array([[1000,1000]])
rule = 1
i = n + 1
print('i\t|\t\tN\t\t|\tf(n)\t|\trule\t|\tM\t|\t\tR')
while M < M_limit:

    z = [f(x) for x in current] # for minimum
    # z = [-1*f(x) for x in current] # for maximum

    # best = np.min(z)
    best = current[np.where(z == np.min(z))]

    # worst = np.max(z)
    worst = current[np.where(z == np.max(z))]

    second_worst = current[np.where(z == np.max(np.delete(z,np.where(z == np.max(z)),axis=0)))]

    if (best == best_prev).all():
        M += 1
        if M >= M_limit: break
    else:
        M = 1

    R = worst # kanonas 1

    sum_of_all_points = (current.T @ np.ones((n+1,1))).T
    N = (2/n)*sum_of_all_points - 2*R
    rule = 1

    # N = (2/n)*np.sum(np.delete(current,R)) - R # oxi akribos sosto

    # if (N == R_prev).all(): # kanonas 2
    if np.allclose(N,R_prev): # kanonas 2
        R = second_worst
        # sum_of_all_points = (current.T @ np.ones((n+1,1))).T
        N = (2/n)*sum_of_all_points - 2*R
        rule = 2
    
    R_prev = R
    best_prev = best
    current[np.where(current == R)] = N
    i += 1
    print(f'{i}\t|\t{np.round(N,3)}\t|\t{np.round(f(N.flatten()),3)}\t|\t{rule}\t|\t{M}\t|\t{np.round(R,3)}')

print(f'Min of f(x) in x = ')

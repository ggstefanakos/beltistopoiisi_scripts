import numpy as np
from prettytable import PrettyTable

def f(x):
    '''
    Objective Function

    x : ndarray shape(n,) n = number of dimensions
        Vector of function variables
    '''
    # f = 2*x[0]**2 + 10*x[1]**2 + 3*np.sin(x[0]) + 8*np.cos(x[1]) -10
    f = (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-3)**2 # has min

    return f

def starting_shape_for_3_dimensions(a,first_point):
    '''
    Generates starting triangle points for 3 dimensional research

    a : float
        Error margin/Polygon vertice length
    first_point : ndarray, shape(n,) n = number of dimensions
        The first point in the research
    '''
    alpha = first_point.copy()
    beta = np.array([alpha[0]+a*np.cos(np.deg2rad(15)),alpha[1]+a*np.sin(np.deg2rad(15)),alpha[2]+0])
    gama = np.array([beta[1],beta[0],beta[2]])
    delta = np.array([alpha[0]+np.sqrt(3)*a/4,alpha[1]+np.sqrt(3)*a/4,alpha[2]+a*np.sqrt(6)/2])
    # delta = np.array([alpha[0]+np.sqrt(3)*a/4,alpha[1]+np.sqrt(3)*a/4,alpha[2]+a*np.sqrt(13)/4]) # mallon lathos

    return np.array([alpha,beta,gama,delta])

def starting_shape_for_2_dimensions(a,fisrt_point):
    '''
    Generates starting triangle points for 2 dimensional research

    a : float
        Error margin/Polygon vertice length
    first_point : ndarray, shape(n,) n = number of dimensions
        The first point in the research
    '''
    alpha = fisrt_point.copy()
    beta = np.array([alpha[0]+a*np.cos(np.deg2rad(15)),alpha[1]+a*np.sin(np.deg2rad(15))])
    gama = np.array([beta[1],beta[0]])

    return np.array([alpha,beta,gama])

def round_up(x):
    '''
    Calculates the next integer from a float (rounds up)

    x : float
        Any float
    '''
    decimals = x - int(x)
    if np.isclose(decimals,0):
        return int(x)
    else:
        return int(x) + 1

rounding_decs = 3 # Number of decimals on the display (the calculations use more decimals)

a = 0.3 # Error margin
goal = 'Min' # or Max


# start = starting_shape_for_2_dimensions(a,np.array([0.0,0.0]))
# start = np.array([[0.0,0.0],[0.28971,0.07761],[0.07761,0.28971]]) # same as line above
start = starting_shape_for_3_dimensions(a,np.array([1,2,3]))
n = start.shape[1] # Number of dimensions
M_limit = round_up(0.05*n**2 + 1.65*n) # Max number of iterations with the same best value
current = start.copy()
M = 1

z_start = [f(x) for x in start]
if goal == 'Min':
    best = start[np.where(z_start == np.min(z_start))]
    worst = start[np.where(z_start == np.max(z_start))]
    second_worst = start[np.where(z_start == np.max(np.delete(z_start,np.where(z_start == np.max(z_start)),axis=0)))]

if goal == 'Max':
    best = start[np.where(z_start == np.max(z_start))]
    worst = start[np.where(z_start == np.min(z_start))]
    second_worst = start[np.where(z_start == np.min(np.delete(z_start,np.where(z_start == np.min(z_start)),axis=0)))]

best_prev = best
R_prev = start[0] + np.array([[1000 for _ in range(n)]]) # random huge values that are for sure diferrent than the ones in start
rule = 0 # Rule number
i = 1 # Point/iteration number

myTable = PrettyTable(["i","N","f(N)","R","rule","Working triangle/tetrahedron","Best","M"]) # Table headers

for x in start: # First rows of table
    if (x != start[-1]).all():
        myTable.add_row([i,np.round(x,rounding_decs),np.round(f(x),rounding_decs),"-","-","-","-",'-'])
    else:
        str_polygon = str(np.round(start,rounding_decs))
        myTable.add_row([i,np.round(x,rounding_decs),np.round(f(x),rounding_decs),"-","-",str_polygon.replace('\n','')[1:-1],np.round(best.flatten(),rounding_decs),M])
    i += 1


while M < M_limit: # Geometric Method (Nelder-Mead)

    R = worst.copy() # Rule 1

    sum_of_all_points = (current.T @ np.ones((n+1,1))).T
    N = (2/n)*sum_of_all_points - 2*R
    rule = 1

    if np.allclose(N,R_prev): # Rule 2
        R = second_worst.copy()
        N = (2/n)*sum_of_all_points - 2*R
        rule = 2
    
    current[np.where(current == R)] = N
    R_prev = R
    best_prev = best
    
    z = [f(x) for x in current]

    if goal == 'Min':
        best = current[np.where(z == np.min(z))]

        worst = current[np.where(z == np.max(z))]

        second_worst = current[np.where(z == np.max(np.delete(z,np.where(z == np.max(z)),axis=0)))]
    if goal == 'Max':
        best = current[np.where(z == np.max(z))]

        worst = current[np.where(z == np.min(z))]

        second_worst = current[np.where(z == np.min(np.delete(z,np.where(z == np.min(z)),axis=0)))]

    

    if (best == best_prev).all():
        M += 1
    else:
        M = 1
    str_polygon = str(np.round(current,rounding_decs))
    myTable.add_row([i,np.round(N.flatten(),rounding_decs),np.round(f(N.flatten()),rounding_decs),np.round(R.flatten(),rounding_decs),rule,str_polygon.replace('\n','')[1:-1],np.round(best.flatten(),rounding_decs),M])
    print(myTable)

    i += 1

    
print(myTable)
if n == 2:
    print(f'\n{goal} of f(x) in x = ({np.round(best[0,0],rounding_decs)} \u00b1 {a}, {np.round(best[0,1],rounding_decs)} \u00b1 {a})^T')
elif n == 3:
    print(f'\n{goal} of f(x) in x = ({np.round(best[0,0],rounding_decs)} \u00b1 {a}, {np.round(best[0,1],rounding_decs)} \u00b1 {a}, {np.round(best[0,2],rounding_decs)} \u00b1 {a})^T')
else:
    print(f'\n{goal} of f(x) in x = {np.round(best.flatten(),rounding_decs)} \u00b1 {a}^T')
print(f'Where f(x) = {np.round(f(best.flatten()),rounding_decs)}')
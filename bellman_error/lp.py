#the purpose of this file is to generate a simple functional
# approximator to V(s,t)

import numpy as np
import time
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver('LinearExample',
                           pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

#x = solver.NumVar(0.0, 10.0, 'x')
#y = solver.NumVar(0.0, 10.0, 'y')
#
#constraint1 = solver.Constraint(-solver.infinity(), 14)
#constraint1.SetCoefficient(x, 1)
#constraint1.SetCoefficient(y, 2)
#
## Constraint 2: 3x - y >= 0.
#constraint2 = solver.Constraint(-solver.infinity(), 10)
#constraint2.SetCoefficient(x, 3)
#constraint2.SetCoefficient(y, 1)
#
#objective = solver.Objective()
#objective.SetCoefficient(x, 5)
#objective.SetCoefficient(y, 6)
#objective.SetMaximization()
#
#solver.Solve()
#
#print('Solution:')
#print('x = ', x.solution_value())
#print('y = ', y.solution_value())

#general initialization
ts = time.time()
np.set_printoptions(precision=4)
np.random.seed(4321)

#business parameter initialization
num_nights = 14
capacity = 100
# product zero is the no-revenue no resource product
# added for simplicity
product_null = 0
# unfortunately, to avoid confusion we need to add a fairly 
# complex product matrix
# if there are N nights, there are N one-night product from 
# 1 to N; there are also N-1 two-night products from N+1 to 2N-1
num_product = num_nights*2
product_resource_map = np.zeros((num_product, num_nights))
for i in range(1,num_nights):
    product_resource_map[i][i-1] = 1.0
    product_resource_map[i][i] = 1.0
for i in range(0,num_nights):    
    product_resource_map[i+num_nights][i] = 1.0
#product_resource_map[num_product-1][num_nights-1] = 1.0

product_revenue = 1000*np.random.uniform(size=[num_product])
product_revenue[product_null] = 0
#total demand
product_demand = np.random.uniform(size=[num_product])*capacity
product_demand[product_null]  = 0

num_steps = int(np.sum(product_demand)/0.01)

#arrival rate (including)
product_prob = np.divide(product_demand,num_steps)
product_prob[0] = 1.0 - np.sum(product_prob)

batch_size = 64

#generate simulation states
data_lhs_0 = np.random.choice(capacity+1, [batch_size, num_nights])
time_lhs = np.random.choice(range(1,num_steps), [batch_size,1])    

problem = 0
#capacity constraints
#cap_supply = data_lhs_0[problem]
#demand constraint
#cap_demand = time_lhs[problem]*product_prob

def lp(cap_supply, cap_demand):
    ts = time.time()
    #variables are number of products sold
    x = [solver.NumVar(0.0, cap_demand[p], "".join(["x",str(p)])) 
            for p in range(num_product)]
    
    #constraints are capacity for each night
    constraints = []   
    for night in range(num_nights):
        #print(cap_supply[night])
        con = solver.Constraint(0.0, float(cap_supply[night]))
        #con = solver.Constraint(0, capacity)
        for p in range(num_product):        
            con.SetCoefficient(x[p], product_resource_map[p][night])
        constraints.append(con)
    
    #objective        
    objective = solver.Objective()
    for p in range(num_product):
        objective.SetCoefficient(x[p], product_revenue[p])        
    objective.SetCoefficient(x[product_null], -1.0)        
    objective.SetMaximization()    
    
    solver.Solve()
    
    if 0:    
        for p in range(num_product):
            print("p=", p, "price = %2.f"%product_revenue[p], "demand = %.2f"%cap_demand[p], ' allocation = %.2f'%(x[p].solution_value()))
            
        print('Solution = %.2f' % objective.Value())
        sol2 = np.sum([product_revenue[p]*x[p].solution_value() for p in range(num_product)])
        print("sol2 = %.2f" % sol2)
        
        print("total time = %.2f"%(time.time()-ts))    
    return objective.Value()

print("solution = %.2f"% lp(data_lhs_0[problem], time_lhs[problem]*product_prob))

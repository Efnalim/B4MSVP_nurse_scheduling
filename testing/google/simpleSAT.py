"""Simple solve."""
from ortools.linear_solver import pywraplp
import json

def main():

    f = open("testing\google\data\\n030w4\Sc-n030w4.json")
    sc_data = json.load(f)
 
    for key, value in sc_data.items():
        print(f'Key: {key}, Value: {value}')
    
    f.close()

    f = open("testing\google\data\\n030w4\WD-n030w4-0.json")
    sc_data = json.load(f)
 
    for key, value in sc_data.items():
        print(f'Key: {key}, Value: {value}')
    
    f.close()

    # Create the linear solver with the GLOP backend.
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return

    # Create the variables x and y.
    x = solver.NumVar(0, 1, "x")
    y = solver.NumVar(0, 2, "y")

    print("Number of variables =", solver.NumVariables())

    # Create a linear constraint, 0 <= x + y <= 2.
    ct = solver.Constraint(0, 2, "ct")
    ct.SetCoefficient(x, 1)
    ct.SetCoefficient(y, 1)

    print("Number of constraints =", solver.NumConstraints())

    # Create the objective function, 3 * x + y.
    objective = solver.Objective()
    objective.SetCoefficient(x, 3)
    objective.SetCoefficient(y, 1)
    objective.SetMinimization()

    status = solver.Solve()

    # if status == pywraplp.OPTIMAL or status == pywraplp.FEASIBLE:
    #     print(f"x = {solver.Value(x)}")
    #     print(f"y = {solver.Value(y)}")
    # else:
    #     print("No solution found.")

    print("Solution:")
    print("Objective value =", objective.Value())
    print("x =", x.solution_value())
    print("y =", y.solution_value())


if __name__ == "__main__":
    main()

    


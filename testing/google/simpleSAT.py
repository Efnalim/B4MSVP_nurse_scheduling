"""Simple solve."""
from ortools.sat.python import cp_model


def SimpleSatProgram():
    """Minimal CP-SAT example to showcase calling the solver."""
    # Creates the model.
    model = cp_model.CpModel()

    # Creates the variables.
    num_vals = 3
    x = model.NewIntVar(0, num_vals - 1, "x")
    y = model.NewIntVar(0, num_vals - 1, "y")
    z = model.NewIntVar(0, num_vals - 1, "z")

    # Creates the constraints.
    model.Add(x != y)

    # Creates a solver and solves the model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"x = {solver.Value(x)}")
        print(f"y = {solver.Value(y)}")
        print(f"z = {solver.Value(z)}")
    else:
        print("No solution found.")


SimpleSatProgram()
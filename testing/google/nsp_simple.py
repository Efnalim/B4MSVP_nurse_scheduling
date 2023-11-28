# PATAT conference
# Timetabling comunity
"""Example of a simple nurse scheduling problem."""
from ortools.sat.python import cp_model

import json

import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker 

shift_to_int = {"Early": 0, "Day": 1, "Late": 2, "Night": 3, "Any": 4}
skill_to_int = {"HeadNurse": 0, "Nurse": 1, "Caretaker": 2, "Trainee": 3}
day_to_int = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
all_weeks = range(4)

def add_shift_skill_req(model, req, shifts, shifts_with_skills, insufficient_staffing, all_nurses, all_days, all_shifts, all_skills):
    shift = shift_to_int[req["shiftType"]]
    skill = skill_to_int[req["skill"]]
    minimal_capacities_in_week = [
        req["requirementOnMonday"]["minimum"],
        req["requirementOnTuesday"]["minimum"],
        req["requirementOnWednesday"]["minimum"],
        req["requirementOnThursday"]["minimum"],
        req["requirementOnFriday"]["minimum"],
        req["requirementOnSaturday"]["minimum"],
        req["requirementOnSunday"]["minimum"],
    ]
    optimal_capacities_in_week = [
        req["requirementOnMonday"]["optimal"],
        req["requirementOnTuesday"]["optimal"],
        req["requirementOnWednesday"]["optimal"],
        req["requirementOnThursday"]["optimal"],
        req["requirementOnFriday"]["optimal"],
        req["requirementOnSaturday"]["optimal"],
        req["requirementOnSunday"]["optimal"],
    ]

    for week in all_weeks:
        for day, min_capacity in enumerate(minimal_capacities_in_week):
            skills_worked = []
            for n in all_nurses:
                skills_worked.append(shifts_with_skills[(n, day + week*7, shift, skill)])
            model.Add(sum(skills_worked) >= min_capacity)
    for week in all_weeks:
        for day, opt_capacity in enumerate(optimal_capacities_in_week):
            skills_worked = []
            for n in all_nurses:
                skills_worked.append(shifts_with_skills[(n, day + week*7, shift, skill)])
            model.Add(opt_capacity - sum(skills_worked) <= insufficient_staffing[(day + week*7, shift, skill)])
    return

def add_shift_succession_reqs(model, shifts, all_nurses, all_days, all_shifts, num_days):
    for n in all_nurses:
        for d in range(num_days - 1):
            for s in all_shifts:
                if(s == 1):
                    model.AddAtMostOne([shifts[(n, d, s)], shifts[(n, d + 1, s - 1)]])
                if(s == 2):
                    model.AddAtMostOne([shifts[(n, d, s)], shifts[(n, d + 1, s - 1)], shifts[(n, d + 1, s - 2)]])
                if(s == 3):
                    model.AddAtMostOne([shifts[(n, d, s)], shifts[(n, d + 1, s - 1)], shifts[(n, d + 1, s - 2)], shifts[(n, d + 1, s - 3)]])
    return 

def add_missing_skill_req(model, nurses_data, shifts_with_skills, all_days, all_shifts, all_skills):
    for index, nurse_data in enumerate(nurses_data):
        for sk in all_skills:
            has_skill = False
            for skill in nurse_data["skills"]:
                if sk == skill_to_int[skill]:
                    has_skill = True
                    break
            if has_skill is False:
                # print(f"  Nurse {nurse_data['id']} does not have skill {sk}")
                for d in all_days:
                    for s in all_shifts:
                        model.Add(shifts_with_skills[(index, d, s, sk)] == 0)

    return

def add_insatisfied_preferences_reqs(model, preferences, unsatisfied_preferences, shifts, all_nurses, all_days, all_shifts, all_skills):
    for preference in preferences:
        nurse_id = int(preference["nurse"].split("_")[1])
        day_id = day_to_int[preference["day"]]
        shift_id = shift_to_int[preference["shiftType"]]

        if shift_id != shift_to_int["Any"]:
            for week in all_weeks:
                model.Add(unsatisfied_preferences[(nurse_id, day_id + week*7, shift_id)] == shifts[(nurse_id, day_id + week*7, shift_id)])
        else:
            for week in all_weeks:
                shifts_worked = []
                for shift in all_shifts:
                    model.Add(unsatisfied_preferences[(nurse_id, day_id + week*7, shift)] == shifts[(nurse_id, day_id + week*7, shift)])
    return

def main():

    # Loading Data
    f1 = open("testing\google\data\\n030w4\Sc-n030w4.json")
    sc_data = json.load(f1)
    # for key, value in sc_data.items():
    #     print(f'Key: {key}, Value: {value}')
    f1.close()

    f2 = open("testing\google\data\\n030w4\WD-n030w4-0.json")
    wd_data = json.load(f2)    
    f2.close()

    num_nurses = len(sc_data["nurses"])
    num_shifts = 4
    num_skills = 4
    num_days = 28
    all_nurses = range(num_nurses)
    all_shifts = range(num_shifts)
    all_days = range(num_days)
    all_skills = range(num_skills)

    # Creates the model.
    model = cp_model.CpModel()

    # Creates shift variables.
    # shifts[(n, d, s)]: nurse 'n' works shift 's' on day 'd'.
    shifts = {}
    for n in all_nurses:
        for d in all_days:
            for s in all_shifts:
                shifts[(n, d, s)] = model.NewBoolVar(f"shift_n{n}_d{d}_s{s}")

    # Creates shifts_with_skills variables.
    # shifts_with_skills[(n, d, s, sk)]: nurse 'n' works shift 's' on day 'd' with skill 'sk'.
    shifts_with_skills = {}
    for n in all_nurses:
        for d in all_days:
            for s in all_shifts:
                for sk in all_skills:
                    shifts_with_skills[(n, d, s, sk)] = model.NewBoolVar(f"shift_n{n}_d{d}_s{s}_sk{sk}")

    # Each nurse works at most one shift per day.
    for n in all_nurses:
        for d in all_days:
            model.AddAtMostOne(shifts[(n, d, s)] for s in all_shifts)
    
    # Each nurse works at most one skill per day.
    for n in all_nurses:
        for d in all_days:
            for s in all_shifts:
                model.AddAtMostOne(shifts_with_skills[(n, d, s, sk)] for sk in all_skills)

    # If nurse is working with skill that shift, she is working that shift.
    for n in all_nurses:
        for d in all_days:
            for s in all_shifts:
                skills_worked = []
                for sk in all_skills:
                    skills_worked.append(shifts_with_skills[(n, d, s, sk)])
                model.Add(sum(skills_worked) == shifts[(n, d, s)])

    insufficient_staffing = {}
    for d in all_days:
        for s in all_shifts:
            for sk in all_skills:
                insufficient_staffing[(d, s, sk)] = model.NewIntVar(0, 10, f"insufficient_staffing_d{d}_s{s}_sk{sk}")
    
    unsatisfied_preferences = {}
    for n in all_nurses:
        for d in all_days:
            for s in all_shifts:
                unsatisfied_preferences[(n, d, s)] = model.NewBoolVar(f"shift_n{n}_d{d}_s{s}")

    for req in wd_data["requirements"]:
        add_shift_skill_req(model, req, shifts, shifts_with_skills, insufficient_staffing, all_nurses, all_days, all_shifts, all_skills)

    add_shift_succession_reqs(model, shifts, all_nurses, all_days, all_shifts, num_days)

    add_missing_skill_req(model, sc_data["nurses"], shifts_with_skills, all_days, all_shifts, all_skills)

    add_insatisfied_preferences_reqs(model, wd_data["shiftOffRequests"], unsatisfied_preferences, shifts, all_nurses, all_days, all_shifts, all_skills)

    model.Minimize( (30 * sum(insufficient_staffing[(d, s, sk)] for d in all_days for s in all_shifts for sk in all_skills))
                   + (10 * sum(unsatisfied_preferences[(n, d, s)] for n in all_nurses for d in all_days for s in all_shifts))
                   )


    # Try to distribute the shifts evenly, so that each nurse works
    # min_shifts_per_nurse shifts. If this is not possible, because the total
    # number of shifts is not divisible by the number of nurses, some nurses will
    # be assigned one more shift.
    # min_shifts_per_nurse = (num_shifts * num_days) // num_nurses
    # if num_shifts * num_days % num_nurses == 0:
    #     max_shifts_per_nurse = min_shifts_per_nurse
    # else:
    #     max_shifts_per_nurse = min_shifts_per_nurse + 1
    # for n in all_nurses:
    #     shifts_worked = []
    #     for d in all_days:
    #         for s in all_shifts:
    #             shifts_worked.append(shifts[(n, d, s)])
    #     model.Add(min_shifts_per_nurse <= sum(shifts_worked))
    #     model.Add(sum(shifts_worked) <= max_shifts_per_nurse)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    solver.parameters.linearization_level = 0
    # Enumerate all solutions.
    solver.parameters.enumerate_all_solutions = True


    class NursesPartialSolutionPrinter(cp_model.CpSolverSolutionCallback):
        """Print intermediate solutions."""

        def __init__(self, shifts, shifts_with_skills, num_nurses, num_days, num_shifts, limit):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._shifts = shifts
            self._shifts_with_skills = shifts_with_skills
            self._num_nurses = num_nurses
            self._num_days = num_days
            self._num_shifts = num_shifts
            self._solution_count = 0
            self._solution_limit = limit

        def on_solution_callback(self):
            self._solution_count += 1
            print(f"Solution {self._solution_count}")   
            # for d in range(self._num_days):
            #     print(f"Day {d}")
            #     for n in range(self._num_nurses):
            #         is_working = False
            #         for s in range(self._num_shifts):
            #             for sk in range(self._num_shifts):
            #                 if self.Value(self._shifts_with_skills[(n, d, s, sk)]):
            #                     is_working = True
            #                     print(f"  Nurse {n} works shift {s} with skill {sk}")
            #         if not is_working:
            #             print(f"  Nurse {n} does not work")
            if self._solution_count >= self._solution_limit:
                print(f"Stop search after {self._solution_limit} solutions")
                self.StopSearch()

        def solution_count(self):
            return self._solution_count

    # Display the first five solutions.
    solution_limit = 1000
    solution_printer = NursesPartialSolutionPrinter(
        shifts, shifts_with_skills, num_nurses, num_days, num_shifts, solution_limit
    )

    solver.parameters.max_time_in_seconds = 960.0

    solver.Solve(model, solution_printer)

    # Statistics.
    print("\nStatistics")
    print(f"  - conflicts      : {solver.NumConflicts()}")
    print(f"  - branches       : {solver.NumBranches()}")
    print(f"  - wall time      : {solver.WallTime()} s")
    print(f"  - solutions found: {solution_printer.solution_count()}")
    print(f"  - minimum of objective function: {solver.ObjectiveValue()}\n")

    schedule_table = np.zeros([num_nurses, num_days * num_shifts]) 
    legend = np.zeros([1, num_skills + 1])

    for d in range(num_days):
        # print(f"Day {d}")
        for n in range(num_nurses):
            is_working = False
            for s in range(num_shifts):
                for sk in range(num_skills):
                    if solver.Value(shifts_with_skills[(n, d, s, sk)]) == 1:
                        is_working = True
                        schedule_table[n][d*num_shifts + s] = 1 - (0.2 * sk)
                        # print(f"  Nurse {n} works shift {s} with skill {sk}")
            # if not is_working:
                # print(f"  Nurse {n} does not work")

    for sk in range(num_skills):
        legend[0][sk] = 1 - (0.2 * sk)

    fig, (ax0, ax1) = plt.subplots(2, 1)
    
    c = ax0.pcolor(schedule_table) 
    ax0.set_title('Schedule') 
    ax0.set_xticks(np.arange(num_days*num_shifts))
    ax0.set_xticklabels(np.arange(num_days*num_shifts) / 4)

    ax0.xaxis.set_major_locator(ticker.MultipleLocator(4))

    c = ax1.pcolor(legend, edgecolors='k', linewidths=5) 
    ax1.set_title('Legend - skils') 
    ax1.set_xticks(np.arange(num_skills + 1) + 0.5)
    ax1.set_xticklabels([ "HeadNurse", "Nurse", "Caretaker", "Trainee", "Not working" ])
    
    fig.tight_layout() 
    plt.show() 


if __name__ == "__main__":
    main()
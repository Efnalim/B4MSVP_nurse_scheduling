# PATAT conference
# Timetabling comunity
"""Example of a simple nurse scheduling problem."""
from ortools.sat.python import cp_model

import json

import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker 
import math

shift_to_int = {"Early": 0, "Day": 1, "Late": 2, "Night": 3, "Any": 4}
skill_to_int = {"HeadNurse": 0, "Nurse": 1, "Caretaker": 2, "Trainee": 3}
contract_to_int = {"FullTime": 0, "PartTime": 1, "HalfTime": 2}
day_to_int = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
all_weeks = range(1)

def load_data():
    f0 = open("testing\google\data\\hidden-JSON\H0-n035w4-0.json")
    h0_data = json.load(f0)
    f0.close()

    f1 = open("testing\google\data\\hidden-JSON\Sc-n035w4.json")
    sc_data = json.load(f1)
    f1.close()

    f2 = open("testing\google\data\\hidden-JSON\WD-n035w4-1.json")
    wd_data_0 = json.load(f2)    
    f2.close()

    f3 = open("testing\google\data\\hidden-JSON\WD-n035w4-4.json")
    wd_data_1 = json.load(f3)    
    f3.close()

    f4 = open("testing\google\data\\hidden-JSON\WD-n035w4-1.json")
    wd_data_2 = json.load(f4)    
    f4.close()

    f5 = open("testing\google\data\\hidden-JSON\WD-n035w4-8.json")
    wd_data_3 = json.load(f5)    
    f5.close()

    # initialize constants
    num_nurses = len(sc_data["nurses"])
    num_shifts = len(sc_data["shiftTypes"])
    num_skills = len(sc_data["skills"])
    num_days = 7
    all_nurses = range(num_nurses)
    all_shifts = range(num_shifts)
    all_days = range(num_days)
    all_skills = range(num_skills)

    constants = {}
    constants["h0_data"] = h0_data
    constants["sc_data"] = sc_data
    constants["wd_data"] = wd_data_0
    constants["all_wd_data"] = [wd_data_0, wd_data_1, wd_data_2, wd_data_3]
    constants["num_nurses"] = num_nurses
    constants["num_shifts"] = num_shifts
    constants["num_skills"] = num_skills
    constants["num_days"] = num_days
    constants["all_nurses"] = all_nurses
    constants["all_shifts"] = all_shifts
    constants["all_days"] = all_days
    constants["all_skills"] = all_skills

    return constants

def init_ilp_vars(model, constants):
    all_nurses = constants["all_nurses"]
    all_shifts = constants["all_shifts"]
    all_days = constants["all_days"]
    all_skills = constants["all_skills"]

    # shifts[(n, d, s)]: nurse 'n' works shift 's' on day 'd'.
    shifts = {}
    for n in all_nurses:
        for d in all_days:
            for s in all_shifts:
                shifts[(n, d, s)] = model.NewBoolVar(f"shift_n{n}_d{d}_s{s}")
    
    working_days = {}
    for n in all_nurses:
        for d in all_days:
            working_days[(n, d)] = model.NewBoolVar(f"shift_n{n}_d{d}")

    # Creates shifts_with_skills variables.
    # shifts_with_skills[(n, d, s, sk)]: nurse 'n' works shift 's' on day 'd' with skill 'sk'.
    shifts_with_skills = {}
    for n in all_nurses:
        for d in all_days:
            for s in all_shifts:
                for sk in all_skills:
                    shifts_with_skills[(n, d, s, sk)] = model.NewBoolVar(f"shift_n{n}_d{d}_s{s}_sk{sk}")

    insufficient_staffing = {}
    for d in all_days:
        for s in all_shifts:
            for sk in all_skills:
                insufficient_staffing[(d, s, sk)] = model.NewIntVar(0, 10, f"insufficient_staffing_d{d}_s{s}_sk{sk}")

    basic_ILP_vars = {}
    basic_ILP_vars["working_days"] = working_days
    basic_ILP_vars["shifts"] = shifts
    basic_ILP_vars["shifts_with_skills"] = shifts_with_skills
    basic_ILP_vars["insufficient_staffing"] = insufficient_staffing
    return basic_ILP_vars

def init_ilp_vars_for_soft_constraints(model, basic_ILP_vars, constants):
    all_nurses = constants["all_nurses"]
    all_shifts = constants["all_shifts"]
    all_days = constants["all_days"]
    num_days = constants["num_days"]
    shifts = basic_ILP_vars["shifts"]
    working_days = basic_ILP_vars["working_days"]
    wd_data = constants["wd_data"]
    sc_data = constants["sc_data"]
    history_data = constants["h0_data"]

    # Preferences for shifts off
    unsatisfied_preferences = {}
    for n in all_nurses:
        for d in all_days:
            for s in all_shifts:
                unsatisfied_preferences[(n, d, s)] = model.NewBoolVar(f"unsatisfied_preferences_n{n}_d{d}_s{s}")

    # Vars for each nurse how many days they worked
    total_working_days = {}
    for n in all_nurses:
        total_working_days[(n)] = model.NewIntVar(0, num_days, f"total_working_days_n{n}")

    for n in all_nurses:
        days_worked = []
        for d in all_days:
            for s in all_shifts:
                days_worked.append(shifts[(n, d, s)])
        model.Add(sum(days_worked) == total_working_days[(n)])

    # Vars for each nurse n indicationg if they were working on weekend w
    working_weekends = {}
    for n in all_nurses:
        working_weekends[(n)] = model.NewBoolVar(f"working_weekends_n{n}")
        model.Add(working_days[(n, 5)] + working_days[(n, 6)] >= working_weekends[(n)])
        model.Add(working_days[(n, 5)] <= working_weekends[(n)])
        model.Add(working_days[(n, 6)] <= working_weekends[(n)])
        # model.Add(sum(shifts_worked_on_saturday) + sum(shifts_worked_on_sunday) >= working_weekends[(n, w)])

    total_working_weekends_over_limit = {}
    for n in all_nurses:
        total_working_weekends_over_limit[(n)] = model.NewIntVar(0, 4, f"total_working_weekends_over_limit_n{n}")
    
    total_incomplete_weekends = {}
    for n in all_nurses:
        total_incomplete_weekends[(n)] = model.NewIntVar(0, 4, f"total_incomplete_weekends_n{n}")

    total_working_days_over_limit = {}
    total_working_days_under_limit = {}
    for n in all_nurses:
        total_working_days_over_limit[(n)] = model.NewIntVar(0, num_days, f"total_working_days_over_limit_n{n}")
        total_working_days_under_limit[(n)] = model.NewIntVar(0, num_days, f"total_working_days_under_limit_n{n}")

    violations_of_max_consecutive_working_days = {}
    violations_of_max_consecutive_working_days_for_nurse = {}
    for n in all_nurses:
        violations_of_max_consecutive_working_days_for_nurse[(n)] = model.NewIntVar(0, num_days, f"violations_of_max_consecutive_working_days_for_nurse{n}")
        all_violation_for_nurse = []
        max_consecutive_working_days = sc_data["contracts"][contract_to_int[sc_data["nurses"][n]["contract"]]]["maximumNumberOfConsecutiveWorkingDays"]
        for d in all_days:
            if d + max_consecutive_working_days >= num_days:
                break
            violations_of_max_consecutive_working_days[(n, d + max_consecutive_working_days)] = model.NewBoolVar(f"violations_of_max_consecutive_working_days_n{n}_d{d + max_consecutive_working_days}")
            all_violation_for_nurse.append(violations_of_max_consecutive_working_days[(n, d + max_consecutive_working_days)])
            working_days_to_sum = []
            for dd in range(max_consecutive_working_days + 1):
                for s in all_shifts:
                    working_days_to_sum.append(shifts[(n, d + dd, s)])
            model.Add(violations_of_max_consecutive_working_days[(n, d + max_consecutive_working_days)] >= (sum(working_days_to_sum) - max_consecutive_working_days))
        
        violations_of_max_consecutive_working_days_from_prev_week = {}
        prev_week_consecutive_working_days = history_data["nurseHistory"][n]["numberOfConsecutiveWorkingDays"]
        if prev_week_consecutive_working_days > 0:
            for d in range(max_consecutive_working_days):
                if prev_week_consecutive_working_days - d == 0:
                    break

                violations_of_max_consecutive_working_days_from_prev_week[(n, d)] = model.NewBoolVar(f"violations_of_max_consecutive_working_days_from_prev_week{n}_d{d}")
                all_violation_for_nurse.append(violations_of_max_consecutive_working_days_from_prev_week[(n, d)])
                
                working_days_to_sum = []
                for dd in range(max_consecutive_working_days - d):
                    working_days_to_sum.append(working_days[(n, d)])
                model.Add(violations_of_max_consecutive_working_days_from_prev_week[(n, d)] >= (sum(working_days_to_sum) + d + 1 - max_consecutive_working_days))

        model.Add(violations_of_max_consecutive_working_days_for_nurse[(n)] == sum(all_violation_for_nurse))
    
    violations_of_max_consecutive_working_shifts = {}
    violations_of_max_consecutive_working_shifts_for_nurse_for_shift_type = {}
    for n in all_nurses:
        for s in all_shifts:
            violations_of_max_consecutive_working_shifts_for_nurse_for_shift_type[(n, s)] = model.NewIntVar(0, num_days, f"violations_of_max_consecutive_working_shifts_for_nurse_for_shift_type_n{n}_s{s}")
            all_violation_for_nurse_for_shift_type = []
            max_consecutive_working_shifts = sc_data["shiftTypes"][s]["maximumNumberOfConsecutiveAssignments"]
            for d in all_days:
                if d + max_consecutive_working_shifts >= num_days:
                    break
                violations_of_max_consecutive_working_shifts[(n, s, d + max_consecutive_working_shifts)] = model.NewBoolVar(f"violations_of_max_consecutive_working_shifts_n{n}_s{s}_d{d + max_consecutive_working_shifts}")
                all_violation_for_nurse_for_shift_type.append(violations_of_max_consecutive_working_shifts[(n, s, d + max_consecutive_working_shifts)])
                working_shifts_to_sum = []
                for dd in range(max_consecutive_working_shifts + 1):
                    working_shifts_to_sum.append(shifts[(n, d + dd, s)])
                model.Add(violations_of_max_consecutive_working_shifts[(n, s, d + max_consecutive_working_shifts)] >= (sum(working_shifts_to_sum) - max_consecutive_working_shifts))
            model.Add(violations_of_max_consecutive_working_shifts_for_nurse_for_shift_type[(n, s)] == sum(all_violation_for_nurse_for_shift_type))

    violations_of_max_consecutive_days_off = {}
    violations_of_max_consecutive_days_off_for_nurse = {}
    for n in all_nurses:
        violations_of_max_consecutive_days_off_for_nurse[(n)] = model.NewIntVar(0, num_days, f"violations_of_max_consecutive_days_off_for_nurse{n}")
        all_violation_for_nurse = []
        max_consecutive_days_off = sc_data["contracts"][contract_to_int[sc_data["nurses"][n]["contract"]]]["maximumNumberOfConsecutiveDaysOff"]
        for d in all_days:
            if d + max_consecutive_days_off >= num_days:
                break
            violations_of_max_consecutive_days_off[(n, d + max_consecutive_days_off)] = model.NewBoolVar(f"violations_of_max_consecutive_days_off_n{n}_d{d + max_consecutive_working_days}")
            all_violation_for_nurse.append(violations_of_max_consecutive_days_off[(n, d + max_consecutive_days_off)])
            days_off_to_sum = []
            for dd in range(max_consecutive_days_off + 1):
                for s in all_shifts:
                    days_off_to_sum.append(shifts[(n, d + dd, s)])
            model.Add(violations_of_max_consecutive_days_off[(n, d + max_consecutive_days_off)] >= 1 - (sum(days_off_to_sum)))
        model.Add(violations_of_max_consecutive_days_off_for_nurse[(n)] == sum(all_violation_for_nurse))

    soft_ILP_vars = {}
    soft_ILP_vars["unsatisfied_preferences"] = unsatisfied_preferences
    soft_ILP_vars["total_working_days"] = total_working_days
    soft_ILP_vars["working_weekends"] = working_weekends
    soft_ILP_vars["total_working_weekends_over_limit"] = total_working_weekends_over_limit
    soft_ILP_vars["total_working_days_over_limit"] = total_working_days_over_limit
    soft_ILP_vars["total_working_days_under_limit"] = total_working_days_under_limit
    soft_ILP_vars["total_incomplete_weekends"] = total_incomplete_weekends
    soft_ILP_vars["violations_of_max_consecutive_working_days_for_nurse"] = violations_of_max_consecutive_working_days_for_nurse
    soft_ILP_vars["violations_of_max_consecutive_days_off_for_nurse"] = violations_of_max_consecutive_days_off_for_nurse
    soft_ILP_vars["violations_of_max_consecutive_working_shifts_for_nurse_for_shift_type"] = violations_of_max_consecutive_working_shifts_for_nurse_for_shift_type

    return soft_ILP_vars

def add_hard_constrains(model, basic_ILP_vars, constants):
    all_nurses = constants["all_nurses"]
    all_shifts = constants["all_shifts"]
    all_days = constants["all_days"]
    all_skills = constants["all_skills"]
    num_days = constants["num_days"]
    sc_data = constants["sc_data"]
    h0_data = constants["h0_data"]
    shifts = basic_ILP_vars["shifts"]
    working_days = basic_ILP_vars["working_days"]
    shifts_with_skills = basic_ILP_vars["shifts_with_skills"]
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
    
    for n in all_nurses:
        for d in all_days:
            shifts_worked = []
            for s in all_shifts:
                shifts_worked.append(shifts[(n, d, s)])
            model.Add(sum(shifts_worked) == working_days[(n, d)])

    add_shift_succession_reqs(model, shifts, all_nurses, all_days, all_shifts, num_days)
    add_missing_skill_req(model, sc_data["nurses"], shifts_with_skills, all_days, all_shifts, all_skills)

    return

def add_soft_constraints(model, basic_ILP_vars, soft_ILP_vars, constants):
    all_nurses = constants["all_nurses"]
    all_days = constants["all_days"]
    all_shifts = constants["all_shifts"]
    all_skills = constants["all_skills"]
    sc_data = constants["sc_data"]
    h0_data = constants["h0_data"]
    wd_data = constants["wd_data"]
    shifts = basic_ILP_vars["shifts"]
    working_days = basic_ILP_vars["working_days"]
    unsatisfied_preferences = soft_ILP_vars["unsatisfied_preferences"]
    total_working_days = soft_ILP_vars["total_working_days"]
    total_working_days_over_limit = soft_ILP_vars["total_working_days_over_limit"]
    total_incomplete_weekends = soft_ILP_vars["total_incomplete_weekends"]
    working_weekends = soft_ILP_vars["working_weekends"]
    total_working_weekends_over_limit = soft_ILP_vars["total_working_weekends_over_limit"]
    total_working_days_under_limit = soft_ILP_vars["total_working_days_under_limit"]
    total_incomplete_weekends = soft_ILP_vars["total_incomplete_weekends"]
    
    add_insatisfied_preferences_reqs(model, wd_data["shiftOffRequests"], unsatisfied_preferences, shifts, all_nurses, all_days, all_shifts, all_skills)

    add_total_working_days_out_of_bounds_constraint(model, sc_data["nurses"], sc_data["contracts"], h0_data["nurseHistory"], total_working_days, total_working_days_over_limit, total_working_days_under_limit, all_nurses)
    
    add_total_incomplete_weekends_constraint(model, sc_data["nurses"], sc_data["contracts"], total_incomplete_weekends, working_weekends, shifts, working_days, all_nurses, all_days, all_shifts)
    
    add_total_working_weekends_soft_constraints(model, sc_data["nurses"], sc_data["contracts"], h0_data["nurseHistory"], total_working_weekends_over_limit, working_weekends, all_nurses)
    return 

def add_total_incomplete_weekends_constraint(model, nurses_data, contracts_data, total_incomplete_weekends, working_weekends, shifts, working_days, all_nurses, all_days, all_shifts):
    incomplete_weekends = {}
    for n in all_nurses:
        isCompleteWeekendRequested = contracts_data[contract_to_int[nurses_data[n]["contract"]]]["completeWeekends"]
        if isCompleteWeekendRequested == 1:
            incomplete_weekends[(n)] = model.NewBoolVar(f"incomplete_weekends_n{n}")
            model.Add(incomplete_weekends[(n)] == 2*working_weekends[(n)] - working_days[(n, 5)] - working_days[(n, 6)])
                
    for n in all_nurses:
        isCompleteWeekendRequested = contracts_data[contract_to_int[nurses_data[n]["contract"]]]["completeWeekends"]
        if isCompleteWeekendRequested == 1:
            incomplete_weekends_n = []
            incomplete_weekends_n.append(incomplete_weekends[(n)])
            model.Add(total_incomplete_weekends[(n)] == sum(incomplete_weekends_n))
    return 

def add_total_working_days_out_of_bounds_constraint(model, nurses_data, contracts_data, history, total_working_days, total_working_days_over_limit, total_working_days_under_limit, all_nurses):
    for n in all_nurses:
        upper_limit = math.ceil(contracts_data[contract_to_int[nurses_data[n]["contract"]]]["maximumNumberOfAssignments"] / 4) + 1
        lower_limit = math.ceil(contracts_data[contract_to_int[nurses_data[n]["contract"]]]["minimumNumberOfAssignments"] / 4) - 1
        model.Add(total_working_days_over_limit[(n)] >= total_working_days[(n)] - upper_limit)
        model.Add(total_working_days_under_limit[(n)] >= lower_limit - total_working_days[(n)])
    return 

def add_shift_skill_req(model, req, basic_ILP_vars, soft_ILP_vars, constants):
    all_nurses = constants["all_nurses"]
    shifts_with_skills = basic_ILP_vars["shifts_with_skills"]
    insufficient_staffing = basic_ILP_vars["insufficient_staffing"]

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

def add_total_working_weekends_soft_constraints(model, nurses_data, contracts_data, history, total_working_weekends_over_limit, working_weekends, all_nurses):
    for n in all_nurses:
        worked_weekends = []
        worked_weekends_limit = contracts_data[contract_to_int[nurses_data[n]["contract"]]]["maximumNumberOfWorkingWeekends"]
        worked_weekends.append(working_weekends[(n)])
        model.Add(total_working_weekends_over_limit[(n)] >= sum(worked_weekends) - worked_weekends_limit + history[n]["numberOfWorkingWeekends"])
        model.Add(total_working_weekends_over_limit[(n)] >= -(sum(worked_weekends) - worked_weekends_limit + history[n]["numberOfWorkingWeekends"]))
    return

def set_objective_function(model, basic_ILP_vars, soft_ILP_vars, constants):
    all_nurses = constants["all_nurses"]
    all_days = constants["all_days"]
    all_shifts = constants["all_shifts"]
    all_skills = constants["all_skills"]

    insufficient_staffing = basic_ILP_vars["insufficient_staffing"]

    unsatisfied_preferences = soft_ILP_vars["unsatisfied_preferences"]
    total_working_days_over_limit = soft_ILP_vars["total_working_days_over_limit"]
    total_incomplete_weekends = soft_ILP_vars["total_incomplete_weekends"]
    total_working_weekends_over_limit = soft_ILP_vars["total_working_weekends_over_limit"]
    total_working_days_under_limit = soft_ILP_vars["total_working_days_under_limit"]
    total_incomplete_weekends = soft_ILP_vars["total_incomplete_weekends"]
    violations_of_max_consecutive_working_days_for_nurse = soft_ILP_vars["violations_of_max_consecutive_working_days_for_nurse"]
    violations_of_max_consecutive_days_off_for_nurse = soft_ILP_vars["violations_of_max_consecutive_days_off_for_nurse"]
    violations_of_max_consecutive_working_shifts_for_nurse_for_shift_type = soft_ILP_vars["violations_of_max_consecutive_working_shifts_for_nurse_for_shift_type"]

    model.Minimize( 
                    (30 * sum(insufficient_staffing[(d, s, sk)] for d in all_days for s in all_shifts for sk in all_skills))
                    + 
                    (10 * sum(unsatisfied_preferences[(n, d, s)] for n in all_nurses for d in all_days for s in all_shifts))
                    +
                    (30 * sum(total_working_weekends_over_limit[(n)] for n in all_nurses))
                    +
                    (30 * sum(total_incomplete_weekends[(n)] for n in all_nurses))
                    +
                    (20 * sum(total_working_days_over_limit[(n)] for n in all_nurses))
                    +
                    (20 * sum(total_working_days_under_limit[(n)] for n in all_nurses))
                    +
                    (30 * sum(violations_of_max_consecutive_working_days_for_nurse[(n)] for n in all_nurses))
                    +
                    (30 * sum(violations_of_max_consecutive_days_off_for_nurse[(n)] for n in all_nurses))
                    +
                    (15 * sum(violations_of_max_consecutive_working_shifts_for_nurse_for_shift_type[(n, s)] for n in all_nurses for s in all_shifts))
                )
    return

def print_results(solver, solution_printer, basic_ILP_vars, soft_ILP_vars, constants):
    num_days = constants["num_days"]
    num_nurses = constants["num_nurses"]
    num_skills = constants["num_skills"]
    num_shifts = constants["num_shifts"]

    shifts_with_skills = basic_ILP_vars["shifts_with_skills"]

    total_working_days = soft_ILP_vars["total_working_days"]
    total_working_days_over_limit = soft_ILP_vars["total_working_days_over_limit"]
    working_weekends = soft_ILP_vars["working_weekends"]
    total_working_weekends_over_limit = soft_ILP_vars["total_working_weekends_over_limit"]
    violations_of_max_consecutive_days_off_for_nurse = soft_ILP_vars["violations_of_max_consecutive_days_off_for_nurse"]
    violations_of_max_consecutive_working_days_for_nurse = soft_ILP_vars["violations_of_max_consecutive_working_days_for_nurse"]

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
    # for n in range(num_nurses):
    #     print(f"  Nurse {n} works {solver.Value(total_working_weekends_over_limit[(n)])} weekends over limit")
    #     # print(f"  Nurse {n} works {solver.Value(working_weekends[(n, 0)])} {solver.Value(working_weekends[(n, 1)])} {solver.Value(working_weekends[(n, 2)])} {solver.Value(working_weekends[(n, 3)])}")
    #     print(f"  Nurse {n} works {solver.Value(total_working_days[(n)])} days in month, that is {solver.Value(total_working_days_over_limit[(n)])} over limit")
    #     print(f"  Nurse {n} has {solver.Value(violations_of_max_consecutive_days_off_for_nurse[(n)])} consecutive days off over limit")
    #     print(f"  Nurse {n} works {solver.Value(violations_of_max_consecutive_working_days_for_nurse[(n)])} consecutive days over limit")

def display_schedule(results, constants, number_weeks):
    num_days = 7 * number_weeks
    num_nurses = constants["num_nurses"]
    num_skills = constants["num_skills"]
    num_shifts = constants["num_shifts"]

    schedule_table = np.zeros([num_nurses, num_days * num_shifts]) 
    legend = np.zeros([1, num_skills + 1])

    for d in range(num_days):
        for n in range(num_nurses):
            for s in range(num_shifts):
                for sk in range(num_skills):
                    if results[(n, d, s, sk)] == 1:
                        schedule_table[n][d*num_shifts + s] = 1 - (0.2 * sk)

    for sk in range(num_skills):
        legend[0][sk] = 1 - (0.2 * sk)

    fig, (ax0, ax1) = plt.subplots(2, 1)
    
    c = ax0.pcolor(schedule_table) 
    ax0.set_title('Schedule') 
    ax0.set_xticks(np.arange(num_days*num_shifts))
    ax0.set_xticklabels(np.arange(num_days*num_shifts) / 4)

    ax0.xaxis.set_major_locator(ticker.MultipleLocator(4))

    c = ax1.pcolor(legend, edgecolors='k', linewidths=5) 
    ax1.set_title('Legend - skills') 
    ax1.set_xticks(np.arange(num_skills + 1) + 0.5)
    ax1.set_xticklabels([ "HeadNurse", "Nurse", "Caretaker", "Trainee", "Not working" ])
    
    fig.tight_layout() 
    plt.show() 

def save_tmp_results(results, solver, constants, basic_ILP_vars, soft_ILP_vars, week_number):
    num_days = constants["num_days"]
    num_nurses = constants["num_nurses"]
    num_skills = constants["num_skills"]
    num_shifts = constants["num_shifts"]
    history_data = constants["h0_data"]
    working_weekends = soft_ILP_vars["working_weekends"]

    shifts_with_skills = basic_ILP_vars["shifts_with_skills"]
    working_days = basic_ILP_vars["working_days"]
    shifts = basic_ILP_vars["shifts"]

    for n in range(num_nurses):
        for d in range(num_days):
            for s in range(num_shifts):
                for sk in range(num_skills):
                    results[(n, d + 7 * week_number, s, sk)] = solver.Value(shifts_with_skills[(n, d, s, sk)])
                    # if(results[(n, d + 7 * week_number, s, sk)]) == 1:
            history_data["nurseHistory"][n]["numberOfAssignments"] += solver.Value(working_days[(n, d)])
        history_data["nurseHistory"][n]["numberOfWorkingWeekends"] += solver.Value(working_weekends[(n)])

        if solver.Value(working_days[(n, 6)]) == 0:
            consecutive_free_days = 1
            d = 5
            while d >= 0 and solver.Value(working_days[(n, d)]) == 0:
                consecutive_free_days += 1
                d -= 1
            history_data["nurseHistory"][n]["numberOfConsecutiveDaysOff"] = consecutive_free_days
            history_data["nurseHistory"][n]["numberOfConsecutiveWorkingDays"] = 0
            history_data["nurseHistory"][n]["numberOfConsecutiveAssignments"] = 0
            history_data["nurseHistory"][n]["lastAssignedShiftType"] = "None"
        else:
            consecutive_work_days = 1
            d = 5
            while d >= 0 and solver.Value(working_days[(n, d)]) == 1:
                consecutive_work_days += 1
                d -= 1
            history_data["nurseHistory"][n]["numberOfConsecutiveWorkingDays"] = consecutive_work_days
            history_data["nurseHistory"][n]["numberOfConsecutiveDaysOff"] = 0

            consecutive_shift = 0
            for s in range(num_shifts):
                if solver.Value(shifts[(n, 6, s)]) == 1:
                    consecutive_shift = s
                    break
            consecutive_shifts = 1
            for shift_name, shift_id in shift_to_int.items():
                if( shift_id == consecutive_shift):
                    history_data["nurseHistory"][n]["lastAssignedShiftType"] = shift_name
                    
            d = 5
            while d >= 0 and solver.Value(shifts[(n, d, consecutive_shift)]) == 1:
                consecutive_shifts += 1
                d -= 1
            history_data["nurseHistory"][n]["numberOfConsecutiveAssignments"] = consecutive_shifts

                        
    return

def compute_one_week(time_limit_for_week, week_number, constants, results):
    # Creates the model.
    model = cp_model.CpModel()

    # Create ILP variables.
    # shifts, shifts_with_skills, insufficient_staffing = init_ilp_vars(model, all_nurses, all_days, all_shifts, all_skills)
    basic_ILP_vars = init_ilp_vars(model, constants)

    # Add hard constrains to model
    add_hard_constrains(model, basic_ILP_vars, constants)
    
    soft_ILP_vars = init_ilp_vars_for_soft_constraints(model, basic_ILP_vars, constants)

    for req in constants["wd_data"]["requirements"]:
        add_shift_skill_req(model, req, basic_ILP_vars, soft_ILP_vars, constants)

    add_soft_constraints(model, basic_ILP_vars, soft_ILP_vars, constants)

    # Sets objective function
    set_objective_function(model, basic_ILP_vars, soft_ILP_vars, constants)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    solver.parameters.linearization_level = 0
    # Enumerate all solutions.
    solver.parameters.enumerate_all_solutions = True

    class NursesPartialSolutionPrinter(cp_model.CpSolverSolutionCallback):
        """Print intermediate solutions."""

        def __init__(self, basic_ILP_vars, soft_ILP_vars, constants, solution_limit):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._shifts = basic_ILP_vars["shifts"]
            self._shifts_with_skills = basic_ILP_vars["shifts_with_skills"]
            self._num_nurses = constants["num_nurses"]
            self._num_days = constants["num_days"]
            self._num_shifts = constants["num_shifts"]
            self._solution_count = 0
            self._solution_limit = solution_limit

        def on_solution_callback(self):
            self._solution_count += 1
            print(f"Solution {self._solution_count} with value: {self.ObjectiveValue()} and time: {self.WallTime()} s")   
            if self._solution_count >= self._solution_limit:
                print(f"Stop search after {self._solution_limit} solutions")
                self.StopSearch()

        def solution_count(self):
            return self._solution_count

    # Display the first five solutions.
    solution_limit = 5000
    solution_printer = NursesPartialSolutionPrinter(basic_ILP_vars, soft_ILP_vars, constants, solution_limit)

    solver.parameters.max_time_in_seconds = time_limit_for_week
    # solver.parameters.max_time_in_seconds = 2 * 10 + 30 * (35 - 20)
    # solver.parameters.max_time_in_seconds = 60.0
    # solver.parameters.max_time_in_seconds = 3*60*60.0
    solver.Solve(model, solution_printer)

    save_tmp_results(results, solver, constants, basic_ILP_vars, soft_ILP_vars, week_number)
    print_results(solver, solution_printer, basic_ILP_vars, soft_ILP_vars, constants)
    return

def main():

    # Loading Data and init constants
    constants = load_data()
    results = {}
    for week_number in range(4):
        constants["wd_data"] = constants["all_wd_data"][week_number]
        compute_one_week(week_number, constants, results)
    display_schedule(results, constants, 4)
    print(json.dumps(constants["h0_data"], indent=4))

if __name__ == "__main__":
    main()
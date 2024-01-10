#!/usr/bin/python

from math import fabs

import itertools
import math

import cplex

shift_to_int = {"Early": 0, "Day": 1, "Late": 2, "Night": 3, "Any": 4}
skill_to_int = {"HeadNurse": 0, "Nurse": 1, "Caretaker": 2, "Trainee": 3}
contract_to_int = {"FullTime": 0, "PartTime": 1, "HalfTime": 2}
day_to_int = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

def init_ilp_vars(model, constants):
    all_nurses = constants["all_nurses"]
    all_shifts = constants["all_shifts"]
    all_days = constants["all_days"]
    all_skills = constants["all_skills"]

    # Creates shifts variables.
    # shifts[n][d][s]: nurse 'n' works shift 's' on day 'd' if 1 does not work if 0.
    vars_to_add = []
    shifts = []
    for n in all_nurses:
        shifts.append([])
        for d in all_days:
            shifts[n].append([])
            for s in all_shifts:
                var_name = f"shift_n{n}_d{d}_s{s}"
                vars_to_add.append(var_name)
                shifts[n][d].append(var_name)
                
    model.variables.add(names=vars_to_add, lb=[0] * len(vars_to_add), ub=[1] * len(vars_to_add), types=["B"] * len(vars_to_add))
    
    # Creates working_days variables.
    # shifts[n][d][s]: nurse 'n' works on day 'd' if 1 does not work if 0.
    vars_to_add = []
    working_days = []
    for n in all_nurses:
        working_days.append([])
        for d in all_days:
            var_name = f"work_day_n{n}_d{d}"
            working_days[n].append(var_name)
            vars_to_add.append(var_name)
    
    model.variables.add(names=vars_to_add, lb=[0] * len(vars_to_add), ub=[1] * len(vars_to_add), types=["B"] * len(vars_to_add))

    # Creates shifts_with_skills variables.
    # shifts_with_skills[(n, d, s, sk)]: nurse 'n' works shift 's' on day 'd' with skill 'sk'.
    vars_to_add = []
    shifts_with_skills = []
    for n in all_nurses:
        shifts_with_skills.append([])
        for d in all_days:
            shifts_with_skills[n].append([])
            for s in all_shifts:
                shifts_with_skills[n][d].append([])
                for sk in all_skills:
                    var_name = f"shift_with_skill_n{n}_d{d}_s{s}_sk{sk}"
                    vars_to_add.append(var_name)
                    shifts_with_skills[n][d][s].append(var_name)
                
    model.variables.add(names=vars_to_add, lb=[0] * len(vars_to_add), ub=[1] * len(vars_to_add), types=["B"] * len(vars_to_add))

    basic_ILP_vars = {}
    basic_ILP_vars["working_days"] = working_days
    basic_ILP_vars["shifts"] = shifts
    basic_ILP_vars["shifts_with_skills"] = shifts_with_skills
    # basic_ILP_vars["insufficient_staffing"] = insufficient_staffing
    return basic_ILP_vars

def add_shift_succession_reqs(model, shifts, all_nurses, all_days, all_shifts, num_days):
    for n in all_nurses:
        for d in range(num_days - 1):
            for s in all_shifts:
                if(s == 1):
                    model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([shifts[n][d][s], shifts[n][d + 1][s - 1]], [1] * 2)],
                    senses=["L"],
                    rhs=[1])
                if(s == 2):
                    model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([shifts[n][d][s], shifts[n][d + 1][s - 1], shifts[n][d + 1][s - 2]], [1] * 3)],
                    senses=["L"],
                    rhs=[1])
                if(s == 3):
                    model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([shifts[n][d][s], shifts[n][d + 1][s - 1], shifts[n][d + 1][s - 2], shifts[n][d + 1][s - 3]], [1] * 4)],
                    senses=["L"],
                    rhs=[1])

def add_missing_skill_req(model, nurses_data, shifts_with_skills, all_days, all_shifts, all_skills):
    for n, nurse_data in enumerate(nurses_data):
        for sk in all_skills:
            has_skill = False
            for skill in nurse_data["skills"]:
                if sk == skill_to_int[skill]:
                    has_skill = True
                    break
            if has_skill is False:
                for d in all_days:
                    for s in all_shifts:
                        model.linear_constraints.add(
                        lin_expr=[cplex.SparsePair([shifts_with_skills[n][d][s][sk]], [1])],
                        senses=["E"],
                        rhs=[0])


def add_hard_constrains(model, basic_ILP_vars, constants):
    all_nurses = constants["all_nurses"]
    all_shifts = constants["all_shifts"]
    all_days = constants["all_days"]
    all_skills = constants["all_skills"]
    num_days = constants["num_days"]
    sc_data = constants["sc_data"]
    wd_data = constants["wd_data"]
    h0_data = constants["h0_data"]
    shifts = basic_ILP_vars["shifts"]
    working_days = basic_ILP_vars["working_days"]
    shifts_with_skills = basic_ILP_vars["shifts_with_skills"]

    # Each nurse works at most one shift per day.
    for n in all_nurses:
        for d in all_days:
            model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(shifts[n][d][:], [1] * len(shifts[n][d][:]))],
            senses=["L"],
            rhs=[1])
    
    # Each nurse works at most one skill per day.
    for n in all_nurses:
        for d in all_days:
            for s in all_shifts:
                model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(shifts_with_skills[n][d][s][:], [1] * len(shifts_with_skills[n][d][s][:]))],
                senses=["L"],
                rhs=[1])

    # If nurse is working with skill that shift, she is working that shift.
    for n in all_nurses:
        for d in all_days:
            for s in all_shifts:
                model.linear_constraints.add(
                lin_expr=[cplex.SparsePair([shifts[n][d][s]] + shifts_with_skills[n][d][s][:], [-1] + [1] * len(shifts_with_skills[n][d][s][:]))],
                senses=["E"],
                rhs=[0])
    
    for n in all_nurses:
        for d in all_days:
            model.linear_constraints.add(
            lin_expr=[cplex.SparsePair([working_days[n][d]] + shifts[n][d][:], [-1] + [1] * len(shifts[n][d][:]))],
            senses=["E"],
            rhs=[0])

    add_shift_succession_reqs(model, shifts, all_nurses, all_days, all_shifts, num_days)
    add_missing_skill_req(model, sc_data["nurses"], shifts_with_skills, all_days, all_shifts, all_skills)

    for req in wd_data["requirements"]:
        add_shift_skill_req_minimal(model, req, basic_ILP_vars, constants)

def add_shift_skill_req_minimal(model, req, basic_ILP_vars, constants):
    all_nurses = constants["all_nurses"]
    # all_weeks = constants["all_weeks"]
    # all_weeks = range(1)
    shifts_with_skills = basic_ILP_vars["shifts_with_skills"]

    s = shift_to_int[req["shiftType"]]
    sk = skill_to_int[req["skill"]]
    minimal_capacities_in_week = [
        req["requirementOnMonday"]["minimum"],
        req["requirementOnTuesday"]["minimum"],
        req["requirementOnWednesday"]["minimum"],
        req["requirementOnThursday"]["minimum"],
        req["requirementOnFriday"]["minimum"],
        req["requirementOnSaturday"]["minimum"],
        req["requirementOnSunday"]["minimum"],
    ]

    for d, min_capacity in enumerate(minimal_capacities_in_week):
        skills_worked = []
        for n in all_nurses:
            skills_worked.append(shifts_with_skills[n][d][s][sk])
        model.linear_constraints.add(
        lin_expr=[cplex.SparsePair(skills_worked, [1] * len(skills_worked))],
        senses=["G"],
        rhs=[min_capacity])

def init_ilp_vars_for_soft_constraints(model, basic_ILP_vars, constants):
    all_nurses = constants["all_nurses"]
    all_shifts = constants["all_shifts"]
    all_skills = constants["all_skills"]
    all_days = constants["all_days"]
    num_days = constants["num_days"]
    shifts = basic_ILP_vars["shifts"]
    working_days = basic_ILP_vars["working_days"]
    wd_data = constants["wd_data"]
    sc_data = constants["sc_data"]
    history_data = constants["h0_data"]

    # Creates insufficient staffing variables.
    # shifts[(d,s,sk)]: number of nurses under optimal number for day d shift s and skill sk
    vars_to_add = []
    insufficient_staffing = {}
    for d in all_days:
        for s in all_shifts:
            for sk in all_skills:
                var_name = f"insufficient_staffing_d{d}_s{s}_sk{sk}"
                vars_to_add.append(var_name)
                insufficient_staffing[(d, s, sk)] = var_name
    model.variables.add(names=vars_to_add, lb=[0] * len(vars_to_add), ub=[10] * len(vars_to_add), types=["N"] * len(vars_to_add))

    # Creates unsatisfied preferences variables.
    # unsatisfied_preferences_n{n}_d{d}_s{s} of nurse n for day d and shift s
    vars_to_add = []
    unsatisfied_preferences = {}
    for n in all_nurses:
        for d in all_days:
            for s in all_shifts:
                var_name = f"unsatisfied_preferences_n{n}_d{d}_s{s}"
                vars_to_add.append(var_name)
                unsatisfied_preferences[(n, d, s)] = var_name
    model.variables.add(names=vars_to_add, lb=[0] * len(vars_to_add), ub=[1] * len(vars_to_add), types=["B"] * len(vars_to_add))

    # Vars for each nurse how many days they worked
    vars_to_add = []
    total_working_days = {}
    for n in all_nurses:
        var_name = f"total_working_days_n{n}"
        total_working_days[(n)] = var_name
        vars_to_add.append(var_name)
    model.variables.add(names=vars_to_add, lb=[0] * len(vars_to_add), ub=[num_days + 1] * len(vars_to_add), types=["N"] * len(vars_to_add))

    for n in all_nurses:
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair([total_working_days[(n)]] + working_days[n][:], [-1] + [1] * len(working_days[n][:]))],
            senses=["E"],
            rhs=[0])

    # Vars for each nurse n indicationg if they were working on weekend this week
    vars_to_add = []
    working_weekends = {}
    for n in all_nurses:
        var_name = f"working_weekends_n{n}"
        working_weekends[(n)] = var_name
        vars_to_add.append(var_name)
    model.variables.add(names=vars_to_add, lb=[0] * len(vars_to_add), ub=[1] * len(vars_to_add), types=["B"] * len(vars_to_add))

    # Vars for each nurse n indicationg if how many weekends were they working up to this week over the limit
    vars_to_add = []
    total_working_weekends_over_limit = {}
    for n in all_nurses:
        var_name = f"total_working_weekends_over_limit_n{n}"
        total_working_weekends_over_limit[(n)] = var_name
        vars_to_add.append(var_name)
    model.variables.add(names=vars_to_add, lb=[0] * len(vars_to_add), ub=[4] * len(vars_to_add), types=["N"] * len(vars_to_add))

    vars_to_add = []
    incomplete_weekends = {}
    for n in all_nurses:
        var_name = f"incomplete_weekends_n{n}"
        vars_to_add.append(var_name)
        incomplete_weekends[(n)] = var_name
    model.variables.add(names=vars_to_add, lb=[0] * len(vars_to_add), ub=[1] * len(vars_to_add), types=["B"] * len(vars_to_add))

    vars_to_add = []
    total_working_days_over_limit = {}
    total_working_days_under_limit = {}
    for n in all_nurses:
        var_name1, var_name2 = f"total_working_days_over_limit_n{n}", f"total_working_days_under_limit_n{n}"
        total_working_days_over_limit[(n)] = var_name1
        total_working_days_under_limit[(n)] = var_name2
        vars_to_add.append(var_name1)
        vars_to_add.append(var_name2)
    model.variables.add(names=vars_to_add, lb=[0] * len(vars_to_add), ub=[7] * len(vars_to_add), types=["N"] * len(vars_to_add))

    vars_to_add = []
    violations_of_max_consecutive_working_days = {}
    # total_violations_of_max_consecutive_working_days_in_week = {}
    # for n in all_nurses:
    #     var_name = f"total_violations_of_max_consecutive_working_days_in_week_n{n}"
    #     vars_to_add.append(var_name)
    #     total_violations_of_max_consecutive_working_days_in_week[(n)] = var_name
    # model.variables.add(names=vars_to_add, lb=[0] * len(vars_to_add), ub=[7] * len(vars_to_add), types=["N"] * len(vars_to_add))

    vars_to_add = []
    for n in all_nurses:
        for d in all_days:
            var_name = f"violations_of_max_consecutive_working_days_n{n}_d{d}"
            vars_to_add.append(var_name)
            violations_of_max_consecutive_working_days[(n, d)] = var_name
    model.variables.add(names=vars_to_add, lb=[0] * len(vars_to_add), ub=[1] * len(vars_to_add), types=["B"] * len(vars_to_add))
    
    #     violations_of_max_consecutive_working_days_from_prev_week = {}
    #     prev_week_consecutive_working_days = history_data["nurseHistory"][n]["numberOfConsecutiveWorkingDays"]
    #     if prev_week_consecutive_working_days > 0:
    #         for d in range(max_consecutive_working_days):
    #             if prev_week_consecutive_working_days - d == 0:
    #                 break

    #             violations_of_max_consecutive_working_days_from_prev_week[(n, d)] = model.NewBoolVar(f"violations_of_max_consecutive_working_days_from_prev_week{n}_d{d}")
    #             all_violation_for_nurse.append(violations_of_max_consecutive_working_days_from_prev_week[(n, d)])
                
    #             working_days_to_sum = []
    #             for dd in range(max_consecutive_working_days - d):
    #                 working_days_to_sum.append(working_days[(n, d)])
    #             model.Add(violations_of_max_consecutive_working_days_from_prev_week[(n, d)] >= (sum(working_days_to_sum) + d + 1 - max_consecutive_working_days))

    #     model.Add(violations_of_max_consecutive_working_days_for_nurse[(n)] == sum(all_violation_for_nurse))
    
    # violations_of_max_consecutive_working_shifts = {}
    # violations_of_max_consecutive_working_shifts_for_nurse_for_shift_type = {}
    # for n in all_nurses:
    #     for s in all_shifts:
    #         violations_of_max_consecutive_working_shifts_for_nurse_for_shift_type[(n, s)] = model.NewIntVar(0, num_days, f"violations_of_max_consecutive_working_shifts_for_nurse_for_shift_type_n{n}_s{s}")
    #         all_violation_for_nurse_for_shift_type = []
    #         max_consecutive_working_shifts = sc_data["shiftTypes"][s]["maximumNumberOfConsecutiveAssignments"]
    #         for d in all_days:
    #             if d + max_consecutive_working_shifts >= num_days:
    #                 break
    #             violations_of_max_consecutive_working_shifts[(n, s, d + max_consecutive_working_shifts)] = model.NewBoolVar(f"violations_of_max_consecutive_working_shifts_n{n}_s{s}_d{d + max_consecutive_working_shifts}")
    #             all_violation_for_nurse_for_shift_type.append(violations_of_max_consecutive_working_shifts[(n, s, d + max_consecutive_working_shifts)])
    #             working_shifts_to_sum = []
    #             for dd in range(max_consecutive_working_shifts + 1):
    #                 working_shifts_to_sum.append(shifts[(n, d + dd, s)])
    #             model.Add(violations_of_max_consecutive_working_shifts[(n, s, d + max_consecutive_working_shifts)] >= (sum(working_shifts_to_sum) - max_consecutive_working_shifts))
    #         model.Add(violations_of_max_consecutive_working_shifts_for_nurse_for_shift_type[(n, s)] == sum(all_violation_for_nurse_for_shift_type))

    # violations_of_max_consecutive_days_off = {}
    # violations_of_max_consecutive_days_off_for_nurse = {}
    # for n in all_nurses:
    #     violations_of_max_consecutive_days_off_for_nurse[(n)] = model.NewIntVar(0, num_days, f"violations_of_max_consecutive_days_off_for_nurse{n}")
    #     all_violation_for_nurse = []
    #     max_consecutive_days_off = sc_data["contracts"][contract_to_int[sc_data["nurses"][n]["contract"]]]["maximumNumberOfConsecutiveDaysOff"]
    #     for d in all_days:
    #         if d + max_consecutive_days_off >= num_days:
    #             break
    #         violations_of_max_consecutive_days_off[(n, d + max_consecutive_days_off)] = model.NewBoolVar(f"violations_of_max_consecutive_days_off_n{n}_d{d + max_consecutive_working_days}")
    #         all_violation_for_nurse.append(violations_of_max_consecutive_days_off[(n, d + max_consecutive_days_off)])
    #         days_off_to_sum = []
    #         for dd in range(max_consecutive_days_off + 1):
    #             for s in all_shifts:
    #                 days_off_to_sum.append(shifts[(n, d + dd, s)])
    #         model.Add(violations_of_max_consecutive_days_off[(n, d + max_consecutive_days_off)] >= 1 - (sum(days_off_to_sum)))
    #     model.Add(violations_of_max_consecutive_days_off_for_nurse[(n)] == sum(all_violation_for_nurse))

    soft_ILP_vars = {}
    soft_ILP_vars["insufficient_staffing"] = insufficient_staffing
    soft_ILP_vars["unsatisfied_preferences"] = unsatisfied_preferences
    soft_ILP_vars["total_working_days"] = total_working_days
    soft_ILP_vars["working_weekends"] = working_weekends
    soft_ILP_vars["total_working_weekends_over_limit"] = total_working_weekends_over_limit
    soft_ILP_vars["total_working_days_over_limit"] = total_working_days_over_limit
    soft_ILP_vars["total_working_days_under_limit"] = total_working_days_under_limit
    soft_ILP_vars["incomplete_weekends"] = incomplete_weekends
    soft_ILP_vars["violations_of_max_consecutive_working_days"] = violations_of_max_consecutive_working_days
    # soft_ILP_vars["violations_of_max_consecutive_days_off_for_nurse"] = violations_of_max_consecutive_days_off_for_nurse
    # soft_ILP_vars["violations_of_max_consecutive_working_shifts_for_nurse_for_shift_type"] = violations_of_max_consecutive_working_shifts_for_nurse_for_shift_type

    return soft_ILP_vars

def add_shift_skill_req_optimal(model, req, basic_ILP_vars, soft_ILP_vars, constants):
    all_nurses = constants["all_nurses"]
    shifts_with_skills = basic_ILP_vars["shifts_with_skills"]
    insufficient_staffing = soft_ILP_vars["insufficient_staffing"]

    s = shift_to_int[req["shiftType"]]
    sk = skill_to_int[req["skill"]]
    optimal_capacities_in_week = [
        req["requirementOnMonday"]["optimal"],
        req["requirementOnTuesday"]["optimal"],
        req["requirementOnWednesday"]["optimal"],
        req["requirementOnThursday"]["optimal"],
        req["requirementOnFriday"]["optimal"],
        req["requirementOnSaturday"]["optimal"],
        req["requirementOnSunday"]["optimal"],
    ]

    for d, opt_capacity in enumerate(optimal_capacities_in_week):
        skills_worked = []
        for n in all_nurses:
            skills_worked.append(shifts_with_skills[n][d][s][sk])
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair([insufficient_staffing[(d, s, sk)]] + skills_worked, [1] + [1] * len(skills_worked))],
            senses=["G"],
            rhs=[opt_capacity])
    
def add_insatisfied_preferences_reqs(model, wd_data, basic_ILP_vars, soft_ILP_vars, constants):
    unsatisfied_preferences = soft_ILP_vars["unsatisfied_preferences"]
    shifts = basic_ILP_vars["shifts"]
    working_days = basic_ILP_vars["working_days"]

    for preference in wd_data["shiftOffRequests"]:
        nurse_id = int(preference["nurse"].split("_")[1])
        day_id = day_to_int[preference["day"]]
        shift_id = shift_to_int[preference["shiftType"]]

        if shift_id != shift_to_int["Any"]:
            # model.Add(unsatisfied_preferences[(nurse_id, day_id, shift_id)] == shifts[(nurse_id, day_id, shift_id)])
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair([unsatisfied_preferences[(nurse_id, day_id, shift_id)], shifts[nurse_id][day_id][shift_id]], [1, -1])],
                senses=["E"],
                rhs=[0])
        # else:
            # model.Add(unsatisfied_preferences[(nurse_id, day_id)] == working_days[(nurse_id, day_id)])
            # model.linear_constraints.add(
            #     lin_expr=[cplex.SparsePair([unsatisfied_preferences[(nurse_id, day_id, shift_id)]] + [working_days[nurse_id][day_id]], [1, -1])],
            #     senses=["E"],
            #     rhs=[0])

def add_total_working_weekends_soft_constraints(model, basic_ILP_vars, soft_ILP_vars, constants, week_number):
    sc_data = constants["sc_data"]
    h0_data = constants["h0_data"]
    total_working_weekends_over_limit = soft_ILP_vars["total_working_weekends_over_limit"]
    working_weekends = soft_ILP_vars["working_weekends"]

    all_nurses = constants["all_nurses"]
    num_weeks = constants["num_weeks"]
    working_days = basic_ILP_vars["working_days"]

    for n in all_nurses:
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair([working_weekends[(n)], working_days[n][5]], [1, -1])],
            senses=["G"],
            rhs=[0])
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair([working_weekends[(n)], working_days[n][6]], [1, -1])],
            senses=["G"],
            rhs=[0])
    
    for n in all_nurses:
        # worked_weekends_limit_for_this_week = sc_data["contracts"][contract_to_int[sc_data["nurses"][n]["contract"]]]["maximumNumberOfWorkingWeekends"]
        worked_weekends_limit_for_this_week = sc_data["contracts"][contract_to_int[sc_data["nurses"][n]["contract"]]]["maximumNumberOfWorkingWeekends"] * ((week_number + 1) / num_weeks)
        worked_weekends_in_previous_weeks = h0_data["nurseHistory"][n]["numberOfWorkingWeekends"]
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair([total_working_weekends_over_limit[(n)], working_weekends[(n)]], [-1, 1])],
            senses=["L"],
            rhs=[worked_weekends_limit_for_this_week - worked_weekends_in_previous_weeks])
        # worked_weekends.append(working_weekends[(n)])
        # model.Add(total_working_weekends_over_limit[(n)] >= sum(worked_weekends) - worked_weekends_limit + worked_weekends_in_previous_weeks)
        # model.Add(total_working_weekends_over_limit[(n)] >= -(sum(worked_weekends) - worked_weekends_limit + worked_weekends_in_previous_weeks))

def add_incomplete_weekends_constraint(model, basic_ILP_vars, soft_ILP_vars, constants):
    nurses_data = constants["sc_data"]["nurses"]
    contracts_data = constants["sc_data"]["contracts"]
    incomplete_weekends = soft_ILP_vars["incomplete_weekends"]
    working_weekends = soft_ILP_vars["working_weekends"]
    working_days = basic_ILP_vars["working_days"]
    all_nurses = constants["all_nurses"]

    for n in all_nurses:
        isCompleteWeekendRequested = contracts_data[contract_to_int[nurses_data[n]["contract"]]]["completeWeekends"]
        if isCompleteWeekendRequested == 1:
            # incomplete_weekends[(n)] = model.NewBoolVar(f"incomplete_weekends_n{n}")
            # model.Add(incomplete_weekends[(n)] == 2*working_weekends[(n)] - working_days[(n, 5)] - working_days[(n, 6)])
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair([incomplete_weekends[(n)], working_weekends[(n)], working_days[n][5], working_days[n][6]], [-1, 2, -1, -1])],
                senses=["E"],
                rhs=[0])
            
def add_total_working_days_out_of_bounds_constraint(model, basic_ILP_vars, soft_ILP_vars, constants, week_number):
    nurses_data = constants["sc_data"]["nurses"]
    contracts_data = constants["sc_data"]["contracts"]
    total_working_days = soft_ILP_vars["total_working_days"]
    total_working_days_over_limit = soft_ILP_vars["total_working_days_over_limit"]
    total_working_days_under_limit = soft_ILP_vars["total_working_days_under_limit"]
    all_nurses = constants["all_nurses"]
    num_weeks = constants["num_weeks"]
    h0_data = constants["h0_data"]
    
    for n in all_nurses:
        worked_days_in_previous_weeks = h0_data["nurseHistory"][n]["numberOfAssignments"]

        upper_limit = math.ceil(contracts_data[contract_to_int[nurses_data[n]["contract"]]]["maximumNumberOfAssignments"] * ((week_number + 1) / num_weeks))
        lower_limit = math.ceil(contracts_data[contract_to_int[nurses_data[n]["contract"]]]["minimumNumberOfAssignments"] * ((week_number + 1) / num_weeks))
        # upper_limit = math.ceil(contracts_data[contract_to_int[nurses_data[n]["contract"]]]["maximumNumberOfAssignments"] * 1)
        # lower_limit = math.ceil(contracts_data[contract_to_int[nurses_data[n]["contract"]]]["minimumNumberOfAssignments"] * 1)
        model.linear_constraints.add(
                lin_expr=[cplex.SparsePair([total_working_days_over_limit[(n)], total_working_days[(n)]], [-1, 1])],
                senses=["L"],
                rhs=[upper_limit - worked_days_in_previous_weeks])
        model.linear_constraints.add(
                lin_expr=[cplex.SparsePair([total_working_days_under_limit[(n)], total_working_days[(n)]], [1, 1])],
                senses=["G"],
                rhs=[lower_limit - worked_days_in_previous_weeks])
    return 

def add_max_consecutive_work_days_constraint(model, basic_ILP_vars, soft_ILP_vars, constants):
    violations_of_max_consecutive_working_days = soft_ILP_vars["violations_of_max_consecutive_working_days"]
    all_nurses = constants["all_nurses"]
    all_days = constants["all_days"]
    num_days = constants["num_days"]
    sc_data = constants["sc_data"]
    working_days = basic_ILP_vars["working_days"]

    for n in all_nurses:
        max_consecutive_working_days = sc_data["contracts"][contract_to_int[sc_data["nurses"][n]["contract"]]]["maximumNumberOfConsecutiveWorkingDays"]
        for d in all_days:
            if d > max_consecutive_working_days:
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([violations_of_max_consecutive_working_days[(n, d)]] + working_days[n][d - max_consecutive_working_days: d + 1], [-1] + [1] * (max_consecutive_working_days + 1))],
                    senses=["L"],
                    rhs=[max_consecutive_working_days])
            # working_days_to_sum = []
            # for dd in range(max_consecutive_working_days + 1):
            #     working_days_to_sum.append(working_days[n][d - dd]])
            # model.Add(violations_of_max_consecutive_working_days[(n, d + max_consecutive_working_days)] >= (sum(working_days_to_sum) - max_consecutive_working_days))
    

def add_soft_constraints(model, basic_ILP_vars, soft_ILP_vars, constants, week_number):
    wd_data = constants["wd_data"]

    for req in wd_data["requirements"]:
        add_shift_skill_req_optimal(model, req, basic_ILP_vars, soft_ILP_vars, constants)
    
    add_insatisfied_preferences_reqs(model, wd_data, basic_ILP_vars, soft_ILP_vars, constants)

    add_total_working_weekends_soft_constraints(model, basic_ILP_vars, soft_ILP_vars, constants, week_number)
    
    add_total_working_days_out_of_bounds_constraint(model, basic_ILP_vars, soft_ILP_vars, constants, week_number)
    
    add_incomplete_weekends_constraint(model, basic_ILP_vars, soft_ILP_vars, constants)

    add_max_consecutive_work_days_constraint(model, basic_ILP_vars, soft_ILP_vars, constants)
    
    return 

def save_tmp_results(results, solver, constants, basic_ILP_vars, soft_ILP_vars, week_number):
    num_days = constants["num_days"]
    num_nurses = constants["num_nurses"]
    num_skills = constants["num_skills"]
    num_shifts = constants["num_shifts"]
    history_data = constants["h0_data"]
    working_weekends = soft_ILP_vars["working_weekends"]
    incomplete_weekends = soft_ILP_vars["incomplete_weekends"]

    shifts_with_skills = basic_ILP_vars["shifts_with_skills"]
    working_days = basic_ILP_vars["working_days"]
    shifts = basic_ILP_vars["shifts"]

    for n in range(num_nurses):
        for d in range(num_days):
            for s in range(num_shifts):
                for sk in range(num_skills):
                    results[(n, d + 7 * week_number, s, sk)] = solver.get_values(shifts_with_skills[n][d][s][sk])
                    # if(results[(n, d + 7 * week_number, s, sk)]) == 1:
            history_data["nurseHistory"][n]["numberOfAssignments"] += solver.get_values(working_days[n][d])
        history_data["nurseHistory"][n]["numberOfWorkingWeekends"] += solver.get_values(working_weekends[(n)])

        if solver.get_values(working_days[n][6]) == 0:
            consecutive_free_days = 1
            d = 5
            while d >= 0 and solver.get_values(working_days[n][d]) == 0:
                consecutive_free_days += 1
                d -= 1
            history_data["nurseHistory"][n]["numberOfConsecutiveDaysOff"] = consecutive_free_days
            history_data["nurseHistory"][n]["numberOfConsecutiveWorkingDays"] = 0
            history_data["nurseHistory"][n]["numberOfConsecutiveAssignments"] = 0
            history_data["nurseHistory"][n]["lastAssignedShiftType"] = "None"
        else:
            consecutive_work_days = 1
            d = 5
            while d >= 0 and solver.get_values(working_days[n][d]) == 1:
                consecutive_work_days += 1
                d -= 1
            history_data["nurseHistory"][n]["numberOfConsecutiveWorkingDays"] = consecutive_work_days
            history_data["nurseHistory"][n]["numberOfConsecutiveDaysOff"] = 0

            consecutive_shift = 0
            for s in range(num_shifts):
                if solver.get_values(shifts[n][6][s]) == 1:
                    consecutive_shift = s
                    break
            consecutive_shifts = 1
            for shift_name, shift_id in shift_to_int.items():
                if( shift_id == consecutive_shift):
                    history_data["nurseHistory"][n]["lastAssignedShiftType"] = shift_name
                    
            d = 5
            while d >= 0 and solver.get_values(shifts[n][d][consecutive_shift]) == 1:
                consecutive_shifts += 1
                d -= 1
            history_data["nurseHistory"][n]["numberOfConsecutiveAssignments"] = consecutive_shifts
                    
def set_objective_function(c, constants, basic_ILP_vars, soft_ILP_vars):
    all_nurses = constants["all_nurses"]
    all_shifts = constants["all_shifts"]
    all_skills = constants["all_skills"]
    all_days = constants["all_days"]

    num_nurses = constants["num_nurses"] 
    num_shifts = constants["num_shifts"]
    num_skills = constants["num_skills"]
    num_days = constants["num_days"]

    shifts = basic_ILP_vars["shifts"]
    insufficient_staffing = soft_ILP_vars["insufficient_staffing"]
    unsatisfied_preferences = soft_ILP_vars["unsatisfied_preferences"]
    total_working_weekends_over_limit = soft_ILP_vars["total_working_weekends_over_limit"]
    incomplete_weekends = soft_ILP_vars["incomplete_weekends"]
    total_working_days_over_limit = soft_ILP_vars["total_working_days_over_limit"]
    total_working_days_under_limit = soft_ILP_vars["total_working_days_under_limit"]
    violations_of_max_consecutive_working_days = soft_ILP_vars["violations_of_max_consecutive_working_days"]

    c.objective.set_linear(list(itertools.chain.from_iterable([
        zip((insufficient_staffing[(d, s, sk)] for d in all_days for s in all_shifts for sk in all_skills), 
            [30] * num_days * num_shifts * num_skills) 
        ,
        zip((unsatisfied_preferences[(n, d, s)] for n in all_nurses for d in all_days for s in all_shifts),
            [10] * num_nurses * num_days * num_shifts) 
        ,
        zip((total_working_weekends_over_limit[(n)] for n in all_nurses),
            [30] * num_nurses) 
        ,
        zip((incomplete_weekends[(n)] for n in all_nurses),
            [30] * num_nurses) 
        ,
        zip((total_working_days_over_limit[(n)] for n in all_nurses),
            [20] * num_nurses) 
        ,
        zip((total_working_days_under_limit[(n)] for n in all_nurses),
            [20] * num_nurses) 
        ,
        zip((violations_of_max_consecutive_working_days[(n, d)] for n in all_nurses for d in all_days),
            [30] * num_nurses) 
    ])))

    return

def setup_problem(c, constants, week_number):
    c.objective.set_sense(c.objective.sense.minimize)

    # Create ILP variables.
    basic_ILP_vars = init_ilp_vars(c, constants)

    # Add hard constrains to model
    add_hard_constrains(c, basic_ILP_vars, constants)

    # soft_ILP_vars = {}
    soft_ILP_vars = init_ilp_vars_for_soft_constraints(c, basic_ILP_vars, constants)
    
    add_soft_constraints(c, basic_ILP_vars, soft_ILP_vars, constants, week_number)

    set_objective_function(c, constants, basic_ILP_vars, soft_ILP_vars)

    return basic_ILP_vars, soft_ILP_vars

def compute_one_week(time_limit_for_week, week_number, constants, results):
    c = cplex.Cplex()
    c.parameters.timelimit.set(time_limit_for_week)
    c.parameters.mip.tolerances.absmipgap.set(0.0)
    c.parameters.emphasis.mip.set(
        c.parameters.emphasis.mip.values.optimality)

    basic_ILP_vars, soft_ILP_vars = setup_problem(c, constants, week_number)


    c.solve()
    sol = c.solution

    save_tmp_results(results, sol, constants, basic_ILP_vars, soft_ILP_vars, week_number)
#!/usr/bin/python

import sys
import json

import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker 
import math

from google.nsp_contest import compute_one_week as compute_one_week_or_tools
from ibm.nsp_cplex import compute_one_week as compute_one_week_cplex

def load_data(number_nurses: int, number_weeks: int, history_data_file_id: int, week_data_files_ids: list):
    """
    Loads and prepairs data for computation.
    Returns a dictionary named 'constants' containing loaded data.
    """

    file_name = "data\hidden-JSON\H0-n0" + str(number_nurses) +"w" + str(number_weeks) + "-" + str(history_data_file_id) + ".json"
    f0 = open(file_name)
    h0_data = json.load(f0)
    f0.close()

    file_name = "data\hidden-JSON\Sc-n0" + str(number_nurses) +"w" + str(number_weeks) + ".json"
    f1 = open(file_name)
    sc_data = json.load(f1)
    f1.close()

    wd_data = []
    for week in range(number_weeks):
        file_name = "data\hidden-JSON\WD-n0" + str(number_nurses) +"w" + str(number_weeks) + "-" + str(week_data_files_ids[week]) + ".json"
        f2 = open(file_name)
        wd_data.append(json.load(f2))
        f2.close()

    # initialize constants
    num_nurses = len(sc_data["nurses"])
    num_shifts = len(sc_data["shiftTypes"])
    num_skills = len(sc_data["skills"])
    num_days = 7
    all_nurses = range(num_nurses)
    all_shifts = range(num_shifts)
    all_days = range(num_days)
    all_skills = range(num_skills)
    all_weeks = range(number_weeks)

    constants = {}
    constants["h0_data"] = h0_data
    constants["sc_data"] = sc_data
    constants["wd_data"] = wd_data[0]
    constants["all_wd_data"] = wd_data
    constants["num_nurses"] = num_nurses
    constants["num_shifts"] = num_shifts
    constants["num_skills"] = num_skills
    constants["num_days"] = num_days
    constants["num_weeks"] = number_weeks
    constants["all_nurses"] = all_nurses
    constants["all_shifts"] = all_shifts
    constants["all_days"] = all_days
    constants["all_skills"] = all_skills
    constants["all_weeks"] = all_weeks

    return constants


def display_schedule(results, constants, number_weeks):
    """
    Displays computed schedule as table in a figure.
    """

    num_days = constants["num_days"] * number_weeks
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

def main(time_limit_for_week, mode, number_nurses: int, number_weeks: int, history_data_file_id: int, week_data_files_ids: list):
    # Loading Data and init constants
    constants = load_data(number_nurses, number_weeks, history_data_file_id, week_data_files_ids)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    if(mode == 0):
        print(f"CPLEX for {number_weeks} weeks ({' '.join(map(str, week_data_files_ids))}) and for {number_nurses} nurses")
    else:
        print(f"OR TOOLS for {number_weeks} weeks ({' '.join(map(str, week_data_files_ids))}) and for {number_nurses} nurses")

    display = True
    if(time_limit_for_week == 0):
        display = False
        time_limit_for_week = 10 + 10 * (constants["num_nurses"] - 20)
        # time_limit_for_week = 10

    # accumulate results over weeks
    results = {}
    for week_number in range(number_weeks):
        constants["wd_data"] = constants["all_wd_data"][week_number]
        if(mode == 0):
            compute_one_week_cplex(time_limit_for_week, week_number, constants, results)
        else:
            compute_one_week_or_tools(time_limit_for_week, week_number, constants, results)

    # display results
    if(display):
        display_schedule(results, constants, number_weeks)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("----------------------------------------------------------------")
    total_value = results[("allweeksoft")]
    for week_number in range(number_weeks):
        print(f"status:          {results[(week_number, 'status')]}")
        print(f"objective value: {results[(week_number, 'value')]}")
        total_value += results[(week_number, "value")]
        print("----------------------------------------------------------------")
    print(f"value total: {total_value}")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

if __name__ == "__main__":
    time_limit_for_week = int(sys.argv[1])
    mode = int(sys.argv[2])
    number_nurses = int(sys.argv[3])
    number_weeks = int(sys.argv[4])
    history_data_file_id = int(sys.argv[5])
    week_data_files_ids = list(map(int, (sys.argv[6:])))
    main(time_limit_for_week, mode, number_nurses, number_weeks, history_data_file_id, week_data_files_ids)
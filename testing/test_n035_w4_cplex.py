#!/usr/bin/python

import sys
import subprocess

from contextlib import contextmanager

original_stdout = sys.stdout

@contextmanager
def redirect_stdout_to_file(file_path):
    original_stdout = sys.stdout
    with open(file_path, 'w') as file:
        sys.stdout = file
        yield
    sys.stdout = original_stdout


try:
    output_file = 'outputs\\output_test_n035_w4_cplex.txt'
    number_of_iteration = 1

    # list of input for benchmark
    with redirect_stdout_to_file(output_file):
        arguments_list = [
            '0 0 35 4 0 1 7 1 8', 
            '0 0 35 4 0 4 2 1 6', 
            '0 0 35 4 0 5 9 5 6',
            '0 0 35 4 0 9 8 7 7',
            '0 0 35 4 1 0 6 9 2',
            '0 0 35 4 2 8 6 7 1',
            '0 0 35 4 2 8 8 7 5',
            '0 0 35 4 2 9 2 2 6',
            '0 0 35 4 2 9 9 2 1'
        ]

        # run the main script in iterations
        for arg in arguments_list:
            for _ in range(number_of_iteration):
                subprocess.run(['python', 'main.py'] + arg.split(' '), stdout=sys.stdout)

    # filter unwanted lines from the output file
    with open(output_file, 'r') as file:
        lines = file.readlines()

    filtered_lines = [line for line in lines if "PARAM" not in line]
    filtered_lines = [line for line in filtered_lines if "Version" not in line]

    with open(output_file, 'w') as file:
        file.writelines(filtered_lines)

except Exception as e:
    sys.stdout = original_stdout
    print(f"An error occurred: {e}")


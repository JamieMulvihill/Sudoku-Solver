# Soduku Solver
import numpy as np
import copy
import time
from sudoku_search import depth_first_search_with_forward_checking
import sudoku_utils
from PartialSudokuState import PartialSudokuState
from collections import deque

def sudoku_solver(sudoku):
    """
    Solves a Sudoku puzzle and returns its unique solution.

    Input
        sudoku : 9x9 numpy array
            Empty cells are designated by 0.

    Output
        9x9 numpy array of integers
            It contains the solution, if there is one. If there is no solution, all array entries should be -1.
    """

    time_limit = 10
    start_time = time.time()
    partial_state = PartialSudokuState(sudoku)

    if not partial_state.ac3():
        return np.full((9, 9), -1, dtype=int)

    while partial_state.apply_naked_pairs():
        if not partial_state.ac3():
            return np.full((9, 9), -1, dtype=int)

    #if not partial_state.is_valid():
        #return np.full((9, 9), -1, dtype=int)

    if not partial_state.quick_validity_check():
        return np.full((9, 9), -1, dtype=int)

    #solution = sudoku_search.depth_first_search(partial_state, start_time, time_limit)
    solution = depth_first_search_with_forward_checking(partial_state, start_time, time_limit)

    if solution is None:
        return np.full((9, 9), -1, dtype=int)
    else:
        return solution.board

# Main execution
sudoku = np.load(r"C:/Users/darre/Desktop/Masters/Foundations/SodukuSolver/data/hard_puzzle.npy")
solutions = np.load(r"C:/Users/darre/Desktop/Masters/Foundations/SodukuSolver/data/hard_solution.npy")

num_success = 0
start_time = time.process_time()

for num in range(len(sudoku)):
    result = sudoku_solver(sudoku[num])
    
    if np.array_equal(result, solutions[num]):
        num_success += 1
    
    print(f"Puzzle {num + 1}:")
    print(f"Puzzle:")
    print(sudoku[num])
    print("Result:")
    print(result)
    print("Solution:")
    print(solutions[num])
    print()

end_time = time.process_time()
print(f"Solved {num_success} out of {len(sudoku)} puzzles.")
print(f"Total time: {end_time - start_time:.4f} seconds")
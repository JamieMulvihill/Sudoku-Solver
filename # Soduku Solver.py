# Soduku Solver
import numpy as np
import copy
import time
from PartialSudokuState import PartialSudokuState
import sudoku_utils
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

    #solution = depth_first_search(partial_state, start_time, time_limit)
    solution = depth_first_search_with_forward_checking(partial_state, start_time, time_limit)

    if solution is None:
        return np.full((9, 9), -1, dtype=int)
    else:
        return solution.board

def pick_next_empty_cell(partial_state):
    """
    Pick the next empty cell to assign a value using MCV heuristic.
    """
    empty_cells = [(row, col) for row in range(9) for col in range(9) if partial_state.board[row, col] == 0]
    return min(empty_cells, key=lambda cell: len(partial_state.get_possible_values(*cell))) if empty_cells else (None, None)

def depth_first_search(partial_state, start_time, time_limit):
    if time.time() - start_time > time_limit:
        return None  # Time limit exceeded

    if np.all(partial_state.board != 0):
        return partial_state

    row, col = sudoku_utils.pick_unassigned_variable(partial_state)
    
    for value in sorted(partial_state.domains[(row, col)]):
        new_state = copy.deepcopy(partial_state)
        new_state.board[row, col] = value
        new_state.domains[(row, col)] = {value}
        
        if new_state.ac3():  # Run AC-3 after each assignment
            result = depth_first_search(new_state, start_time, time_limit)
            if result is not None:
                return result

    return None

def depth_first_search_with_forward_checking(partial_state, start_time, time_limit):
    if time.time() - start_time > time_limit:
        return None  # Time limit exceeded

    if np.all(partial_state.board != 0):
        return partial_state

    row, col = sudoku_utils.pick_unassigned_variable(partial_state)
    
    for value in sorted(partial_state.domains[(row, col)]):
        new_state = copy.deepcopy(partial_state)
        new_state.board[row, col] = value
        new_state.domains[(row, col)] = {value}
        
        if sudoku_utils.forward_check(new_state, row, col, value) and new_state.ac3():
            result = depth_first_search_with_forward_checking(new_state, start_time, time_limit)
            if result is not None:
                return result

    return None

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
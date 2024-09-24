import time
import copy
import sudoku_utils
import numpy as np


def depth_first_search_with_forward_checking(partial_state, start_time, time_limit):
    if time.time() - start_time > time_limit:
        return None  # Time limit exceeded

    if np.all(partial_state.board != 0):
        return partial_state  # Board is complete

    row, col = sudoku_utils.pick_next_empty_cell_smallest_domain(partial_state)
    
    # Try each value in the domain of the selected cell
    for value in sorted(partial_state.domains[(row, col)]):
        new_state = copy.deepcopy(partial_state) # Create a deep copy of the current state to avoid modifying it directly
        new_state.board[row, col] = value # Assign the value to the cell
        new_state.domains[(row, col)] = {value} # Update the domain for this cell
        
        # Perform forward checking and run AC-3 to prune domains
        if sudoku_utils.forward_check(new_state, row, col, value) and new_state.ac3():
            result = depth_first_search_with_forward_checking(new_state, start_time, time_limit) # Recursively apply DFS with the new state
            if result is not None:
                return result

    return None  # No solution found

def depth_first_search(partial_state, start_time, time_limit):
    if time.time() - start_time > time_limit:
        return None  # Time limit exceeded

    if np.all(partial_state.board != 0):
        return partial_state # Board is complete

    row, col = sudoku_utils.pick_unassigned_variable(partial_state)
    
    for value in sorted(partial_state.domains[(row, col)]):
        new_state = copy.deepcopy(partial_state)
        new_state.board[row, col] = value
        new_state.domains[(row, col)] = {value}
        
        if new_state.ac3():  # Run AC-3 after each assignment
            result = depth_first_search(new_state, start_time, time_limit)
            if result is not None:
                return result

    return None # No solution found
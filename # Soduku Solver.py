# Soduku Solver

import numpy as np
import copy
import random
import time

# Load sudokus
sudoku = np.load(r"C:/Users/darre/Desktop/Masters/Foundations/SodukuSolver/data/hard_puzzle.npy")
print("very_easy_puzzle.npy has been loaded into the variable sudoku")
print(f"sudoku.shape: {sudoku.shape}, sudoku[0].shape: {sudoku[0].shape}, sudoku.dtype: {sudoku.dtype}")

# Load solutions for demonstration
solutions = np.load(r"C:/Users/darre/Desktop/Masters/Foundations/SodukuSolver/data/hard_puzzle.npy")
print()

# Print the first 9x9 sudoku...
print("First sudoku:")
print(sudoku[0], "\n")

# ...and its solution
print("Solution of first sudoku:")
print(solutions[0])

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
    ### YOUR CODE HERE
    
    #sudoku_copy = copy.deepcopy(sudoku)
    solved_sudoku = []

    #print(sudoku_copy.shape)
    return solved_sudoku

def pick_next_empty_cell(partial_state):
    """
    Pick the next empty cell to assign a value. This can be based on 
    the number of possible values remaining (Most Constrained Variable heuristic).
    """
    empty_cells = [(row, col) for row in range(9) for col in range(9) if partial_state.board[row, col] == 0]
    # Can implement other heuristics to pick the best cell, here it's random.
    print(empty_cells)
    return random.choice(empty_cells)

def order_values(partial_state, row, col):
    """
    Get possible values for a particular cell in the order we should try them.
    """
    values = partial_state.get_possible_values(row, col)
    random.shuffle(values)
    print(values)
    return values

def depth_first_search(partial_state):
    """
    Perform a depth-first search on partial Sudoku states, trying each possible value
    for the next empty cell. If the state is valid, continue searching. If it's a goal state,
    return the solution.
    """
    # Check if the puzzle is already solved
    if partial_state.is_goal():
        return partial_state

    # Pick the next empty cell to fill
    row, col = pick_next_empty_cell(partial_state)
    values = order_values(partial_state, row, col)
    print(values)

    for value in values:
        # Try assigning this value to the cell
        new_state = partial_state.set_value(row, col, value)
        
        # Check if this state is valid and continue searching
        if not new_state.is_invalid():
            print("is valid")
            result = depth_first_search(new_state)
            if result is not None and result.is_goal():
                return result
        else:
            print("is not valid")

    # No solution found
    return None

class SudokuState():
    def __init__(self, configuration):
        self.configuration = np.copy(configuration)
        self.domains = self._init_domains()

    def _init_domains(self):

        domains = {}
        for i in range(9):
            for j in range(9):
                if self.configuration[i,j] == 0:
                    domains[(i,j)] = set(range(1,10))
                else:
                    domains[(i, j)] = {self.board[i, j]}
        return domains 
    
    def get_possible_values(self, row, col):
        return list(self.domains[(row, col)])
    
    def set_value(self, row, col, value):
        self.configuration[row, col] = value
        self._update_domains(row, col, value)

    def _update_domains(self, row, col, value):

        # remove from row and col
        for i in range(9):
            self.domains[(row, i)].discard(value)
            self.domains[(i, col)].discard(value)

        # remove from 3x3 grid
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                self.domains[(i, j)].discard(value)

    def is_valid(self, row, col, value):
        if value in self.configuration[row, :]:
            return False
        if value in self.configuration[:, col]:
            return False
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        if value in self.board[start_row:start_row+3, start_col:start_col+3]:
            return False 
        
         # Check the row for conflicts, excluding the current cell
    #if value in self.board[row, :] and self.board[row, col] != value:
        #return False
    
    # Check the column for conflicts, excluding the current cell
    #if value in self.board[:, col] and self.board[row, col] != value:
        #return False
    
    # Check the 3x3 subgrid, excluding the current cell
    #start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    #subgrid = self.board[start_row:start_row + 3, start_col:start_col + 3]
    
    #if value in subgrid and self.board[row, col] != value:
        #return False

    def is_solved(self):
        return np.all(self.configuration != 0)
    
    def copy(self):
        return SudokuState(self.configuration)
    
class PartialSudokuState:
    def __init__(self, board):
        self.board = np.copy(board)
        self.domains = self._init_domains()
    
    def _init_domains(self):
        """Initialize possible values (domains) for each empty cell."""
        #domains = {}
        #for r in range(9):
            #for c in range(9):
                #if self.board[r, c] == 0:
                    #domains[(r, c)] = set(range(1, 10))
                #else:
                    #domains[(r, c)] = {self.board[r, c]}
        
        #print(domains)
        #return domains

        domains = {}
        for r in range(9):
            for c in range(9):
                cell_value = self.board[r, c]
                # Convert np.int8 to Python int
                if isinstance(cell_value, np.int8):
                    cell_value = int(cell_value)
                    if cell_value == 0:
                        domains[(r, c)] = set(range(1, 10))
                    else:
                        domains[(r, c)] = {cell_value}
        
        #print(domains)
        return domains

    def get_possible_values(self, row, col):
        """Return possible values for a specific cell."""
        return list(self.domains[(row, col)])

    def set_value(self, row, col, value):
        """Set a value to a cell and return a new state."""
        new_state = PartialSudokuState(self.board)
        new_state.board[row, col] = value
        new_state._update_domains(row, col, value)
        return new_state

    def _update_domains(self, row, col, value):
        """Update the domains after assigning a value to a cell."""
        for r in range(9):
            self.domains[(r, col)].discard(value)
        for c in range(9):
            self.domains[(row, c)].discard(value)
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                self.domains[(r, c)].discard(value)

    def is_goal(self):
        """Check if the puzzle is fully solved (no empty cells)."""
        return np.all(self.board != 0)

    def is_invalid(self):
        """Check if the current state violates Sudoku constraints."""
        # Check rows for duplicates
        for r in range(9):
            row = self.board[r, :]
            if len(set(row[row != 0])) != len(row[row != 0]):
                return True

        # Check columns for duplicates
        for c in range(9):
            col = self.board[:, c]
            if len(set(col[col != 0])) != len(col[col != 0]):
                return True

        # Check 3x3 boxes for duplicates
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box = self.board[box_row:box_row+3, box_col:box_col+3].flatten()
                if len(set(box[box != 0])) != len(box[box != 0]):
                    return True
                
        return False

num_success = 0
start_time = time.process_time()
for num in range(3, len(sudoku)):
    partial_state = PartialSudokuState(sudoku[num])
    solution = depth_first_search(partial_state)

    if solution is not None and solution.is_goal():
        print("Solved Sudoku:")
        print(solution.board)
        num_success += 1
    else:
        print("No solution found")



#print(sudoku[0])
#sudoku_solver(sudoku[0])
end_time = time.process_time()
print("This sudoku took {} seconds to solve.\n".format(end_time-start_time))
print(num_success)
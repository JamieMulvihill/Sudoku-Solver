# Soduku Solver
import numpy as np
import copy
import random
import time

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
    partial_state = PartialSudokuState(sudoku)

    if not partial_state.is_valid():
        return np.full((9, 9), -1, dtype=int)

    solution = depth_first_search(partial_state)

    if solution is None:
        return np.full((9, 9), -1, dtype=int)
    else:
        return solution.board

def pick_next_empty_cell(partial_state):
    """
    Pick the next empty cell to assign a value. This can be based on 
    the number of possible values remaining (Most Constrained Variable heuristic).
    """
    empty_cells = [(row, col) for row in range(9) for col in range(9) if partial_state.board[row, col] == 0]
    if not empty_cells:
        return None, None
    return min(empty_cells, key=lambda cell: len(partial_state.get_possible_values(*cell)))

def order_values(partial_state, row, col):
    """
    Get possible values for a particular cell in the order we should try them.
    """
    def count_conflicts(value):
        conflicts = 0
        for r in range(9):
            if r != row and value in partial_state.domains[(r, col)]:
                conflicts += 1
        for c in range(9):
            if c != col and value in partial_state.domains[(row, c)]:
                conflicts += 1
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if (r, c) != (row, col) and value in partial_state.domains[(r, c)]:
                    conflicts += 1
        return conflicts
    
    values = partial_state.get_possible_values(row, col)
    return sorted(values, key=count_conflicts)

def depth_first_search(partial_state):
    """
    Perform a depth-first search on partial Sudoku states, trying each possible value
    for the next empty cell. If the state is valid, continue searching. If it's a goal state,
    return the solution.
    """
    if partial_state.is_goal():
        return partial_state

    row, col = pick_next_empty_cell(partial_state)
    if row is None and col is None:
        return partial_state if not partial_state.is_invalid() else None
    
    for value in order_values(partial_state, row, col):
        new_state = partial_state.set_value(row, col, value)
        if forward_checking(new_state, row, col, value):
            result = depth_first_search(new_state)
            if result is not None and result.is_goal():
                return result

    return None

def forward_checking(state, row, col, value):
    """
    Perform forward checking after assigning a value to a cell.
    Returns False if this leads to an invalid state, True otherwise.
    """
    for c in range(9):
        if c != col and state.board[row, c] == value:
            return False
        if c != col and state.board[row, c] == 0:
            if value in state.domains[(row, c)]:
                state.domains[(row, c)].remove(value)
                if len(state.domains[(row, c)]) == 0:
                    return False

    for r in range(9):
        if r != row and state.board[r, col] == value:
            return False
        if r != row and state.board[r, col] == 0:
            if value in state.domains[(r, col)]:
                state.domains[(r, col)].remove(value)
                if len(state.domains[(r, col)]) == 0:
                    return False

    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if (r, c) != (row, col) and state.board[r, c] == value:
                return False
            if (r, c) != (row, col) and state.board[r, c] == 0:
                if value in state.domains[(r, c)]:
                    state.domains[(r, c)].remove(value)
                    if len(state.domains[(r, c)]) == 0:
                        return False

    return True

class PartialSudokuState:
    def __init__(self, board):
        self.board = np.copy(board)
        self.domains = self._init_domains()
    
    def _init_domains(self):
        domains = {}
        for r in range(9):
            for c in range(9):
                cell_value = self.board[r, c]
                if isinstance(cell_value, np.int8):
                    cell_value = int(cell_value)
                if cell_value == 0:
                    domains[(r, c)] = set(range(1, 10))
                else:
                    domains[(r, c)] = {cell_value}
        return domains
    
    def is_valid(self):
        return self._no_duplicates_in_rows() and \
               self._no_duplicates_in_columns() and \
               self._no_duplicates_in_boxes()
    
    def _no_duplicates_in_rows(self):
        for row in self.board:
            if self._has_duplicates(row):
                return False
        return True
    
    def _no_duplicates_in_columns(self):
        for col in self.board.T:
            if self._has_duplicates(col):
                return False
        return True
    
    def _no_duplicates_in_boxes(self):
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                box = self.board[i:i+3, j:j+3].flatten()
                if self._has_duplicates(box):
                    return False
        return True
    
    def _has_duplicates(self, arr):
        seen = set()
        for num in arr:
            if num != 0:
                if num in seen:
                    return True
                seen.add(num)
        return False

    def get_possible_values(self, row, col):
        return list(self.domains[(row, col)])

    def set_value(self, row, col, value):
        new_state = PartialSudokuState(self.board)
        new_state.board[row, col] = value
        new_state._update_domains(row, col, value)
        return new_state

    def _update_domains(self, row, col, value):
        for r in range(9):
            if r != row:
                self.domains[(r, col)].discard(value)
        for c in range(9):
            if c != col:
                self.domains[(row, c)].discard(value)
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if (r, c) != (row, col):
                    self.domains[(r, c)].discard(value)

    def is_goal(self):
        return np.all(self.board != 0)

    def is_invalid(self):
        for r in range(9):
            row = self.board[r, :]
            if len(set(row[row != 0])) != len(row[row != 0]):
                return True

        for c in range(9):
            col = self.board[:, c]
            if len(set(col[col != 0])) != len(col[col != 0]):
                return True

        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box = self.board[box_row:box_row+3, box_col:box_col+3].flatten()
                if len(set(box[box != 0])) != len(box[box != 0]):
                    return True
                
        return False

# Main execution
sudoku = np.load(r"C:/Users/darre/Desktop/Masters/Foundations/SodukuSolver/data/medium_puzzle.npy")
solutions = np.load(r"C:/Users/darre/Desktop/Masters/Foundations/SodukuSolver/data/medium_solution.npy")

num_success = 0
start_time = time.process_time()

for num in range(len(sudoku)):
    result = sudoku_solver(sudoku[num])
    
    if np.array_equal(result, solutions[num]):
        num_success += 1
    
    print(f"Puzzle {num + 1}:")
    print("Result:")
    print(result)
    print("Solution:")
    print(solutions[num])
    print()

end_time = time.process_time()
print(f"Solved {num_success} out of {len(sudoku)} puzzles.")
print(f"Total time: {end_time - start_time:.2f} seconds")
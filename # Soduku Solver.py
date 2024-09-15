# Soduku Solver
import numpy as np
import copy
import random
import time
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

    #if not partial_state.is_valid():
        #return np.full((9, 9), -1, dtype=int)

    if not partial_state.quick_validity_check():
        return np.full((9, 9), -1, dtype=int)

    solution = depth_first_search(partial_state, start_time, time_limit)

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
    """
    Perform a depth-first search on partial Sudoku states.
    """
    if time.time() - start_time > time_limit:
        return None  # Time limit exceeded

    if partial_state.is_goal():
        return partial_state

    row, col = pick_next_empty_cell(partial_state)
    if row is None and col is None:
        return partial_state if not partial_state.is_invalid() else None
    
    for value in partial_state.get_possible_values(row, col):
        new_state = partial_state.set_value(row, col, value)
        if new_state.is_valid():
            result = depth_first_search(new_state, start_time, time_limit)
            if result is not None and result.is_goal():
                return result

    return None

class PartialSudokuState:
    def __init__(self, board):
        self.board = np.copy(board)
        self.domains = self._init_domains()
    
    def _init_domains(self):
        domains = {}
        for r in range(9):
            for c in range(9):
                if self.board[r, c] == 0:
                    domains[(r, c)] = set(range(1, 10)) - self._get_used_values(r, c)
                else:
                    domains[(r, c)] = {self.board[r, c]}
        return domains
    
    def _get_used_values(self, row, col):
        used = set(self.board[row]) | set(self.board[:, col])
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        used |= set(self.board[box_row:box_row+3, box_col:box_col+3].flatten())
        return used - {0}

    def quick_validity_check(self):
        # Check if any cell has an empty domain
        return all(len(domain) > 0 for domain in self.domains.values())

    def is_valid(self):
        for i in range(9):
            if len(set(self.board[i]) - {0}) != len(self.board[i][self.board[i] != 0]):
                return False
            if len(set(self.board[:, i]) - {0}) != len(self.board[:, i][self.board[:, i] != 0]):
                return False
        
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                box = self.board[i:i+3, j:j+3].flatten()
                if len(set(box) - {0}) != len(box[box != 0]):
                    return False
        
        return True

    def get_possible_values(self, row, col):
        return list(self.domains[(row, col)])

    def set_value(self, row, col, value):
        new_state = PartialSudokuState(self.board)
        new_state.board[row, col] = value
        new_state._update_domains(row, col, value)
        return new_state

    def _update_domains(self, row, col, value):
        self.domains[(row, col)] = {value}
        for r in range(9):
            if r != row and value in self.domains[(r, col)]:
                self.domains[(r, col)].remove(value)
        for c in range(9):
            if c != col and value in self.domains[(row, c)]:
                self.domains[(row, c)].remove(value)
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if (r, c) != (row, col) and value in self.domains[(r, c)]:
                    self.domains[(r, c)].remove(value)

    def is_goal(self):
        return np.all(self.board != 0)

    def is_invalid(self):
        return not self.is_valid()
    
    def ac3(self):
        queue = deque([(r, c) for r in range(9) for c in range(9)])
        while queue:
            (row, col) = queue.popleft()
            if self.revise(row, col):
                if len(self.domains[(row, col)]) == 0:
                    return False
                neighbors = self.get_neighbors(row, col)
                for neighbor in neighbors:
                    if neighbor != (row, col):
                        queue.append(neighbor)
        return True

    def revise(self, row, col):
        revised = False
        for value in list(self.domains[(row, col)]):
            if not self.has_consistent_assignment(row, col, value):
                self.domains[(row, col)].remove(value)
                revised = True
        return revised

    def has_consistent_assignment(self, row, col, value):
        for r in range(9):
            if r != row and len(self.domains[(r, col)]) == 1 and value in self.domains[(r, col)]:
                return False
        for c in range(9):
            if c != col and len(self.domains[(row, c)]) == 1 and value in self.domains[(row, c)]:
                return False
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if (r, c) != (row, col) and len(self.domains[(r, c)]) == 1 and value in self.domains[(r, c)]:
                    return False
        return True

    def get_neighbors(self, row, col):
        neighbors = []
        for r in range(9):
            if r != row:
                neighbors.append((r, col))
        for c in range(9):
            if c != col:
                neighbors.append((row, c))
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if (r, c) != (row, col):
                    neighbors.append((r, c))
        return neighbors

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
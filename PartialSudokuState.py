import numpy as np
from collections import deque

class PartialSudokuState:
    def __init__(self, board):
        self.board = np.copy(board) # Create a copy of the board to avoid modifying the original
        self.domains = self._init_domains()  # Initialize the domains for each cell
    
    def _init_domains(self):
        # Initialize the domains for each cell in the Sudoku board.
        domains = {}
        for r in range(9):
            for c in range(9):
                if self.board[r, c] == 0:
                    domains[(r, c)] = set(range(1, 10)) - self._get_used_values(r, c)   # Exclude used values
                else:
                    domains[(r, c)] = {self.board[r, c]} # Fixed values have a domain of one
        return domains
    
    def _get_used_values(self, row, col):
        # Get values that are already used in the eaach row, column, and 3x3 box.
        used = set(self.board[row]) | set(self.board[:, col])
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        used |= set(self.board[box_row:box_row+3, box_col:box_col+3].flatten())
        return used - {0}

    def is_valid_check(self):
        # Check if any cell has an empty domain
        return all(len(domain) > 0 for domain in self.domains.values())

    def get_possible_values(self, row, col):
        # get the list of possible values for the given cell based in its domain.
        return list(self.domains[(row, col)])

    def set_value(self, row, col, value):
        # Set a value for a cell and update the domains and retun the new Partial Sudoku State with the change
        new_state = PartialSudokuState(self.board) # Create a new state
        new_state.board[row, col] = value # Assign the value
        new_state._update_domains(row, col, value) # Update domains after assignment
        return new_state

    def _update_domains(self, row, col, value):
        self.domains[(row, col)] = {value} # Set the domain of the assigned cell to just the assigned value
        for r in range(9):
            if r != row and value in self.domains[(r, col)]:
                self.domains[(r, col)].remove(value) # Remove value from other cells in the same column
        for c in range(9):
            if c != col and value in self.domains[(row, c)]:
                self.domains[(row, c)].remove(value) # Remove value from other cells in the same row
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if (r, c) != (row, col) and value in self.domains[(r, c)]:
                    self.domains[(r, c)].remove(value)  # Remove value from other cells in the same box

    def is_goal(self):
        #Check if the current state is a goal state
        return np.all(self.board != 0)

    def is_invalid(self):
        # Check if the current state is invalid
        return not self.is_valid()
    
    def ac3(self):
        # AC-3 algorithm to reduce the domains of the variables, the idea is to ensure domains are consistant with each other, rteturns false if any domain bceomes empty
        queue = deque([(r, c) for r in range(9) for c in range(9)]) # Initialize the queue with all cells
        while queue:
            (row, col) = queue.popleft()  # Get the next cell to process
            if self.revise(row, col): # Attempt to revise the domain
                if len(self.domains[(row, col)]) == 0:  # Check for empty domain
                    return False
                neighbours = self.get_neighbours(row, col) # Get neighbours of the cell
                for neighbour in neighbours:
                    if neighbour != (row, col):
                        queue.append(neighbour) # Add neighbors to the queue for processing
        return True

    def revise(self, row, col):
        # Revise the domain based on its neighbours, remove values if they dont lead to consistant assignment, return true if have removed
        revised = False
        for value in list(self.domains[(row, col)]):
            if not self.has_consistent_assignment(row, col, value):
                self.domains[(row, col)].remove(value) # Remove inconsistent
                revised = True
        return revised

    def has_consistent_assignment(self, row, col, value):
        # Check if assigning is consistant with Neighbours
        for r in range(9):
            if r != row and len(self.domains[(r, col)]) == 1 and value in self.domains[(r, col)]:
                return False # Conflict in the same column
        for c in range(9):
            if c != col and len(self.domains[(row, c)]) == 1 and value in self.domains[(row, c)]:
                return False  # Conflict in the same row
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if (r, c) != (row, col) and len(self.domains[(r, c)]) == 1 and value in self.domains[(r, c)]:
                    return False  # Conflict in the same box
        return True

    def get_neighbours(self, row, col):
        # Getter for the neighbours of a cell
        neighbours = []
        for r in range(9):
            if r != row:
                neighbours.append((r, col)) # Add cells in same column
        for c in range(9):
            if c != col:
                neighbours.append((row, c)) # Add cells in same row
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if (r, c) != (row, col):
                    neighbours.append((r, c)) # Add cells in same box
        return neighbours
    
    def apply_naked_pairs(self):
        # Nake Pairs technique to reduce domains, eliminates possible values for other cells in the same column/row/box("Unit"), returns true if changes were made
        changed = False
        for unit in self.get_all_units(): # Iterate through all units
            pairs = self.find_naked_pairs(unit) # Identify naked pairs in the unit
            for pair, cells in pairs.items():
                if len(cells) == 2:
                    for cell in unit:
                        if cell not in cells: # Skip the cells that are part of the pair
                            removed = self.domains[cell] & set(pair)
                            if removed:
                                self.domains[cell] -= set(pair)
                                changed = True
        return changed

    def find_naked_pairs(self, unit):
        # Identify naked pairs in a unit, returns a dictionary of pairs and their associated cells.
        pairs = {}
        for cell in unit:
            if len(self.domains[cell]) == 2: # Only consider cells with 2 possible values
                pair = tuple(sorted(self.domains[cell]))
                if pair in pairs:
                    pairs[pair].append(cell) # Add cell to existing pair
                else:
                    pairs[pair] = [cell] # Create a new entry for the pair
        return pairs

    def get_all_units(self):
        # Gather all units (rows, columns, and boxes) into a single list
        units = []
        # Rows
        units.extend([[(r, c) for c in range(9)] for r in range(9)])
        # Columns
        units.extend([[(r, c) for r in range(9)] for c in range(9)])
        # Boxes
        units.extend([[(r, c) for r in range(box_r, box_r + 3) for c in range(box_c, box_c + 3)] 
                      for box_r in range(0, 9, 3) for box_c in range(0, 9, 3)])  
        return units
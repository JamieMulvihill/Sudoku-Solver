import numpy as np
from collections import deque

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
    
    def apply_naked_pairs(self):
        changed = False
        for unit in self.get_all_units():
            pairs = self.find_naked_pairs(unit)
            for pair, cells in pairs.items():
                if len(cells) == 2:
                    for cell in unit:
                        if cell not in cells:
                            removed = self.domains[cell] & set(pair)
                            if removed:
                                self.domains[cell] -= set(pair)
                                changed = True
        return changed

    def find_naked_pairs(self, unit):
        pairs = {}
        for cell in unit:
            if len(self.domains[cell]) == 2:
                pair = tuple(sorted(self.domains[cell]))
                if pair in pairs:
                    pairs[pair].append(cell)
                else:
                    pairs[pair] = [cell]
        return pairs

    def get_all_units(self):
        units = []
        # Rows
        units.extend([[(r, c) for c in range(9)] for r in range(9)])
        # Columns
        units.extend([[(r, c) for r in range(9)] for c in range(9)])
        # Boxes
        units.extend([[(r, c) for r in range(box_r, box_r + 3) for c in range(box_c, box_c + 3)] 
                      for box_r in range(0, 9, 3) for box_c in range(0, 9, 3)])  
        return units
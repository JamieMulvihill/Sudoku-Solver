def pick_next_empty_cell_fewest_values(partial_state):
    # Pick the next empty cell to assign a value using the Minimum Remaining Values (MRV) heuristic.
    # The MRV heuristic aims to choose the cell with the fewest legal possible values.
    empty_cells = [(row, col) for row in range(9) for col in range(9) if partial_state.board[row, col] == 0]
    return min(empty_cells, key=lambda cell: len(partial_state.get_possible_values(*cell))) if empty_cells else (None, None)

def pick_next_empty_cell_smallest_domain(partial_state):
    # Pick the next empty cell to assign a value using the Minimum Remaining Values (MRV) heuristic.
    # The MRV heuristic aims to choose the the next empty cell with the smallest domain size.
    return min(
        ((r, c) for r in range(9) for c in range(9) if partial_state.board[r, c] == 0),
        key=lambda cell: len(partial_state.domains[cell])
    )

def forward_check(partial_state, row, col, value):
    # Remove the assigned value from the domains of affected cells
    for r in range(9):
        if r != row and value in partial_state.domains[(r, col)]:
            if len(partial_state.domains[(r, col)]) == 1:
                return False # Domain would become empty
            partial_state.domains[(r, col)].remove(value)
    
    for c in range(9):
        if c != col and value in partial_state.domains[(row, c)]:
            if len(partial_state.domains[(row, c)]) == 1:
                return False # Domain would become empty
            partial_state.domains[(row, c)].remove(value)
    
    # Check the 3x3 box containing the cell
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if (r, c) != (row, col) and value in partial_state.domains[(r, c)]:
                if len(partial_state.domains[(r, c)]) == 1:
                    return False
                partial_state.domains[(r, c)].remove(value)
    
    return True  # Forward check succeeded

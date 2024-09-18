def pick_next_empty_cell(partial_state):
    """
    Pick the next empty cell to assign a value using MCV heuristic.
    """
    empty_cells = [(row, col) for row in range(9) for col in range(9) if partial_state.board[row, col] == 0]
    return min(empty_cells, key=lambda cell: len(partial_state.get_possible_values(*cell))) if empty_cells else (None, None)

def pick_unassigned_variable(state):
        return min(
        ((r, c) for r in range(9) for c in range(9) if state.board[r, c] == 0),
        key=lambda cell: len(state.domains[cell])
    )

def forward_check(state, row, col, value):
    for r in range(9):
        if r != row and value in state.domains[(r, col)]:
            if len(state.domains[(r, col)]) == 1:
                return False
            state.domains[(r, col)].remove(value)
    
    for c in range(9):
        if c != col and value in state.domains[(row, c)]:
            if len(state.domains[(row, c)]) == 1:
                return False
            state.domains[(row, c)].remove(value)
    
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if (r, c) != (row, col) and value in state.domains[(r, c)]:
                if len(state.domains[(r, c)]) == 1:
                    return False
                state.domains[(r, c)].remove(value)
    
    return True

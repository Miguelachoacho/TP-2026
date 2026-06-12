from simpleai.search import CspProblem, backtrack

def is_border(position, rows, cols):
    row, col = position
    return row == 0 or row == rows - 1 or col == 0 or col == cols - 1


def adjacent(first, second):
    return abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1


def neighbors(position, rows, cols):
    row, col = position
    result = []
    for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        next_row = row + delta_row
        next_col = col + delta_col
        if 0 <= next_row < rows and 0 <= next_col < cols:
            result.append((next_row, next_col))
    return result


def build_camp(camp_size, habs, generators, labs, deposits, airlocks, craters):
    rows, cols = camp_size
    crater_set = set(craters)
    all_cells = sorted(
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if (row, col) not in crater_set
    )
    border_cells = [cell for cell in all_cells if is_border(cell, rows, cols)]
    interior_cells = [cell for cell in all_cells if not is_border(cell, rows, cols)]
    hab_cells = [
        cell
        for cell in interior_cells
        if any(neighbor not in crater_set for neighbor in neighbors(cell, rows, cols))
    ]
    lab_cells = [
        cell
        for cell in all_cells
        if any(neighbor not in crater_set for neighbor in neighbors(cell, rows, cols))
    ]
    hab_exits = {
        cell: [neighbor for neighbor in neighbors(cell, rows, cols) if neighbor not in crater_set]
        for cell in hab_cells
    }
    lab_supports = {
        cell: [neighbor for neighbor in neighbors(cell, rows, cols) if neighbor not in crater_set]
        for cell in lab_cells
    }

    total_modules = habs + generators + labs + deposits + airlocks
    if total_modules == 0:
        return []
    if total_modules > len(all_cells):
        return None
    if airlocks > len(border_cells):
        return None
    if habs > len(hab_cells):
        return None
    if labs > 0 and deposits == 0:
        return None
    if labs > 0 and not lab_cells:
        return None

    occupied = set()
    reserved_exits = set()
    required_deposits = set()

    hab_positions = []
    lab_positions = []
    air_positions = []
    generator_positions = []
    extra_deposit_positions = []

    def blocked_cells():
        return occupied | reserved_exits | required_deposits

    def enough_free_cells(remaining_modules):
        return len(all_cells) - len(blocked_cells()) >= remaining_modules

    def place_labs(start_index):
        if len(lab_positions) == labs:
            return place_habs(0)

        remaining_labs = labs - len(lab_positions)
        remaining_after = remaining_labs - 1 + habs + airlocks + generators

        for index in range(start_index, len(lab_cells)):
            lab_cell = lab_cells[index]
            if lab_cell in blocked_cells():
                continue

            support_cells = [
                cell
                for cell in lab_supports[lab_cell]
                if cell not in occupied and cell not in reserved_exits
            ]
            if not support_cells:
                continue

            occupied.add(lab_cell)
            lab_positions.append(lab_cell)

            for support_cell in support_cells:
                added_support = support_cell not in required_deposits
                if added_support:
                    required_deposits.add(support_cell)

                extra_deposits_remaining = max(0, deposits - len(required_deposits))
                if len(required_deposits) <= deposits and enough_free_cells(remaining_after + extra_deposits_remaining):
                    result = place_labs(index + 1)
                    if result is not None:
                        return result

                if added_support:
                    required_deposits.remove(support_cell)

            lab_positions.pop()
            occupied.remove(lab_cell)

        return None

    def place_habs(start_index):
        if len(hab_positions) == habs:
            return place_airlocks(0)

        remaining_habs = habs - len(hab_positions)
        remaining_after = remaining_habs - 1 + airlocks + generators

        for index in range(start_index, len(hab_cells)):
            hab_cell = hab_cells[index]
            if hab_cell in blocked_cells():
                continue

            exit_cells = [
                cell
                for cell in hab_exits[hab_cell]
                if cell not in occupied and cell not in required_deposits
            ]
            if not exit_cells:
                continue

            occupied.add(hab_cell)
            hab_positions.append(hab_cell)

            for exit_cell in exit_cells:
                added_exit = exit_cell not in reserved_exits
                if added_exit:
                    reserved_exits.add(exit_cell)

                extra_deposits_remaining = max(0, deposits - len(required_deposits))
                if enough_free_cells(remaining_after + extra_deposits_remaining):
                    result = place_habs(index + 1)
                    if result is not None:
                        return result

                if added_exit:
                    reserved_exits.remove(exit_cell)

            hab_positions.pop()
            occupied.remove(hab_cell)

        return None

    def place_airlocks(start_index):
        if len(air_positions) == airlocks:
            return place_generators(0)

        remaining_after = airlocks - len(air_positions) - 1 + generators

        for index in range(start_index, len(border_cells)):
            air_cell = border_cells[index]
            if air_cell in blocked_cells():
                continue

            occupied.add(air_cell)
            air_positions.append(air_cell)

            extra_deposits_remaining = max(0, deposits - len(required_deposits))
            if enough_free_cells(remaining_after + extra_deposits_remaining):
                result = place_airlocks(index + 1)
                if result is not None:
                    return result

            air_positions.pop()
            occupied.remove(air_cell)

        return None

    def place_generators(start_index):
        if len(generator_positions) == generators:
            return place_extra_deposits(0)

        for index in range(start_index, len(all_cells)):
            generator_cell = all_cells[index]
            if generator_cell in blocked_cells():
                continue
            if any(adjacent(generator_cell, hab_cell) for hab_cell in hab_positions):
                continue
            if any(adjacent(generator_cell, other_generator) for other_generator in generator_positions):
                continue

            occupied.add(generator_cell)
            generator_positions.append(generator_cell)

            result = place_generators(index + 1)
            if result is not None:
                return result

            generator_positions.pop()
            occupied.remove(generator_cell)

        return None

    def place_extra_deposits(start_index):
        extra_needed = deposits - len(required_deposits)
        if extra_needed < 0:
            return None
        if len(extra_deposit_positions) == extra_needed:
            result = []
            for row, col in hab_positions:
                result.append(("hab", row, col))
            for row, col in generator_positions:
                result.append(("gen", row, col))
            for row, col in lab_positions:
                result.append(("lab", row, col))
            for row, col in sorted(required_deposits):
                result.append(("dep", row, col))
            for row, col in extra_deposit_positions:
                result.append(("dep", row, col))
            for row, col in air_positions:
                result.append(("air", row, col))
            return result

        available_cells = [
            cell
            for cell in all_cells[start_index:]
            if cell not in blocked_cells() and cell not in extra_deposit_positions
        ]
        if len(available_cells) < extra_needed - len(extra_deposit_positions):
            return None

        for index in range(start_index, len(all_cells)):
            deposit_cell = all_cells[index]
            if deposit_cell in blocked_cells() or deposit_cell in extra_deposit_positions:
                continue

            extra_deposit_positions.append(deposit_cell)
            result = place_extra_deposits(index + 1)
            if result is not None:
                return result
            extra_deposit_positions.pop()

        return None

    return place_labs(0)


# if __name__ == "__main__":
#     resultado = build_camp(
#         camp_size=(5, 6),
#         habs=2,
#         generators=1,
#         labs=1,
#         deposits=2,
#         airlocks=1,
#         craters=[(2, 2), (2, 3)],
#     )

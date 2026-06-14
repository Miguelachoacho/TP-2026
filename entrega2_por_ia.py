from itertools import combinations

from simpleai.search import (
    CspProblem, backtrack, MOST_CONSTRAINED_VARIABLE, LEAST_CONSTRAINING_VALUE
)


def build_camp(camp_size, habs, generators, labs, deposits, airlocks, craters):
    rows, cols = camp_size
    crater_set = set(craters)

    def is_border(pos):
        r, c = pos
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    def get_neighbors(pos):
        r, c = pos
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                result.append((nr, nc))
        return result

    def are_adjacent(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1

    all_cells = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in crater_set
    ]
    border_cells = [p for p in all_cells if is_border(p)]
    interior_cells = [p for p in all_cells if not is_border(p)]

    total = habs + generators + labs + deposits + airlocks
    if total == 0:
        return []

    # Early impossibility checks
    if len(all_cells) < total:
        return None
    if airlocks > len(border_cells):
        return None
    if habs > 0 and not interior_cells:
        return None
    if labs > 0 and deposits == 0:
        return None

    # Hab domain: interior cells that have at least one non-crater neighbor
    # (ensures the evacuation constraint can potentially be satisfied)
    hab_domain = [
        p for p in interior_cells
        if any(n not in crater_set for n in get_neighbors(p))
    ]
    if habs > len(hab_domain):
        return None

    # --- Build variables and domains ---
    module_vars = []
    domains = {}

    for i in range(habs):
        v = f"hab_{i}"
        module_vars.append(v)
        domains[v] = hab_domain[:]

    for i in range(generators):
        v = f"gen_{i}"
        module_vars.append(v)
        domains[v] = all_cells[:]

    for i in range(labs):
        v = f"lab_{i}"
        module_vars.append(v)
        domains[v] = all_cells[:]

    for i in range(deposits):
        v = f"dep_{i}"
        module_vars.append(v)
        domains[v] = all_cells[:]

    for i in range(airlocks):
        v = f"air_{i}"
        module_vars.append(v)
        domains[v] = border_cells[:]

    # Auxiliary escape variables (one per hab).
    # esc_i represents the free cell that serves as hab_i's evacuation route.
    # Using auxiliary binary variables avoids a large n-ary global constraint.
    esc_vars = []
    for i in range(habs):
        v = f"esc_{i}"
        esc_vars.append(v)
        domains[v] = all_cells[:]

    all_vars = module_vars + esc_vars

    hab_vars = [f"hab_{i}" for i in range(habs)]
    gen_vars = [f"gen_{i}" for i in range(generators)]
    lab_vars = [f"lab_{i}" for i in range(labs)]
    dep_vars = [f"dep_{i}" for i in range(deposits)]

    constraints = []

    # Constraint 1: No two modules in the same cell (all pairs of module vars)
    for v1, v2 in combinations(module_vars, 2):
        constraints.append(([v1, v2], lambda _vars, vals: vals[0] != vals[1]))

    # Constraint 5: No generator adjacent to a hab
    for gv in gen_vars:
        for hv in hab_vars:
            constraints.append(
                ([gv, hv], lambda _vars, vals: not are_adjacent(vals[0], vals[1]))
            )

    # Constraint 6: No two generators adjacent to each other
    for gv1, gv2 in combinations(gen_vars, 2):
        constraints.append(
            ([gv1, gv2], lambda _vars, vals: not are_adjacent(vals[0], vals[1]))
        )

    # Constraint 7: Each lab must be adjacent to at least one deposit (n-ary)
    for lv in lab_vars:
        cvars = [lv] + dep_vars
        constraints.append(
            (cvars, lambda _vars, vals: any(are_adjacent(vals[0], d) for d in vals[1:]))
        )

    # Constraint 8: Hab evacuation via auxiliary escape variable (all binary)
    for i in range(habs):
        hv = f"hab_{i}"
        ev = f"esc_{i}"
        # The escape cell must be adjacent to its hab
        constraints.append(
            ([hv, ev], lambda _vars, vals: are_adjacent(vals[0], vals[1]))
        )
        # The escape cell must not coincide with any module (must stay free)
        for mv in module_vars:
            constraints.append(
                ([ev, mv], lambda _vars, vals: vals[0] != vals[1])
            )

    # Symmetry breaking: order same-type variables to cut the search space
    for i in range(habs - 1):
        constraints.append(
            ([f"hab_{i}", f"hab_{i+1}"], lambda _vars, vals: vals[0] < vals[1])
        )
    for i in range(generators - 1):
        constraints.append(
            ([f"gen_{i}", f"gen_{i+1}"], lambda _vars, vals: vals[0] < vals[1])
        )
    for i in range(labs - 1):
        constraints.append(
            ([f"lab_{i}", f"lab_{i+1}"], lambda _vars, vals: vals[0] < vals[1])
        )
    for i in range(deposits - 1):
        constraints.append(
            ([f"dep_{i}", f"dep_{i+1}"], lambda _vars, vals: vals[0] < vals[1])
        )
    for i in range(airlocks - 1):
        constraints.append(
            ([f"air_{i}", f"air_{i+1}"], lambda _vars, vals: vals[0] < vals[1])
        )

    problem = CspProblem(all_vars, domains, constraints)
    solution = backtrack(
        problem,
        variable_heuristic=MOST_CONSTRAINED_VARIABLE,
        value_heuristic=LEAST_CONSTRAINING_VALUE,
    )

    if solution is None:
        return None

    result = []
    for v in module_vars:
        tipo = v.split("_")[0]
        r, c = solution[v]
        result.append((tipo, r, c))

    return result

from itertools import combinations
from simpleai.search import CspProblem, backtrack


def build_camp(camp_size, habs, generators, labs, deposits, airlocks, craters):
    fil, col = camp_size
    craters_set = set(craters)

    # Seteo cuanto de cada construccion tengo
    hab_vars = [f"hab_{i}" for i in range(habs)]
    gen_vars = [f"gen_{i}" for i in range(generators)]
    lab_vars = [f"lab_{i}" for i in range(labs)]
    dep_vars = [f"dep_{i}" for i in range(deposits)]
    air_vars = [f"air_{i}" for i in range(airlocks)]
    variables = hab_vars + gen_vars + lab_vars + dep_vars + air_vars

    # dominios en parte
    dominiocompleto = [ 
        (f, c)
        for f in range(fil)
        for c in range(col)
        if (f, c) not in craters_set
    ]
    borde = [
        (f, c) for (f, c) in dominiocompleto
        if f == 0 or f == fil - 1 or c == 0 or c == col - 1  # Dominio solo para Esclusas 
    ]
    interior = [ ##Domonio solo para las habitaciones
        (f, c) for (f, c) in dominiocompleto
        if f != 0 and f != fil - 1 and c != 0 and c != col - 1
    ]

    domains = {}
    for variable in hab_vars:
        domains[variable] = interior
    for variable in air_vars:
        domains[variable] = borde
    for variable in gen_vars + lab_vars + dep_vars:
        domains[variable] = dominiocompleto

    # Restricciones
    constraints = []

    def no_superposicion(variables, values):  # llegan 2 modulos y deben estar en celdas distintas
        mod1, mod2 = values
        return mod1 != mod2

    for variable1, variable2 in combinations(variables, 2):
        constraints.append(((variable1, variable2), no_superposicion))

    def gen_no_adyacente_hab(variables, values):  # generador y habitacional no pueden ser adyacentes
        fil1, col1 = values[0]
        fil2, col2 = values[1]
        return abs(fil1 - fil2) + abs(col1 - col2) > 1

    for gen in gen_vars:
        for hab in hab_vars:
            constraints.append(((gen, hab), gen_no_adyacente_hab))

    def gen_no_adyacente_gen(variables, values):  # dos generadores no pueden ser adyacentes
        fil1, col1 = values[0]
        fil2, col2 = values[1]
        return abs(fil1 - fil2) + abs(col1 - col2) > 1

    for gen1, gen2 in combinations(gen_vars, 2):
        constraints.append(((gen1, gen2), gen_no_adyacente_gen))

    def lab_adyacente_dep(variables, values):  # cada laboratorio debe tener un deposito adyacente
        fil, col = values[0]
        deps = values[1:]
        return any(abs(fil - fd) + abs(col - cd) == 1 for fd, cd in deps)

    for lab in lab_vars:
        constraints.append((tuple([lab] + dep_vars), lab_adyacente_dep))

    def hab_con_salida(variables, values):  # cada habitacional necesita una celda adyacente libre
        filHab, colHab = values[0]
        otros = set(values[1:])
        adyacentes = [(filHab + 1, colHab), (filHab - 1, colHab), (filHab, colHab + 1), (filHab, colHab - 1)]
        return any(
            0 <= f < fil and 0 <= c < col
            and (f, c) not in craters_set
            and (f, c) not in otros
            for f, c in adyacentes
        )

    otros = gen_vars + lab_vars + dep_vars + air_vars
    for hab in hab_vars:
        constraints.append((tuple([hab] + otros), hab_con_salida))

    problem = CspProblem(variables, domains, constraints)
    solution = backtrack(
        problem,
        variable_heuristic='MOST_CONSTRAINED_VARIABLE',
        value_heuristic='LEAST_CONSTRAINING_VALUE',
    )

    if solution is None:
        return None

    resultado = []
    for variable, pos in solution.items():
        tipo = variable.split("_")[0]
        fila, columna = pos
        resultado.append((tipo, fila, columna))

    return resultado


if __name__ == "__main__":

    resultado = build_camp(
        camp_size= (5,6),
        habs=2,
        generators=1,
        labs=1,
        deposits=1,
        airlocks=2,
        craters=[(2, 2), (2, 3)]
        )

    if resultado is None:
        print("No existe distribucion valida para esta configuracion.")
    else:
        for modulo in resultado:
            print(f"  {modulo}")

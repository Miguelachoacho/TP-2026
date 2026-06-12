from simpleai.search import SearchProblem, astar


MAX_BATERIA = 20
MAX_CARGA = 2


def _quitar_una_muestra(muestras, posicion):
	restantes = list(muestras)
	restantes.remove(posicion)
	return tuple(restantes)


class ProblemaRover(SearchProblem):
	def __init__(self, estado_inicial, zonas_sombra):
		self.zonas_sombra = set(zonas_sombra)
		super().__init__(estado_inicial)

	def actions(self, state):
		posicion, bateria, taladro, carga, muestras_igneas, muestras_sedimentarias = state
		fila, columna = posicion
		acciones = []

		# Nunca se permiten acciones que dejen la bateria en 0 o menos.
		if bateria >= 2:
			acciones.extend(
				[
					("moverse", (fila - 1, columna)),
					("moverse", (fila + 1, columna)),
					("moverse", (fila, columna - 1)),
					("moverse", (fila, columna + 1)),
				]
			)

			if taladro != "termico" and muestras_igneas:
				acciones.append(("equipar", "termico"))
			if taladro != "percusion" and muestras_sedimentarias:
				acciones.append(("equipar", "percusion"))

			if carga == 2 or (carga == 1 and not muestras_igneas and not muestras_sedimentarias):
				acciones.append(("depositar", None))

		if bateria >= 5:
			acciones.extend(
				[
					("sobremarcha", (fila - 2, columna)),
					("sobremarcha", (fila + 2, columna)),
					("sobremarcha", (fila, columna - 2)),
					("sobremarcha", (fila, columna + 2)),
				]
			)

		if bateria >= 4 and carga < MAX_CARGA:
			if posicion in muestras_igneas and taladro == "termico":
				acciones.append(("recolectar", "ignea"))
			if posicion in muestras_sedimentarias and taladro == "percusion":
				acciones.append(("recolectar", "sedimentaria"))

		if posicion not in self.zonas_sombra and bateria < MAX_BATERIA:
			acciones.append(("recargar", None))

		return acciones

	def result(self, state, action):
		posicion, bateria, taladro, carga, muestras_igneas, muestras_sedimentarias = state
		tipo_accion, parametro = action

		if tipo_accion == "moverse":
			return (parametro, bateria - 1, taladro, carga, muestras_igneas, muestras_sedimentarias)

		if tipo_accion == "sobremarcha":
			return (parametro, bateria - 4, taladro, carga, muestras_igneas, muestras_sedimentarias)

		if tipo_accion == "equipar":
			return (posicion, bateria - 1, parametro, carga, muestras_igneas, muestras_sedimentarias)

		if tipo_accion == "recolectar":
			if parametro == "ignea":
				return (
					posicion,
					bateria - 3,
					taladro,
					carga + 1,
					_quitar_una_muestra(muestras_igneas, posicion),
					muestras_sedimentarias,
				)

			return (
				posicion,
				bateria - 3,
				taladro,
				carga + 1,
				muestras_igneas,
				_quitar_una_muestra(muestras_sedimentarias, posicion),
			)

		if tipo_accion == "depositar":
			return (posicion, bateria - 1, taladro, 0, muestras_igneas, muestras_sedimentarias)

		# recargar
		return (
			posicion,
			min(MAX_BATERIA, bateria + 10),
			taladro,
			carga,
			muestras_igneas,
			muestras_sedimentarias,
		)

	def is_goal(self, state):
		_, _, _, carga, muestras_igneas, muestras_sedimentarias = state
		return carga == 0 and not muestras_igneas and not muestras_sedimentarias

	def heuristic(self, state):
		# Heuristica admisible: solo suma costos inevitables.
		_, _, _, carga, muestras_igneas, muestras_sedimentarias = state
		pendientes = len(muestras_igneas) + len(muestras_sedimentarias)

		costo_recoleccion_minimo = 2 * pendientes
		costo_deposito_minimo = pendientes + carga
		return costo_recoleccion_minimo + costo_deposito_minimo

	def cost(self, state, action, state2):
		tipo_accion, _ = action

		if tipo_accion in ("moverse", "sobremarcha"):
			return 1
		if tipo_accion == "equipar":
			return 3
		if tipo_accion == "recolectar":
			return 2
		if tipo_accion == "depositar":
			return state[3]
		return 4


def planear_rover(rover_inicio, bateria_inicial, zonas_sombra, muestras_igneas, muestras_sedimentarias):
	estado_inicial = (
		rover_inicio,
		bateria_inicial,
		"ninguno",
		0,
		tuple(muestras_igneas),
		tuple(muestras_sedimentarias),
	)

	problema = ProblemaRover(estado_inicial, zonas_sombra)
	resultado = astar(problema, graph_search=True)

	if resultado is None:
		return []

	return [accion for accion, _ in resultado.path() if accion is not None]

from simpleai.search import (
    SearchProblem,
    breadth_first,
    depth_first,
    limited_depth_first,
    uniform_cost,
    iterative_limited_depth_first,
    greedy,
    astar,
)
from simpleai.search.viewers import BaseViewer, WebViewer

def planear_rover(rover_inicio, bateria_inicial, zonas_sombra, muestras_igneas, muestras_sedimentarias):
    EstadoInicial = (rover_inicio, bateria_inicial, None, 0,  tuple(muestras_igneas), tuple(muestras_sedimentarias)) ## El estado es Pos_robot, bateria, taladro Equipado, carga ,muestras

    class ProblemaRover(SearchProblem):

        def actions(self, state):
            acciones = []
            x, y = state[0]
            movimientos = [(-1, 0),  # arriba
                           (1, 0),  # abajo
                           (0, -1),  # izquierda
                           (0, 1)]  # derecha                           
            for dx, dy in movimientos:
                if state[1] > 1: ##Siempre que tenga bateria opta por moverse
                    acciones.append(("moverse", (x + dx, y + dy)))

            movimientosOverride = [(-2, 0),  # arriba 2 pos
                                   (2, 0),  # abajo 2 pos 
                                   (0, -2),  # izquierda 2 pos 
                                   (0, 2)]  # derecha 2 pos
            for dx, dy in movimientosOverride:
                if state[1] > 4:
                    acciones.append(("sobremarcha", (x + dx, y + dy)))

            if state[2] != "termico" and state[1]> 1 and state[4]: #Si no tiene el taladro equipado ,tiene la bateria suficiete y hay muestras igneas
               acciones.append(("equipar", "termico"))
            if state[2] != "percusion"and state[1]> 1 and state[5]:
                acciones.append(("equipar", "percusion"))

            if state[1]> 3 and state[0] in state[4] and state[2] == "termico" and 0 <= state[3] < 2:  #Si tengo equipado el taladro correcto y tengo lugar
                acciones.append(("recolectar", "ignea"))
            if state[1]> 3 and state[0] in state[5] and state[2] == "percusion" and 0 <= state[3] < 2:  #Si tengo equipado el taladro correcto y tengo lugar
                acciones.append(("recolectar", "sedimentaria"))

            if state[0] not in zonas_sombra and state[1] < 11 :
                acciones.append(("recargar", None))
            
            if state[1]> 1 and( state[3] == 2 or (len(state[4]) == 0 and len(state[5])==0 and state[3]>0)): #Comparo que tenga dos muestras o si no hay ninguna en el mapa y tiene carga
                acciones.append(("depositar",None))

            return acciones

        def result(self, state, action):
            acc,obj = action
            posRobot = list(state[0]) # Paso a lista para poder manipular como se mueve el robot
            taladroEquipado = state[2]
            carga = state[3]
            bateria = state[1]
            muestrasIg = list (state[4])
            muestrasSed = list (state[5])

            if acc == "moverse": 
                posRobot = obj
                bateria -= 1
            if acc == "sobremarcha":
                posRobot = obj
                bateria -= 4
            if acc == "equipar":
                taladroEquipado = obj
                bateria -= 1
            if acc == "recolectar":
                carga += 1
                bateria -= 3
                if obj == "ignea":
                    muestrasIg.remove(state[0])
                if obj == "sedimentaria":
                    muestrasSed.remove(state[0])
            if acc == "recargar": # Como el tope es 20, si tiene más de 10, solo cargo hasta 20
                bateria = min(20, bateria + 10)
            if acc == "depositar": 
                carga = 0
                bateria -= 1
            return tuple(posRobot),bateria,taladroEquipado,carga,tuple(muestrasIg),tuple(muestrasSed)

        def is_goal(self, state):
            return state[3] + len( state[4]) + len(state[5]) == 0  ## Si la suma de las muestras y la carga da 0 retorna true
        
        def heuristic(self, state):
            pos = state[0]
            taladro_equipado = state[2]
            carga = state[3]
            muestras_igneas = state[4]
            muestras_sedimentarias = state[5]
            muestras = list(muestras_igneas) + list(muestras_sedimentarias) #Sumo la cantidad de muestras

            if not muestras: ##Si no hay muestras en el mapa debe retornar lo que tiene cargado
                return carga
            
            dist_min_rover_a_muestra = min( #Distancia de manhhatan de la pos actual y la pos muestra
                abs(pos[0] - mx) + abs(pos[1] - my)
                for mx, my in muestras
            )
            costo_min_mov = (dist_min_rover_a_muestra +1 ) // 2
            costo_recoleccion = 2 * len(muestras) ## Cantidad de muestras por lo que cuesta
            costo_deposito_minimo = carga + len(muestras) #Si hay n muestras y carga de a 2, minimamente voy a depositar n/2 veces
            costo_taladro = 0
                #Calculo el costo de equipar un taladro cuando no es correcto
            if taladro_equipado is None and muestras:
                costo_taladro = 3
            elif taladro_equipado == "termico" and muestras_sedimentarias:
                costo_taladro = 3
            elif taladro_equipado == "percusion" and muestras_igneas:
                costo_taladro = 3

            return costo_min_mov + costo_recoleccion + costo_deposito_minimo + costo_taladro

        def cost(self, state, action, state2):
            acc, obj = action ##Subdivido el action en accion y objeto. ej. "Moverse", (2,1)
            muestrasCargadas = state[3]
            if acc == "moverse" or acc == "sobremarcha":
                return 1 
            if acc == "equipar":
                return 3
            if acc == "recolectar":
                return 2 
            if acc == "depositar":
                return muestrasCargadas
            else:
                return 4

    problema = ProblemaRover(EstadoInicial)
    result = astar(problema,graph_search=True)

    acciones = [accion for accion, _estado in result.path() if accion is not None]
    return acciones



# # todas las coordenadas son en formato (fila, columna)
# if __name__ == "__main__":
#     acciones = planear_rover(
#         rover_inicio=(0, 0),
#         bateria_inicial=20,
#         zonas_sombra=[(0, 1), (0, 2)],
#         muestras_igneas=[(1, 1), (1, 2)],
#         muestras_sedimentarias=[(2, 3)],
# #     )

# print("Acciones a realizar:")
# for i, accion in enumerate(acciones):
#     print(f"{i+1}. {accion}")
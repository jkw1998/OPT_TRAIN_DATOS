import numpy as np
import pandas as pd
from pyomo.environ import *

# =============================================================================
# 1. CONFIGURACIÓN GENERAL
# =============================================================================
DIAS_A_GENERAR = 1000
HORAS = 24
ruta_ipopt = r'C:\Users\James Kagunda\Desktop\TFG\CODING\VPP\Ipopt-3.14.19-win64-msvs2022-md\bin\ipopt.exe'

def solve_single_day(day_index):
    t_arr = np.arange(1, 25)
    
    # --- GENERACIÓN DE DATOS DINÁMICOS (MÁS REALISTAS) ---
    
    # 1. PRECIOS: Modelo de caminata aleatoria sobre base senoidal
    # Esto evita que todos los días tengan el pico a la misma hora exacta
    shift = np.random.uniform(-4, 4)
    base_precios = 55 + 25 * np.sin((t_arr - 8 + shift) / 24 * 2 * np.pi)
    precios = {t: float(max(5, base_precios[t-1] + np.random.normal(0, 8))) for t in t_arr}
    
    # 2. FV: "Días de nubes" (Simulamos caídas bruscas de producción)
    clima_factor = np.random.choice([0.2, 0.5, 0.9, 1.1], p=[0.1, 0.2, 0.5, 0.2])
    fv = {}
    for t in t_arr:
        if 7 <= t <= 18:
            # 15% de probabilidad de que pase una nube y la producción caiga un 70%
            nube = 0.3 if np.random.rand() > 0.85 else 1.0
            val = 7 * clima_factor * np.sin((t-6)/12 * np.pi) * nube
            fv[t] = float(max(0, val + np.random.normal(0, 0.1)))
        else:
            fv[t] = 0.0
            
    # 3. DEMANDA: Ruido horario para evitar perfiles idénticos
    demanda = {t: float(1.4 + 0.6 * np.random.random() + 0.4 * np.sin(t/24 * 4 * np.pi)) for t in t_arr}

    # --- MODELO PYOMO ---
    model = ConcreteModel()
    T = range(1, 25)

    # Variables
    model.Pd = Var(T, bounds=(-15, 15))
    model.Pg = Var(T, bounds=(0, 5))
    model.Pl = Var(T, bounds=(0.5, 2.0))
    model.Es = Var(T, bounds=(0.2, 1.0))
    model.Psc = Var(T, bounds=(0, 0.4))
    model.Psd = Var(T, bounds=(0, 0.6))
    model.v = Var(T, domain=Binary, initialize=1)

    # Condiciones iniciales fijas por día
    Pg_0, Pl_0, Es_0 = 2.0, 1.5, 0.5

    def obj_rule(m):
        total_obj = 0
        for t in T:
            m.v[t].fix(1) # Forzamos generador encendido
            p_g_prev = Pg_0 if t == 1 else m.Pg[t-1]
            p_l_prev = Pl_0 if t == 1 else m.Pl[t-1]
            
            Eg = (p_g_prev + m.Pg[t]) / 2
            El = (p_l_prev + m.Pl[t]) / 2
            
            utilidad = 150 * El - 30 * (El**2)
            costo_gen = 5 * (Eg**2) + 10 * Eg + 50
            
            # MEJORA: Penalización cuadrática pequeña por usar la batería
            # Esto "suaviza" el SoC y evita que salte de 0.2 a 1.0 sin razón
            costo_bateria = 0.8 * (m.Psc[t]**2 + m.Psd[t]**2)
            
            total_obj += (precios[t] * m.Pd[t] + utilidad - costo_gen - costo_bateria)
        return total_obj

    model.obj = Objective(rule=obj_rule, sense=maximize)
    model.cons = ConstraintList()

    for t in T:
        p_g_prev = Pg_0 if t == 1 else model.Pg[t-1]
        es_prev = Es_0 if t == 1 else model.Es[t-1]
        
        # Balance: Generación + FV + Descarga == Demanda + Carga + Intercambio Red
        model.cons.add( (p_g_prev + model.Pg[t])/2 + fv[t] + model.Psd[t] == demanda[t] + model.Psc[t] + model.Pd[t] )
        
        # Dinámica de batería con pérdidas (Eficiencia 88%)
        model.cons.add( model.Es[t] == es_prev + 0.88 * model.Psc[t] - (model.Psd[t] / 0.88) )
        
        # Rampas de potencia
        model.cons.add( model.Pg[t] - p_g_prev <= 2.5 )
        model.cons.add( p_g_prev - model.Pg[t] <= 2.5 )

    # Restricción: No dejar la batería vacía al final (Sostenibilidad)
    model.cons.add(model.Es[24] >= Es_0)

    # Resolución con IPOPT
    solver = SolverFactory('ipopt', executable=ruta_ipopt)
    solver.options['tol'] = 1e-7 # Precisión alta para captar el ruido
    
    res = solver.solve(model)
    
    if res.solver.termination_condition == TerminationCondition.optimal:
        return ([precios[t] for t in T] + 
                [fv[t] for t in T] + 
                [demanda[t] for t in T] + 
                [value(model.Es[t]) for t in T])
    return None

# =============================================================================
# 2. EJECUCIÓN Y GUARDADO
# =============================================================================
dataset = []
print(f"Generando {DIAS_A_GENERAR} dias con optimizacion individual y ruido real...")

for d in range(1, DIAS_A_GENERAR + 1):
    fila = solve_single_day(d)
    if fila:
        dataset.append(fila)
    if d % 50 == 0:
        print(f"Progreso: {d} dias ({(d/DIAS_A_GENERAR)*100:.0f}%)")

# Nombres de columnas
cols = ([f'Precio_H{t}' for t in range(1, 25)] + 
        [f'FV_H{t}' for t in range(1, 25)] + 
        [f'Demanda_H{t}' for t in range(1, 25)] + 
        [f'SOC_Final_H{t}' for t in range(1, 25)])

df = pd.DataFrame(dataset, columns=cols)
df.to_csv("Dataset_Final_Realista_1000.csv", index=False)

print("\n ¡Dataset listo! Revisa 'Dataset_Final_Realista_1000.csv' para ver la variabilidad.")
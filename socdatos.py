import numpy as np
import pandas as pd
from pyomo.environ import *

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
DIAS_A_GENERAR = 1000
HORAS = 24
ruta_ipopt = r'C:\Users\James Kagunda\Desktop\TFG\CODING\VPP\Ipopt-3.14.19-win64-msvs2022-md\bin\ipopt.exe'

def solve_single_day(day_index):
    t_arr = np.arange(1, 25)
    
    # --- GENERACIÓN DE DATOS CON ALTA VARIABILIDAD (NO REPETITIVOS) ---
    
    # 1. PRECIOS: Desfase aleatorio del pico y ruido de volatilidad
    shift = np.random.uniform(-3, 3) # El pico de precio se mueve de hora cada día
    volatilidad = np.random.uniform(10, 30)
    precios = {t: float(60 + volatilidad * np.sin((t - 8 + shift) / 24 * 2 * np.pi) + np.random.normal(0, 5)) for t in t_arr}
    
    # 2. FOTOVOLTAICA: Selección de tipo de clima (Factor "Día de lluvia" vs "Día Soleado")
    clima = np.random.choice([0.1, 0.4, 0.8, 1.2], p=[0.1, 0.2, 0.4, 0.3])
    # Añadimos ruido gaussiano hora a hora para simular nubes pasajeras
    fv = {t: float(max(0, (6 * clima * np.sin((t-6)/12 * np.pi)) + np.random.normal(0, 0.15))) if 6 <= t <= 18 else 0.0 for t in t_arr}
    
    # 3. DEMANDA: Perfil errático individual
    demanda = {t: float(1.3 + 0.7 * np.random.random() + 0.4 * np.sin(t/24 * 4 * np.pi)) for t in t_arr}

    # --- MODELO PYOMO (Optimización Individual por día/hora) ---
    model = ConcreteModel()
    T = range(1, 25)

    # Variables con rangos que permiten optimización fina
    model.Pd = Var(T, bounds=(-15, 15))
    model.Pg = Var(T, bounds=(0, 5))
    model.Pl = Var(T, bounds=(0.5, 2.5))
    model.Es = Var(T, bounds=(0.2, 1.0))
    model.Psc = Var(T, bounds=(0, 0.35))
    model.Psd = Var(T, bounds=(0, 0.55))
    model.v = Var(T, domain=Binary)

    # Condiciones iniciales
    Pg_0, Pl_0, Es_0 = 2.0, 1.5, 0.5

    def obj_rule(m):
        beneficio = 0
        for t in T:
            # Forzamos v=1 para simplificar, o podrías dejar que el solver decida
            m.v[t].fix(1) 
            
            p_g_prev = Pg_0 if t == 1 else m.Pg[t-1]
            p_l_prev = Pl_0 if t == 1 else m.Pl[t-1]
            
            # Promedios para energía integrada
            Eg = (p_g_prev + m.Pg[t]) / 2
            El = (p_l_prev + m.Pl[t]) / 2
            
            utilidad = 160 * El - 35 * (El**2) # Ligeramente ajustado
            coste = 6 * (Eg**2) + 12 * Eg + 40
            
            # Beneficio horario individualizado
            beneficio += (precios[t] * m.Pd[t] + utilidad - coste)
        return beneficio
    
    model.obj = Objective(rule=obj_rule, sense=maximize)
    model.cons = ConstraintList()

    for t in T:
        p_g_prev = Pg_0 if t == 1 else model.Pg[t-1]
        es_prev = Es_0 if t == 1 else model.Es[t-1]
        
        # Balance de energía (Carga/Descarga batería + FV + Red)
        model.cons.add( (p_g_prev + model.Pg[t])/2 + fv[t] + model.Psd[t] == demanda[t] + model.Psc[t] + model.Pd[t] )
        
        # Dinámica de la batería (Rendimiento 85%)
        model.cons.add( model.Es[t] == es_prev + 0.85 * model.Psc[t] - (model.Psd[t] / 0.85) )
        
        # Restricciones de rampa (Evita cambios bruscos irreales)
        model.cons.add( model.Pg[t] - p_g_prev <= 2.5 )
        model.cons.add( p_g_prev - model.Pg[t] <= 2.5 )

    # Obligamos a que el día termine con al menos el SOC inicial para evitar "vaciado oportunista"
    model.cons.add(model.Es[24] >= Es_0)

    # RESOLUCIÓN
    solver = SolverFactory('ipopt', executable=ruta_ipopt)
    # Aumentamos la precisión para optimización individualizada
    solver.options['tol'] = 1e-8
    
    res = solver.solve(model)
    
    if res.solver.termination_condition == TerminationCondition.optimal:
        precios_list = [precios[t] for t in T]
        fv_list = [fv[t] for t in T]
        demanda_list = [demanda[t] for t in T]
        soc_list = [value(model.Es[t]) for t in T]
        
        return precios_list + fv_list + demanda_list + soc_list
    return None

# =============================================================================
# 2. BUCLE PRINCIPAL
# =============================================================================
dataset = []
print(f"Iniciando optimizacion de {DIAS_A_GENERAR} dias independientes...")

for d in range(1, DIAS_A_GENERAR + 1):
    fila = solve_single_day(d)
    if fila:
        dataset.append(fila)
    
    if d % 50 == 0:
        print(f"Progreso: {d} dias completados ({(d/DIAS_A_GENERAR)*100:.1f}%)")

# --- GUARDADO ---
cols = ([f'Precio_H{t}' for t in range(1, 25)] + 
        [f'FV_H{t}' for t in range(1, 25)] + 
        [f'Demanda_H{t}' for t in range(1, 25)] + 
        [f'SOC_Final_H{t}' for t in range(1, 25)])

df = pd.DataFrame(dataset, columns=cols)
df.to_csv("Dataset_Optimizado_1000_NUEVO.csv", index=False)

print("\n Dataset generado correctamente.")
print(f"Resultados guardados en: 'Dataset_Optimizado_1000_NUEVO.csv'")
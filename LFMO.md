# LFMO para simulación y optimización de políticas de mantenimiento

Implementación en Julia de un modelo de reparaciones de mantenimiento con donde los tiempos de vidas siguen una distribución **Levy Frailty Marshall Olkins**, enfocado en sistemas multicomponente con fallas dependientes y simultáneas. Se representa una red cuyos componentes pueden fallar de forma conjunta, construye el MDP asociado y se obtiene políticas de mantenimiento mediante el algoritmo SAVIA.

## Archivos

- [`LFMO_Act.jl`](LFMO_Act.jl): modelo LFMO, muestreo de fallas, representación de estados, acciones y construcción del MDP.
- [`SAVIA.jl`](SAVIA.jl): implementación de los algoritmos SAVIA y SAVIA+ para obtener una política aproximada.
- [`RunPolicy.jl`](RunPolicy.jl): conecta el modelo LFMO con SAVIA+ y permite calcular una política para una red dada.
- [`SimLFMO.jl`](SimLFMO.jl): simulación de políticas, métricas de largo plazo y evaluación Monte Carlo.

## Flujo general

```text
Red y parámetros
      ↓
LFMO_Act.jl
      ↓
Construcción del MDP
      ↓
SAVIA.jl
      ↓
Política de mantenimiento
      ↓
SimLFMO.jl
      ↓
Evaluación por simulación
```

Las acciones se representan como:

- `1`: no reparar (`NR`)
- `2`: reparar (`R`)

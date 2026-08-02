# LFMO: simulación y optimización de políticas de mantenimiento

Este repositorio reúne una implementación en Julia de un modelo **LFMO** para sistemas multicomponente con fallas dependientes y simultáneas.

El objetivo es representar una red cuyos componentes pueden fallar de forma conjunta, construir el MDP asociado y obtener políticas de mantenimiento mediante el algoritmo SAVIA.

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

## Dependencias

```julia
using Pkg

Pkg.add([
    "StatsBase",
    "Distributions",
    "JLD2"
])
```

## Uso básico

```julia
include("RunPolicy.jl")
using .RunPolicy

EDGES = [
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 4)
]

resultado = obtener_politica_red(
    EDGES,
    1,
    4,
    true;
    μ = 1.0,
    λ = 0.4,
    γ = 0.4,
    η_factor = 0.9,
    Cc = 0.3,
    fact_Cs = 1.0,
    ε = 0.05,
    δ = 0.1,
    max_it = 10
)

politica = resultado.πₒ
```

Las acciones se representan como:

- `1`: no reparar (`NR`)
- `2`: reparar (`R`)

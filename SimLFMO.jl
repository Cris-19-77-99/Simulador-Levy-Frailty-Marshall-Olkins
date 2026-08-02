module SimLFMO

using Random
using Statistics

include(joinpath(@__DIR__, "LFMO_Act.jl"))
using .LFMOProject: construir_ϕ

export simular_sistema_red,
       simular_red,
       mean_ci95,
       mc_policy_red,
       cargar_politica_txt


"""
    simular_sistema_red(
        politica,
        EDGES,
        s_src,
        t_dst,
        directed,
        μ,
        λ,
        γ,
        Cc,
        fact_Cs,
        horizonte;
        semilla=nothing
    )

Simula una red sometida a fallas dependientes generadas por un modelo LFMO.

La política debe ser un diccionario que asocie cada estado de la red con
`"R"` (reparar) o `"NR"` (no reparar). Las reparaciones correctivas y
preventivas reinician el sistema y determinan el final de un ciclo.
"""
function simular_sistema_red(
    politica,
    EDGES,
    s_src::Int,
    t_dst::Int,
    directed,
    μ,
    λ,
    γ,
    Cc,
    fact_Cs,
    horizonte;
    semilla=nothing
)

    horizonte > 0 || throw(ArgumentError("El horizonte debe ser positivo."))
    μ >= 0 || throw(ArgumentError("μ debe ser no negativo."))
    λ > 0 || throw(ArgumentError("λ debe ser positivo."))
    γ > 0 || throw(ArgumentError("γ debe ser positivo."))
    Cc >= 0 || throw(ArgumentError("Cc debe ser no negativo."))
    fact_Cs >= 0 || throw(ArgumentError("fact_Cs debe ser no negativo."))

    semilla === nothing || Random.seed!(semilla)

    ϕ = construir_ϕ(EDGES, s_src, t_dst; undirected=directed)
    n = length(EDGES)

    n > 0 || throw(ArgumentError("La red debe contener al menos una arista."))

    # Umbrales exponenciales de los componentes
    ϵ = -log.(rand(n))

    # Estado de la red: 1 operativo, 0 fallado
    s = ones(Int, n)
    ϵ_bool = falses(n)

    # Trayectoria del subordinador
    t = [0.0]
    x = [0.0]

    # Eventos y costos
    tiempos_muerte = Float64[]
    tiempos_reparacion = Float64[]
    tiempos_renovacion = [0.0]
    costos = Float64[]

    count_fallas = 0
    Cs = fact_Cs*n*Cc

    proximo_ϵ() = all(ϵ_bool) ? Inf : minimum(ϵ[.!ϵ_bool])

    function costo_instantaneo()
        fallados = count(==(0), s)
        costo_componentes = fallados*Cc
        costo_sistema = ϕ(s) == 0 ? Cs : 0.0

        return costo_componentes + costo_sistema
    end

    function reiniciar_umbral!(ϵ_actual)
        for j in 1:n
            if ϵ[j] <= ϵ_actual
                ϵ[j] = ϵ_actual - log(rand())
            end

            s[j] = 1
            ϵ_bool[j] = false
        end
    end

    function registrar_renovacion!(tiempo_evento, ϵ_actual, tipo)
        push!(costos, costo_instantaneo())
        push!(tiempos_renovacion, tiempo_evento)

        if tipo == :correctiva
            push!(tiempos_muerte, tiempo_evento)
        elseif tipo == :preventiva
            push!(tiempos_reparacion, tiempo_evento)
        else
            throw(ArgumentError("Tipo de renovación no reconocido."))
        end

        reiniciar_umbral!(ϵ_actual)
    end

    ϵ_min = proximo_ϵ()

    while t[end] < horizonte

        # Tiempo hasta el siguiente salto del proceso de Poisson
        Δt = -log(rand())/λ
        t_next = min(t[end] + Δt, horizonte)

        # Evolución continua del subordinador
        x_drif = x[end] + μ*(t_next - t[end])

        # Fallas producidas por el drift
        while (x[end] < ϵ_min) && (x_drif >= ϵ_min)

            t_cruce = t[end] + (ϵ_min - x[end])/μ

            k = argmin([
                ϵ_bool[j] ? Inf : ϵ[j]
                for j in 1:n
            ])

            s[k] = 0
            ϵ_bool[k] = true
            count_fallas += 1

            if ϕ(s) == 0

                registrar_renovacion!(
                    t_cruce,
                    ϵ_min,
                    :correctiva
                )

                ϵ_min = proximo_ϵ()

            else

                accion = get(politica, Tuple(s), "NR")

                if accion == "R"

                    registrar_renovacion!(
                        t_cruce,
                        ϵ_min,
                        :preventiva
                    )

                    ϵ_min = proximo_ϵ()

                else
                    ϵ_min = proximo_ϵ()
                end
            end
        end

        t_next >= horizonte && break

        # Salto del subordinador
        salto = -log(rand())/γ
        x_salto = x_drif + salto

        indices = [
            j for j in 1:n
            if !ϵ_bool[j] &&
               ϵ[j] <= x_salto &&
               ϵ[j] > x_drif
        ]

        if !isempty(indices)

            for j in indices
                s[j] = 0
                ϵ_bool[j] = true
                count_fallas += 1
            end

            if ϕ(s) == 0

                registrar_renovacion!(
                    t_next,
                    x_salto,
                    :correctiva
                )

                ϵ_min = proximo_ϵ()

            else

                accion = get(politica, Tuple(s), "NR")

                if accion == "R"

                    registrar_renovacion!(
                        t_next,
                        x_salto,
                        :preventiva
                    )

                    ϵ_min = proximo_ϵ()

                else
                    ϵ_min = proximo_ϵ()
                end
            end
        end

        push!(x, x_salto)
        push!(t, t_next)
    end

    duraciones_ciclo = diff(tiempos_renovacion)

    Ccycle = isempty(costos) ? NaN : mean(costos)

    Tcycle = isempty(duraciones_ciclo) ?
             NaN :
             mean(duraciones_ciclo)

    LTMC = sum(costos)/horizonte
    LTMN = count_fallas/horizonte

    return (
        LTMC=LTMC,
        LTMN=LTMN,
        Ccycle=Ccycle,
        Tcycle=Tcycle,
        costos=costos,
        tiempos_muerte=tiempos_muerte,
        tiempos_reparacion=tiempos_reparacion,
        tiempos_renovacion=tiempos_renovacion,
        n=n
    )
end


"""
    simular_red(
        politica,
        EDGES,
        s_src,
        t_dst,
        directed;
        μ=1.0,
        λ=0.4,
        γ=0.4,
        Cc=0.3,
        fact_Cs=1.0,
        H=1000.0,
        seed=nothing
    )

Ejecuta una simulación y devuelve únicamente las métricas principales.
"""
function simular_red(
    politica,
    EDGES,
    s_src,
    t_dst,
    directed;
    μ=1.0,
    λ=0.4,
    γ=0.4,
    Cc=0.3,
    fact_Cs=1.0,
    H=1000.0,
    seed=nothing
)

    resultado = simular_sistema_red(
        politica,
        EDGES,
        s_src,
        t_dst,
        directed,
        μ,
        λ,
        γ,
        Cc,
        fact_Cs,
        H;
        semilla=seed
    )

    return (
        LTMC=resultado.LTMC,
        LTMN=resultado.LTMN,
        Ccycle=resultado.Ccycle,
        Tcycle=resultado.Tcycle
    )
end


"""
    mean_ci95(v)

Calcula la media y un intervalo de confianza normal aproximado del 95 %.
"""
function mean_ci95(v)

    n = length(v)

    n == 0 && return (
        mean=NaN,
        lo=NaN,
        hi=NaN,
        n=0
    )

    m = mean(v)

    n == 1 && return (
        mean=m,
        lo=m,
        hi=m,
        n=1
    )

    d = 1.96*std(v)/sqrt(n)

    return (
        mean=m,
        lo=m-d,
        hi=m+d,
        n=n
    )
end


"""
    mc_policy_red(
        politica,
        EDGES,
        s_src,
        t_dst,
        directed;
        R=200,
        pars...
    )

Evalúa una política mediante `R` réplicas Monte Carlo.
"""
function mc_policy_red(
    politica,
    EDGES,
    s_src,
    t_dst,
    directed;
    R=200,
    pars...
)

    R > 0 || throw(ArgumentError("R debe ser positivo."))

    mets = [
        simular_red(
            politica,
            EDGES,
            s_src,
            t_dst,
            directed;
            pars...,
            seed=10+i
        )
        for i in 1:R
    ]

    LTMCs = getfield.(mets, :LTMC)
    LTMNs = getfield.(mets, :LTMN)
    Ccycles = getfield.(mets, :Ccycle)
    Tcycles = getfield.(mets, :Tcycle)

    Ccycles_ok = filter(x -> !isnan(x), Ccycles)
    Tcycles_ok = filter(x -> !isnan(x), Tcycles)

    return (
        LTMC=mean_ci95(LTMCs),
        LTMN=mean_ci95(LTMNs),

        Ccycle=isempty(Ccycles_ok) ?
               (mean=NaN, lo=NaN, hi=NaN, n=0) :
               mean_ci95(Ccycles_ok),

        Tcycle=isempty(Tcycles_ok) ?
               (mean=NaN, lo=NaN, hi=NaN, n=0) :
               mean_ci95(Tcycles_ok),

        raw=(
            LTMC=LTMCs,
            LTMN=LTMNs,
            Ccycle=Ccycles,
            Tcycle=Tcycles
        )
    )
end


"""
    cargar_politica_txt(filename)

Carga una política almacenada en líneas de la forma:

`1,0,1 => R`
"""
function cargar_politica_txt(filename)

    politica = Dict{Tuple{Vararg{Int}}, String}()

    for line in eachline(filename)

        line = strip(line)
        isempty(line) && continue

        estado_txt, accion = split(
            line,
            " => ";
            limit=2
        )

        estado = Tuple(
            parse.(
                Int,
                strip.(split(estado_txt, ","))
            )
        )

        politica[estado] = strip(accion)
    end

    return politica
end


end # module

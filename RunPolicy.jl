module RunPolicy

include(joinpath(@__DIR__, "LFMO_Act.jl"))
include(joinpath(@__DIR__, "SAVIA.jl"))

using .LFMOProject
using .SAVIA

export obtener_politica_red

function obtener_politica_red(
    EDGES,
    s0::Int,
    t0::Int,
    undirected::Bool;
    law = :expo,
    μ,
    λ,
    γ,
    η_factor,
    Cc,
    fact_Cs,
    ε,
    δ,
    max_it,
    checkpoint_path = nothing,
    time_limit_s = nothing,
    save_every_i = 1
)
    law == :expo || error("Por ahora solo está soportada law=:expo. Llegó: $law")
    μ >= 0 || error("μ debe ser no negativo")
    λ > 0 || error("λ debe ser positivo")
    γ > 0 || error("γ debe ser positivo")
    0 < η_factor < 1 || error("η_factor debe satisfacer 0 < η_factor < 1")
    Cc >= 0 || error("Cc debe ser no negativo")
    fact_Cs >= 0 || error("fact_Cs debe ser no negativo")
    ε > 0 || error("ε debe ser positivo")
    0 < δ < 1 || error("δ debe satisfacer 0 < δ < 1")
    max_it >= 0 || error("max_it debe ser no negativo")
    save_every_i >= 1 || error("save_every_i debe ser positivo")

    ϕ = LFMOProject.construir_ϕ(
        EDGES,
        s0,
        t0;
        undirected = undirected
    )

    n = length(EDGES)

    params = Dict(
        :μ => μ,
        :λ => λ,
        :γ => γ
    )

    Λ = LFMOProject.construir_Λ(n, law, params)

    η = η_factor / LFMOProject.ψ(n, law, params)

    Cs = fact_Cs * n * Cc

    r_fun(s, a) = (
        a === :R ?
        -Cc * LFMOProject.F(s) + (ϕ(s) == 0 ? -Cs : 0.0)
        :
        0.0
    )

    model = LFMOProject.LFMOModel(
        n,
        law,
        params,
        Λ,
        η,
        ϕ
    )

    mdp = LFMOProject.lfmo_mtilde(model; r_fun = r_fun)

    Q, T, πₒ, info = SAVIA.savia_plus(
        mdp;
        ε = ε,
        δ = δ,
        max_it = max_it,
        checkpoint_path = checkpoint_path,
        time_limit_s = time_limit_s,
        save_every_i = save_every_i
    )

    return (;
        Q,
        T,
        πₒ,
        info,
        n,
        η,
        Λ,
        params,
        mdp
    )
end

end # module RunPolicy

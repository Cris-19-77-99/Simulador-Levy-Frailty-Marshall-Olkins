module LFMOProject

using Random
using StatsBase
using Distributions

export LFMOModel, G_MDP,
       ψ, Δψ, construir_Λ,
       sample_lfmo,
       W, F,
       decodificar_estado, codificar_estado,
       acciones_permitidas,
       validar_η!,
       sampler_mtilde,
       lfmo_mtilde,
       construir_ϕ

# ============================================================
# 1) LFMO / sampling
# ============================================================

function ψ(b::Real, law::Symbol, param)
    if law == :expo
        μ = param[:μ]
        λ = param[:λ]
        γ = param[:γ]
        return b * (μ + λ / (γ + b))

    elseif law == :pareto
        error("Aún no implementado: ley :pareto")

    else
        error("Ley no reconocida: $law")
    end
end

Δψ(x, law::Symbol, param) = ψ(x, law, param) - ψ(x - 1, law, param)

function safe_cat(p)
    p_safe = max.(p, 0)
    total = sum(p_safe)

    total > 0 || error("Las probabilidades deben tener suma positiva")

    return Categorical(p_safe ./ total)
end

function construir_Λ(n::Int, law::Symbol, param)
    n >= 1 || error("n debe ser positivo")

    return Dict(
        d => [
            sum(
                (-1)^i *
                binomial(I - 1, i) *
                Δψ(d - I + i + 1, law, param)
                for i in 0:(I - 1)
            )
            for I in 1:d
        ]
        for d in 1:n
    )
end

function sample_t_lfmo(d::Int, law::Symbol, param)
    rate = ψ(d, law, param)
    rate > 0 || error("La tasa ψ(d) debe ser positiva")

    return rand(Exponential(1 / rate))
end

function sample_num_death_lfmo(d::Int, Λ, law::Symbol, param)
    denom = ψ(d, law, param)
    denom > 0 || error("La tasa ψ(d) debe ser positiva")

    p = [binomial(d, i) * Λ[d][i] / denom for i in 1:d]

    return rand(safe_cat(p))
end

function sample_idx_death_lfmo(idx, num_death::Int)
    return sample(idx, num_death; replace = false)
end

function sample_lfmo(s::AbstractVector, Λ, law::Symbol, param)
    s_new = copy(s)
    idx = findall(!iszero, s_new)
    d = length(idx)

    if d == 0
        return s_new, idx, 0.0
    end

    T = sample_t_lfmo(d, law, param)
    num_death = sample_num_death_lfmo(d, Λ, law, param)
    idx_death = sample_idx_death_lfmo(idx, num_death)

    s_new[idx_death] .= zero(eltype(s_new))
    idx_new = findall(!iszero, s_new)

    return s_new, idx_new, T
end

# ============================================================
# 2) Estados, acciones y estructuras del MDP
# ============================================================

W(s) = count(==(1), s)
F(s) = count(==(0), s)

struct LFMOModel{T, Φ, P, L}
    n::Int
    law::L
    param::P
    Λ
    η::T
    ϕ::Φ
end

struct G_MDP{T, Fsampler}
    S::Int
    A::Int
    r::Matrix{T}
    sampler::Fsampler
    As::Vector{NTuple{2, Bool}}
end

function decodificar_estado(idx::Int, n::Int)
    idx >= 1 || error("idx debe ser positivo")
    n >= 1 || error("n debe ser positivo")

    v = Vector{Int64}(undef, n)
    t = UInt64(idx - 1)

    for j in 1:n
        v[j] = Int64(((t >> (j-1)) & 0x01) == 0x01)
    end

    return v
end

codificar_estado(v) = begin
    n = length(v)
    idx = UInt64(0)
    # Para cada posición j, si v[j] ≠ 0 encendemos el bit (j-1) en idx.
    for j in 1:n
        # "(v[j] != 0 ? 1 : 0)" toma el valor 1 si el bit está encendido, 0 si no.
        # "<< (j-1)" en término de la cadena de bits esta operacion coloca ese 1 en la posición de bit correcta.
        # |= lo “mezcla” (OR) con los bits ya establecidos en idx.
        idx |= UInt64(v[j] != 0 ? 1 : 0) << (j-1)
    end
    return Int(idx + 1)  #Salida el entero que es representado por la cadena de bits pero + 1 para facilitar codigo.
end


function acciones_permitidas(s, ϕ)::NTuple{2, Bool}
    if all(!iszero, s)
        return (true, false)       # todo operativo: solo NR
    elseif ϕ(s) == 0
        return (false, true)       # sistema caído: solo R
    else
        return (true, true)        # sistema funcionando con fallas: ambas
    end
end

function validar_η!(model::LFMOModel)
    upper = 1 / ψ(model.n, model.law, model.param)

    if !(0 < model.η < upper)
        error("η debe satisfacer 0 < η < 1/ψ(n). η=$(model.η), upper=$upper")
    end

    return nothing
end

# ============================================================
# 3) Sampler discreto normalizado
# ============================================================


function sampler_mtilde(s, a, model)
    validar_η!(model)

    n     = model.n
    law   = model.law
    param = model.param
    Λ     = model.Λ
    η     = model.η
    ϕ     = model.ϕ

    if ϕ(s) == 1
        if a === :NR
            b = W(s)
            p_jump = η * ψ(b, law, param)

            if rand() > p_jump
                return copy(s)
            else
                s′, _, _ = sample_lfmo(s, Λ, law, param)
                return s′
            end

        elseif a === :R
            p_jump = η * ψ(n, law, param)

            if rand() > p_jump
                return copy(s)
            else
                s_sup = ones(Int8, n)
                s′, _, _ = sample_lfmo(s_sup, Λ, law, param)
                return s′
            end

        else
            error("Acción no reconocida: $a")
        end
    else
        if a === :R
            p_jump = η * ψ(n, law, param)

            if rand() > p_jump
                return copy(s)
            else
                s_sup = ones(Int8, n)
                s′, _, _ = sample_lfmo(s_sup, Λ, law, param)
                return s′
            end

        elseif a === :NR
            error("La acción :NR no está permitida cuando ϕ(s) = 0")

        else
            error("Acción no reconocida: $a")
        end
    end
end



function lfmo_mtilde(model::LFMOModel; r_fun)
    validar_η!(model)

    n = model.n
    S = 2^n
    A = 2

    As = Vector{NTuple{2, Bool}}(undef, S)
    r = fill(-10000.0, S, A)

    for sidx in 1:S
        s = decodificar_estado(sidx, n)
        As[sidx] = acciones_permitidas(s, model.ϕ)

        if As[sidx][1]
            r[sidx, 1] = r_fun(s, :NR)
        end

        if As[sidx][2]
            r[sidx, 2] = r_fun(s, :R)
        end
    end

    sampler = function (s_idx::Int, a_idx::Int)
        if !As[s_idx][a_idx]
            return s_idx
        end

        s = decodificar_estado(s_idx, n)
        a = (a_idx == 1 ? :NR : :R)

        s′ = sampler_mtilde(s, a, model)

        return codificar_estado(s′)
    end

    return G_MDP(S, A, r, sampler, As)
end

# ============================================================
# 4) Redes: construir ϕ
# ============================================================

function construir_ϕ(
    EDGES,
    s::Int,
    t::Int;
    N_NODES::Union{Int, Nothing} = nothing,
    undirected::Bool = true
)
    isempty(EDGES) && error("EDGES no puede estar vacío")

    nn = isnothing(N_NODES) ? maximum(vcat(first.(EDGES), last.(EDGES))) : N_NODES

    nn >= 1 || error("N_NODES debe ser positivo")
    1 <= s <= nn || error("El nodo de origen está fuera de rango")
    1 <= t <= nn || error("El nodo de destino está fuera de rango")

    adj = [Tuple{Int, Int}[] for _ in 1:nn]

    for (eid, (u, v)) in enumerate(EDGES)
        1 <= u <= nn || error("El nodo $u está fuera de rango")
        1 <= v <= nn || error("El nodo $v está fuera de rango")

        push!(adj[u], (v, eid))

        if undirected
            push!(adj[v], (u, eid))
        end
    end

    visited = falses(nn)
    queue = Int[]
    sizehint!(queue, nn)

    return function (x)
        length(x) == length(EDGES) || error("El estado debe tener una entrada por arista")

        fill!(visited, false)
        empty!(queue)

        visited[s] = true
        push!(queue, s)

        while !isempty(queue)
            u = popfirst!(queue)
            u == t && return 1

            for (v, eid) in adj[u]
                if x[eid] != 0 && !visited[v]
                    visited[v] = true
                    push!(queue, v)
                end
            end
        end

        return 0
    end
end

end # module LFMOProject

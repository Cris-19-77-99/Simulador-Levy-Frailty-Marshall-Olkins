module SAVIA

using Random, Dates
using StatsBase, Distributions, Combinatorics
using Printf
using JLD2

export savia_plus

span_norm(x) = maximum(x) - minimum(x)

function span_norm_finite(x)
    xf = x[isfinite.(x)]
    isempty(xf) && return 0.0
    return maximum(xf) - minimum(xf)
end

factible(As, s, a) = As isa AbstractMatrix ? As[s, a] : As[s][a]

function maxA(Q, As)
    S, A = size(Q)
    h = Vector{eltype(Q)}(undef, S)
    for s in 1:S
        mejor = -Inf
        for a in 1:A
            factible(As, s, a) || continue
            v = Q[s, a]
            if v > mejor
                mejor = v
            end
        end
        mejor == -Inf && error("No hay acciones factibles en el estado $s")
        h[s] = mejor
    end
    return h
end

function π_de_Q(Q, As)
    S, A = size(Q)
    πₒ = Vector{Int}(undef, S)

    for s in 1:S
        mejor_a = 0
        mejor_v = -Inf
        for a in 1:A
            factible(As, s, a) || continue
            v = Q[s, a]
            if v > mejor_v
                mejor_v = v
                mejor_a = a
            end
        end
        mejor_a == 0 && error("No hay acciones factibles en el estado $s")
        πₒ[s] = mejor_a  # 0 si no hay acción factible en s
    end
    return πₒ
end


β_k(k) = k / (k + 2)
c_k(k) = 5 * (k + 2) * (log(k + 2))^2

function SAMPLE(mdp, d, m)
    m >= 1 || error("m debe ser positivo")

    S, A = mdp.S, mdp.A
    D = zeros(Float64, S, A)

    oracle_calls = 0

    for s in 1:S, a in 1:A
        factible(mdp.As, s, a) || continue
        acc = 0.0
        for _ in 1:m
            s1 = mdp.sampler(s, a)
            1 <= s1 <= S || error("El sampler devolvió un estado fuera de rango")
            acc += d[s1]
            oracle_calls += 1
        end
        D[s, a] = acc / m
    end
    return D, oracle_calls
end


function savia(mdp; Q0=nothing, n, ε, δ,
              mk_min::Int=20, sp_tol=nothing, log_every::Int=200)

    n >= 0 || error("n debe ser no negativo")
    ε > 0 || error("ε debe ser positivo")
    0 < δ < 1 || error("δ debe satisfacer 0 < δ < 1")
    mk_min >= 1 || error("mk_min debe ser positivo")
    log_every >= 1 || error("log_every debe ser positivo")

    S, A = mdp.S, mdp.A
    Q0 === nothing && (Q0 = zeros(Float64, S, A))
    size(Q0) == (S, A) || error("Q0 debe tener dimensiones S × A")
    Q0 = Float64.(Q0)

    # tolerancia interna ligada a ε (si no entregas sp_tol)
    sp_tol === nothing && (sp_tol = ε / 50)
    sp_tol >= 0 || error("sp_tol debe ser no negativo")

    α = log(2 * S * A * (n + 1) / δ)

    T_prev = copy(mdp.r)
    h_prev = zeros(Float64, S)

    Q = copy(Q0)
    T = copy(T_prev)
    h = copy(h_prev)

    hist = (; mk=Int[], sp_dk=Float64[], sp_res_emp=Float64[],
          oracle_calls=Int[], oracle_calls_acum=Int[])

    for k in 0:n
        β = β_k(k)

        # Q <- (1-β)Q0 + βT_prev (solo acciones factibles)
        for s in 1:S, a in 1:A
            if factible(mdp.As, s, a)
                Q[s, a] = (1 - β) * Q0[s, a] + β * T_prev[s, a]
            else
                Q[s, a] = T_prev[s, a]  # típicamente -Inf
            end
        end

        h = maxA(Q, mdp.As)
        d = h .- h_prev

        sp = span_norm_finite(d)

        mk = ceil(Int, α * c_k(k) * (sp^2) / (ε^2))
        mk = max(mk, mk_min)

        D, calls_k = SAMPLE(mdp, d, mk)
        T = T_prev .+ D

        push!(hist.mk, mk)
        push!(hist.sp_dk, sp)

        # residuo robusto por estado (max por estado)
        res_emp = span_norm_finite(maxA(Q, mdp.As) .- maxA(T, mdp.As))
        push!(hist.sp_res_emp, res_emp)
        push!(hist.oracle_calls, calls_k)

        if isempty(hist.oracle_calls_acum)
            push!(hist.oracle_calls_acum, calls_k)
        else
            push!(hist.oracle_calls_acum, hist.oracle_calls_acum[end] + calls_k)
        end

        # logging controlado (evita "correr en banda")
        if n > 2^12 && (k % log_every == 0)
            @printf("k=%d   mk=%d   sp=%.3e   res=%.3e\n", k, mk, sp, res_emp)
        end

        # criterio de parada razonable (ligado a ε)
        if sp ≤ sp_tol && k > 2
            πₒ = π_de_Q(Q, mdp.As)
            return (Q, T, πₒ, hist)
        end

        T_prev .= T
        h_prev .= h
    end

    πₒ = π_de_Q(Q, mdp.As)
    return (Q, T, πₒ, hist)
end

function savia_plus(mdp; Q0=nothing, ε, δ, max_it,
                    mk_min::Int=20, sp_tol=nothing, log_every::Int=200,
                    checkpoint_path=nothing,
                    time_limit_s=nothing,
                    save_every_i::Int=1)

    ε > 0 || error("ε debe ser positivo")
    0 < δ < 1 || error("δ debe satisfacer 0 < δ < 1")
    max_it >= 0 || error("max_it debe ser no negativo")
    mk_min >= 1 || error("mk_min debe ser positivo")
    log_every >= 1 || error("log_every debe ser positivo")
    save_every_i >= 1 || error("save_every_i debe ser positivo")
    isnothing(time_limit_s) || time_limit_s > 0 || error("time_limit_s debe ser positivo")

    S, A = mdp.S, mdp.A
    Q0 === nothing && (Q0 = zeros(Float64, S, A))
    size(Q0) == (S, A) || error("Q0 debe tener dimensiones S × A")
    Q0 = Float64.(Q0)

    tiempos  = Float64[]
    residuos = Float64[]
    ns       = Int[]

    oracle_calls      = Int[]
    oracle_calls_acum = Int[]
    hists             = Any[]

    Q  = copy(Q0)
    T  = copy(mdp.r)
    πₒ = π_de_Q(Q, mdp.As)

    last_i   = -1
    last_ni  = 0
    last_δi  = NaN
    last_res = NaN

    t0 = time()

    function save_checkpoint(; status="running")
        isnothing(checkpoint_path) && return
        info_ckpt = (
            i = last_i,
            n = last_ni,
            δi = last_δi,
            res = last_res,
            tiempos = copy(tiempos),
            residuos = copy(residuos),
            ns = copy(ns),
            oracle_calls = copy(oracle_calls),
            oracle_calls_acum = copy(oracle_calls_acum),
            status = status,
            elapsed = time() - t0
        )
        jldsave(checkpoint_path;
            Q=Q, T=T, πₒ=πₒ, info_ckpt=info_ckpt)
    end

    for i in 0:max_it
        if !isnothing(time_limit_s) && (time() - t0 ≥ time_limit_s)
            @info "Time limit alcanzado antes de i=$i. Se devuelve última política disponible."
            save_checkpoint(status="time_limit_before_iteration")
            return (
                Q, T, πₒ,
                (; i=last_i, n=last_ni, δi=last_δi, res=last_res,
                   tiempos=tiempos, residuos=residuos, ns=ns,
                   oracle_calls=oracle_calls,
                   oracle_calls_acum=oracle_calls_acum,
                   hists=hists,
                   stopped_by="time_limit")
            )
        end

        ni = 2^i
        δi = δ / c_k(i)

        hist_i = nothing
        t = @elapsed begin
            Q, T, πₒ, hist_i = savia(mdp; Q0=Q, n=ni, ε=ε, δ=δi,
                                     mk_min=mk_min, sp_tol=sp_tol, log_every=log_every)
        end

        res = span_norm_finite(maxA(Q, mdp.As) .- maxA(T, mdp.As))
        if !isfinite(res)
            res = Inf
        end

        calls_i = isempty(hist_i.oracle_calls_acum) ? 0 : hist_i.oracle_calls_acum[end]

        push!(tiempos, t)
        push!(residuos, res)
        push!(ns, ni)
        push!(oracle_calls, calls_i)
        push!(hists, hist_i)

        if isempty(oracle_calls_acum)
            push!(oracle_calls_acum, calls_i)
        else
            push!(oracle_calls_acum, oracle_calls_acum[end] + calls_i)
        end

        last_i = i
        last_ni = ni
        last_δi = δi
        last_res = res

        @info "Iteración i=$i  n=$ni  residuo=$res  tiempo=$(t) s  calls=$calls_i"

        if (i % save_every_i == 0)
            save_checkpoint(status="running")
        end

        if res ≤ 14ε
            save_checkpoint(status="converged")
            return (
                Q, T, πₒ,
                (; i=i, n=ni, δi=δi, res=res,
                   tiempos=tiempos, residuos=residuos, ns=ns,
                   oracle_calls=oracle_calls,
                   oracle_calls_acum=oracle_calls_acum,
                   hists=hists,
                   stopped_by="criterion")
            )
        end

        if !isnothing(time_limit_s) && (time() - t0 ≥ time_limit_s)
            @info "Time limit alcanzado después de i=$i. Se devuelve última política disponible."
            save_checkpoint(status="time_limit_after_iteration")
            return (
                Q, T, πₒ,
                (; i=i, n=ni, δi=δi, res=res,
                   tiempos=tiempos, residuos=residuos, ns=ns,
                   oracle_calls=oracle_calls,
                   oracle_calls_acum=oracle_calls_acum,
                   hists=hists,
                   stopped_by="time_limit")
            )
        end
    end

    save_checkpoint(status="max_it")
    return (
        Q, T, πₒ,
        (; i=last_i, n=last_ni, δi=last_δi, res=last_res,
           tiempos=tiempos, residuos=residuos, ns=ns,
           oracle_calls=oracle_calls,
           oracle_calls_acum=oracle_calls_acum,
           hists=hists,
           stopped_by="max_it")
    )
end
end

using Distributions

@kwdef mutable struct RandomTcEnvironment
    C::Float64
    freq::Float64
    cv_wt::Float64 = 1.
    τ::Float64
    dots::Vector{Float64} = []
    t::Float64 = 0.
end

function Base.show(io::IO, env::RandomTcEnvironment)
    print(io, "RandomTcEnvironment(C=$(env.C), freq=$(env.freq), cv_wt=$(env.cv_wt), τ=$(env.τ), dots=($(length(env.dots))), t=$(env.t))")
end

# Avoid environment duplication during `remake`
Base.deepcopy_internal(env::RandomTcEnvironment, ::IdDict) = env

function simulate_env!(env::RandomTcEnvironment, tmax)
    wt_dist = if iszero(env.cv_wt)
        Dirac(1 / env.freq)
    else
        Gamma(1 / env.cv_wt^2, env.cv_wt^2 / env.freq)
    end

    while env.t <= tmax
        dt = rand(wt_dist)

        if env.t + dt >= tmax
            env.t = tmax
            break
        end

        env.t += dt
        push!(env.dots, env.t)
    end
end

function (env::RandomTcEnvironment)(t)
    if t > env.t
        @warn "Evaluating environment simulated until time $(env.t) at time $t"
    end

    i1 = searchsortedfirst(env.dots, t - env.τ + 1e-6)
    i1 > length(env.dots) && return 0.
    i2 = searchsortedlast(env.dots, t)
    i2 < i1 && return 0.

    env.C * (i2 - i1 + 1)
end

function create_env(defaults; kwargs...)
    kwargs = combine_params(defaults, kwargs, [ :C, :freq, :cv_wt, :τ ])
    RandomTcEnvironment(; kwargs...)
end

function get_tstops(env::RandomTcEnvironment; tspan=(0, Inf), ϵ=1e-6)
    i1 = searchsortedfirst(env.dots, tspan[1] - env.τ - ϵ)
    i1 > length(env.dots) && return Float64[]
    i2 = searchsortedlast(env.dots, tspan[2] + ϵ)
    i2 < i1 && return Float64[]

    [ env.dots[i1:i2]; env.dots[i1:i2] .+ env.τ ] .+ ϵ
end

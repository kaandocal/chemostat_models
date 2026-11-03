using StaticArrays
using Distributions

include("wrappednormal.jl")

const Point = SVector{2,Float64}
const NU_CUTOFF = 1e-3

@kwdef mutable struct RandomNuEnvironment
    ν_p::Float64
    intensity::Float64
    τ_ν::Float64
    D_ν::Float64
    size::Point
    dots::Vector{Pair{Float64,Point}} = []
    t::Float64 = 0.
end

function Base.show(io::IO, env::RandomNuEnvironment)
    print(io, "RandomNuEnvironment(ν_p=$(env.ν_p), intensity=$(env.intensity), τ_ν=$(env.τ_ν), D_ν=$(env.D_ν), size=$(env.size), dots=($(length(env.dots))), t=$(env.t))")
end

# Avoid environment duplication during `remake`
Base.deepcopy_internal(env::RandomNuEnvironment, ::IdDict) = env

function simulate_env!(env::RandomNuEnvironment, tmax)
    if iszero(env.t)
        env.t = env.τ_ν * log(NU_CUTOFF)
    end

    while env.t <= tmax
        λ = env.intensity * prod(env.size)
        dt = rand(Exponential(1 / λ))

        if env.t + dt >= tmax
            env.t = tmax
            break
        end

        env.t += dt
        pos = rand(2) .* env.size
        push!(env.dots, env.t => pos)
    end
end

function compute_nutrients(env::RandomNuEnvironment, t, x)
    if t > env.t
        @warn "Evaluating environment simulated until time $(env.t) at time $t"
    end

    t_min = t + log(NU_CUTOFF) * env.τ_ν
    i1 = searchsortedfirst(env.dots, t_min; by=first)
    i1 > length(env.dots) && return 0.
    i2 = searchsortedlast(env.dots, t; by=first)
    i2 < i1 && return 0.

    env.ν_p * sum(@view env.dots[i1:i2]) do (s, y)
        σ = sqrt(2 * env.D_ν * (t - s))
        exp(-(t - s) / env.τ_ν) * green_function_x(env, y, x, σ)
    end
end

function green_function_x(env::RandomNuEnvironment, x, y, σ)
    @fastmath prod(1:2) do j
        sf = 2pi / env.size[j]
        dist = WrappedNormal(x[j] * sf, σ * sf; tol=1e-3)
        sf * pdf(dist, y[j] * sf)
    end
end

(env::RandomNuEnvironment)(x, y, t) = compute_nutrients(env, t, Point(x, y))

function create_env(defaults; kwargs...)
    kwargs = combine_params(defaults, kwargs, [ :ν_p, :intensity, :τ_ν, :D_ν, :size ])
    RandomNuEnvironment(; kwargs...)
end

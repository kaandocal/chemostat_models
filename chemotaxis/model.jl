using Catalyst
using ModelingToolkit
using OrdinaryDiffEq
using Setfield
using Chemostats
import SciMLBase

const MTK = ModelingToolkit

include("../utils.jl")
include("env.jl")

const ROTDIFF_DT = 0.1

p = Dict(
    :v_max => 0.02,
    :σ_tum => 0.63,
    :τ_tum => 0.15,
    :D_rot => 0.19,

    :τ_m => 15,
    :a_0 => 0.57,
    :h => 6,
    :σ_m => 0.19,

    :K_on => 3,
    :K_off => 0.018,
    :g => 10,
    :β_0 => 0.86,

    :ν_p => 500,
    :intensity => 1e-6,
    :τ_ν => 1800 / log(2),
    :D_ν => 5.5e-4,
    :size => [ 20, 20 ],

    :λ_max => log(2) / (20 * 60),
    :V_d => 2,
    :K_ν => 1.5,
)

# Motility and growth
rn_mot = @network_component mot begin
    @variables x(t) = 0. y(t) = 0. θ(t) = 0. V(t) = V_d / 2 u(t) = 1e-6
    @parameters v_max σ_tum τ_tum D_rot λ_max K_ν V_d mod(t)::Int = 0 (env::$RandomNuEnvironment)(..) t0 = 0.

    @equations begin
        D(u) ~ -λs * mod - (1 - mod)
        D(x) ~ cos(θ) * v_max * mod
        D(y) ~ sin(θ) * v_max * mod
        D(θ) ~ 0                # Rotational diffusion is applied in rn_rotdiff

        ν ~ env(x, y, t + t0)
        D(V) ~ λ_max * ν / (ν + K_ν) * V
    end
end

# Switch between runs (mod = 1) and tumbles (mod = 0)
function tumble!(x, obs, ::Nothing, int)
    if x.mod == 1
        @set! x.mod = 0
        add_tstop!(int, int.t + obs.τ_tum)
        @set! x.u = obs.τ_tum
        @set! x.θ = mod2pi(x.θ + randn() * obs.σ_tum)
    else
        @set! x.mod = 1
        @set! x.u = -log(rand())
    end

    x
end

rn_tumble = let
    tumble_affect = MTK.ImperativeAffect(tumble!;
                                         modified = (; θ = :θ, u = :u, mod = :mod),
                                         observed = (; σ_tum = :σ_tum, τ_tum = :τ_tum))

    t = default_t()
    @variables u(t)
    tumble_event = MTK.SymbolicContinuousCallback(u ~ 0, tumble_affect)
    ReactionSystem(t; name=:tumble, continuous_events=tumble_event)
end

# Rotational diffusion and methylation noise
function rotdiff!(x, obs, ::Nothing, int)
    @set! x.θ = mod2pi(x.θ + randn() * sqrt(2 * ROTDIFF_DT * obs.D_rot))
    @set! x.m = x.m + randn() * obs.σ_m * sqrt(ROTDIFF_DT)
    x
end

rn_rotdiff = let
    rotdiff_affect = MTK.ImperativeAffect(rotdiff!;
                                          modified = (; θ = :θ, m = :m),
                                          observed = (; D_rot = :D_rot, σ_m = :σ_m))

    t = default_t()
    rotdiff_event = ROTDIFF_DT => rotdiff_affect
    ReactionSystem(t; name=:rotdiff, discrete_events=rotdiff_event)
end

# Division at volume V_d
function terminate_int!(x, obs, ::Nothing, int)
    terminate!(int)
    x
end

rn_div = let
    div_affect = MTK.ImperativeAffect(terminate_int!)

    t = default_t()
    @variables V(t)
    @parameters V_d
    div_event = MTK.SymbolicContinuousCallback(V ~ V_d, div_affect)
    ReactionSystem(t; name=:div, continuous_events=div_event)
end

# Receptor activity and adaptation
log_f(ν, K_on, K_off) = log((1 + ν/K_off) / (1 + ν/K_on))

rn_rec = @network_component rec begin
    @variables m_0(t) a(t) λs(t) m(t) = log((1 - a_0) / a_0)
    @parameters a_0 h τ_m K_on K_off g β_0 σ_m

    @equations begin
        D(m) ~ -(m - m_0) / τ_m

        m_0 ~ log((1 - a_0) / a_0) - h * log_f(ν, K_on, K_off)

        a ~ 1 / (1 + exp(m + h * log_f(ν, K_on, K_off)))

        λs ~ β_0 * exp(g * (a / a_0 - 1))
    end
end

function divide(int)
    env = int.ps[:env]

    map(1:2) do _
        u0 = [ :V => int[:V] / 2,
               :x => mod(int[:x], env.size[1]),
               :y => mod(int[:y], env.size[2]) ]

        p = [ :t0 => int.ps[:t0] + int.t ]

        (; u0, p)
    end
end

rn = complete(extend(rn_mot, [ rn_rec, rn_tumble, rn_div, rn_rotdiff ]))
osys = structural_simplify(convert(ODESystem, rn))

function create_prob(rn; maxiters=1e9, save_mode=:none, save_idxs=nothing, alg=BS3(), kwargs...)
    env = create_env(p; kwargs...)
    u0, ps = problem_inputs(rn, p, kwargs, env)

    prob = ODEProblem{true,SciMLBase.FullSpecialize}(rn, u0, (0, 1.), ps;
                                                     alg, maxiters, save_idxs, save_kwargs(save_mode)...)

    @set! prob.kwargs[:callback].continuous_callbacks[1].interp_points = 0
    @set! prob.kwargs[:callback].continuous_callbacks[1].abstol = 1e-3
    if save_mode != :full
        @set! prob.kwargs[:callback].continuous_callbacks[1].save_positions = (false, false)
    end

    prob
end

# Need to manually copy the timed callback (hack)
function Chemostats.reinit(old_int, prob; t0, p=nothing, u0=nothing)
    int = Chemostats.copy_integrator(prob, old_int.alg, t0)

    sys = prob.f.sys
    rotdiff_cb = MTK.generate_timed_callback(rn_rotdiff.discrete_events[1], sys, unknowns(sys), parameters(sys))
    opts = int.opts
    int.opts = @set opts.callback.discrete_callbacks[1] = rotdiff_cb

    Chemostats.apply_reinit_overrides!(int, old_int; p, u0)
end

using Catalyst
using ModelingToolkit
using Distributions
using OrdinaryDiffEq
using Setfield
using Chemostats
using Chemostats: savevalues!, CellState, get_state, set_state!, finalise!
import SciMLBase

const MTK = ModelingToolkit

include("../utils.jl")
include("env.jl")

p = let μM_to_N = 1.4e-15 * 6.022e23 * 1e-6
    Dict(
        :λ_0 => log(2) / 45,
        :V_d => 2,

        :D_T => 0.015,
        :α => 2,
        :K_A => 10,

        :k_R => 0.2,
        :k_A => 0.2,

        :K_RRO1 => 1e-6 * μM_to_N,
        :K_RRO2 => 5e-7 * μM_to_N,
        :K_RTc => 1e-3,
        :X_Tc => 1,
        :X_A => 15,

        :C => 100,
        :freq => 1/100.,
        :τ => 20.,
        :cv_wt => 0.0,

        :ρ => 0.76,
        :ϕ_R_max => 0.55,
        :ϕ_R_min => 0.07,
        :κ_t0 => 4.5 / 60,
    )
end

div2(x) = div(x + 1e-3, 2)
@register_symbolic div2(x)

# Growth mechanics mostly follow Scott and Hwa 2011
rn_grw = @network_component grw begin
    @variables V(t) = 1 κ_t(t) λ(t) Tc(t) ϕ_R(t) κ_n(t)
    @species A(t) = 0
    @parameters λ_0 κ_t0 ϕ_R_min ϕ_R_max ρ V_d X_Tc X_A

    @equations begin
        D(V) ~ λ * V

        κ_t ~ κ_t0 * 1 / (1 + Tc/X_Tc) * 1 / (1 + A/(V * X_A))
        ϕ_R ~ ϕ_R_min + (ϕ_R_max - ϕ_R_min) * κ_n / (κ_n + κ_t)

        κ_n ~ κ_t0 * λ_0 * ρ / ((ϕ_R_max - ϕ_R_min) * κ_t0 - λ_0 * ρ)
        λ ~ κ_t/ρ * (ϕ_R - ϕ_R_min)

        f ~ (λ/λ_0)^2
    end
end

# Tetracycline uptake, TetR binding and tetR/tetA transcription
rn_ode = @network_component ode begin
    @variables Tc(t) = 0 V(t) = 1 RR(t) RRTc(t) Tc_f(t) p_Rf(t) tx_1(t) tx_12(t) O_Z(t) O_ff(t) O_bf(t) O_fb(t) O_bb(t) u_R(t) = 1e-6 u_A(t) = 1. f(t)
    @species A(t) = 0 R(t) = 1

    @parameters k_R k_A α K_A D_T K_RRO1 K_RRO2 K_RTc (env::$RandomTcEnvironment)(..) t0=0. μM_to_N=1e-15 * 6.022e23 * 1e-6

    @equations begin
        D(u_R) ~ -f * V * k_R * (tx_1 + tx_12) / 2
        D(u_A) ~ -f * V * k_A * tx_12
        D(Tc) ~ D_T * Tc_E - (D_T + α * A / (abs(Tc) + K_A) + λ) * Tc

        Tc_E ~ env(t0 + t)

        RR ~ div2(R)
        RRTc ~ (RR + (Tc + K_RTc) * V * μM_to_N - sqrt((RR - (Tc + K_RTc) * V * μM_to_N)^2 + 4 * RR * K_RTc * V * μM_to_N)) / 2
        p_Rf ~ 1 - RRTc / (RR + 1e-6)
        Tc_f ~ Tc - RRTc / (V * μM_to_N)

        O_ff ~ 1
        O_bf ~ RR / K_RRO1
        O_fb ~ RR / K_RRO2
        O_bb ~ RR * (RR - 1) / (K_RRO2 * K_RRO1)
        O_Z ~ O_ff + O_bf + O_fb + O_bb

        tx_1 ~ (O_ff + O_fb + (1 - p_Rf) * (O_bf + O_bb)) / O_Z
        tx_12 ~ (O_ff + (1 - p_Rf) * (O_bf + O_fb) + (1 - p_Rf)^2 * O_bb) / O_Z
    end
end

# Stochastic production of TetR and TetA molecules
function tx_R!(x, obs, ::Nothing, int)
    @set! x.R += 1
    @set! x.u_R = -log(rand())
    x
end

function tx_A!(x, obs, ::Nothing, int)
    @set! x.A += 1
    @set! x.u_A = -log(rand())
    x
end

rn_tx = let
    affect_R = MTK.ImperativeAffect(tx_R!; modified = (; u_R = :u_R, R = :R), observed = (; ))
    affect_A = MTK.ImperativeAffect(tx_A!; modified = (; u_A = :u_A, A = :A), observed = (; ))

    t = default_t()
    @variables u_R(t) u_A(t)
    event_R = MTK.SymbolicContinuousCallback(u_R ~ 0, affect_R)
    event_A = MTK.SymbolicContinuousCallback(u_A ~ 0, affect_A)
    ReactionSystem(t; name=:tx, continuous_events=[ event_R, event_A ])
end

# Binomial partitioning of TetR and TetA at division.
# If there are at least two TetR dimers, each daughter receives at least one.
function divide(int)
    dimers = round(Int, div2(int[:R]))
    monomers = round(Int, int[:R] - 2 * dimers)

    R1 = if dimers >= 2
        2 + 2 * rand(Binomial(dimers - 2, 0.5))
    else
        2 * rand(Binomial(dimers, 0.5))
    end + rand(Binomial(monomers, 0.5))

    A1 = rand(Binomial(round(Int, int[:A]), 0.5))

    Rs = (R1, round(Int, int[:R] - R1))
    As = (A1, round(Int, int[:A] - A1))

    map(1:2) do i
        u0 = [ :V => int[:V] / 2,
               :R => Rs[i],
               :A => As[i],
               :u_A => -log(rand()),
               :u_R => -log(rand()) ]

        p = [ :t0 => int.ps[:t0] + int.t ]
        (; u0, p)
    end
end

rn = complete(extend(rn_ode, [ rn_tx, rn_grw ]))
osys = structural_simplify(convert(ODESystem, rn))

# Division at volume V_d
const getV = MTK.getsym(osys, osys.V)
const getVd = MTK.getp(osys, osys.V_d)
cb_cond(u, t, int) = getV(u) - getVd(int)
cb_div = ContinuousCallback(cb_cond, terminate!; save_positions=(true, false))

function create_prob(rn; tmax=1., maxiters=1e9, alg=BS3(), save_mode=:none, save_idxs=nothing, kwargs...)
    env = create_env(p; kwargs...)
    u0, ps = problem_inputs(rn, p, kwargs, env)

    prob = ODEProblem{true,SciMLBase.FullSpecialize}(rn, u0, (0, tmax), ps; callback=cb_div, dtmax=1.,
                                                     alg, maxiters, save_idxs, save_kwargs(save_mode)...)

    for i in eachindex(prob.kwargs[:callback].continuous_callbacks)
        @set! prob.kwargs[:callback].continuous_callbacks[i].interp_points = 0
        @set! prob.kwargs[:callback].continuous_callbacks[i].abstol = 1e-6

        if save_mode != :full
            @set! prob.kwargs[:callback].continuous_callbacks[i].save_positions = (false, false)
        end
    end

    prob
end

# Override of Chemostats.step! for DECell that adds the discontinuities of the
# external tetracycline concentration as tstops
function Chemostats.step!(cell::DECell, dt, p)
    @assert !isnothing(cell.int)

    isnothing(p) || @warn "Parameter arguments to DECell are ignored" maxlog=1

    if get_state(cell) != CellState.Alive
        @warn "Tried to simulate cell in state $(get_state(cell))"
        return
    end

    for t in get_tstops(cell.int.ps[:env]; tspan=(cell.int.t, cell.int.t + dt))
        t > cell.int.t && SciMLBase.add_tstop!(cell.int, t)
    end

    step!(cell.int, dt, true)
    savevalues!(cell)

    if SciMLBase.done(cell.int)
        if !SciMLBase.successful_retcode(cell.int.sol)
            @warn "Cell solver errored, removing cell..."
            set_state!(cell, CellState.Killed)
            finalise!(cell)
        else
            set_state!(cell, CellState.EndOfLife)
        end
    end
end

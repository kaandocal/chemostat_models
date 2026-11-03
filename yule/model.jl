# The Yule process
# Here, every cell divides after an exponentially distributed time τ ~ Exp(1),
# so the population grows with rate Λ = 1.
using Chemostats
using OrdinaryDiffEq

# Each cell is represented by its remaining time to division `u`
countdown(u, p, t) = -one(u)

cb_div = Chemostats.DivideCallback((u, t, int) -> u)

prob = ODEProblem(countdown, -log(rand()), (0., Inf); callback=cb_div, alg=Tsit5())

# Both daughters get new exponentially distributed times to division
divide(int) = [ (u0 = -log(rand()), p = nothing) for _ in 1:2 ]

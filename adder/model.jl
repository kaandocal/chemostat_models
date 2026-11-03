# The adder model
# Cells grow exponentially and divide once they have added a volume 
# Δ ~ Gamma(5, 0.2) (mean 1) to their birth volume. Each daughter inherits a
# uniformly distributed fraction of the parent's volume. 
# The population growth rate is equal to the single-cell growth rate, Λ = 1.
using Chemostats
using Distributions
using OrdinaryDiffEq

# Each cell is represented by its volume `u`, which grows exponentially
grow(u, p, t) = u

# A cell divides once it reaches its division volume `V_d`, stored as a parameter
cb_div = Chemostats.DivideCallback((u, t, int) -> u - int.p.V_d)

const dist_Δ = Gamma(5, 0.2)
const dist_frac = Beta(1)

prob = ODEProblem(grow, 1., (0., Inf), (V_d = 2.,); callback=cb_div, alg=Tsit5())

function divide(int)
    frac = rand(dist_frac)
    V1 = frac * int.u
    V2 = (1 - frac) * int.u

    # Each daughter samples its own division volume
    [ (u0 = V1, p = (V_d = V1 + rand(dist_Δ),)),
      (u0 = V2, p = (V_d = V2 + rand(dist_Δ),)) ]
end

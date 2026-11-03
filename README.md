# Probing bacterial fitness in virtual chemostats

This repository contains the models for the paper:

> Kaan Öcal, Lars Maas, Michael P.H. Stumpf. **"Probing bacterial fitness in virtual chemostats."** [*bioRxiv*](https://www.biorxiv.org/content/10.64898/2026.10.31.763286v1) (2026).

The population simulations use [Chemostats.jl](https://github.com/kaandocal/Chemostats.jl).

## Models

* Linear birth, or Yule process (folder `yule`)
* Adder model (folder `adder`)
* Tetracycline model (folder `tetracycline`)
* Chemotaxis model (folder `chemotaxis`)

The linear birth and adder models can also be found in [Chemostats.jl](https://github.com/kaandocal/Chemostats.jl).

**Note:** The chemotaxis model is currently *very* slow.

## Usage

The code was written for Julia 1.12. To install all dependencies, run
```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
in this folder. Each model can then be loaded with `include`. For the Yule and adder models, this defines a problem `prob` and a division function `divide`:
```julia
using Chemostats
include("adder/model.jl")

chem = Chemostat([ DECell(prob, prob.kwargs[:alg], divide) ])
Chemostats.simulate!(chem, 1000., Chemostats.Strict(10))

est_Λ(chem)     # should be close to 1
```

The chemotaxis and tetracycline models are defined using [Catalyst.jl](https://github.com/SciML/Catalyst.jl), and can be instantiated using `create_prob`.
In both models, the environment has to be simulated ahead of time. To avoid floating-point precision issues for long runs, set `reset_t=true`:
```julia
using Chemostats
include("chemotaxis/model.jl")

prob = create_prob(osys; τ_m = 10)
env = prob.ps[:env]

chem = Chemostat([ DECell(prob, prob.kwargs[:alg], divide; reset_t=true) ])

for t in 1000:1000:10_000
    simulate_env!(env, t)
    Chemostats.simulate!(chem, t, Chemostats.Strict(10))
end

est_Λ(chem)
```


"""
    combine_params(defaults, overrides, names)

Merge `overrides` into `defaults` and return a `Dict` of the entries whose keys are in `names`.
"""
function combine_params(defaults, overrides, names)
    merged = merge(defaults, overrides)
    Dict(k => v for (k, v) in merged if k in names)
end

"""
    problem_inputs(rn, defaults, overrides, env)

Return the initial conditions and parameters of the system `rn`, taking values from `overrides`
before `defaults`. The environment `env` is added as the parameter `env`. Warns about entries
of `overrides` that are used neither by `rn` nor by `env`.
"""
function problem_inputs(rn, defaults, overrides, env)
    u0_names = [ unk.f.name for unk in unknowns(rn) ]
    p_names = [ par.name for par in parameters(rn) if hasproperty(par, :name) ]

    u0 = combine_params(defaults, overrides, u0_names)
    ps = combine_params(defaults, overrides, p_names)

    unused = setdiff(keys(overrides), u0_names, keys(ps), propertynames(env))
    isempty(unused) || @warn "The following parameters were not used: $unused"

    u0, [ collect(ps); :env => env ]
end

"""
    save_kwargs(save_mode)

Solver keyword arguments for `save_mode`, which is `:none` (only save the start and end of
each cell), `:full` (save every step) or a collection of time points.
"""
function save_kwargs(save_mode)
    if save_mode == :none
        (; save_everystep = false, dense = false)
    elseif save_mode == :full
        (;)
    else
        (; saveat = save_mode, save_start = true, save_end = true)
    end
end

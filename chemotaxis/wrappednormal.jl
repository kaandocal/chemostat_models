# Wrapped normal distribution, adapted from code by Seth Axen
using Distributions
using LogExpFunctions

abstract type WrappedNormalDensityMethod end
struct WrappedMethod <: WrappedNormalDensityMethod end
struct JacobiMethod <: WrappedNormalDensityMethod end

"""
    WrappedNormal(μ, σ; [tol])

The wrapped normal distribution with mean `μ` and standard deviation `σ` of the underlying
normal distribution.

`tol` specifies the absolute error of the approximation to the density function and is used
to choose between the wrapped and the Jacobi theta series and the number of terms used.

Reference: G. Kurz, I. Gilitschenski, U. D. Hanebeck. "Efficient evaluation of the probability
density function of a wrapped normal distribution," 2014 Sensor Data Fusion: Trends, Solutions,
Applications (SDF), pp1-5. doi: 10.1109/SDF.2014.6954713
"""
struct WrappedNormal{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    method::WrappedNormalDensityMethod
    nterms::Int
end

function WrappedNormal(μ::T, σ::T; tol=eps(T)) where {T<:Real}
    nw = _num_terms_wrappednormal_wrapped(μ, σ, tol)
    nj = _num_terms_wrappednormal_jacobi(μ, σ, tol)
    if nw < nj
        return WrappedNormal{T}(μ, σ, WrappedMethod(), ceil(Int, nw))
    else
        return WrappedNormal{T}(μ, σ, JacobiMethod(), ceil(Int, nj))
    end
end
WrappedNormal(μ::Real, σ::Real; kwargs...) = WrappedNormal(promote(μ, σ)...; kwargs...)

function Distributions.logpdf(d::WrappedNormal, x::Real)
    z = rem2pi(x - d.μ, RoundNearest)
    if d.method isa WrappedMethod
        return _logpdf_wrappednormal_wrapped(z, d.σ, d.nterms)
    else
        return _logpdf_wrappednormal_jacobi(z, d.σ, d.nterms)
    end
end

Distributions.pdf(d::WrappedNormal, x::Real) = exp(logpdf(d, x))

function _num_terms_wrappednormal_wrapped(μ, σ, ϵ)
    T = Base.promote_eltype(μ, σ, ϵ)
    return 1 + ((σ / sqrt(2)) / π) * max(1, sqrt(-log(2) - log(ϵ) - (3//2) * T(log(π))))
end

function _num_terms_wrappednormal_jacobi(μ, σ, ϵ)
    T = Base.promote_eltype(μ, σ, ϵ)
    return max(1, sqrt(-T(log(2)) / 2 - log(π) - log(σ) - log(ϵ))) / σ
end

function _logpdf_wrappednormal_wrapped(z, σ, n)
    T = Base.promote_eltype(z, σ)
    logc = -T(log(2π)) / 2 - log(σ)
    log_series = logsumexp((muladd(k, 2π, z) / σ)^2 / -2 for k in (-n):n)
    return log_series + logc
end

function _logpdf_wrappednormal_jacobi(z, σ, n)
    ρ = exp(-σ^2 / 2)
    log_series = log1p(2 * sum(ρ^(k^2) * cos(k * z) for k in 1:n))
    return log_series - log(2π)
end

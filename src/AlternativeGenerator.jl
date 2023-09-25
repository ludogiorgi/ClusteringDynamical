module AlternativeGenerator
using MarkovChainHammer.Utils: autocovariance
using ProgressBars, LsqFit, LinearAlgebra

export fit_autocorrelation, alternative_generator

model1(x, p) = @. exp(x * p[1])
model2(x, p) = @. exp(x * p[1]) * cos(p[2] * x)

function fit_autocorrelation(Λ, W, X, dt, fitting_window_factor; indices=2:length(Λ))
    λlist = ComplexF64[]
    for index in indices
        g = W[end-index+1, :]
        λ = Λ[end-index+1]
        Nmax = ceil(Int, -1 / real(dt * λ))
        g_timeseries = [real(g[x]) for x in X]
        g_autocor = autocovariance(g_timeseries; timesteps= fitting_window_factor * Nmax)
        g_autocor = g_autocor / g_autocor[1]
        if abs(imag(λ)) > sqrt(eps(1.0))
            modeli = model2
            pguess = [real(λ), imag(λ)]
        else
            modeli = model1
            pguess = [real(λ)]
        end
        xdata = collect(range(0, fitting_window_factor * Nmax - 1, length=length(g_autocor))) .* dt
        ydata = g_autocor
        tmp = curve_fit(modeli, xdata, ydata, pguess)
        tmp.param
        if abs(imag(λ)) > sqrt(eps(1.0))
            push!(λlist, tmp.param[1] + tmp.param[2] * im)
        else
            push!(λlist, tmp.param[1] + 0 * im)
        end
    end
    Λ̃ = [reverse(λlist)..., Λ[end]]
    ll = eigenvalue_correction(Λ̃)
    return ll
end

function eigenvalue_correction(Λ)
    i = 1
    Λ̃ = copy(Λ)
    while i <= length(Λ̃)
        if abs(imag(Λ̃[i])) > sqrt(eps(1.0))
            Λ̃[i+1] = conj(Λ̃[i])
            i += 2
        else
            i += 1
        end
    end
    return Λ̃
end

alternative_generator(Q::AbstractArray, X, dt, fitting_window_factor) = alternative_generator(eigen(Q), X, dt, fitting_window_factor)
function alternative_generator(E::Eigen, X, dt, fitting_window_factor)
    Λ, V = E
    W = inv(V)
    Λ̃ = fit_autocorrelation(Λ, W, X, dt, fitting_window_factor)
    tmp = V * Diagonal(Λ̃) * W
    Q̃ = real.(tmp)
    if norm(imag.(tmp)) / norm(Q̃) > sqrt(eps(1.0))
        warn("Warning: eigenvalue correction failed")
    end
    return Q̃
end

end # module AlternativeGenerator

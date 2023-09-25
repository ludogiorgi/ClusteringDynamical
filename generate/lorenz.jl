function lorenz(x)
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    return [10.0 * (x2 - x1), x1 * (28.0 - x3) - x2, x1 * x2 - 8.0 / 3.0 * x3]
end

function lorenz_data(timesteps, Δt, res, ϵ)
    x_f = zeros(3, timesteps)
    x_f[:, 1] = [14.0, 15.0, 27.0]
    step = RungeKutta4(3)
    for i in ProgressBar(2:timesteps)
        xOld = x_f[:, i-1]
        step(lorenz, xOld, Δt)
        𝒩 = randn(3)
        @inbounds @. x_f[:, i] = step.xⁿ⁺¹ + ϵ * sqrt(Δt) * 𝒩
    end
    L2 = floor(Int, timesteps / 10)
    Dt = Δt * res
    x = zeros(3, L2)
    for i in 1:L2
        @inbounds x[:, i] .= x_f[:, res*i]
    end

    return x, Dt
end

lorenz_data(; timesteps=10^7, Δt=0.005, res=1, ϵ=0.0) = lorenz_data(timesteps, Δt, res, ϵ)
x, dt = lorenz_data(timesteps=10^6)

@info "saving data for Lorenz"
hfile = h5open(pwd() * "/data/lorenz.hdf5", "w")
hfile["x"] = x
hfile["dt"] = dt
close(hfile)
@info "done saving data for Lorenz"
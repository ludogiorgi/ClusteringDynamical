@info "running potential well case"
function U(v)
    Ax = 1.05
    Ay = 1.1
    Az = 1.15
    x = v[1]
    y = v[2]
    z = v[3]
    return (x - Ax)^2 * (x + Ax)^2 + (y - Ay)^2 * (y + Ay)^2 + (z - Az)^2 * (z + Az)^2
end

∇U(x) = gradient(Enzyme.Reverse, U, x)
force(x) = -∇U(x)

function potential_data(timesteps, Δt, res, ϵ)
    x_f = zeros(3, timesteps)
    step = RungeKutta4(3)
    for i in ProgressBar(2:timesteps)
        xOld = x_f[:, i-1]
        step(force, xOld, Δt)
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
potential_data(; timesteps=1000000, Δt=0.03, res=10, ϵ=0.75) = potential_data(timesteps, Δt, res, ϵ)
x, dt = potential_data(timesteps=10^6)

@info "saving data for potential well"
hfile = h5open(pwd() * "/data/potential_well.hdf5", "w")
hfile["x"] = x
hfile["dt"] = dt
close(hfile)
@info "done saving data for potential well"
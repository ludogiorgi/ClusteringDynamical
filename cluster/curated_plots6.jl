using MultiDimensionalClustering.AlternativeGenerator
import MarkovChainHammer.TransitionMatrix: holding_times
Q̃ = mean(BayesianGenerator(X_LN; dt = dt))
Qred = alternative_generator(Q̃, X_LN, dt, 100) 
##
step = 100
exp(Q̃ * step * dt)
exp(Qred * step * dt)
perron_frobenius(X_LN; step = step)
ht = holding_times(X_LN; dt=dt)
##
timepf = []
timeexpqt = []
timeexpqt2 = []
steps = 1:10:40000
for step in ProgressBar(steps)
    pij = perron_frobenius(X_LN; step = step)
    push!(timepf, pij)
    push!(timeexpqt, exp(Q̃ * step * dt))
    push!(timeexpqt2, exp(Qred * step * dt))
end
##
lw = 10
fig = Figure(resolution = (2000, 1500))
m = length(union(X_LN))
labelsize = 40
yaxis_names = ["P₁₁", "P₂₁", "P₃₁", "P₁₂", "P₂₂", "P₃₂", "P₁₃", "P₂₃", "P₃₃"] .* "(t)"
axis_options = (; xlabel="time", xgridstyle=:dash, ygridstyle=:dash, ygridwidth=5, xgridwidth=5, titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
axs = []
τ = τs[3] * dt
title = "Newton-Leipnik Reduced Order Model Matrix Entries"
ga = fig[1,1] = GridLayout()
ax = Axis(fig[1, 1]; title = title, titlesize = 50, titlegap = 50,
leftspinevisible = false,
rightspinevisible = false,
bottomspinevisible = false,
topspinevisible = false,)
hidedecorations!(ax)
for i in 1:m^2
    ii = (i-1)%m+1 
    jj = div((i-1), m) + 1
    # println(ii, jj)
    ax  = Axis(ga[ii, jj]; ylabel = yaxis_names[i], axis_options...)
    push!(axs, ax)
    pf = [timepf[j][i] for j in eachindex(timepf)]
    eqt = [timeexpqt[j][i] for j in eachindex(timepf)]
    eqt2 = [timeexpqt2[j][i] for j in eachindex(timepf)]
    GLMakie.lines!(ax, steps * dt, pf, color = (:black, 0.5), linewidth = lw, label = "Pᵢⱼ(t)")
    GLMakie.lines!(ax, steps * dt, eqt, color = (:blue, 0.5), linewidth = lw, label = "exp(Q t)")
    GLMakie.lines!(ax, steps * dt, eqt2, color = (:red, 0.5), linewidth = lw, label = "exp(Qₚ t)")
    GLMakie.ylims!(ax, (0,1))
end
axislegend(axs[1]; position = :rt, labelsize = labelsize)
display(fig)
##
save("figure/NewtonReducedModel.png", fig)

##
Λs = []
Vs = []
for i in ProgressBar(eachindex(timepf))
    Λ̃, V =  eigen(timepf[i])
    push!(Λs, log.(abs.(Λ̃))/(dt * steps[i]))
    push!(Vs, V)
end

l1 = [Λ[1] for Λ in Λs]
l2 = [Λ[2] for Λ in Λs]
l3 = [Λ[3] for Λ in Λs]
Λp = eigvals(Qred)
Λ, V = eigen(Q̃)
##
fig = Figure()
ax = Axis(fig[1,1])
lw = 10
GLMakie.lines!(ax, l2, color = (:black, 0.15), linewidth = lw)
GLMakie.hlines!(ax, Λ[2], color = (:red, 0.5), linewidth = lw)
GLMakie.hlines!(ax, Λp[2], color = (:blue, 0.5), linewidth = lw)
ax = Axis(fig[1,2])
GLMakie.lines!(ax, l1, color = (:black, 0.15), linewidth = lw)
GLMakie.hlines!(ax, Λ[1], color = (:red, 0.5), linewidth = lw)
GLMakie.hlines!(ax, Λp[1], color = (:blue, 0.5), linewidth = lw)
display(fig)
##
Qpp = inv(V) * Diagonal([l1[end], l2[end], l3[end]]) * V

##
fig = Figure(resolution = (2000, 1500))
m = length(union(X_LN))
labelsize = 40
yaxis_names = "P" .* ["₁₁", "₂₁", "₃₁", "₁₂", "₂₂", "₃₂", "₁₃", "₂₃", "₃₃"] .* "(t)"
axis_options = (; xlabel="time", xgridstyle=:dash, ygridstyle=:dash, ygridwidth=5, xgridwidth=5, titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
axs = []
τ = τs[3] * dt
title = "Newton-Leipnik Reduced Order Model Eigenvector Matrix Entries"
ga = fig[1,1] = GridLayout()
ax = Axis(fig[1, 1]; title = title, titlesize = 50, titlegap = 50,
leftspinevisible = false,
rightspinevisible = false,
bottomspinevisible = false,
topspinevisible = false,)
hidedecorations!(ax)
pfV = []
for i in eachindex(timepf)
    l, v = eigen(timepf[i])
    for j in 1:3
        v[:, j] .*= sign(real(v[1, j]))
    end
    push!(pfV, real.(v))
end
##
for i in 1:m^2
    ii = (i-1)%m+1 
    jj = div((i-1), m) + 1
    # println(ii, jj)
    ax  = Axis(ga[ii, jj]; ylabel = yaxis_names[i], axis_options...)
    push!(axs, ax)
    pf = [pfV[j][i] for j in eachindex(timepf)]
    GLMakie.lines!(ax, steps * dt, pf, color = (:black, 0.5), linewidth = lw, label = "Vᵢⱼ(t)")
    GLMakie.ylims!(ax, (-1,1))
end

display(fig)
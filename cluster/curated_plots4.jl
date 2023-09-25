using HDF5, GLMakie, FFTW, MultiDimensionalClustering, Random, ProgressBars
using MultiDimensionalClustering.CommunityDetection, LinearAlgebra, Statistics
using ParallelKMeans
using MarkovChainHammer.BayesianMatrix
using MarkovChainHammer.Trajectory: generate
directory = pwd() 
hfile = h5open(directory * "/data/ks_medium_res3.hdf5")
Random.seed!(12345)
x = read(hfile["u"])
close(hfile)
##
fig = Figure(resolution=(300, 1300))
ax = Axis(fig[1, 1]; xlabel="x", ylabel="t")
skip = 20
indices = 1:skip:60000
ts = (indices .- 1) .* skip
xs = collect(1:64)
GLMakie.heatmap!(ax, x[:, indices], colormap=:balance, colorrange=(-2.5, 2.5), interpolate=true)
display(fig)
##
# [norm(x[:, i]) for i in 1:size(x)[2]]
# clusters = kmeans(x[:, 1:10:end], 100;  max_iters=10^3)
##
X = copy(x)
##
function split(X)
    numstates = 2
    r0 = kmeans(X, numstates; max_iters=10000)
    child_0 = (r0.assignments .== 1)
    child_1 = (!).(child_0)
    children = [view(X, :, child_0), view(X, :, child_1)]
    return r0.centers, children
end
level_global_indices(level) = 2^(level-1):2^level-1
##
levels = 12
parent_views = []
centers_list = Vector{Vector{Float64}}[]
push!(parent_views, X)
## Level 1
centers, children = split(X)
push!(centers_list, [centers[:, 1], centers[:, 2]])
push!(parent_views, children[1])
push!(parent_views, children[2])
## Levels 2 through levels
for level in ProgressBar(2:levels)
    for parent_global_index in level_global_indices(level)
        centers, children = split(parent_views[parent_global_index])
        push!(centers_list, [centers[:, 1], centers[:, 2]])
        push!(parent_views, children[1])
        push!(parent_views, children[2])
    end
end
@info "done with k-means"
##
struct StateTreeEmbedding{S, T}
    markov_states::S
    levels::T
end
function (embedding::StateTreeEmbedding)(current_state)
    global_index = 1 
    for level in 1:embedding.levels
        new_index = argmin([norm(current_state - markov_state) for markov_state in embedding.markov_states[global_index]])
        global_index = child_global_index(new_index, global_index)
    end
    return local_index(global_index, embedding.levels)
end

# assumes binary tree
local_index(global_index, levels) = global_index - 2^levels + 1 # markov index from [1, 2^levels]
# parent local index is markov_index(global_index, levels-1)
# child local index is 2*markov_index(global_index, levels-1) + new_index - 1
# global index is 2^levels + 1 + child local index
child_global_index(new_index, global_parent_index, level) = (2 * (local_index(global_parent_index, level - 1)-1) + new_index - 1) + 2^(level) 
# simplified:
child_global_index(new_index, global_parent_index) = 2 * global_parent_index + new_index - 1 
# global_indices per level
level_global_indices(level) = 2^(level-1):2^level-1
parent_global_index(child_index) = div(child_index, 2) # both global
centers_matrix = zeros(length(centers_list[1][1]), length(centers_list[1]), length(centers_list))
for i in eachindex(centers_list)
    centers_matrix[:, :, i] = hcat(centers_list[i]...)
end
centers_list = [[centers_matrix[:,1, i], centers_matrix[:,2, i]] for i in 1:size(centers_matrix)[3]]
# constructing embedding with 2^levels number of states
# note that we can also choose a number less than levels
levels = 11
embedding = StateTreeEmbedding(centers_list, levels)
##
@info "applying embedding"
markov_embedding = [embedding(x[:,i]) for i in 1:size(x)[2]]
@info "done with embedding"
##
Q = mean(BayesianGenerator(markov_embedding))
Λ, V =  eigen(Q)
W = inv(V)
koopman1 = W[end-1, :]
koopman2 = W[end-2, :]
koopman3 = W[end-3, :]
koopman4 = W[end-4, :]
##
k1_t = [real(koopman1[embedding_index]) for embedding_index in markov_embedding]
k2_t = [real(koopman2[embedding_index]) for embedding_index in markov_embedding]
k3_t = [real(koopman3[embedding_index]) for embedding_index in markov_embedding]
k4_t = [real(koopman4[embedding_index]) for embedding_index in markov_embedding]
##
L = 34 
dt =  0.017 * skip 
shift = 10000
inds = 1+shift:10000+shift
ts = collect(0:length(inds)-1) .* dt
xs = collect(0:63) / 64 * L 
Δx = xs[2] - xs[1]
energy = [sum(x[:, i] .^2 .* Δx) for i in 1:size(x)[2]] 
##
fig = Figure()
ax1 = Axis(fig[1,1])
scatter!(ax1, k1_t[inds])
ax2 = Axis(fig[1,2])
scatter!(ax2, energy[inds], color = k1_t[inds], colormap = :balance)
ax21 = Axis(fig[2,1])
scatter!(ax21, k2_t[inds])
ax22 = Axis(fig[2,2])
scatter!(ax22, k3_t[inds])
ax3 = Axis(fig[1:2, 3])
GLMakie.heatmap!(ax3, x[:, inds], colormap = :balance, colorrange = (-2,2))
display(fig)
##
τs = reverse((-1 ./ real.(Λ[end-20:end]))[1:end-1])
##
P1 = exp(Q * τs[1])
P2 = exp(Q)
##
@info "applying LN1"
q_min = 0.0
F, G, H, PI = leicht_newman_with_tree(P1, q_min)
##
@info "applying LN2"
q_min = 0.0
F2, G2, H2, PI2 = leicht_newman_with_tree(P2, q_min)
##
X_LN = classes_timeseries(F, markov_embedding) # 4 groups
X_LN = classes_timeseries(F2, markov_embedding) # 18 groups
##
p = real.(V[:,end] ./ sum(V[:, end]))
##
shift = 10000
skip = 5
inds = 1+shift:skip:12000+shift
colormap = :glasbey_hv_n256

L = 34 
dt =  0.017 * skip 
ts = collect(0:length(inds)-1) .* dt
xs = collect(0:63) / 64 * L 
Δx = xs[2] - xs[1]
energy = [sum(x[:, i] .^2 .* Δx) for i in 1:size(x)[2]] 
##

firstline = ts[230]
secondline = ts[700]
thirdline = ts[1825]
fourthline = ts[2100]
opacity = 0.5
linewidth = 10
linecolor = :yellow
linecolor2 = :black
labelsize = 40

axis_options = (; titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
axis2_options = (; titlesize = labelsize, xlabel = "space", ylabel = "time", ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
fig = Figure(resolution = (2120, 1432))
ax1 = Axis(fig[1,2]; title = "Coarse-Grained Cluster Dynamics", xlabel = "time", axis_options..., ylabel = "Cluster Label")
scatter!(ax1, ts, X_LN[inds]; color = X_LN[inds], colormap = colormap)
ax1.yticks = ([2, 4, 6, 8, 10, 12, 14, 16, 18])# ([-75, -50, -25, 0, 25, 50, 75], ["75S", "50S", "25S", "0", "25N", "50N", "75N"])
vlines!(ax1, firstline, linewidth = 10, color = (linecolor2, opacity))
vlines!(ax1, secondline, linewidth = 10, color = (linecolor2, opacity))
vlines!(ax1, thirdline, linewidth = 10, color = (linecolor2, opacity))
vlines!(ax1, fourthline, linewidth = 10, color = (linecolor2, opacity))
GLMakie.ylims!(0, 20)
ax2 = Axis(fig[2,2]; title = "Energy Dynamics", xlabel = "time", axis_options..., ylabel = "Energy")
lines!(ax2, ts, energy[inds], color = X_LN[inds], colormap = colormap)
vlines!(ax2, firstline, linewidth = 10, color = (linecolor2, opacity))
vlines!(ax2, secondline, linewidth = 10, color = (linecolor2, opacity))
vlines!(ax2, thirdline, linewidth = 10, color = (linecolor2, opacity))
vlines!(ax2, fourthline, linewidth = 10, color = (linecolor2, opacity))
#=
ax21 = Axis(fig[2,1])
scatter!(ax21, k2_t[inds])
ax22 = Axis(fig[2,2])
scatter!(ax22, k3_t[inds])
=#
ax3 = Axis(fig[1:2, 1]; title = "Space-Time Dynamics", axis2_options...)
GLMakie.heatmap!(ax3, xs, ts, x[:, inds], colormap = :balance, colorrange = (-3,3), interpolate = true)
hlines!(ax3, firstline, linewidth = linewidth, color = linecolor)
hlines!(ax3, secondline, linewidth = linewidth, color = linecolor)
hlines!(ax3, thirdline, linewidth = linewidth, color = linecolor)
hlines!(ax3, fourthline, linewidth = linewidth, color = linecolor)
display(fig)
##
save("figure/KuramotoSivashinsky.png", fig)
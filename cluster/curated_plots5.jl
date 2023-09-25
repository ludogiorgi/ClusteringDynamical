using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using MultiDimensionalClustering.CommunityDetection, ProgressBars
using MarkovChainHammer
using Main.MarkovChainHammer.BayesianMatrix:BayesianGenerator
using Main.MarkovChainHammer.TransitionMatrix: steady_state
using Main.MarkovChainHammer.TransitionMatrix:perron_frobenius
using SparseArrays
Random.seed!(12345)
##
include("curated_plots_convenience_functions.jl")
##
@info "opening data"
hfile = h5open("data/lorenz.hdf5")
x = read(hfile["x"])
dt = read(hfile["dt"])
close(hfile)
##
@info "embedding"
embedding = state_tree_embedding(x, 11)
@info "applying embedding"
markov_embedding = [embedding(x[:,i]) for i in 1:size(x)[2]]
@info "eigenvalue decomposition"
Q = mean(BayesianGenerator(markov_embedding))
Λ, V =  eigen(Q)
τs = reverse((-1 ./ real.(Λ[end-20:end]))[1:end-1])
##
P1 = exp(Q * τs[3]) 
q_min = 0.0
@info "leicht_newman_with_tree"
F, G, H, PI = leicht_newman_with_tree(P1, q_min)
@info "done with Leicht Newmann algorithm"
node_labels, adj, adj_mod, edge_numbers = graph_from_PI(PI)
X_LN = classes_timeseries(F, markov_embedding)
##
q_min = 0.0
P1⁺ = exp(Q * (τs[3] +1)) 
P1⁻ = exp(Q * (τs[3] -1))
F⁺, _, _, _ = leicht_newman_with_tree(P1⁺, q_min)
F⁻, _, _, _ = leicht_newman_with_tree(P1⁻, q_min)
X_LN⁺ = classes_timeseries(F⁺, markov_embedding)
X_LN⁻ = classes_timeseries(F⁻, markov_embedding)
score1, _ =  label_ordering(X_LN⁺, X_LN)
score2, _ =  label_ordering(X_LN⁻, X_LN)
score = (score1 + score2) / 2
##
fig = Figure(resolution = (3200, 800))
layout = Buchheim()
colormap = :glasbey_hv_n256
set_theme!(backgroundcolor=:white)

ax11 = Axis(fig[1,2])
G = SimpleDiGraph(adj)
transparancy = 0.4 * adj_mod.nzval[:] / adj_mod.nzval[1] .+ 0.1
nlabels_fontsize = 35
edge_color = [(:red, transparancy[i]) for i in 1:edge_numbers]
nlabels = [@sprintf("%2.2f", node_labels[i]) for i in 1:nv(G)]
graphplot!(ax11, G, layout=layout, nlabels=nlabels, node_size=100,
    node_color=(:orange, 0.9), edge_color=edge_color, edge_width=5,
    arrow_size=45, nlabels_align=(:center, :center),
    nlabels_textsize=nlabels_fontsize, tangents=((0, -1), (0, -1)))
# cc = cameracontrols(ax11.scene)
hidedecorations!(ax11)
hidespines!(ax11);

ax11 = LScene(fig[1,1]; show_axis = false)
res = 10
markersize = 10.0
GLMakie.scatter!(ax11, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], markersize=markersize, color=markov_embedding[1:res:end], colormap = colormap)
rotate_cam!(ax11.scene, (0.0, -10.5, 0.0))
ax13 = LScene(fig[1,3]; show_axis = false)
res = 10
markersize = 10.0
GLMakie.scatter!(ax13, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], markersize=markersize, color=cgrad(colormap)[X_LN[1:res:end]])
rotate_cam!(ax13.scene, (0.0, -10.5, 0.0))

ax = Axis(fig[1,4])
Q_graph = mean(BayesianGenerator(X_LN, dt=dt))
g_Q = DiGraph(Q_graph')
Q_prim = zeros(size(Q_graph))
for i in 1:size(Q_graph)[1]
    Q_prim[:, i] .= -Q_graph[:, i] / Q_graph[i, i]
    Q_prim[i, i] = -1 / Q_graph[i, i]
end
elabels = string.([round(Q_prim[i]; digits=2) for i in 1:ne(g_Q)])
# [@sprintf("%.0e", node_labels[i]) for i in 1:nv(G)]
transparancy = [Q_prim[i] for i in 1:ne(g_Q)]
elabels_color = [(:black, transparancy[i] > eps(100.0)) for i in 1:ne(g_Q)]
#edge_color_Q = [(:black, transparancy[i]) for i in 1:ne(g_Q)]
edge_color_Q = [(cgrad(colormap)[(i-1)÷(size(Q_graph)[1])+1], transparancy[i]) for i in 1:ne(g_Q)]
node_color = [(cgrad(colormap)[i]) for i in 1:nv(g_Q)]
edge_attr = (; linestyle=[:dot, :dash, :dash, :dash, :dot, :dash, :dash, :dash, :dot])
elabels_fontsize = 40
nlabels_fontsize = 40
node_size = 100.0
edge_width_Q = [10.0 for i in 1:ne(g_Q)]
arrow_size_Q = [40.0 for i in 1:ne(g_Q)]
node_labels_Q = repr.(1:nv(g_Q))
kwargs_edges  = (;  elabels_color=elabels_color, elabels_textsize=elabels_fontsize, edge_color=edge_color_Q, edge_width=edge_width_Q) # elabels=elabels,
kwargs_nodes  = (; node_color=node_color, node_size=node_size)#  nlabels=node_labels_Q, nlabels_textsize=nlabels_fontsize)
kwargs_arrows =  (; arrow_size=arrow_size_Q)
graphplot!(ax, g_Q; kwargs_edges..., kwargs_nodes..., kwargs_arrows..., layout = Stress())
hidedecorations!(ax)
hidespines!(ax)
display(fig)
##
@info "opening data"
hfile = h5open("data/newton.hdf5")
x = read(hfile["x"])
dt = read(hfile["dt"])
close(hfile)
##
@info "embedding"
embedding = state_tree_embedding(x, 11)
@info "applying embedding"
markov_embedding = [embedding(x[:,i]) for i in 1:size(x)[2]]
@info "eigenvalue decomposition"
Q = mean(BayesianGenerator(markov_embedding))
Λ, V =  eigen(Q)
τs = reverse((-1 ./ real.(Λ[end-20:end]))[1:end-1])
##
P1 = exp(Q * τs[3]) 
q_min = 0.0
@info "leicht_newman_with_tree"
F, G, H, PI = leicht_newman_with_tree(P1, q_min)
node_labels, adj, adj_mod, edge_numbers = graph_from_PI(PI)
X_LN = classes_timeseries(F, markov_embedding)
##
q_min = 0.0
P1⁺ = exp(Q * (τs[3] +1)) 
P1⁻ = exp(Q * (τs[3] -1))
F⁺, _, _, _ = leicht_newman_with_tree(P1⁺, q_min)
F⁻, _, _, _ = leicht_newman_with_tree(P1⁻, q_min)
X_LN⁺ = classes_timeseries(F⁺, markov_embedding)
X_LN⁻ = classes_timeseries(F⁻, markov_embedding)
score1, _ =  label_ordering(X_LN⁺, X_LN)
score2, _ =  label_ordering(X_LN⁻, X_LN)
score = (score1 + score2) / 2
##
colormap = :glasbey_hv_n256
set_theme!(backgroundcolor=:white)

ax11 = Axis(fig[2,2])
G = SimpleDiGraph(adj)
transparancy = 0.4 * adj_mod.nzval[:] / adj_mod.nzval[1] .+ 0.1
nlabels_fontsize = 35
edge_color = [(:red, transparancy[i]) for i in 1:edge_numbers]
nlabels = [@sprintf("%2.2f", node_labels[i]) for i in 1:nv(G)]
graphplot!(ax11, G, layout=layout, nlabels=nlabels, node_size=100,
    node_color=(:orange, 0.9), edge_color=edge_color, edge_width=5,
    arrow_size=45, nlabels_align=(:center, :center),
    nlabels_textsize=nlabels_fontsize, tangents=((0, -1), (0, -1)))
# cc = cameracontrols(ax11.scene)
hidedecorations!(ax11)
hidespines!(ax11);

ax11 = LScene(fig[2,1]; show_axis = false)
res = 10
markersize = 10.0
GLMakie.scatter!(ax11, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], markersize=markersize, color=markov_embedding[1:res:end], colormap = colormap)
# rotate_cam!(ax11.scene, (0.0, -10.5, 0.0))
ax13 = LScene(fig[2,3]; show_axis = false)
res = 10
markersize = 10.0
GLMakie.scatter!(ax13, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], markersize=markersize, color=cgrad(colormap)[X_LN[1:res:end]])
# rotate_cam!(ax13.scene, (0.0, -10.5, 0.0))

ax = Axis(fig[2,4])
Q_graph = mean(BayesianGenerator(X_LN, dt=dt))
g_Q = DiGraph(Q_graph')
Q_prim = zeros(size(Q_graph))
for i in 1:size(Q_graph)[1]
    Q_prim[:, i] .= -Q_graph[:, i] / Q_graph[i, i]
    Q_prim[i, i] = -1 / Q_graph[i, i]
end
elabels = string.([round(Q_prim[i]; digits=2) for i in 1:ne(g_Q)])
# [@sprintf("%.0e", node_labels[i]) for i in 1:nv(G)]
transparancy = [Q_prim[i] for i in 1:ne(g_Q)]
elabels_color = [(:black, transparancy[i] > eps(100.0)) for i in 1:ne(g_Q)]
#edge_color_Q = [(:black, transparancy[i]) for i in 1:ne(g_Q)]
edge_color_Q = [(cgrad(colormap)[(i-1)÷(size(Q_graph)[1])+1], transparancy[i]) for i in 1:ne(g_Q)]
node_color = [(cgrad(colormap)[i]) for i in 1:nv(g_Q)]
edge_attr = (; linestyle=[:dot, :dash, :dash, :dash, :dot, :dash, :dash, :dash, :dot])
elabels_fontsize = 40
nlabels_fontsize = 40
node_size = 100.0
edge_width_Q = [10.0 for i in 1:ne(g_Q)]
arrow_size_Q = [40.0 for i in 1:ne(g_Q)]
node_labels_Q = repr.(1:nv(g_Q))
kwargs_edges  = (;  elabels_color=elabels_color, elabels_textsize=elabels_fontsize, edge_color=edge_color_Q, edge_width=edge_width_Q) # elabels=elabels,
kwargs_nodes  = (; node_color=node_color, node_size=node_size)#  nlabels=node_labels_Q, nlabels_textsize=nlabels_fontsize)
kwargs_arrows =  (; arrow_size=arrow_size_Q)
graphplot!(ax, g_Q; kwargs_edges..., kwargs_nodes..., kwargs_arrows..., layout = Stress())
hidedecorations!(ax)
hidespines!(ax)
display(fig)
##

save("figure/NewtonLorenz.png", fig)


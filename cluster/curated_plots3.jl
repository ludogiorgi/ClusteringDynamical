using HDF5, GLMakie, GraphMakie, Statistics
using MarkovChainHammer.BayesianMatrix

include("GenerateFineCluster.jl")
include("GenerateCoarseClusterTree.jl")
include("GenerateCoarseCluster.jl")

using Main.GenerateFineCluster:read_fine_cluster, plot_fine_cluster, plot_fine_cluster_array, fine_cluster
using Main.GenerateCoarseClusterTree:read_coarse_cluster_tree, plot_coarse_cluster_tree, coarse_cluster_tree
using Main.GenerateCoarseCluster:coarse_cluster, plot_coarse_cluster1, plot_coarse_cluster2, read_coarse_cluster
using HDF5, LaTeXStrings, GLMakie
using SparseArrays, NetworkLayout, Graphs, Printf, NetworkLayout, GraphMakie, Graphs
using Main.GenerateFineCluster:read_fine_cluster, plot_fine_cluster, fine_cluster
using Main.GenerateCoarseClusterTree:read_coarse_cluster_tree, plot_coarse_cluster_tree, coarse_cluster_tree
using Main.GenerateCoarseCluster:coarse_cluster, plot_coarse_cluster1, plot_coarse_cluster2, read_coarse_cluster
using HDF5, ProgressBars, InvertedIndices
##
function read_data(file)
    hfile = h5open(pwd()*"/data/" *file* ".hdf5")
    x = read(hfile["x"])
    dt = read(hfile["dt"])
    close(hfile)
    return x, dt
end
##
# Read data
file = "potential_well"
X1, Xc1 = read_fine_cluster(file)
x1, dt1 = read_data(file)

file = "lorenz"
X2, Xc2 = read_fine_cluster(file)
x2, dt2 = read_data(file)

file = "newton"
X3, Xc3 = read_fine_cluster(file)
x3, dt3 = read_data(file)
##
res = 4
fig = Figure(resolution = (1374, 960)) 
colormap = :glasbey_hv_n256
markersize = 5
ax1 = LScene(fig[1,1]; show_axis = false)
x = x1 
X = X1
scatter!(ax1, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], markersize=markersize, color=X[1:res:end], colormap = colormap)

ax2 = LScene(fig[1, 2]; show_axis = false)
x = x2 
X = X2
scatter!(ax2, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], markersize=markersize, color=X[1:res:end], colormap = colormap)
rotate_cam!(ax2.scene, (0.0, -7.0, 0.0))

ax3 = LScene(fig[1, 3]; show_axis = false)
x = x3 
X = X3
scatter!(ax3, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], markersize=markersize, color=X[1:res:end], colormap = colormap)
# rotated version
ax1 = LScene(fig[2,1]; show_axis = false)
x = x1 
X = X1
scatter!(ax1, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], markersize=markersize, color=X[1:res:end], colormap = colormap)
rotate_cam!(ax1.scene, (0.0, -10.5, 0.0))

ax2 = LScene(fig[2, 2]; show_axis = false)
x = x2 
X = X2
scatter!(ax2, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], markersize=markersize, color=X[1:res:end], colormap = colormap)
rotate_cam!(ax2.scene, (0.0, -10.5, 0.0))

ax3 = LScene(fig[2, 3]; show_axis = false)
x = x3 
X = X3
scatter!(ax3, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], markersize=markersize, color=X[1:res:end], colormap = colormap)
rotate_cam!(ax3.scene, (0.0, -2.5, 0.0))
display(fig)
##
save("figure/PotentialWellNewtonLorenz.png", fig)

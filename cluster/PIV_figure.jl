include("GenerateFineCluster.jl")
include("GenerateCoarseClusterTree.jl")
include("GenerateCoarseCluster.jl")

using Main.GenerateFineCluster:read_fine_cluster, plot_fine_cluster, plot_fine_cluster_array, fine_cluster
using Main.GenerateCoarseClusterTree:read_coarse_cluster_tree, plot_coarse_cluster_tree, coarse_cluster_tree
using Main.GenerateCoarseCluster:coarse_cluster, plot_coarse_cluster1, plot_coarse_cluster2, read_coarse_cluster
using HDF5, LaTeXStrings, GLMakie
using HDF5, GLMakie, Dierckx
using LinearAlgebra, Statistics, Random, LaTeXStrings
using MultiDimensionalClustering.CommunityDetection, ProgressBars
using MarkovChainHammer.BayesianMatrix: BayesianGenerator
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix: perron_frobenius
using MultiDimensionalClustering.AlternativeGenerator
##
hfile = h5open("data/PIV.hdf5")
x = read(hfile["x"])
dt = read(hfile["dt"])
U = read(hfile["U"])
close(hfile)
X, Xc = read_fine_cluster("PIV")
P = perron_frobenius(X,step=1)
nc, ln_nc = leicht_newman(P,2)
X_LN = classes_timeseries(ln_nc, X)
##
newf = U * x
# 63 x 79  data split into 2 fields
# https://juliagraphics.github.io/ColorSchemes.jl/stable/catalogue/ 
fig = Figure(resolution=(2000, 3000))
timeskip = 20
for i in 1:6
    jj = (i - 1) ÷ 3 + 1
    ii = (i - 1) % 3 + 1
    index = (i - 1) * timeskip + 1
    timevalue = round(Int, (index - 1) * dt * 1000)
    titlestring = "t = $(timevalue)"
    ax = Axis(fig[ii, jj]; title=titlestring, titlesize=60)
    u = reshape(newf[1:4977, index], (63, 79))
    v = reshape(newf[4978:end, index], (63, 79))
    spline_u = Spline2D(1:63, 1:79, u)
    spline_v = Spline2D(1:63, 1:79, v)
    stream(x, y) = Point2f(spline_v(x, y), spline_u(x, y))
    streamplot!(ax, stream, 1:63, 1:79, arrow_size=20, linewidth=2.5, colormap = :plasma)
    hidedecorations!(ax)
end
save("figure/PIV_data.png", fig)
##
newf = U * Xc
# 63 x 79  data split into 2 fields
# https://juliagraphics.github.io/ColorSchemes.jl/stable/catalogue/ 
fig = Figure(resolution=(4000, 4000))
timeskip = 1
clust = 0
for i in 1:16
    jj = (i - 1) ÷ 4 + 1
    ii = (i - 1) % 4 + 1
    index = (i - 1) * timeskip + 1
    for j in eachindex(ln_nc)
        if (index in ln_nc[j]) == true
            clust = j
        end
    end
    timevalue = round(Int, (index - 1) * dt * 1000)
    titlestring = "$(clust)"
    ax = Axis(fig[ii, jj]; title=titlestring, titlesize=80)
    u = reshape(newf[1:4977, index], (63, 79))
    v = reshape(newf[4978:end, index], (63, 79))
    spline_u = Spline2D(1:63, 1:79, u)
    spline_v = Spline2D(1:63, 1:79, v)
    stream(x, y) = Point2f(spline_v(x, y), spline_u(x, y))
    streamplot!(ax, stream, 1:63, 1:79, arrow_size=40, linewidth=4, colormap = :plasma)
    hidedecorations!(ax)
end
save("figure/PIV_cluster.png", fig)

##
randn(5,5)
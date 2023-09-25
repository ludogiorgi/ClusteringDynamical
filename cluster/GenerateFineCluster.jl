module GenerateFineCluster
using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using LaTeXStrings
export read_fine_cluster, plot_fine_cluster, plot_fine_cluster_array, fine_cluster

function norm_ord(X,Xc)
    L = length(X)
    n_clusters = length(union(X))
    E = zeros(n_clusters)
    for i = 1:n_clusters E[i] = norm(Xc[:,i],2) end
    E_ind = sortperm(E)
    clusters_ord = zeros(Int,L)
    for i = 1:L
        clusters_ord[i] = findall(y->y == X[i],E_ind)[1]
    end
    return clusters_ord,Xc[:,E_ind]
end

function read_fine_cluster(file)
    hfile = h5open(pwd() * "/data/" * file * "_fine_cluster.hdf5")
    X = read(hfile["X"])
    Xc = read(hfile["Xc"])
    close(hfile)
    return X, Xc
end

function plot_fine_cluster(x,X,title,figure_number; res = 1, mks = 5, azimuth=0.0pi, elevation=0.0pi)
    L = length(X)
    fig = Figure(resolution=(1500, 1500*L))
    colormap = :glasbey_hv_n256
    for i in 1:L
        ax = Axis3(fig[1, i], xticklabelsize=40, yticklabelsize=40, zticklabelsize=40, xlabelsize=60, ylabelsize=60, zlabelsize=60, azimuth = azimuth, elevation=elevation)
        scatter!(ax, x[i][1, 1:res:end], x[i][2, 1:res:end], x[i][3, 1:res:end], color=cgrad(colormap)[(X[i][1:res:end] .% 256 .+1)]; markersize=mks)
        Label(fig[1, i, Top()], title; textsize=100)
    end
    save("figure/figure" * string(figure_number[1])*string(figure_number[2]) * ".png", fig)
end

function plot_fine_cluster_array(x,X,title,figure_number,mks,res,azimuth,elevation)
    L = length(X)
    fig = Figure(resolution=(1800*L, 1500))
    colormap = :glasbey_hv_n256
    for i in 1:L
        ax = Axis3(fig[1, i], xticklabelsize=40, yticklabelsize=40, zticklabelsize=40, xlabelsize=60, ylabelsize=60, zlabelsize=60, azimuth = azimuth[i], elevation=elevation[i])
        scatter!(ax, x[i][1, 1:res[i]:end], x[i][2, 1:res[i]:end], x[i][3, 1:res[i]:end], color=cgrad(colormap)[(X[i][1:res[i]:end] .% 256 .+1)]; markersize=mks[i])
        Label(fig[1, i, Top()], title[i]; textsize=100)
    end
    colgap!(fig.layout, 0)
    rowgap!(fig.layout, 0)
    save("figure/figure" * string(figure_number[1])*string(figure_number[2]) * ".png", fig)
end

function fine_cluster(x; n_clusters = 2000, file = false)
    if n_clusters > Int(round(length(x[1,:])/10))
        n_clusters = Int(round(length(x[1,:])/10))
    end
    Random.seed!(12345)
    kmn = kmeans(x, n_clusters; max_iters=10^6)
    X = kmn.assignments
    Xc = kmn.centers
    X,Xc = norm_ord(X,Xc)
    if file != false
        hfile = h5open(pwd()*"/data/" * file * "_fine_cluster.hdf5", "w")
        hfile["X"] = X
        hfile["Xc"] = Xc
        close(hfile)
    end
    return X, Xc
end
end
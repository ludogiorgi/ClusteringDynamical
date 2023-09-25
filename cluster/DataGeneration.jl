include("GenerateFineCluster.jl")
include("GenerateCoarseClusterTree.jl")
include("GenerateCoarseCluster.jl")

using Main.GenerateFineCluster:read_fine_cluster, plot_fine_cluster, fine_cluster
using Main.GenerateCoarseClusterTree:read_coarse_cluster_tree, plot_coarse_cluster_tree, coarse_cluster_tree
using Main.GenerateCoarseCluster:coarse_cluster, plot_coarse_cluster1, plot_coarse_cluster2, read_coarse_cluster
using HDF5, ProgressBars, InvertedIndices

function read_data(file)
    hfile = h5open(pwd()*"/data/" *file* ".hdf5")
    x = read(hfile["x"])
    dt = read(hfile["dt"])
    close(hfile)
    return x, dt
end

x = []
dt = []
files = ["potential_well", "lorenz", "newton", "kuramoto", "PIV"]
for i in 1:5
    temp1, temp2 = read_data(files[i])
    push!(x, temp1)
    push!(dt, temp2)
end
######## DANGER #########
for i in 1:5
   fine_cluster(x[i]; n_clusters = 30, file = files[i])
end

X = []
Xc = []
for i in (1:5)
    temp1, temp2 = read_fine_cluster(files[i])
    push!(X, temp1)
    push!(Xc, temp2)
end
######## DANGER #########
for i in 1:5
    coarse_cluster_tree(X[i],dt[i],8; file=files[i])
end

X_LN_array = []
adj_array = []
adj_mod_array = []
node_labels_array = []
edge_numbers_array = []
τ = []
for i in 1:5
    temp1, temp2, temp3, temp4, temp5, temp6 = read_coarse_cluster_tree(files[i])
    push!(X_LN_array, temp1)
    push!(adj_array, temp2)
    push!(adj_mod_array, temp3)
    push!(node_labels_array, temp4)
    push!(edge_numbers_array, temp5)
    push!(τ, temp6)
end
######## DANGER #########
qmins = []
push!(qmins, [2 ,2 ,4 ,4 ,4 ,4 ,4 , 8])
push!(qmins, [3 ,3 ,3 ,4 ,4 ,4 ,4 , 4])
push!(qmins, [2 ,4 ,4 ,4 ,4 ,4 ,4 , 4])
push!(qmins, [2 ,2 ,2 ,4 ,4 ,4 ,4 , 2])  
push!(qmins, [3 ,3 ,3 ,3 ,3 ,3 ,3 ,3])
indices = [1,2,3,4,5,6,7,8]
for i in 1:5
    coarse_cluster(X[i], dt[i], indices, τ[i], qmins[i]; file=files[i])
end
module GenerateCoarseClusterTree
using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using MultiDimensionalClustering.CommunityDetection, ProgressBars
using MarkovChainHammer.BayesianMatrix: BayesianGenerator
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix: perron_frobenius
using MultiDimensionalClustering.AlternativeGenerator
using SparseArrays, NetworkLayout, Graphs, Printf, NetworkLayout, GraphMakie, Graphs, LaTeXStrings
export read_coarse_cluster_tree, plot_coarse_cluster_tree, coarse_cluster_tree


function read_coarse_cluster_tree(file)
    hfile = h5open(pwd()*"/data/" * file * "_coarse_cluster_tree.hdf5")
    τ = read(hfile["τ"])
    n_timescales = length(τ)
    adj_array = []
    adj_mod_array = []
    edge_numbers_array = []
    node_labels_array = []
    X_LN_array = []
    for t = 1:n_timescales
        edge_numbers = read(hfile["tree_edge_numbers_tms=($t)"])
        push!(edge_numbers_array,edge_numbers)
        N = read(hfile["tree_matrix_size_tms=($t)"])
        adj = spzeros(Int64, N, N)
        adj_mod = spzeros(Float64, N, N)
        for i in 1:edge_numbers
            ii = read(hfile["tree_i"*string(i)*"_tms=($t)"])
            jj = read(hfile["tree_j"*string(i)*"_tms=($t)"])
            modularity_value = read(hfile["tree_modularity"*string(i)*"_tms=($t)"])
            adj[ii, jj] += 1
            adj_mod[ii, jj] = modularity_value
        end
        push!(adj_array, adj)
        push!(adj_mod_array, adj_mod)
        node_labels = zeros(Float64, N)
        for i in 1:edge_numbers
            ii = read(hfile["tree_i"*string(i)*"_tms=($t)"])
            modularity_value = read(hfile["tree_modularity"*string(i)*"_tms=($t)"])
            node_labels[ii] = modularity_value
        end
        push!(node_labels_array,node_labels)
        X_LN = read(hfile["X_LN_tms=($t)"])
        push!(X_LN_array, X_LN)
    end
    close(hfile)
    return X_LN_array, adj_array, adj_mod_array, node_labels_array, edge_numbers_array, τ
end

function timescales(λ)
    λ = real.(λ)
    ind = sortperm(λ,rev=true)
    λ = λ[ind]
    τ = Float64[]
    push!(τ, -1/λ[2])
    λ_len = length(λ)
    for i in 3:λ_len
        if (λ[i-1] - λ[i]) > 0.0001 push!(τ, -1/λ[i]) end
    end
    return τ
end

function plot_coarse_cluster_tree(x,X_LN_array, adj_array, adj_mod_array, node_labels_array, edge_numbers_array, τ, title, figure_number; res = 5, mks = 10, azimuth=0.0pi, elevation=0.0pi)
    fig = Figure(resolution=(5000, 4000))
    layout = Buchheim()
    colormap = :glasbey_hv_n256
    set_theme!(backgroundcolor=:white)
    n_timescales = length(τ)
    fct = 10
    for t = 1:n_timescales
        if t <= Int(n_timescales/2)
            ax = Axis(fig[2:fct+1, t]; title=latexstring("\\textbf{t=$(round(τ[t],sigdigits=3))}"), titlesize=60)
        else
            ax = Axis(fig[2*fct+2:3*fct+1, t-Int(n_timescales/2)]; title=latexstring("\\textbf{t=$(round(τ[t],sigdigits=3))}"), titlesize=60)
        end
        G = SimpleDiGraph(adj_array[t])
        transparancy = 0.4 * adj_mod_array[t].nzval[:] / adj_mod_array[t].nzval[1] .+ 0.1
        nlabels_fontsize = 35
        edge_color = [(:red, transparancy[i]) for i in 1:edge_numbers_array[t]]
        nlabels = [@sprintf("%.01e", node_labels_array[t][i]) for i in 1:nv(G)]
        graphplot!(ax, G, layout=layout, nlabels=nlabels, node_size=150,
            node_color=(:orange, 0.9), edge_color=edge_color, edge_width=5,
            arrow_size=45, nlabels_align=(:center, :center),
            nlabels_textsize=nlabels_fontsize, tangents=((0, -1), (0, -1)))
        hidedecorations!(ax)
        hidespines!(ax);
        if t <= Int(n_timescales/2)
            ax = Axis3(fig[fct+2:2*fct+1, t], xticklabelsize=40, yticklabelsize=40, zticklabelsize=40, xlabelsize=40, ylabelsize=40, zlabelsize=40, azimuth = azimuth, elevation=elevation)
        else
            ax = Axis3(fig[3*fct+2:4*fct+1, t-Int(n_timescales/2)], xticklabelsize=40, yticklabelsize=40, zticklabelsize=40, xlabelsize=40, ylabelsize=40, zlabelsize=40, azimuth = azimuth, elevation=elevation)
        end
        scatter!(ax, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], color=cgrad(colormap)[X_LN_array[t][1:res:end]]; markersize=mks)
    end
    Label(fig[1, 1:Int(n_timescales/2), Top()], title; textsize=100)
    colgap!(fig.layout, 25)
    rowgap!(fig.layout, 25)
    save("figure/figure" * string(figure_number[1])*string(figure_number[2]) * ".png", fig)
end

function coarse_cluster_tree(X, dt, n_timescales; file=false)
    Q = mean(BayesianGenerator(X; dt=dt))
    Λ, _ = eigen(Q)
    τ = timescales(Λ)
    t_steps = Int.(round.(τ ./dt)) 
    q_min = 1e-16   
    adj_array = []
    adj_mod_array = []
    edge_numbers_array = []
    node_labels_array = []
    X_LN_array = []
    if file != false
        filename = "/data/" * file * "_coarse_cluster_tree.hdf5"
        hfile = h5open(pwd()*filename, "w")
        hfile["τ"] = τ[1:n_timescales]
    end
    for t in 1:n_timescales
        P = perron_frobenius(X,step=t_steps[t])
        F, _, _, PI = leicht_newman_with_tree(P, q_min)
        edge_numbers = length(PI)
        push!(edge_numbers_array,edge_numbers)
        N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])
        adj = spzeros(Int64, N, N)
        adj_mod = spzeros(Float64, N, N)
        for i in 1:edge_numbers
            ii = PI[i][1]
            jj = PI[i][2]
            modularity_value = PI[i][3]
            adj[ii, jj] += 1
            adj_mod[ii, jj] = modularity_value
        end
        push!(adj_array, adj)
        push!(adj_mod_array, adj_mod)
        node_labels = zeros(Float64, N)
        for i in 1:edge_numbers
            ii = PI[i][1]
            modularity_value = PI[i][3]
            node_labels[ii] = modularity_value
        end
        push!(node_labels_array,node_labels)
        X_LN = classes_timeseries(F, X)
        push!(X_LN_array, X_LN)
        if file != false
            hfile["X_LN_tms=($t)"] = X_LN
            hfile["tree_edge_numbers_tms=($t)"] = length(PI)
            hfile["Flength_tms=($t)"] = length(F)
            hfile["tree_matrix_size_tms=($t)"] = N
            for i in eachindex(PI)
                hfile["tree_i"*string(i)*"_tms=($t)"] = PI[i][1]
                hfile["tree_j"*string(i)*"_tms=($t)"] = PI[i][2]
                hfile["tree_modularity"*string(i)*"_tms=($t)"] = PI[i][3]
            end
        end
    end
    if file != false
        close(hfile)
    end
    return X_LN_array, adj_array, adj_mod_array, node_labels_array, edge_numbers_array, τ[1:n_timescales]
end

end
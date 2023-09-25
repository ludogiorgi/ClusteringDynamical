include("GenerateFineCluster.jl")
include("GenerateCoarseClusterTree.jl")
include("GenerateCoarseCluster.jl")

using Main.GenerateFineCluster:read_fine_cluster, plot_fine_cluster, plot_fine_cluster_array, fine_cluster
using Main.GenerateCoarseClusterTree:read_coarse_cluster_tree, plot_coarse_cluster_tree, coarse_cluster_tree
using Main.GenerateCoarseCluster:coarse_cluster, plot_coarse_cluster1, plot_coarse_cluster2, read_coarse_cluster
using HDF5, LaTeXStrings, GLMakie
using SparseArrays, NetworkLayout, Graphs, Printf, NetworkLayout, GraphMakie, Graphs

##
function read_data(file)
    hfile = h5open(pwd()*"/data/" *file* ".hdf5")
    x = read(hfile["x"])
    dt = read(hfile["dt"])
    close(hfile)
    return x, dt
end

function plot_fine_clusterP(x,X,title,figure_number,mks,res,azimuth,elevation)
    fig = Figure(resolution=(5000, 3000))
    colormap = :glasbey_hv_n256
    for i in 1:3
        ax = Axis3(fig[1, 2*(i-1)+1:2*i], xticklabelsize=40, yticklabelsize=40, zticklabelsize=40, xlabelsize=60, ylabelsize=60, zlabelsize=60, azimuth = azimuth[i], elevation=elevation[i])
        scatter!(ax, x[i][1, 1:res[i]:end], x[i][2, 1:res[i]:end], x[i][3, 1:res[i]:end], color=cgrad(colormap)[(X[i][1:res[i]:end] .% 256 .+1)]; markersize=mks[i])
        Label(fig[1, 2*(i-1)+1:2*i, Top()], title[i]; textsize=100)
    end
    for i in 1:2
        ax = Axis3(fig[2, 3*(i-1)+1:3*i], xticklabelsize=40, yticklabelsize=40, zticklabelsize=40, xlabelsize=60, ylabelsize=60, zlabelsize=60, azimuth = azimuth[i+3], elevation=elevation[i+3])
        scatter!(ax, x[i+3][1, 1:res[i+3]:end], x[i+3][2, 1:res[i+3]:end], x[i+3][3, 1:res[i+3]:end], color=cgrad(colormap)[(X[i+3][1:res[i+3]:end] .% 256 .+1)]; markersize=mks[i+3])
        Label(fig[2, 3*(i-1)+1:3*i, Top()], title[i+3]; textsize=100)
    end
    colgap!(fig.layout, 0)
    rowgap!(fig.layout, 0)
    save("figure/figure" * string(figure_number[1])*string(figure_number[2]) * ".png", fig)
end

x = []
dt = []
files = ["potential_well", "lorenz", "newton", "kuramoto", "PIV"]
for i in 1:5
    temp1, temp2 = read_data(files[i])
    push!(x, temp1)
    push!(dt, temp2)
end

X = []
Xc = []
for i in (1:5)
    temp1, temp2 = read_fine_cluster(files[i])
    push!(X, temp1)
    push!(Xc, temp2)
end

X_LN_array1 = []
adj_array = []
adj_mod_array = []
node_labels_array = []
edge_numbers_array = []
τ = []
for i in 1:5
    temp1, temp2, temp3, temp4, temp5, temp6 = read_coarse_cluster_tree(files[i])
    push!(X_LN_array1, temp1)
    push!(adj_array, temp2)
    push!(adj_mod_array, temp3)
    push!(node_labels_array, temp4)
    push!(edge_numbers_array, temp5)
    push!(τ, temp6)
end

X_LN_array2 = []
Q_array = []
Q_pert_array = []
Pt_array = []
Qt_array = []
Qt_pert_array = []
score = []
for i in 1:5
    temp1, temp2, temp3, temp4, temp5, temp6, temp7 = read_coarse_cluster(files[i])
    push!(X_LN_array2, temp1)
    push!(Q_array, temp2)
    push!(Q_pert_array, temp3)
    push!(Pt_array, temp4)
    push!(Qt_array, temp5)
    push!(Qt_pert_array, temp6)
    push!(score, temp7)
end
##
titles = [latexstring("\\textbf{\\textrm{Multi-well potential}}") 
        latexstring("\\textbf{\\textrm{Lorenz 63}}") 
        latexstring("\\textbf{\\textrm{Newton-Leipnik}}") 
        latexstring("\\textbf{\\textrm{Kuramoto-Sivashinsky}}") 
        latexstring("\\textbf{\\textrm{PIV}}")]
mks = [5,5,3,3,20]
azimuth = [0.2pi, 0.05pi, 0.4pi, 1.2pi, 0.1pi]
elevation = [0.2pi, 0.2pi, 0.2pi, 0.1pi, 0.1pi]
res = [1 ,1 ,1 ,1 ,1]

figure_number = [1 1]
plot_fine_clusterP(x,X, titles, figure_number, mks, res, azimuth, elevation)

##
titles = [latexstring("\\textbf{\\textrm{Multi-well potential}}") 
        latexstring("\\textbf{\\textrm{Lorenz 63}}") 
        latexstring("\\textbf{\\textrm{Newton-Leipnik}}") 
        latexstring("\\textbf{\\textrm{Kuramoto-Sivashinsky}}") 
        latexstring("\\textbf{\\textrm{PIV}}")]

mks_vals = [5,5,3,3,20]
azimuth = [0.2pi, 0.05pi, 0.4pi, 1.2pi, 0.05pi]
elevation = [0.2pi, 0.2pi, 0.2pi, 0.1pi, 0.2pi]
for f_index in 1:5
    figure_number = [2 f_index]
    plot_coarse_cluster_tree(x[f_index],X_LN_array1[f_index], adj_array[f_index], adj_mod_array[f_index], 
                        node_labels_array[f_index], edge_numbers_array[f_index], τ[f_index], titles[f_index], figure_number; res = 1, mks = mks_vals[f_index], azimuth=azimuth[f_index], elevation=elevation[f_index])
end
##
titles = [latexstring("\\textbf{\\textrm{Multi-well potential}}") 
        latexstring("\\textbf{\\textrm{Lorenz 63}}") 
        latexstring("\\textbf{\\textrm{Newton-Leipnik}}") 
        latexstring("\\textbf{\\textrm{Kuramoto-Sivashinsky}}") 
        latexstring("\\textbf{\\textrm{PIV}}")]

mks_vals = [5,5,3,3,20]
azimuth = [0.2pi, 0.05pi, 0.4pi, 1.2pi, 0.05pi]
elevation = [0.2pi, 0.2pi, 0.4pi, 0.1pi, 0.2pi]
Lt = [5000, 5000, 20000, 5000, 1000]
for f_index in 3:3
    figure_number = [3 f_index]
    plot_coarse_cluster1P(x[f_index], X[f_index], τ[f_index], dt[f_index], X_LN_array2[f_index], Q_array[f_index], score[f_index], titles[f_index], figure_number; res = 1,mks = mks_vals[f_index], azimuth=azimuth[f_index], elevation=elevation[f_index], Lt=Lt[f_index])
end
##
indices = [2,3,3,7,3]
for f_index in 1:5
    titles = [latexstring("\\textbf{\\textrm{Multi-well potential}} \\;(\\textbf{t} = \\textbf{$(round(τ[f_index][indices[f_index]],sigdigits=3))})") 
        latexstring("\\textbf{\\textrm{Lorenz 63}} \\;(\\textbf{t} = \\textbf{$(round(τ[f_index][indices[f_index]],sigdigits=3))})") 
        latexstring("\\textbf{\\textrm{Newton-Leipnik}} \\;(\\textbf{t} = \\textbf{$(round(τ[f_index][indices[f_index]],sigdigits=3))})") 
        latexstring("\\textbf{\\textrm{Kuramoto-Sivashinsky}} \\;(\\textbf{t} = \\textbf{$(round(τ[f_index][indices[f_index]],sigdigits=3))})") 
        latexstring("\\textbf{\\textrm{PIV}} \\;(\\textbf{t} = \\textbf{$(round(τ[f_index][indices[f_index]],sigdigits=3))})")]
    figure_number = [4 f_index]
    plot_coarse_cluster2(dt[f_index], Pt_array[f_index], Qt_array[f_index], Qt_pert_array[f_index], indices[f_index], titles[f_index], figure_number)
end
##
function plot_coarse_cluster1P(x, X, τ, dt, X_LN_array, Q_array, score, title, figure_number; res = 1, mks = 5, azimuth=0.0pi, elevation=0.0pi, Lt=1000)
    indices_len = length(τ)
    g_Qarr = []
    kwargs_edges = []
    kwargs_nodes = []
    kwargs_arrows = []
    colormap = :glasbey_hv_n256
    for ii = 1:indices_len
        Q_graph = Q_array[ii]
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
        node_size = 200.0
        edge_width_Q = [10.0 for i in 1:ne(g_Q)]
        arrow_size_Q = [40.0 for i in 1:ne(g_Q)]
        node_labels_Q = repr.(1:nv(g_Q))
        push!(kwargs_edges, (; elabels=elabels, elabels_color=elabels_color, elabels_textsize=elabels_fontsize, edge_color=edge_color_Q, edge_width=edge_width_Q))
        push!(kwargs_nodes, (; node_color=node_color, node_size=node_size, nlabels=node_labels_Q, nlabels_textsize=nlabels_fontsize))
        push!(kwargs_arrows, (; arrow_size=arrow_size_Q))
        push!(g_Qarr, g_Q)
    end
    set_theme!(backgroundcolor=:white)
    fig = Figure(resolution=(4000, 6000))
    fct = 10
    for i = 1:indices_len
        if i <= Int(indices_len/2)
            ax = Axis(fig[2:fct+1, i]; title=latexstring("\$\\textbf{t=$(round(τ[i],sigdigits=3))}\\,\\textbf{(\\Delta=$(round(score[i],sigdigits=3)))}\$"), titlesize=60)
        else
            ax = Axis(fig[3*fct+3:4*fct+2, i-Int(indices_len/2)]; title=latexstring("\$\\textbf{t=$(round(τ[i],sigdigits=3))}\\,\\textbf{(\\Delta=$(round(score[i],sigdigits=3)))}\$"), titlesize=60)
        end
        hidedecorations!(ax); hidespines!(ax)
        graphplot!(ax, g_Qarr[i]; kwargs_edges[i]..., kwargs_nodes[i]..., kwargs_arrows[i]...)
    end
    for i = 1:indices_len
        if i <= Int(indices_len/2)
            ax = Axis3(fig[fct+2:2*fct+1, i], xticklabelsize=40, yticklabelsize=40, zticklabelsize=40, xlabelsize=40, ylabelsize=40, zlabelsize=40, azimuth = azimuth, elevation=elevation)
        else
            ax = Axis3(fig[4*fct+3:5*fct+2, i-Int(indices_len/2)], xticklabelsize=40, yticklabelsize=40, zticklabelsize=40, xlabelsize=40, ylabelsize=40, zlabelsize=40, azimuth = azimuth, elevation=elevation)
        end
        #scatter!(ax, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], color=cgrad(colormap)[X_LN_array[i][1:res:end,1]]; markersize=mks)
        scatter!(ax, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], color=cgrad(colormap)[X_LN_array[i][1:res:end,2]]; markersize=mks)
        #scatter!(ax, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], color=cgrad(colormap)[X_LN_array[i][1:res:end,3]]; markersize=mks)
    end
    for i = 1:indices_len
        if i <= Int(indices_len/2)
            ax = Axis(fig[2*fct+2:3*fct+1, i], xticklabelsize=40, yticklabelsize=40, xlabelsize=40, ylabelsize=40, ylabel=latexstring("\$i_c\$"), xlabel=L"t")
        else
            ax = Axis(fig[5*fct+3:6*fct+2, i-Int(indices_len/2)], xticklabelsize=40, yticklabelsize=40, xlabelsize=40, ylabelsize=40, ylabel=latexstring("\$i_c\$"), xlabel=L"t")
        end
        t_ax = [dt:dt:Lt*dt...]
        lines!(ax,t_ax,X[1:Lt],color=cgrad(colormap)[X_LN_array[i][1:Lt,2]], linewidth=2)
    end
    Label(fig[1, 1:Int(indices_len/2), Top()], title; textsize=100)
    save("figure/figure" * string(figure_number[1])*string(figure_number[2]) * ".png", fig)
end
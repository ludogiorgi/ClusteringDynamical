module GenerateCoarseCluster
using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using Combinatorics
using MultiDimensionalClustering.CommunityDetection, ProgressBars
using MarkovChainHammer.BayesianMatrix: BayesianGenerator
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix: perron_frobenius
using MultiDimensionalClustering.AlternativeGenerator
using SparseArrays, NetworkLayout, Graphs, Printf, NetworkLayout, GraphMakie, Graphs, InvertedIndices, LaTeXStrings

export coarse_cluster, plot_coarse_cluster1, plot_coarse_cluster2, read_coarse_cluster

function label_ordering(cc_minus, cc_plus)
    cc_plus_len = length(union(cc_plus))
    perms_start = [1:cc_plus_len...]
    perms_array = []
    perms = permutations(perms_start)
    for p in perms
        push!(perms_array,p)
    end
    cc_plus_temp = copy(cc_plus)
    cc_plus_ord = copy(cc_plus)
    ScoreOld = 1.
    ScoreNew = 1.
    for i in eachindex(perms_array)
        for j in eachindex(cc_plus)
            cc_plus_temp[j] = findall(y->y==cc_plus[j],perms_array[i])[1]
        end
        ScoreNew = sum(diff_check.(cc_minus, cc_plus_temp))/length(cc_minus)
        if ScoreNew < ScoreOld
            cc_plus_ord = copy(cc_plus_temp)
            ScoreOld = ScoreNew
        end
    end
    return ScoreOld, cc_plus_ord
end

function diff_check(a,b)
    if a == b 
        return 0.
    else 
        return 1.
    end
end

function Q_performance(X_LN,Q_LN,Q_LN_pert,dt,factor)
    l,_ = eigen(Q_LN_pert)
    n_tau = Int(ceil(-1/real(l[end-1])/dt))*factor
    nc = size(Q_LN_pert)[1]
    Parr = zeros(Float64,n_tau,nc,nc)
    Qarr = zeros(Float64, n_tau,nc,nc)
    Qarr_pert = zeros(Float64,n_tau,nc,nc)   
    for i = 0:n_tau-1
        Parr[i+1,:,:] = perron_frobenius(X_LN,step=i+1)
        Qarr[i+1,:,:] = exp(Q_LN * dt* i)
        Qarr_pert[i+1,:,:] = exp(Q_LN_pert * dt* i)
    end
    return Parr, Qarr, Qarr_pert
end

function coarse_cluster(X, dt, indices, τ, qmins; factor=10, iteration1=3, iteration2=5, file=false)
    indices_len = length(indices)
    X_len = length(X)
    t_steps = zeros(Int64, 3, indices_len)
    for i in eachindex(indices)
        t_steps[1,i] = Int(round(τ[i] *0.9/dt))
        t_steps[2,i] = Int(round(τ[i] /dt))
        t_steps[3,i] = Int(round(τ[i] *1.1/dt))
    end
    X_LN = zeros(Int64, X_len, 3)
    X_LN_array = []
    Q_array = []
    Q_pert_array = []
    Pt_array = []
    Qt_array = []
    Qt_pert_array = []
    if file != false
        clustername = "/data/" * file * "_coarse_cluster.hdf5"
        hfile = h5open(pwd() * clustername, "w")
    end
    for i = 1:indices_len
        for j = 1:3
            P = perron_frobenius(X,step=t_steps[j,i])
            nc, ln_nc = leicht_newman(P,qmins[i])
            X_LN[:,j] = classes_timeseries(ln_nc, X)
        end
        if maximum([length(union(X_LN[:,1])), length(union(X_LN[:,2])), length(union(X_LN[:,3]))]) <= 8
            score1, X_LN[:,1] = label_ordering(X_LN[:,2], X_LN[:,1])
            score2, X_LN[:,3] = label_ordering(X_LN[:,2], X_LN[:,3])
            score = (score1 + score2) / 2
        end
        Q_LN = mean(BayesianGenerator(X_LN[:,2];dt=dt))
        Q_LN_pert = copy(Q_LN)
        for _ in 1:iteration1
            Q_LN_pert = alternative_generator(Q_LN_pert,X_LN[:,2],dt,iteration2)
        end
        push!(Q_array, Q_LN)
        push!(Q_pert_array, Q_LN_pert)
        Pt, Qt, Qt_pert = Q_performance(X_LN[:,2],Q_LN,Q_LN_pert,dt,factor)
        push!(Pt_array, Pt)
        push!(Qt_array, Qt)
        push!(Qt_pert_array, Qt_pert)
        push!(X_LN_array, X_LN)
        if file != false
            hfile["X_LN_tms=$i"] = X_LN
            hfile["Q_tms=$i"] = Q_LN
            hfile["Q_pert_tms=$i"] = Q_LN_pert
            hfile["Pt_array_tms=$i"] = Pt
            hfile["Qt_array_tms=$i"] = Qt
            hfile["Qt_pert_tms=$i"] = Qt_pert
            hfile["score_tms=$i"] = score
        end
    end
    if file != false
        hfile["t_steps"] = t_steps
        close(hfile)
    end
    return X_LN_array, Q_array, Q_pert_array, Pt_array, Qt_array, Qt_pert_array
end

function plot_coarse_cluster1(x, X, τ, dt, X_LN_array, Q_array, score, title, figure_number; res = 1, mks = 5, azimuth=0.0pi, elevation=0.0pi, Lt=1000)
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
            ax = Axis(fig[2*fct+2:3*fct+1, i], xticklabelsize=40, yticklabelsize=40, xlabelsize=40, ylabelsize=40)
        else
            ax = Axis(fig[5*fct+3:6*fct+2, i-Int(indices_len/2)], xticklabelsize=40, yticklabelsize=40, xlabelsize=40, ylabelsize=40)
        end
        t_ax = [dt:dt:Lt*dt...]
        lines!(ax,t_ax,X[1:Lt],color=cgrad(colormap)[X_LN_array[i][1:Lt,2]])
    end
    Label(fig[1, 1:Int(indices_len/2), Top()], title; textsize=100)
    save("figure/figure" * string(figure_number[1])*string(figure_number[2]) * ".png", fig)
end

function plot_coarse_cluster2(dt, Pt_array, Qt_array, Qt_pert_array, index, title, figure_number)
    PFdim = length(Pt_array[index][1,1,:])
    PFlen = length(Pt_array[index][:,1,1])
    xax = [dt:dt:PFlen*dt...]
    set_theme!(backgroundcolor=:white)
    fig = Figure(resolution=(4000, 2800))
    for i in 1:PFdim
        for j in 1:PFdim
            ax = Axis(fig[j, i], ylabel=latexstring("\$P_{$i,$j}(t)\$"), xlabel=L"t", xticklabelsize=40, yticklabelsize=40, xlabelsize=60, ylabelsize=60)
            lines!(ax,xax, Pt_array[index][:,i,j],color=:red, linewidth=5, label=latexstring("P(t)"))
            lines!(ax,xax, Qt_array[index][:,i,j],color=:black, linewidth=5, label=latexstring("\\textrm{exp}(Q t)"))
            lines!(ax,xax, Qt_pert_array[index][:,i,j],color=:blue, linewidth=5, label=latexstring("\\textrm{exp}(Q_{\\textrm{pert}}t)"))
            if i == 1 && j == 1
                axislegend(ax; labelsize=80)
            end
        end
    end
    Label(fig[1, 1:PFdim, Top()], title; textsize=100)
    save("figure/figure" * string(figure_number[1])*string(figure_number[2]) * ".png", fig)
end

function read_coarse_cluster(file)
    hfile = h5open(pwd()*"/data/" * file * "_coarse_cluster.hdf5")
    t_steps = read(hfile["t_steps"])
    indices_len = size(t_steps)[2]
    X_LN_array = []
    Q_array = []
    Q_pert_array = []
    Pt_array = []
    Qt_array = []
    Qt_pert_array = []
    score = []
    for i = 1:indices_len
        push!(X_LN_array, read(hfile["X_LN_tms=$i"]))
        push!(Q_array, read(hfile["Q_tms=$i"]))
        push!(Q_pert_array, read(hfile["Q_pert_tms=$i"]))
        push!(Pt_array, read(hfile["Pt_array_tms=$i"]))
        push!(Qt_array, read(hfile["Qt_array_tms=$i"]))
        push!(Qt_pert_array, read(hfile["Qt_pert_tms=$i"]))
        push!(score, read(hfile["score_tms=$i"]))
    end
    close(hfile)
    return X_LN_array, Q_array, Q_pert_array, Pt_array, Qt_array, Qt_pert_array, score
end

end
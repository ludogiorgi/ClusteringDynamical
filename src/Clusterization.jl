module Clusterization

using ParallelKMeans, LinearAlgebra, Statistics, Random, HDF5, ProgressBars
using MultiDimensionalClustering.CommunityDetection:leicht_newman, classes_timeseries, greatest_common_cluster
using MultiDimensionalClustering.AlternativeGenerator
using MarkovChainHammer.BayesianMatrix:BayesianGenerator
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix:perron_frobenius

export cluster, write_cluster, read_cluster

struct Cluster_timeseries
    xc
    GCC
    Perr
    PerrGen
    PerrGenPert
end

function Q_performance(X_LN,Q_LN,Q_LN_pert,nc,dt,factor)
    l,_ = eigen(Q_LN_pert)
    n_tau = Int(ceil(-1/real(l[end-1])/dt))*factor
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

function cluster(x, t_step, nc, dt; k_clusters=1000, n_threads=12, pm_steps = 4, performance=false, factor=6, k_means=true, iteration1=1, iteration2=4)
    indices = [t_step-pm_steps:t_step+pm_steps...] 
    if k_means   
        kmn = kmeans(x, k_clusters; n_threads=n_threads, max_iters=10^6)
        X = kmn.assignments
    else
        X = x
    end
    P = perron_frobenius(X,step=t_step)
    nc, ln_nc = leicht_newman(P,nc)
    X_LN = classes_timeseries(ln_nc, X)
    ln_nc_indices = greatest_common_cluster(X, nc, indices; progress_bar = false)
    GCC = classes_timeseries(ln_nc_indices, X)
    P_LN = perron_frobenius(X_LN,step=t_step)
    Q_LN = mean(BayesianGenerator(X_LN;dt=dt))
    Q_LN_pert = copy(Q_LN)
    for j = 1:iteration1
        Q_LN_pert = alternative_generator(Q_LN_pert, X_LN, dt,iteration2)
    end
    cluster_timeseries = Cluster_timeseries(X_LN, GCC, P_LN, Q_LN, Q_LN_pert)
    if performance
        Parr, Qarr, Qarr_pert = Q_performance(X_LN,Q_LN,Q_LN_pert,nc,dt,factor)
        return nc, cluster_timeseries, Parr, Qarr, Qarr_pert
    else
        return nc, cluster_timeseries
    end
end

function write_cluster(x, t_step, nc_input, dt, file_name; k_means=false, performance=true, progress_bars = true)
    nc_output = zeros(Int64,length(t_step),length(nc_input))
    if progress_bars == true 
        iterator_i = ProgressBar(eachindex(t_step))
        iterator_j = ProgressBar(eachindex(nc_input))
    else 
        iterator_i = eachindex(t_step)
        iterator_j = eachindex(nc_input)
    end
    for i in iterator_i
        for j in iterator_j
            nc, CT, Parr, Qarr, Qarr_pert = cluster(x, t_step[i], nc_input[j], dt; k_means=k_means, performance=performance,iteration1=10,iteration2=2)
            hfile = h5open(pwd() * file_name * "-t_step=$(t_step[i])-nc=$(nc_input[j]).hdf5", "w")
            hfile["xc"] = CT.xc
            hfile["GCC"] = CT.GCC
            hfile["PerrGen"] = CT.PerrGen
            hfile["Parr"] = Parr
            hfile["Qarr"] = Qarr
            hfile["Qarr_pert"] = Qarr_pert
            close(hfile)
            nc_output[i,j] = nc
        end
    end
    return nc_output
end

function read_cluster(t_step,nc_output,file_name)
    xc_array = []
    GCC_array = []
    PerrGen_array = []
    Parr_array = []
    Qarr_array = []
    Qarr_pert_array = []
    for i in eachindex(t_step)
        for j in eachindex(nc_output[i,:])
            hfile = h5open(pwd() * file_name * "-t_step=$(t_step[i])-nc=$(nc_output[i,j]).hdf5")
            xc = read(hfile["xc"])
            GCC = read(hfile["GCC"])
            PerrGen = read(hfile["PerrGen"])
            Parr = read(hfile["Parr"])
            Qarr = read(hfile["Qarr"])
            Qarr_pert = read(hfile["Qarr_pert"])
            close(hfile)
            push!(xc_array,xc)
            push!(GCC_array,GCC)
            push!(PerrGen_array,PerrGen)
            push!(Parr_array,Parr)
            push!(Qarr_array,Qarr)
            push!(Qarr_pert_array,Qarr_pert)
        end
    end
    return xc_array, GCC_array, PerrGen_array, Parr_array, Qarr_array, Qarr_pert_array
end
end

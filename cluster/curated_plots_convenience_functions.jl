using Combinatorics
function diff_check(a,b)
    if a == b 
        return 0.
    else 
        return 1.
    end
end
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
##


function my_split(X)
    numstates = 2
    r0 = kmeans(X, numstates; max_iters=10000)
    child_0 = (r0.assignments .== 1)
    child_1 = (!).(child_0)
    children = [view(X, :, child_0), view(X, :, child_1)]
    return r0.centers, children
end
level_global_indices(level) = 2^(level-1):2^level-1
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

function state_tree_embedding(X, levels)
    parent_views = []
    centers_list = Vector{Vector{Float64}}[]
    push!(parent_views, X)
    ## Level 1
    centers, children = my_split(X)
    push!(centers_list, [centers[:, 1], centers[:, 2]])
    push!(parent_views, children[1])
    push!(parent_views, children[2])
    ## Levels 2 through levels
    for level in ProgressBar(2:levels)
        for parent_global_index in level_global_indices(level)
            centers, children = my_split(parent_views[parent_global_index])
            push!(centers_list, [centers[:, 1], centers[:, 2]])
            push!(parent_views, children[1])
            push!(parent_views, children[2])
        end
    end
    @info "done with k-means"
    centers_matrix = zeros(length(centers_list[1][1]), length(centers_list[1]), length(centers_list))
    for i in eachindex(centers_list)
        centers_matrix[:, :, i] = hcat(centers_list[i]...)
    end
    centers_list = [[centers_matrix[:,1, i], centers_matrix[:,2, i]] for i in 1:size(centers_matrix)[3]]
    # constructing embedding with 2^levels number of states
    # note that we can also choose a number less than levels
    embedding = StateTreeEmbedding(centers_list, levels)
    return embedding 
end
##
function graph_from_PI(PI)
    N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])
    adj = spzeros(Int64, N, N)
    adj_mod = spzeros(Float64, N, N)
    for i in ProgressBar(eachindex(PI))
        ii = PI[i][1]
        jj = PI[i][2]
        modularity_value = PI[i][3]
        adj[ii, jj] += 1
        adj_mod[ii, jj] = modularity_value
    end 
    N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])
    node_labels = zeros(N)
    for i in eachindex(PI)
        node_labels[PI[i][1]] = PI[i][3]
    end
    return node_labels, adj, adj_mod, length(PI)
end
#=
@info "applying embedding"
markov_embedding = [embedding(x[:,i]) for i in 1:size(x)[2]]
@info "done with embedding"
##
Q = mean(BayesianGenerator(markov_embedding))
Λ, V =  eigen(Q)
##
τs = reverse((-1 ./ real.(Λ[end-20:end]))[1:end-1])
##
P1 = exp(Q * τs[1])
P2 = exp(Q * τs[3])
##
@info "applying LN1"
q_min = 0.0
F, G, H, PI = leicht_newman_with_tree(P1, q_min)
##
@info "applying LN2"
q_min = 0.0
F2, G2, H2, PI2 = leicht_newman_with_tree(P2, q_min)
##
X_LN = classes_timeseries(F, markov_embedding) # 2 groups
X_LN = classes_timeseries(F2, markov_embedding) # 4 groups
=#
module CommunityDetection 
    using LinearAlgebra, ProgressBars
    using MarkovChainHammer.TransitionMatrix: perron_frobenius
    export leicht_newman, classes_timeseries, greatest_common_cluster, leicht_newman_with_tree

    function modularity_matrix(A)
        N = size(A)[1]
        b = zeros(N, N)
        for i = 1:N, j = 1:N
            b[i, j] = (A[i, j] - (1 - A[i, i]) * (sum(A[j, :])) / N) / N
        end
        B = Symmetric(b + b')
        return B
    end

    function principal_vector(B::Symmetric)
        s = ones(Int, size(B)[1])
        Λ, V = eigen(B)
        v₁ = V[:, sortperm(real.(Λ))[end]]
        s[v₁.<=0] .= -1
        return s
    end

    function modularity(B, s)
        return s' * (B * s)
    end

    function modularity_eig(B)
        Λ, V = eigen(B)
        return maximum(Λ)
    end

    modularity(B::Symmetric) = modularity(B, principal_vector(B))

    function split_community(B, indices, q_min)
        Bg = B[indices, :][:, indices]
        Bg = Bg - Diagonal(sum(Bg, dims=1)[:])
        Bg = Symmetric(Bg + Bg')
        s = principal_vector(Bg)
        q = modularity(Bg)
        qq = q_min

        if (q > q_min)
            ind1 = [i for (j, i) in enumerate(indices) if s[j] == 1]
            ind2 = [i for (j, i) in enumerate(indices) if s[j] == -1]
            qq = q
            return ind1, ind2, qq
        end
        return [], [], qq
    end

    function leicht_newman(A, q_min::Float64)
        B = modularity_matrix(A)
        n = size(A)[1]
        W, F, G = [collect(1:n)], [], []
        qOld = 0.0
        H = []
        while (length(W) > 0)
            w = popfirst!(W)
            ind1, ind2, q = split_community(B, w, q_min)
            if (length(ind1) > 0) & (length(ind2) > 0)
                W = [ind1, ind2, W...]
                push!(H, [ind1, ind2, q])
                push!(G, q)
                if q > 0
                    qOld = q
                end
            else
                push!(F, w)
            end
        end
        return F, G, H
    end

    function leicht_newman(X, qmin::Float64, indices; progress_bar = false)
        LN = []
        if progress_bar 
            iterator = ProgressBar(indices)
        else
            iterator = indices
        end
        for i in iterator 
            P = perron_frobenius(X; step = i)
            lntmp = leicht_newman(P, qmin)
            push!(LN, lntmp)
        end
        return LN
    end                                                                                                                 

    function leicht_newman_with_tree(A, q_min::Float64)
        B = modularity_matrix(A)
        n = size(A)[1]
        W, F, G, P1, P2 = [collect(1:n)], [], [], [1], []
        qOld = 0.0
        H = []
        global_index = 1
        while (length(W) > 0)
            w = popfirst!(W)
            p1 = popfirst!(P1)
            ind1, ind2, q = split_community(B, w, q_min)
            if (length(ind1) > 0) & (length(ind2) > 0)
                W = [ind1, ind2, W...]
                P1 = [global_index + 1, global_index + 2, P1...]
                P2 = push!(P2, (p1, global_index + 1, q))
                P2 = push!(P2, (p1, global_index + 2, q))
                global_index += 2
                push!(H, [ind1, ind2, q])
                if q > 0
                    qOld = q
                end
            else
                push!(F, w)
                push!(G, qOld)
            end
        end
        return F, G, H, P2
    end
                                                                
    function q_next(index,PI)
        i = 1
        while i <= length(PI) && PI[i][1] != PI[index][2]
            i += 1
        end
        if i > length(PI)
            return 0.
        else
            return PI[i][3]
        end
    end

    function add_element(PI_ind,q,PI,index)
        for i in eachindex(PI)
            if PI[i][1] == index
                push!(PI_ind, i)
                push!(q, q_next(i,PI))
            end
        end
        q_ind = sortperm(q,rev=true)
        PI_ind = PI_ind[q_ind]
        q = q[q_ind]
        return PI_ind,q
    end                                                            
                                                                
    function leicht_newman(A, nc::Int64)
        _, G, H, PI = leicht_newman_with_tree(A, 0.)
        if nc > length(G) 
            println("Maximum number of clusters is ", length(G)+1, ". The number of clusters has been now changed to ", length(G)+1)
            nc = length(G)
        end
        PI_ind = []
        q = []
        PI_ind,q = add_element(PI_ind,q,PI,1) 
        for j = 1:nc-2
            ind_rem = popfirst!(PI_ind)
            popfirst!(q)
            PI_ind,q = add_element(PI_ind,q,PI,PI[ind_rem][2]) 
        end
        PI_cluster = []
        for i in eachindex(PI)
            h = mod(i,2)
            if h == 0 h = 2 end
            push!(PI_cluster, [Int(floor((i+1)/2)),h])
        end
        H_ind = []
        for i in PI_ind
            push!(H_ind,PI_cluster[i])
        end
        ln_nc = []
        for i in eachindex(H_ind)
            push!(ln_nc, H[H_ind[i][1]][H_ind[i][2]])
        end                                                           
        return nc, ln_nc
    end
                                                                        
    function leicht_newman(X, nc::Int64, indices; progress_bar = false)
        LN = []
        if progress_bar 
            iterator = ProgressBar(indices)
        else
            iterator = indices
        end
        Threads.@threads for i in iterator 
            P = perron_frobenius(X; step = i)
            nc_temp, lntmp = leicht_newman(P, nc)
            push!(LN, lntmp)
        end
        return LN
    end  

    function leicht_newman_intersection(LN)
        inter_array = copy(LN)
        for h = eachindex(inter_array[1:end-1])
            inter_array1 = []
            for i = eachindex(inter_array[1:end-1])
                inter_array2 = []
                for j = eachindex(inter_array[i]), k = eachindex(inter_array[i+1])
                    inter_temp = intersect(inter_array[i][j], inter_array[i+1][k])
                    if length(inter_temp) > 0
                        push!(inter_array2, inter_temp)
                    end
                end
                push!(inter_array1, inter_array2)
            end
            inter_array = inter_array1
        end
        return inter_array[1]
    end

    function greatest_common_cluster(X, qmin::Float64, indices; progress_bar=false)
        LNs = leicht_newman(X, qmin, indices; progress_bar=progress_bar)
        return leicht_newman_intersection(LNs)
    end
                                                                                                                            
     function greatest_common_cluster(X, nc::Int64, indices; progress_bar=false)
        LNs = leicht_newman(X, nc, indices; progress_bar=progress_bar)
        return leicht_newman_intersection(LNs)
    end                                                                                                                       

    function classes_timeseries(LN, X)
        L = length(X)
        ln_classes = zeros(Int64, L)
        Threads.@threads for i = 1:L
            for j in eachindex(LN)
                if X[i] in LN[j][:]
                    ln_classes[i] = j
                end
            end
        end
        return ln_classes
    end

end

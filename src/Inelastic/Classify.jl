module Classify
#excitation-type classifiers + calibration
using LinearAlgebra
using Statistics
using SparseArrays
using Arpack

using ...Core.Types: TopologyMetricSpec, FeatureBundle, STATIC_INVARIANT_SCHEMA, DYNAMIC_INVARIANT_SCHEMA, ScalarInvariant, VectorInvariant, QPresult

using ..Topology: topology_distance, L2Norm, AbsoluteDistance, L1Distance, CosineDistance

export classifyQP
#------------------------------------#
#               Utility              #
#------------------------------------#
#
# Pairwise distance matrix
function pairwise_distance( 
        features::Vector{FeatureBundle},
        metric::TopologyMetricSpec
    )
    N=length(features); D = zeros(Float64,N,N)
    Threads.@threads for i in 1:N
        for j in i+1:N
            d = topology_distance(features[i], features[j], metric)
            D[i,j] = d
            D[j,i] = d
        end
    end
    return D
end
#
# kNN affinity Graph 
function knn_affinity(
        D::Matrix{Float64}, k::Int; 
        σ::Union{Nothing,Float64}=nothing)
    N=size(D,1)
    rows = Vector{Int}(undef, N*k) #new
    cols = Vector{Int}(undef, N*k) #new
    vals = Vector{Float64}(undef, N*k) #new
    ctr = 1 #new
    for i in 1:N
        idx = partialsortperm(@view(D[i,:]), 1:k+1)
        for j in idx
            i == j && continue
            w = isnothing(σ) ? 1.0 : exp(-D[i,j]^2 / (2σ^2))
            rows[ctr] = i; cols[ctr] = j; vals[ctr] = w #new
            ctr += 1 #new
        end
    end
    resize!(rows,ctr-1); resize!(cols,ctr-1); resize!(vals, ctr-1)

    #kNN graph
    W = sparse(rows, cols, vals, N, N) #NEW
    #return sparse(rows, cols, vals, N, N) #OLD
    
    #Symmetrize kNN graph for spectral clustering
    return max.(W,W') #NEW
end
#
#-------------------------------------------------------#
#               Spectral Clustering Core                #
#-------------------------------------------------------#
function spectral_clustering(
        W::SparseMatrixCSC{Float64, Int},
        ncluster::Int
    )
    d = sum(W, dims=2); invD = spdiagm(0 => vec(1.0 ./ sqrt.(d .+ eps())))
    L = LinearAlgebra.I - invD*W*invD #normalized laplacian
    

    #evals, evecs = eigs(L, nev=ncluster+1, which=:SM) #ARPACK (OLD)
    nev = min(ncluster + 1, size(L,1)); evals, evecs = eigs(L, nev=nev, which=:SM) #NEW
    
    
    evals = real(evals); evecs = real(evecs)
    #idx = sortperm(evals); evals=evals[idx]; U=evecs[:,idx] #OLD
    idx = sortperm(evals); evals=evals[idx]; evecs=evecs[:,idx] #NEW
    
    #only use first nclusters eigenvectors of sorted spectrum
    #U = evecs[:, 1:ncluster] #OLD
    U = evecs[:, 1:min(ncluster, size(evecs,2))] #NEW
    U = U ./ sqrt.(sum(abs2, U; dims=2) .+ eps())
    labels = kmeans_assign(U, ncluster)
    
    return labels, evals
end
#
#
#Helper: Initialize k-means++ algorithm. (performance optimization)
function _kmeans_init(X::Matrix{Float64}, k::Int)
    N, d = size(X); chosen = Int[]
    push!(chosen, rand(1:N))
    dists = fill(Inf,N)
    for _ in 2:k
        c = X[chosen[end],:]
        @inbounds for i in 1:N
            dists[i] = min(dists[i], sum(abs2, X[i,:] .- c))
        end
        total = sum(dists); r = rand()*total
        acc = 0.0; next = N
        for i in 1:N
            acc += dists[i]
            if acc >= r; next = i; break; end
        end
        push!(chosen, next)
    end
    return X[chosen,:]
end
#
#
#assign labels using kmeans
function kmeans_assign(X::Matrix{Float64}, k::Int; maxiter=100)
    #OLD
    #N,d = size(X); centers=X[rand(1:N, k), :]; 
    
    #NEW
    N,d = size(X); centers = _kmeans_init(X,k)

    labels=zeros(Int,N)
    for _ in 1:maxiter
        for i in 1:N
            minDist = typemax(Float64); best = 0
            for j in 1:k
                dist = sum(abs2, X[i,:] .- centers[j,:])
                if dist < minDist; minDist = dist; best = j; end
            end
            labels[i] = best
        end
        for j in 1:k #update
            inds = findall(labels .== j)
            !isempty(inds) && (centers[j,:] .= vec(sum(X[inds,:], dims=1))./length(inds))
        end
    end
    return labels
end
#
#calculate neighborhood label entropy for each sample i, computed over
#k-nearest neighbors (excluding self). Uses natural log, so H ∈ [0, log(nclasses)].
function knn_entropy(labels::Vector{Int}, D::Matrix{Float64}, k::Int)
    N=length(labels); H=zeros(Float64,N); ks = 1:k+1; nclusters=maximum(labels)
    Threads.@threads for i in 1:N
        idx = partialsortperm(D[i,:], ks)#include self (first entry)
        neighbors = labels[idx[2:end]]   #exclude self (after first entry)
        hist = zeros(Int,nclusters)
        @inbounds for lbl in neighbors; hist[lbl] += 1; end
        invk = 1/length(neighbors); hi = 0.0
        @inbounds for c in hist; if c > 0; p=c*invk; hi -= p*log(p); end; end
        H[i] = hi
    end
    return H
end
#
#dataset validator
function validate_bundles!(F::Vector{FeatureBundle}, metric::TopologyMetricSpec)
    isempty(F) && error("No FeatureBundles provided!")
    
    # --- Static --- #
    if !isnothing(F[1].static)
        keys_ref = keys(F[1].static.invariants)
        for fb in F
            keys(fb.static.invariants) == keys_ref || error("Inconsistent static invariant keys across FeatureBundles!")
        end
        for k in keys_ref
            v=F[1].static.invariants[k]
            v isa AbstractVector || continue
            for fb in F
                length(fb.static.invariants[k]) == length(v) || error("Invariant $k has inconsistent vector length!")
            end
        end
    end
    
    # --- Dynamic --- #
    if !isnothing(F[1].dynamic)
        keys_ref = keys(F[1].dynamic.invariants)
        for fb in F
            keys(fb.dynamic.invariants) == keys_ref || error("Inconsistent dynamic invariant keys across FeatureBundles!")
        end
        for k in keys_ref
            v = F[1].dynamic.invariants[k]
            v isa AbstractVector || continue
            for fb in F
                length(fb.dynamic.invariants[k]) == length(v) || error("Dynamic invariant $k has inconsistent vector length!")
            end
        end
    end
    return nothing
end
#
#Ensure that the TopologyMetricSpec is compatible with the static 
#invariant schema inferred from FeatureBundles. This does NOT modify
#structure, only checks semantic compatibility. 
function validate_metric_schema!(F::Vector{FeatureBundle}, metric::TopologyMetricSpec)
    # --- Static --- #
    if !isnothing(F[1].static)
        sinvs = F[1].static.invariants
        for(key, dist) in metric.static
            haskey(sinvs,key) || error("Invariant $key given to TopologyMetric is not present in FeatureBundle!")
            kind = get(STATIC_INVARIANT_SCHEMA, key, nothing)
            kind === nothing && error("Invariant $key given to TopologyMetric is not registered in STATIC_INVARIANT_SCHEMA!")
            val = sinvs[key]; _check_kind_distance_compat(key,val,kind,dist)
        end
    end
    # --- Dynamic --- #
    if !isnothing(F[1].dynamic)
        dinvs = F[1].dynamic.invariants
        for (key, dist) in metric.dynamic
            haskey(dinvs, key) || error("Invariant $key given to TopologyMetric is not present in dynamic FeatureBundle!")
            kind = get(DYNAMIC_INVARIANT_SCHEMA, key, nothing)
            kind === nothing && error("Invariant $key is not registered in DYNAMIC_INVARIANT_SCHEMA!")
            val = dinvs[key]; _check_kind_distance_compat(key,val,kind,dist)
        end
    end
    return nothing
end

#Helper: Enforces invariant/distance compatibility
function _check_kind_distance_compat(key, val, kind, dist)
    if kind isa ScalarInvariant
        val isa Real || error("Invariant $key is scalar in schema but of type $(typeof(val))!")
        dist isa AbsoluteDistance || error("Invariant $key is scalar but metric is of type $(typeof(dist))!")
    elseif kind isa VectorInvariant
        val isa AbstractVector || error("Invariant $key is a vector in schema but of type $(typeof(val))!")
        dist isa Union{L1Distance, L2Norm, CosineDistance} || error("Invariant $key is a vector but metric is of type $(typeof(dist))!")
    end
end
#-------------------------------------#
#             Public API              #
#-------------------------------------#
"""
    classifyQP(features, metric; k=10, nclusters)

Performs quasi-particle classification using kNN + spectral clustering
on an arrya of `FeatureBundle`s.

Workflow:
    - Validates bundle consistency and metric/schema compatibility
    - Pairwise topology distance matrix (parallelised)
    - kNN affinity graph (binary or Gaussian kernel via `σ`)
    - Normalized Laplacian spectral clustering via ARPACK
    - Per-sample confidence from kNN label entropy

Keyword Arguments:
    - `k`         ~ number of nearest neighbors
    - `nclusters` ~ number of spectral clusters (must be ≥ 2)
    - `σ`         ~ Gaussian kernel bandwidth for affinity weights (nothing = binary)

Returns a `QPResult`.
"""
function classifyQP(
        features::Vector{FeatureBundle},
        metric::TopologyMetricSpec;
        k::Int=10, nclusters::Int,
        σ::Union{Nothing,Float64}=nothing
    )::QPresult
    nclusters ≥ 2 || error("nclusters must be ≥ 2, got nclusters=$nclusters !")
    k ≥ 1 || error("k must be ≥ 1, got k=$k !")
    length(features) > nclusters || error("Less FeatureBundles ($(length(features))) than clusters ($nclusters)!")

    #Validate
    validate_bundles!(features, metric); validate_metric_schema!(features, metric)

    #Distance & Affinity Graph Matrix 
    D = pairwise_distance(features, metric); A = knn_affinity(D, k; σ)
    
    #Spectral Clustering ~ Laplacian eigenvals & eigengap(s)
    labels, eigenvals = spectral_clustering(A, nclusters)
    
    #OLD
    eigengap = (nclusters < length(eigenvals)) ? (eigenvals[nclusters+1] - eigenvals[nclusters]) : 0.0
    #OLD NEW
    #eigengap = (nclusters < length(eigenvals)) ? (eigenvals[nclusters] - eigenvals[nclusters-1]) : 0.0
    #OLD NEW NEW
    #idx = findfirst(x -> x > 1e-4, eigenvals)
    #eigengap = isnothing(idx) ? 0.0 : eigenvals[idx]
    
    #NEW NEW NEW
    #idx = findfirst(>(1e-4), eigenvals)
    #if isnothing(idx) || idx ≥ length(eigenvals)
    #    #nclusters=1
    #    eigengap=0.0
    #else
    #    gaps = diff(eigenvals)
    #    shift = argmax(gaps[idx:end])
    #    best_k = idx + shift - 1
    #    #nclusters = best_k
    #    eigengap = gaps[best_k]
    #end
    
    #NEW NEW NEW NEW
    #gaps = diff(eigenvals); shift = argmax(gaps[1:end]); best_k = shift - 1; eigengap = gaps[best_k]
    
    #NEW NEW NEW NEW NEW (ALL OF THESE ARE BAD) :(
    #gaps = diff(eigenvals); kmax = min(length(eigenvals)-1,20) 
    #scores = similar(gaps, kmax)
    #for k in 2:kmax; scores[k] = gaps[k] / max(eigenvals[k], eps()); end
    #best_k = argmax(scores); eigengap = gaps[best_k]

    #Graph label entropy & confidence
    entropy = knn_entropy(labels, D, k); confidence = 1 ./ (1 .+ entropy)
    threshold = 0.5*log(nclusters); ambiguous = entropy .> threshold
    
    #meta-data
    meta = Dict(:spectral=>Dict(:eigenvals=>eigenvals, :eigengap=>eigengap),
                :uncertainty=>Dict(:confidence=>confidence,
                                   :entropy => entropy, 
                                   :ambiguous=>ambiguous,
                                   :threshold=>threshold),
                :settings=>Dict(:metric=>metric, :nclusters=>nclusters, :k=>k)
               )
    return QPresult(labels, confidence, eigenvals, meta, k)
end
#
end #module

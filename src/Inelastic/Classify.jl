module Classify
#excitation-type classifiers + calibration
using LinearAlgebra
using Statistics
using SparseArrays
using Arpack

using ..Types
using ..Topology
#import .Topology: topology_distance

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
    rows = Vector{Float64}(undef, N*k) #new
    cols = Vector{Float64}(undef, N*k) #new
    vals = Vector{Float64}(undef, N*k) #new
    ctr = 1 #new
    for i in 1:N
        idx = partialsortperm(D[i,:], 1:k+1)
        for j in idx
            i == j && continue
            w = isnothing(σ) ? 1.0 : exp(-D[i,j]^2 / (2σ^2))
            rows[ctr] = i; cols[ctr] = j; vals[ctr] = w #new
            ctr += 1 #new
        end
    end
    resize!(rows,ctr-1); resize!(cols,ctr-1); resize!(vals, ctr-1) #new
    return sparse(rows, cols, vals, N, N)
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
    #evals, evecs = eigen(Matrix(L)); idx = sortperm(evals) #old
    #U = evecs[:,idx[1:ncluster]] #old
    evals, evecs = eigs(L, nev=ncluster, which=:SM) #ARPACK
    U = real(evecs); evals = real(evals) #ARPACK ℂ-valued by default
    idx = sortperm(evals)
    U = U ./ sqrt.(sum(abs2, U; dims=2) .+ eps())
    labels = kmeans_assign(U, ncluster)
    return labels, evals[idx]
end
#
#assign labels using kmeans
function kmeans_assign(X::Matrix{Float64}, k::Int; maxiter=100)
    N,d = size(X); centers=X[rand(1:N, k), :]; labels=zeros(Int,N)
    for _ in 1:maxiter
        for i in 1:N
            #old multi-allocation (memory intensive)
            #labels[i] = argmin([norm(X[i,:] .- centers[j,:]) for j in 1:k])
            minDist = typemax(Float64); best = 0
            for j in 1:k
                dist = sum(abs2, X[i,:] .- centers[j,:])
                if dist < minDist; minDist = dist; best = j; end
            end
            labels[i] = best
        end
        for j in 1:k #update
            inds = findall(labels .== j)
            #!isempty(inds) && (centers[j,:] .= mean(X[inds,:], dims=1)) #old
            !isempty(inds) && (centers[j,:] .= sum(X[inds,:], dims=1)./length(inds))
        end
    end
    return labels
end
#
#calculate neighborhood label entropy for each sample i, computed over
#k-nearest neighbors (excluding self). Uses natural log, so H ∈ [0, log(nclasses)].
function knn_entropy(labels::Vector{Int}, D::Matrix{Float64}, k::Int)
    N=length(labels); H=zeros(Float64,N); ks = 1:k+1
    Threads.@threads for i in 1:N
        idx = partialsortperm(D[i,:], ks)#inclue self (first entry)
        neighbors = labels[idx[2:end]]   #exclude self (after first entry)
        p = counts(neighbors) ./ length(neighbors) #label distribution
        H[i] = -sum(p .* log.(p .+ eps()))#Shannon-entropy
    end
    return H
end
#
#dataset validator
function validate_bundles!(F::Vector{FeatureBundle}, metric::TopologyMetricSpec)
    isempty(F) && error("No FeatureBundles provided!")
    keys_ref = keys(F[1].static.invariants)
    for fb in F
        keys(fb.static.invariants) == keys_ref || error("Inconsistent invariant keys across FeatureBundles!")
    end
    for k in keys_ref
        v=F[1].static.invariants[k]
        v isa AbstractVector || continue
        for fb in F
            length(fb.static.invariants[k]) == length(v) || error("Invariant $k has inconsistent vector length!")
        end
    end
    return nothing
end
#
#Ensure that the TopologyMetricSpec is compatible with the static 
#invariant schema inferred from FeatureBundles. This does NOT modify
#structure, only checks semantic compatibility. 
#
#(Currently, static invariants only.)
function validate_metric_schema!(F::Vector{FeatureBundle}, metric::TopologyMetricSpec)
    sinvs = F[1].static.invariants
    for(key, dist) in metric.static
        haskey(sinvs,key) || error("Invariant $key given to TopologyMetric is not present in FeatureBundle!")
        kind = get(STATIC_INVARIANT_SCHEMA, key, nothing)
        kind === nothing && error("Invariant $key given to TopologyMetric is not registered in STATIC_INVARIANT_SCHEMA!")
        val = sinvs[key]
        #enforce invariant/distance compatibility
        if kind isa ScalarInvariant
            val isa Real || error("Invariant $key is scalar in schema but of type $(typeof(val))!")
            dist isa AbsoluteDistance || error("Invariant $key is scalar but metric is of type $(typeof(dist))!")
        elseif kind isa VectorInvariant
            val isa AbstractVector || error("Invariant $key is a vector in schema but of type $(typeof(val))!")
            dist isa L2Norm || error("Invariant $key is a vector but metric is of type $(typeof(dist))!")
        end
    end
    return nothing
end
#-------------------------------------#
#             Public API              #
#-------------------------------------#
"""
    classifyQP(features, metric; k=10, nclusters)

Performs quasi-particle classification using kNN + spectral clustering.
"""
function classifyQP(
        features::Vector{FeatureBundle},
        metric::TopologyMetricSpec;
        k::Int=10, nclusters::Int
    )::QPresult
    #Validate
    validate_bundles!(features, metric); validate_metric_schema!(features, metric)
    #Distance & Affinity Graph Matrix 
    D = pairwise_distance(features, metric); A = knn_affinity(D, k)
    #Spectral Clustering ~ Laplacian eigenvals & eigengap(s)
    labels, eigenvals = spectral_clustering(A, nclusters)
    eigengap = (nclusters < length(eigenvals)) ? (eigenvals[nclusters+1] - eigenvals[nclusters]) : 0.0
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
    return QPresult(labels, confidence, eigvals, meta, k)
end
#
end #module

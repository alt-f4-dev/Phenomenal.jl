module Topology
using LinearAlgebra
using ...Core.Types
using ..PHysicalTDABridge: compute_static_topology, compute_dynamic_topology
#-------------------------#
# Distance Specifications #
#-------------------------#
struct WassersteinPD <: AbstractFeatureDistance
    p::Int
end

struct L2Distance <: AbstractFeatureDistance end
struct L1Distance <: AbstractFeatureDistance end
struct CosineDistance <: AbstractFeatureDistance end
struct AbsoluteDistance <: AbstractFeatureDistance end
struct L2Norm <: AbstractFeatureDistance end

#Scalars: entropy, total persistence, nbands, etc...
function feature_distance(a::Real, b::Real, ::AbsoluteDistance)
    return abs(a - b)
end
#Vectors: Betti curvature, band-weight vectors, etc...
function feature_distance(a::AbstractVector, b::AbstractVector, ::L2Norm)
    @assert length(a) == length(b) "L2Norm: vector length mismatch ($(length(a)) vs $(length(b)))"
    return norm(a .- b)
end

function feature_distance(a::AbstractVector, b::AbstractVector, ::L1Distance)
    @assert length(a) == length(b) "L1Distance: vector length mismatch ($(length(a)) vs $(length(b)))"
    return sum(abs, a .- b)
end

#Arrays: Persistence Images
function feature_distance(a::AbstractArray, b::AbstractArray, ::L2Distance)
    return norm(a .- b)
end

function feature_distance(a::AbstractArray, b::AbstractArray, ::CosineDistance)
    va = vec(a); vb = vec(b)
    denom = norm(va)*norm(vb)
    denom < eps() && return 1.0 #treat as maximally distant
    return 1 - ( dot(va,vb)/denom )
end


#Cosine distance for vectors (unused but could be useful)
function feature_distance(a::AbstractVector, b::AbstractVector, dist::CosineDistance)
    return feature_distance(reshape(a, :, 1), reshape(b, :, 1), dist::CosineDistance)
end

#------------------------#
# FeatureBundle Distance #
#------------------------#
function topology_distance(A::FeatureBundle, B::FeatureBundle, spec::TopologyMetricSpec)
    total = 0.0
    #Static
    if A.static !== nothing && B.static !== nothing
        for (key, metric) in spec.static
            a = A.static.invariants[key]
            b = B.static.invariants[key]

            w = get(spec.weights, key, 1.0)
            total += w*feature_distance(a,b,metric)
        end
    end
    #Dynamic
    if A.dynamic !== nothing && B.dynamic !== nothing
        for (key, metric) in spec.dynamic
            a = A.dynamic.invariants[key]
            b = B.dynamic.invariants[key]

            w = get(spec.weights, key, 1.0)
            total += w*feature_distance(a,b,metric)
        end
    end
    return total
end
#-----------------------------------------------#
#               Feature Extraction              #
#-----------------------------------------------#
#static features only for now
function extract_features(I::IntensityData, spec::FeatureSpec)::FeatureBundle
    st = spec.static
    dy = spec.dynamic
    if st === nothing && dy === nothing
        error("At least one StaticFeatureSpec or DynamicFeatureSpec must be given!")
    end
    #-----------------#
    #Static Extraction#
    #-----------------#
    static_bundle = nothing; static_meta = nothing
    if st !== nothing
        invs, meta = compute_static_topology(I, st)
        validate_static_invariants!(invs,st)
        static_bundle = Types.StaticFeatureBundle(invs, nothing)
        static_meta = meta
    end
    #------------------#
    #Dynamic Extraction#
    #------------------#
    dynamic_bundle=nothing
    if dy !== nothing
        dyn = compute_dynamic_topology(I,dy)
        validate_dynamic_invariants!(dyn.invariants,dy)
        dynamic_bundle = dyn
    end
    return Types.FeatureBundle(static_bundle, dynamic_bundle, 
                               (static=st, dynamic=dy), 
                               Dict(:source=>get(I.meta, :source, :unknown), 
                                    :topology=>static_meta,
                                    :dynamic=>dy === nothing ? nothing : dy.params))
end
end #module

module Topology
using LinearAlgebra
using ...Core.Types
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

#Scalars: entropy, total persistence, etc...
function feature_distance(a::Real, b::Real, ::AbsoluteDistance)
    return abs(a - b)
end
#Vectors: Betti curvature, etc...
function feature_distance(a::AbstractVector, b::AbstractVector, ::L2Norm)
    @assert length(a) == length(b)
    return norm(a .- b)
end
#Persistence Images
function feature_distance(a::AbstractArray, b::AbstractArray, ::L2Distance)
    return norm(a .- b)
end

function feature_distance(a::AbstractArray, b::AbstractArray, ::CosineDistance)
    va = vec(a); vb = vec(b)
    return 1 - ( dot(va,vb)/(norm(va)*norm(vb)) )
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
    st === nothing && error("StaticFeatureSpec required!")
    invs, meta = compute_static_topology(I, st)
    validate_static_invariants!(invs,st)
    static_bundle = StaticFeatureBundle(invs, nothing)
    return FeatureBundle(static_bundle, nothing, (static=st, dynamic=nothing), 
                         Dict(:source=>get(I.meta, :source, :unknown), 
                              :topology=>meta))
end
end #module

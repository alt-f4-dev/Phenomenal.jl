#-----------------------------#
# Top-Level Phenomenal Module #
#-----------------------------#
module Phenomenal

#----------------------#
# Core Types/Contracts #
#----------------------#
module Core
include("Core/Types.jl")
end

#--------------------#
# Forward Simulators #
#--------------------#
module Forward
include("Forward/PhunnyBridge.jl")
end

#--------------------#
# Inelastic Analysis #
#--------------------#
module Inelastic
include("Inelastic/PHysicalTDABridge.jl")
include("Inelastic/Topology.jl")
include("Inelastic/Classify.jl")
end

#------------#
# Import API #
#------------#
using .Core.Types: IntensityData, FeatureSpec, StaticFeatureSpec, DynamicFeatureSpec,
                   TopologyMetricSpec, QPresult, ModelHypothesis, QuerySpec

using .Forward.PhunnyBridge: simulate

using .Inelastic.Topology: extract_features
using .Inelastic.Classify: classifyQP

#------------#
# Public API #
#------------#
export IntensityData, FeatureSpec, StaticFeatureSpec, DynamicFeatureSpec, TopologyMetricSpec
export QPresult, ModelHypothesis, QuerySpec
export classify

"""
    classify(data, fs, metric; k, nclusters)

Topology-aware quasi-particle classification from intensity data.
Returns QPresult.
"""
function classify(
        data::Vector{IntensityData}, fs::FeatureSpec, 
        metric::TopologyMetricSpec; k::Int=10, nclusters::Int,
        σ::Union{Nothing,Float64} = nothing)::QPresult
    features = extract_features.(data, Ref(fs))
    return classifyQP(features, metric; k=k, nclusters=nclusters, σ=σ)
end

"""
    classify(mh, qs, gs, metric; k, nclusters)

Simulate intensity data from a ModelHypothesis (mh) and classify
quasi-particles using topology-aware inference.
"""
function classify(
        mh::ModelHypothesis, qs::QuerySpec,
        fs::FeatureSpec, metric::TopologyMetricSpec;
        k::Int=10, nclusters::Int, σ::Union{Nothing,Float64} = nothing)::QPresult
    I = simulate(mh, qs)
    return classify([I], fs, metric; k=k, nclusters=nclusters, σ=σ)
end

end #module 

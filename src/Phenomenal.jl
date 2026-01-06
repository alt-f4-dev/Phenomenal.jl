module Phenomenal
#----------------------#
# Core Types/Contracts #
#----------------------#
include("Core/Types.jl")
#
#--------------------#
# Forward Simulators #
#--------------------#
include("Forward/PhunnyBridge.jl")
#
#--------------------#
# Inelastic Analysis #
#--------------------#
include("Inelastic/Topology.jl")
include("Inelastic/Classify.jl")
#
#------------#
# Import API #
#------------#
using .Types: IntensityData, FeatureSpec, StaticFeatureSpec, TopologyMetricSpec,
              QPresult, ModelHypothesis, QuerySpec
using .PhunnyBridge: simulate
#
#------------#
# Public API #
#------------#
export IntensityData, FeatureSpec, StaticFeatureSpec, TopologyMetricSpec
export QPresult, ModelHypothesis, QuerySpec
export classify
"""
    classify(data, fs, metric; k, nclusters)

Topology-aware quasi-particle classification from intensity data.
Returns QPresult.
"""
function classify(
        data::Vector{IntensityData}, fs::FeatureSpec, 
        metric::TopologyMetricSpec; k::Int=10, nclusters::Int)::QPresult
    features = Topology.extract_features.(data, Ref(fs))
    return Classify.classifyQP(features, metric; k=k, nclusters=nclusters)
end
"""
    classify(mh, qs, gs, metric; k, nclusters)

Simulate intensity data from a ModelHypothesis (mh) and classify
quasi-particles using topology-aware inference.
"""
function classify(
        mh::ModelHypothesis, qs::QuerySpec,
        fs::FeatureSpec, metric::TopologyMetricSpec;
        k::Int=10, nclusters::Int)::QPresult
    I = PhunnyBridge.simulate(mh, qs)
    return classify([I], fs, metric; k=k, nclusters=nclusters)
end
#
end #module 

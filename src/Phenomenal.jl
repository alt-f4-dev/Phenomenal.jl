module Phenomenal
#module assembly (only non-empty files included)
include("Core/Types.jl")
#
include("Forward/PhunnyBridge.jl")
#
include("Inelastic/Topology.jl")
include("Inelastic/Classify.jl")
#
#Public API
using .Types
#
using .PhunnyBridge
#
using .Classify
using .Topology
#
export IntensityData, FeatureSpec, StaticFeatureSpec, TopologyMetricSpec, QPresult, ModelHypothesis, QuerySpec
export classify
"""
    classify(data, fs, metric; k, nclusters)

Topology-aware quasi-particle classification from intensity data.
Returns QPresult.
"""
function classify(
        data::Vector{IntensityData}, fs::FeatureSpec, 
        metric::TopologyMetricSpec; k::Int=10, nclusters::Int)::QPresult
    features = extract_features.(data, Ref(fs))
    return classifyQP(features, metric; k=k, nclusters=nclusters)
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
    I = simulate(mh, qs)
    return classify([I], fs, metric; k=k, nclusters=nclusters)
end
#
end #module Phenomenal

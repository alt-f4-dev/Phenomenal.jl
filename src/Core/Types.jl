module Types
export IntensityData, FeatureBundle, FeatureSpec, Constraints
export QuerySpec, ModelHypothesis
export AbstractFeatureDistance, AbstractTopologyMetric, TopologyMetricSpec
export feature_distance, topology_distance, extract_features
export InvariantKind, ScalarInvariant, VectorInvariant
export STATIC_INVARIANT_SCHEMA, required_static_keys, validate_static_invariants!
export DYNAMIC_INVARIANT_SCHEMA, required_dynamic_keys, validate_dynamic_invariants!
export QPresult
#-------------------------------------------------------------#
#                   Intensity Data Structure                  # 
#-------------------------------------------------------------#
"""
    struct IntensityData{T,N}

A canonical container for intensity data.

- `data`: N-dimensional intensity array
- `axes`: Named tuples of coordinate vectors (e.g., qx,qy,qz,ω)
- `meta`: Metadata such as provenance, experiment settings, instrument info
"""
struct IntensityData{T,N}
    data::Array{T,N}    #I(q,ω) ~ 4D
    axes::NamedTuple    #axes = (h=vec, k=vec, ℓ=vec, ω=vec)
    meta::Dict{Symbol,Any} #History/Experiment tags
end
#Example:
#       Iqω = rand(64,64,64,200)
#       h = range(-3,3,length=64)
#       k = range(-3,3,length=64)
#       ℓ = range(-3,3,length=64)
#       ω = range(0,80,200)
#       axes = (h=h, k=k, ℓ=ℓ, ω=ω)
#       meta = Dict(:source => :simulation)
#
#       Data = IntensityData(Iqω, axes, meta)


#-----------------------------------------------#
#               Topological Features            #
#-----------------------------------------------#
"""
    struct StaticFeatureSpec

Specification for topological feature extraction from IntensityData.
"""
struct StaticFeatureSpec
    dims::Vector{Int}           #homology dimensions
    filtration::Symbol          #:superlevel, :sublevel
    invariants::Vector{Symbol}  #:entropy, :betti, :curvature, etc.
    storePDs::Bool              #retain persistence diagrams?
    params::Dict{Symbol,Any}    #threshold, normalization, etc
end

struct DynamicFeatureSpec
    invariants::Vector{Symbol} #:bandcount, :bandweights, etc.
    params::Dict{Symbol,Any}
end

struct FeatureSpec
    static::Union{Nothing,StaticFeatureSpec}
    dynamic::Union{Nothing,DynamicFeatureSpec}
end

struct StaticFeatureBundle
    invariants::Dict{Symbol,Any}
    diagrams::Union{Nothing,Any}
end
"""
    struct DynamicFeatureBundle

Container for dynamic (spectral) features extracted from a single IntensityData object.

- `bands`:
    Semantic, orientation-aware intermediate representation of band structure
    (e.g. autosplitbands output, persistence-guided band partitions).
    This field is NOT used directly by metric or clustering code.

- `invariants`:
    Metric-safe dynamic invariants derived from `bands`.
    Values must be scalar or fixed-length vectors, suitable for distance computation.

Users should not need to worry about these details. This information is for developers.
If a user finds themselves working with this, please reach out to contribute. 
"""
struct DynamicFeatureBundle
    bands::Dict{Symbol,Any} #semantic/intermediate layer
    invariants::Dict{Symbol,Any} #metric-facing invariants
end
"""
    struct FeatureBundle

Container for invariant features extracted from intensity data.
Intended to be model-agnostic and stable under symmetry operations.

- `spec`: Feature specification for extraction
- `source` Provenance (dataset ID, slice info, etc.)
"""
struct FeatureBundle
    static::Union{Nothing,StaticFeatureBundle}
    dynamic::Union{Nothing,DynamicFeatureBundle}
    spec::NamedTuple
    source::Dict{Symbol,Any}
end
#--------------------------------------------------#
#               Constraints & Queries              #
#--------------------------------------------------#
"""
    struct Constraints

Collection of inferred constraints on admissible models.

- `elastic`: Dict of elastic constraints
- `inelastic`: Dict of inelastic constraints
- `confidence`: reliability weights
"""
struct Constraints
    name::Symbol
    origin::Symbol #:elastic, :static, :dynamic
    hard::Bool #hard vs soft constraint 
    predicate::Function # ModelHypothesis -> (:pass, :fail, :confused)
    metadata::Dict{Symbol,Any}
end
#Example:
#       elastic = Dict(:centering => [:F => 0.95], :lattice => (a=5.65, σ = 0.02), :DW => :isotropic)
#       inelastic = Dict(:class => [:phonon => 0.98], :topology => fingerprint)

"""
    struct QuerySpec

Specificies which momentum-energy patch to simulate.

- `axes`: NamedTuple of axis ranges
- `resolution`: Dict of resolution parameters
- `cuts`: Dict of slice/path requests
"""
struct QuerySpec
    axes::NamedTuple             #requested axes + ranges
    resolution::Dict{Symbol,Any} #instrumental resolution
    cuts::Dict{Symbol,Any}       #slices, planes, paths
end
#Example: 
#       QuerySpec(axes=(qx=(-3,3,200),qy=(-3,3,200),ω=(0,80,400)),
#                 resolution=Dict(:dE => 1.5, :dQ => 0.02),
#                 cuts = Dict(:plane=>(:qx,:qy), :at=>:qz=>0.0))


"""
    struct ModelHypothesis

A fully specified candidate physical model.

- `structure`: lattice, basis, space group
- `dynamics`: phonon FCMs, magnitudes, etc.
- `instrument`: resolution, background models
- `complexity`: penalty or score cost
"""
struct ModelHypothesis
    structure::Dict{Symbol,Any}   #lattice, basis, symmetry, atoms
    freedom::Any                  #phonons, spins, orbitals
    dynamics::Any                 #FCMs, Hamiltonians, couplings
    forward::Dict{Symbol,Function}#:elastic, :inelastic simulators
    complexity::Float64           #model complexity penalty
end
#Example:
#       structure = Dict(:lattice => L, :basis => basis, :centering => :F)
#       dynamics = Dict(:type => :phonon, :FCM => Φ)
#       instrument = Dict(:resolution => resolution)
#       complexity = 12.0
#
#       ModelHypothesis(structure=structure, dynamics=dynamics, 
#                       instrument=instrument, complexity=complexity)




#---------------------------------------------------#
#             Topology Feature Metrics              #
#---------------------------------------------------#

"""
Abstract distance acting on a single invariant contained
inside a FeatureBundle (e.g. PDs, Betti curves, scalars).
"""
abstract type AbstractFeatureDistance end

"""
Abstract specification of a composite topology metric.
Defines how FeatureBundles are compared.
"""
abstract type AbstractTopologyMetric end


"""
Specification for a topology-aware metric on FeatureBundle.

- `static`: distances acting on StaticFeatureBundle invariants
- `dynamic`: distances acting on DynamicFeatureBundle invariants
- `weights`: epistemic aggregation weights
"""
struct TopologyMetricSpec <: AbstractTopologyMetric
    static::Dict{Symbol,AbstractFeatureDistance}
    dynamic::Dict{Symbol,AbstractFeatureDistance}
    weights::Dict{Symbol,Float64}
end


"""
Distance evaluation contract for invariant-level comparisons.
Concrete methods live in Inelastic/Topology.jl.
"""
function feature_distance end


"""
Distance evaluation contract for FeatureBundles.
Concrete method lives in Inelastic/Topology.jl.
"""
function topology_distance end


#-----------------------------------------------#
#         Quasi-particle Classification         #
#-----------------------------------------------#
#feature extraction
function extract_features end
#
#Invariants Data Structure
abstract type InvariantKind end
struct ScalarInvariant <: InvariantKind end
struct VectorInvariant <: InvariantKind end

#----------------------------------------#
# Static Invariants: Persistent Homology #
# features extracted from I(q,ω).        #
#----------------------------------------#
const STATIC_INVARIANT_SCHEMA = Dict{Symbol,InvariantKind}(:betti0 => VectorInvariant(), 
                                                           :betti1 => VectorInvariant(),
                                                           :entropy0 => ScalarInvariant(), 
                                                           :entropy1 => ScalarInvariant())

function required_static_keys(spec::StaticFeatureSpec)
    keys = Symbol[]
    for p in [0,1]
        if :betti in spec.invariants
            push!(keys, Symbol(:betti,p))
        end
        if :entropy in spec.invariants
            push!(keys, Symbol(:entropy,p))
        end
    end
    return keys
end

function validate_static_invariants!(invs::Dict{Symbol,Any}, spec::StaticFeatureSpec)
    req = required_static_keys(spec)
    for k in req
        haskey(invs, k) || error("Missing required static invariant: $k")
    end
    for (k,kind) in STATIC_INVARIANT_SCHEMA
        haskey(invs,k) || continue
        v = invs[k]
        if kind isa ScalarInvariant
            v isa Real || error("Invariant $k must be scalar, got $(typeof(v))!")
            isfinite(v) || error("Invariant $k is not finite!")
        elseif kind isa VectorInvariant
            v isa AbstractVector || error("Invariant $k must be a vector!")
            isempty(v) && error("Invariant $k is empty!")
            all(isfinite, v) || error("Invariant $k contains non-finite values!")
        end
    end
    return nothing
end
#---------------------------------------#
# Dynamic Invariants: Spectral dynamics #
# extracted from I(q,ω) via PH-guided   #
# band-segmentation + summaries.        #
#---------------------------------------#
const DYNAMIC_INVARIANT_SCHEMA = Dict{Symbol,InvariantKind}(:nbands=>ScalarInvariant(),
                                                            :BandWeights=>VectorInvariant(),
                                                            :BandIntensity=>VectorInvariant(),
                                                            :BandCenters=>VectorInvariant(),
                                                            :BandWidths=>VectorInvariant(),
                                                            :BandGaps=>VectorInvariant(),
                                                            :PerBandPersistence=>VectorInvariant(),
                                                            :TotalBandPersistence=>ScalarInvariant(),
                                                            :BandMaxFrac=>ScalarInvariant(),
                                                            :BandWeightEntropy=>ScalarInvariant(),
                                                            :BandPersistenceEntropy=>ScalarInvariant(),
                                                            :BandWeightMax=>ScalarInvariant(),
                                                            :nbandsCurve=>VectorInvariant(),
                                                            :BandWeightEntropyCurve=>VectorInvariant(),
                                                            :BandWeightMaxCurve=>VectorInvariant()
                                                           )
#----------------------------------------------------------------------------------------------------#
#Core Band Structure:
#  nbands → Number of bands/branches of band-segmented integrated intensity
#
#Per-band Intensity / Spectral Summaries: 
#  BandWeights → Normalized per-band weights
#  BandIntensity → Absolute integrated per-band intensity
#  BandCenters → Band center positions
#  BandWidths → Band widths (spreads)
#  BandGaps → Spectral gaps between adjacent bands (ordering convention defined in extraction)
# 
#Persistent Homology Measures:
#  PerBandPersistence → Per-band persistence strength (e.g. lifetime)
#  TotalBandPersistence → Total persistence summed across bands
#  BandMaxFrac → Ratio of strongest band persistence to total band persistence
#  BandWeightEntropy → Entropy of normalized band weights (spectral-weight segments)
#  BandPersistenceEntropy → Entropy of normalized band persistence (topological segments)
#  BandWeightMax → Maximum normalized band weight (single-mode dominance)
#
#
#Curve-like Measures: (May remove these)
#  nbandsCurve → q-resolved or ω-resolved band count curve
#  BandWeightEntropyCurve → q-resolved or ω-resolved entropy curve
#  BandWeightMaxCurve → q-resolved pr ω-resolved max-weight
#
#---------------------------------------------------------------------------------------------------#
function required_dynamic_keys(spec::DynamicFeatureSpec)
    return (:nbands, :BandIntensity, :BandWeightEntropy, :BandPersistenceEntropy)
end
function validate_dynamic_invariants!(invs::Dict{Symbol,Any}, spec::DynamicFeatureSpec)
    requirements = required_dynamic_keys(spec)
    for k in requirements
        haskey(invs,k) || error("Missing required dynamic invariant: $k")
    end
    for (k,kind) in DYNAMIC_INVARIANT_SCHEMA
        haskey(invs, k) || continue
        v = invs[k]
        if kind isa ScalarInvariant
            v isa Real || error("Dynamic invariant $k must be scalar, got $(typeof(v))!")
            isfinite(v) || error("Dynamic invariant $k is not finite!")
        elseif kind isa VectorInvariant
            v isa AbstractVector || error("Dynamic invariant $k must be a vector, got $(typeof(v))")
            isempty(v) && error("Dynamic invariant $k is empty!")
            all(isfinite, v) || error("Dynamic invariant $k contains non-finite elements!")
        end
    end
    return nothing
end

"""
Quasi-particle classification result container.
"""
struct QPresult
    labels::Vector{Int}         #cluster labels
    confidence::Vector{Float64} #local confidence
    eigvals::Vector{Float64}    #Laplacian eigenvalues
    meta::Dict{Symbol,Any}      #metric, feature spec, etc..
    k::Int                      # kNN parameter
end
end #module

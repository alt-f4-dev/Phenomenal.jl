module PhenomenalPHysicalTDABridge

using Statistics

import PHysicalTDA
using ..Core.Types: IntensityData, StaticFeatureSpec, validate_static_invariants!

export compute_static_topology

"""
    compute_static_topology(I::IntensityData, spec::StaticFeatureSpec)

Static topology bridge:
- projects 4D (h,k,ℓ,ω) ⟶ 2D (q,ω)
- computes topology via PHysicalTDA
- returns (PDs, invariants, meta)
"""
function compute_static_topology(I::IntensityData, spec::StaticFeatureSpec)
    Iqω, proj_meta = project(I, spec)
    maxdim = isempty(spec.dims) ? 1 : maximum(spec.dims)
    superlevel = spec.filtration === :superlevel
    threshold = get(spec.params, :threshold, nothing)
    normalize = get(spec.params, :normalize, false)
    PDs = PHysicalTDA.pd_array_intensities(Iqω; maxdim=maxdim, 
                                           threshold=threshold,
                                           superlevel=superlevel,
                                           normalize=normalize)
    invs = Dict{Symbol, Any}(); dims = spec.dims
    invset = Set(spec.invariants)

    τ = get(spec.params, :τgrid, nothing)
    if :betti in invset
        β = PHysicalTDA.betti_curve(PDs, τ; dims=dims)
        for p in dims
            invs[Symbol(:betti,p)]=β[p]
        end
    end
    if :entropy in invset
        S, _ = PHysicalTDA.persistence_entropy(PDs; dims=dims)
        for p in dims
            invs[Symbol(:entropy,p)] = S[p]
        end
    end
    validate_static_invariants!(invs,spec)
    meta = Dict{Symbol,Any}(:projection => proj_meta,
                            :dims => dims, 
                            :filtration => spec.filtration, 
                            :normalize => normalize)
    return invs, meta
end
#
# 4D -> 2D Projections
function project(I::IntensityData, spec::StaticFeatureSpec)
    mode = get(spec.params, :projection, :plane)
    if mode === :plane
        A = I.data
        over = get(spec.params, :collapse, (:k, :ℓ))
        Iqω = PHysicalTDA.collapse(A;over=over,op=sum)
        meta = Dict(:type=>:plane, :collapse=>over)
    elseif mode === :qpath
        #want to allow users to specify qpath in spec.params or StaticFeatureSpec
        qpath = get(spec.params, :qpath, nothing)
        qpath === nothing && error(":projection => :qpath requires that :qpath => qpath in params.spec!")
        interp = get(spec.params, :interp, :nearest)
        Iqω = sample_intensity_along_qpath(I, qpath; interp=interp)
        meta = Dict(:type=>:qpath, 
                    :interp=>:nearest, 
                    :Nq=>length(qpath))
    else
        error("Unknown projection mode: $mode")
    end
    ndims(Iqω) == 2 || error("Projection did not yield an image!")
    return Array{Float64}(Iqω), meta
end
"""
    sample_intensity_along_qpath(I, qpath; interp=:nearest)

Samples a 4D intensity cube I(h,k,ℓ,ω) along a specified q-path,
assuming that q = (h,k,ℓ). Returns a 2D projection I(q,ω).
"""
function sample_intensity_along_qpath(
        I::IntensityData, 
        qpath::AbstractVector; 
        interp::Symbol = :nearest)
    Nq = length(qpath); Nω = size(I,4); ωaxis = 1:Nω 
    H = I.axes.h; K = I.axes.k; L = I.axes.ℓ
    data = I.data; Iqω = zeros(Float64, Nq, Nω)
    @inline findidx(A,a) = argmin(abs.(A .- a))
    for (qₙ, q) in enumerate(qpath)
        h,k,ℓ = q
        hₙ = findidx(H,h)
        kₙ = findidx(K,k)
        ℓₙ = findidx(L,ℓ)
        @inbounds for ωₙ in ωaxis
            Iqω[qₙ,ωₙ] = data[hₙ, kₙ, ℓₙ, ωₙ]
        end
    end
    return Iqω
end
end #module

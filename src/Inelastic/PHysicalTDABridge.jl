module PHysicalTDABridge

using Statistics
using LinearAlgebra

import PHysicalTDA
using ...Core.Types: IntensityData, StaticFeatureSpec, DynamicFeatureSpec, DynamicFeatureBundle, validate_dynamic_invariants!, validate_dynamic_bands!, validate_static_invariants!

export compute_static_topology, compute_dynamic_topology

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
"""
    compute_dynamic_topology(I::IntensityData, spec::DynamicFeatureSpec)

Dynamic topology bridge:
    - projects 4D I(h,k,ℓ,ω) → 2D I(q,ω)
    - computes topology via PHysicalTDA
    -returns (PDs, invariants, meta)

Extracts dynamic (spectral-band) invariants from 2D I(q,ω) projection via
persistence-homology-guided band segmentation (`PHysicalTDA.autosplitbands`).



The extraction proceeds in three stages:
  1. **Projection**: 4D I(h,k,ℓ,ω) → 2D I(q,ω) via the same projection
     machinery used by the static branch (`:plane` or `:qpath`).
  2. **Band segmentation**: `autosplitbands` is called on I(q,ω), yielding
     energy-first (splitE) and momentum-first (splitQ) band decompositions.
     Both are stored in the `bands` semantic layer.
  3. **Invariant computation**: metric-safe invariants are derived from the
     chosen orientation's `bandIq` matrix (energy-integrated intensity per
     band as a function of q). Orientation is selected via
     `spec.params[:orientation]` ∈ {:energy, :momentum}, defaulting to
     `:energy`. The invariants match the `DYNAMIC_INVARIANT_SCHEMA`.
 
Band persistence is estimated from the per-band integrated intensity treated
as a 1D signal, and 0D PH is reused to get per-band lifetime.
 
## Parameters in `spec.params`
 
| Key                | Type     | Default    | Description                                |
|--------------------|----------|------------|--------------------------------------------|
| `:projection`      | Symbol   | `:plane`   | Forwarded to `project`                     |
| `:collapse`        | Tuple    | `(:k,:ℓ)`  | Axes to collapse for `:plane` mode         |
| `:qpath`           | Vector   | required   | q-path for `:qpath` mode                   |
| `:orientation`     | Symbol   | `:energy`  | Which `autosplitbands` result to use       |
| `:pers_ratio_min`  | Float64  | `2.0`      | Band detection persistence ratio threshold |
| `:curve_invariants`| Bool     | `false`    | Compute q-resolved curve invariants        |
 
## Returns
`DynamicFeatureBundle` with:
- `.bands`      → raw `autosplitbands` output (semantic layer, not metric-facing)
- `.invariants` → validated `Dict{Symbol,Any}` matching `DYNAMIC_INVARIANT_SCHEMA`

"""
function compute_dynamic_topology(I::IntensityData, spec::DynamicFeatureSpec)
    #------------------#
    #4D → 2D Projection#
    #------------------#
    proj_params = Dict{Symbol,Any}( :projection => get(spec.params, :projection, :plane),
                                    :collapse => get(spec.params, :collapse, (:k, :ℓ))
                                  )
    if haskey(spec.params, :qpath)
        proj_params[:qpath] = spec.params[:qpath]
        proj_params[:interp] = get(spec.params, :interp, :nearest)
    end
    proj_spec = StaticFeatureSpec(Int[], :superlevel, Symbol[], false, proj_params)
    Iqω, proj_meta = project(I,proj_spec)
    nq,nω=size(Iqω); ωs=collect(Float64,I.axes.ω)
    length(ωs) == nω || error("ω-axis length $(length(ωs)) ≠ Iqω column count nω=$nω!")
    
    #---------------------------#
    #PH-guided Band Segmentation#
    #---------------------------#
    #   rawbands = (splitE=(bandIq, bandIω), splitQ=(bandIq, bandIω))
    #   splitE: energy-first segmentation → each row is a band, good for phonons
    #   splitQ: momentum-first segmentation → each row is a band, good for spins 
    ρ₀ = Float64(get(spec.params, :pers_ratio_min, 2.0))    
    rawbands = PHysicalTDA.autosplitbands(Iqω, ωs; pers_ratio_min=ρ₀)
    
    splitE = rawbands.splitE; splitQ = rawbands.splitQ
    bands_dict = Dict{Symbol,Any}(:splitE_bandIq => splitE.bandIq,
                                  :splitE_bandIω => splitE.bandIω,
                                  :splitQ_bandIq => splitQ.bandIq,
                                  :splitQ_bandIω => splitQ.bandIω,
                                  :splitE        => splitE,
                                  :splitQ        => splitQ,
                                  :autosplitbands => rawbands,
                                  :projection    => proj_meta,
                                  :Iqω           => Iqω,
                                  :ωs            => ωs,
                                  :pers_ratio_min => ρ₀ )
    
    validate_dynamic_bands!(bands_dict)

    #------------------#
    #Compute Invariants#
    #------------------#
    orientation = get(spec.params, :orientation, :energy)
    if orientation === :energy
        split = rawbands.splitE
    elseif orientation === :momentum
        split = rawbands.splitQ
    else
        error("Unknown :orientation=$orientation; use :energy or :momentum")
    end

    #   Iq :: (Nbands × nq) ~ Energy-integrated per-band intensity
    #   Iω :: (Nbands × nω) ~ Momentum-integrated per-band intensity
    Iq = split.bandIq; Iω = split.bandIω; Nbands = size(Iq, 1)
    
    #OLD
    #bands_dict = Dict{Symbol, Any}(:autosplitbands => rawbands,
    #                               :orientation => orientation,
    #                               :projection => proj_meta,
    #                               :Iqω => Iqω, :ωs => ωs)
    
    #NEW
    bands_dict[:orientation] = orientation

    #------------------#
    #Compute Invariants#
    #------------------#
    invs = _compute_dynamic_invariants(Iq, Iω, ωs, spec)
    validate_dynamic_invariants!(invs, spec)
    return DynamicFeatureBundle(bands_dict, invs)
end

#------------------------------------------------------------------------------------------------------------------------------#
#       _compute_dynamic_invariants(bandIq, bandIω, ωs, spec)
#
#
# Internal: derives all `DYNAMIC_INVARIANT_SCHEMA` invariants from the 
#           (Nbands × nq) and (Nbands × nω) band matrices produced by
#           `autosplitbands`
#
#
# Invariants computed:
#       - :nbands                       ~ Number of bands                               (scalar)
#       - :BandIntensity                ~ Total integrated intensity per band           (vector) Nbands -> length
#       - :BandWeights                  ~ Normalized BandIntensity                      (vector) Nbands 
#       - :BandCenters                  ~ Intensity-weighted ω centers per band         (vector) Nbands
#       - :BandWidths                   ~ Intensity-weighted ω std per band             (vector) Nbands 
#       - :BandGaps                     ~ ω-gap between band centers                    (vector) Nbands
#       - :PerBandPersistence           ~ derived from 0D PH on bandIq[b,:]             (vector) Nbands
#       - :TotalBandPersistence         ~ sum of per-band persistence                   (scalar)
#       - :BandMaxFrac                  ~ max(BandPersistence)/TotalPersistence         (scalar)
#       - :BandWeightEntropy            ~ Shannon entropy of BandWeights                (scalar)
#       - :BandPersistenceEntropy       ~ Shannon entropy of norm(BandPersistence)      (scalar)
#       - :BandWeightMax                ~ maximum(BandWeights)                          (scalar)
#       - :nbandsCurve                  ~ q-resolved band count                         (vector) nq [if :curve_invariants]
#       - :BandWeightEntropyCurve       ~ q-resolved entropy of BandWeights             (vector) nq [if :curve_invariants]
#       - :BandWeightMaxCurve           ~ q-resolved maximum(BandWeights)               (vector) nq [if :curve_invariants]
#
#-------------------------------------------------------------------------------------------------------------------------------#
function _compute_dynamic_invariants(
        Iq::Matrix{Float64}, Iω::Matrix{Float64}, 
        ωs::Vector{Float64}, spec::DynamicFeatureSpec)::Dict{Symbol,Any}
    Nbands, nq = size(Iq); _, nω = size(Iω)
    invset = Set(spec.invariants); invs = Dict{Symbol,Any}()

    #Helper: Shannon entropy w/o PHysicalTDA bridge
    function _shannon(p::AbstractVector{Float64})
        S = 0.0; tot=sum(p)
        tot ≤ 0.0 && return 0.0
        @inbounds for pᵢ in p
            pᵢ> 0.0 && (S -= (pᵢ/tot)*log2(pᵢ/tot))
        end
        return S
    end

    #--------#
    # nbands #
    #--------#
    invs[:nbands] = Float64(Nbands)

    #-----------------------------#
    # BandIntensity & BandWeights #
    #-----------------------------#
    BandIntensity = vec(sum(Iq,dims=2)) #∑ₙIᵇ(qₙ) ~ weighted by Δω by PHysicalTDA.jl
    invs[:BandIntensity] = BandIntensity
    
    TotalIntensity = sum(BandIntensity)
    BandWeights = TotalIntensity > 0.0 ? BandIntensity ./ TotalIntensity : fill(1.0/Nbands, Nbands)
    invs[:BandWeights] = BandWeights

    #--------------------------#
    # BandCenters & BandWidths #
    #--------------------------#
    BandCenters = zeros(Float64, Nbands) #BandCenters[b] ≡ ∑ₙωₙIᵇ(ωₙ)/∑ₙIᵇ(ωₙ)
    BandWidths = zeros(Float64, Nbands)  #BandWidths[b]  ≡ sqrt(∑ₙ(ωₙ-center)²*Iᵇ(ωₙ) / ∑ₙIᵇ(ωₙ))
    @inbounds for b in 1:Nbands
        row = @view Iω[b,:]; rownorm = sum(row)
        if rownorm > 0.0
            μ = dot(ωs, row)/rownorm
            σ = sqrt(sum((ωs[j] - μ)^2 * row[j] for j in 1:nω)/rownorm)
            BandCenters[b] = μ; BandWidths[b] = σ
        else
            BandCenters[b] = (ωs[1] + ωs[end])/2.0
            BandWidths[b] = 0.0
        end
    end
    invs[:BandCenters] = BandCenters; invs[:BandWidths] = BandWidths
    invs[:BandGaps] = Nbands > 1 ? diff(sort(BandCenters)) : [0.0]

    #-------------------------------------------#
    # PerBandPersistence, TotalBandPersistence  #
    # BandMaxFrac, & BandPersistenceEntropy     #
    #-------------------------------------------#
    #Run 0D PH on each band's q-resolved profile 
    #Iq[b,:] to get a persistence lifetime for that band.
    #
    #The longest barcode lifetime in the superlevel filtration 
    #measures band robustness along q-axis.

    PerBandPersistence = zeros(Float64, Nbands)
    @inbounds for b in 1:Nbands
        profile = @view Iq[b,:]
        if all(iszero, profile)
            PerBandPersistence[b] = 0.0
            continue
        end
        PDᵇ = PHysicalTDA.pd_array_intensities(profile; 
                                               maxdim=0, 
                                               superlevel=true, 
                                               threshold=nothing,
                                               normalize=false) #superlevel, threshold, normalize should be defined by user specs
        D0 = PDᵇ[1]
        if isempty(D0)
            PerBandPersistence[b] = 0.0
        else
            maxlife = 0.0
            for x in D0
                bval = PHysicalTDA.birth(x); dval = PHysicalTDA.death(x) 
                isfinite(dval) && dval > bval && (maxlife = max(maxlife, dval - bval))
            end
            PerBandPersistence[b] = maxlife
        end
    end
    TotalBandPersistence = sum(PerBandPersistence)
    BandMaxFrac = TotalBandPersistence > 0.0 ? maximum(PerBandPersistence)/TotalBandPersistence : 0.0
    invs[:PerBandPersistence] = PerBandPersistence
    invs[:TotalBandPersistence] = TotalBandPersistence
    invs[:BandMaxFrac] = BandMaxFrac

    #-----------------------------------#
    # BandWeightEntropy & BandWeightMax #
    #-----------------------------------#
    invs[:BandPersistenceEntropy] = _shannon(PerBandPersistence)
    invs[:BandWeightEntropy] = _shannon(BandWeights)
    invs[:BandWeightMax] = isempty(BandWeights) ? 0.0 : maximum(BandWeights)

    #---------------------------------------#
    # Curve-resolved Invariantes (optional) #
    #---------------------------------------#
    if get(spec.params, :curve_invariants, false)
        nbandsCurve = vec(sum(Iq .> 0.0, dims=1))
        invs[:nbandsCurve] = Float64.(nbandsCurve)

        BandWeightEntropyCurve = zeros(Float64, nq)
        BandWeightMaxCurve = zeros(Float64, nq)
        @inbounds for i in 1:nq
            col = @view Iq[:,i]; colnorm = sum(col)
            if colnorm > 0.0
                BandWeightEntropyCurve[i] = _shannon(col)
                BandWeightMaxCurve[i] = maximum(col)/colnorm
            else
                BandWeightEntropyCurve[i] = 0.0
                BandWeightMaxCurve[i] = 0.0
            end
        end
        invs[:BandWeightEntropyCurve] = BandWeightEntropyCurve
        invs[:BandWeightMaxCurve] = BandWeightMaxCurve
    end
    return invs
end


    


"""
    project(I::IntensityData, spec::StaticFeatureSpec)

Input: 
    - Four-dimensional I(H,K,L,E)
    - Static feature specifications

Output:
    - Projected (3D, 2D, or 1D) intensity
"""
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
function sample_intensity_along_qpath(I::IntensityData, qpath::AbstractVector; interp::Symbol = :nearest)
    Nq = length(qpath); Nω = size(I.data, 4); ωaxis = 1:Nω 
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

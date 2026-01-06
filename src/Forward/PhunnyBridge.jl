module PhunnyBridge
include("../Core/Types.jl")
using .Types
#Import necessary packages
using StaticArrays
import Phunny: onephonon_dsf

#Export API
export simulate

"""
    function simulate(ModelHypothesis, QuerySpec)::IntensityData

Given a ModelHypothesis and a QuerySpec, call Phunny to compute
S(q,ω) over the requested grid and return an IntensityData.
"""
function simulate(mh::ModelHypothesis, QS::QuerySpec)::IntensityData
    #Extract Model Specifications
    model = mh.structure[:model]
    ϕ = mh.dynamics[:force_constants]
    T = mh.instrument[:temperature]
    η = mh.instrument[:η, 0.5]

    #Desired Axes
    qx,qy,qz = QS.axes.h, QS.axes.k, QS.axes.ℓ    
    ω = QS.axes.ω; Nq = length(qx)*length(qy)*length(qz)
    qlist = Vector{SVector{3,Float64}}(undef, Nq)
    
    #Preprocess momentum values
    qlist = [SVector{3,Float64}(x,y,z) for x in qx for y in qy for z in qz]
    dims = (length(qx), length(qy), length(qz))

    #Prepare output array
    Sqω = zeros(Float64, length(qx), length(qy), length(qz), length(ω))

    #Construct 4D S(q,ω)
    Base.Threads.@threads for idx in 1:length(qlist)
        q = qlist[idx]
        Sω = onephonon_dsf(model,ϕ, q, ω; T=T, η=η)
        ix, iy, iz = ind2sub(dims,idx)
        Sqω[ix,iy,iz,:] .= Sω
    end
    axes = (h=qx, k=qy, ℓ=qz, ω=ω)
    meta = Dict(:source => :simulation, :modelID => objectid(model))
    return IntensityData(Sqω, axes, meta)
end
end #module

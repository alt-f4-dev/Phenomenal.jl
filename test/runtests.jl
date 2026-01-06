using Test
using Phenomenal
@testset "Phenomenal loads" begin
    @test isdefined(Main, :Phenomenal)
    @test isdefined(Phenomenal, :classify)
end

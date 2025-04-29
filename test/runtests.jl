using Test
using CatallaxyModel

@testset "Catallaxy Model Types" begin
    # Test type construction and invariants
end

@testset "Agent Logic" begin
    # Test agent decision logic
end

@testset "Market Mechanics" begin
    # Test order book updates
end

@testset "Pricing Engines" begin
    using CatallaxyModel.Types
    using CatallaxyModel.Pricing
    using Random
    
    # Deterministic MC test with explicit RNG
    state = MarketState(
        Dict("OPT" => Price(100.0)),
        Dict("OPT" => 0.2),
        Dict("OPT" => 1.0),
        Dict("OPT" => 0.01),
        0.0
    )
    opt = OptionContract("OPT", 100.0, 0.25, false, true)
    config = SimConfig(1000, 42, 0.01, 0.01, Dict(:behavioral_exercise => true))
    rng = MersenneTwister(42)  # Explicit RNG for reproducibility
    
    # Test option pricing
    price = price_american_mc(opt, state, config, rng)
    @test price > 0
    @test typeof(price) == Float64
    
    # Test Greeks calculation
    greeks = calculate_option_greeks(opt, state, config, rng)
    @test haskey(greeks, :delta)
    @test haskey(greeks, :gamma)
    @test haskey(greeks, :vega)
    @test haskey(greeks, :theta)
    @test greeks[:delta] < 0  # Put option should have negative delta
    
    # Test reproducibility
    rng2 = MersenneTwister(42)  # Same seed
    price2 = price_american_mc(opt, state, config, rng2)
    @test price ≈ price2  # Should be identical with same seed
    
    # Test different volatility
    state_highvol = deepcopy(state)
    state_highvol.volatility["OPT"] = 0.4  # Higher volatility
    price_highvol = price_american_mc(opt, state_highvol, config, rng)
    @test price_highvol > price  # Put option value increases with volatility
end

@testset "Simulation Loop" begin
    using CatallaxyModel.Types
    using CatallaxyModel.Simulation
    using DataFrames
    
    # Run simulation with small number of paths for speed
    config = SimConfig(100, 123, 0.01, 0.01, Dict(:behavioral_exercise => true))
    results = run_simulation(config)
    
    # Test results structure
    @test isa(results, DataFrame)
    @test nrow(results) == 50  # 50 simulation steps
    @test hasproperty(results, :spot_price)
    @test hasproperty(results, :option_price)
    @test hasproperty(results, :option_delta)
    
    # Test values
    @test all(results.spot_price .> 0)  # All prices positive
    @test all(results.option_price .> 0)  # All option prices positive
    
    # Test plotting function
    p = plot_simulation_results(results)
    @test isa(p, Plots.Plot)
end

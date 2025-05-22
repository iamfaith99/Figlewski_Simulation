using Test
using CatallaxyModel

@testset "Catallaxy Model Types" begin
    using CatallaxyModel.Types # Make Price, OrderBook directly available

    # Test Price struct
    @test Price(10.0).value == 10.0
    @test Price(0.0).value == 0.0
    @test_throws ArgumentError Price(-1.0) # Negative price not allowed

    p1 = Price(10.0)
    p2 = Price(20.0)
    @test p1 < p2
    @test p1 < 20.0
    @test 10.0 < p2
    @test p1 == Price(10.0)
    @test p1 == 10.0
    @test 10.0 == p1
    @test (p2 - p1) == 10.0

    # Test OrderBook default constructor
    ob = OrderBook()
    @test isempty(ob.bids)
    @test isempty(ob.asks)
    @test eltype(ob.bids) == Tuple{Price{Float64}, Float64, Int, Symbol}
    @test eltype(ob.asks) == Tuple{Price{Float64}, Float64, Int, Symbol}
end

@testset "Agent Logic" begin
    # Test agent decision logic
end

@testset "Market Mechanics" begin
    using CatallaxyModel.Types # For Price, OrderBook, OptionContract, SimConfig, MarketState
    using CatallaxyModel.Market # For update_orderbook

    # Setup basic market components
    test_opt = OptionContract("TESTUNDERLYING", 100.0, 0.25, false, true) # Put Option
    test_option_key = "OPTION_$(test_opt.strike)_$(test_opt.is_call ? 'C' : 'P')"

    test_config = SimConfig(
        100, UInt(42), 0.05, 0.005, 0.01, 0.005, 
        0.1, 0.0001, 20, Dict(:behavioral_exercise => true)
    )
    test_state = MarketState(
        Dict(test_opt.underlying => Price(100.0)),
        Dict(test_opt.underlying => 0.2),
        Dict(test_opt.underlying => 1000.0),
        0.0
    )

    @testset "Order Book Update - No Trade" begin
        ob_no_trade = OrderBook()
        orders_no_trade = [
            Dict(:type => :option, :side => :buy, :price => 9.0, :size => 1.0, :agent_id => 1, :agent_role => :retail, :option_key => test_option_key),
            Dict(:type => :option, :side => :sell, :price => 11.0, :size => 1.0, :agent_id => 2, :agent_role => :retail, :option_key => test_option_key)
        ]
        
        updated_ob, trades = update_orderbook(ob_no_trade, orders_no_trade, test_state, test_config, test_opt)
        
        @test isempty(trades)
        @test length(updated_ob.bids) == 1
        @test updated_ob.bids[1][1] == Price(9.0) # Price is first element
        @test length(updated_ob.asks) == 1
        @test updated_ob.asks[1][1] == Price(11.0) # Price is first element
    end

    @testset "Order Book Update - Full Trade" begin
        ob_trade = OrderBook()
        orders_trade = [
            Dict(:type => :option, :side => :buy, :price => 10.0, :size => 1.0, :agent_id => 1, :agent_role => :retail, :option_key => test_option_key),
            Dict(:type => :option, :side => :sell, :price => 10.0, :size => 1.0, :agent_id => 2, :agent_role => :retail, :option_key => test_option_key)
        ]
        
        updated_ob_trade, trades_full = update_orderbook(ob_trade, orders_trade, test_state, test_config, test_opt)
        
        @test length(trades_full) == 1
        trade = trades_full[1]
        @test trade[:trade_price] == 10.0
        @test trade[:trade_size] == 1.0
        @test trade[:buyer_id] == 1
        @test trade[:seller_id] == 2
        @test trade[:type] == :option
        @test trade[:option_key] == test_option_key
        
        @test isempty(updated_ob_trade.bids)
        @test isempty(updated_ob_trade.asks)
    end

    @testset "Order Book Update - Partial Trade" begin
        ob_partial = OrderBook()
        # Buyer wants 2 at 10.0, Seller offers 1 at 10.0
        orders_partial = [
            Dict(:type => :option, :side => :buy, :price => 10.0, :size => 2.0, :agent_id => 1, :agent_role => :retail, :option_key => test_option_key),
            Dict(:type => :option, :side => :sell, :price => 10.0, :size => 1.0, :agent_id => 2, :agent_role => :retail, :option_key => test_option_key)
        ]
        
        updated_ob_partial, trades_partial = update_orderbook(ob_partial, orders_partial, test_state, test_config, test_opt)
        
        @test length(trades_partial) == 1
        trade_p = trades_partial[1]
        @test trade_p[:trade_price] == 10.0
        @test trade_p[:trade_size] == 1.0 # Trade size is limited by seller
        
        @test length(updated_ob_partial.bids) == 1 # Buyer's remaining order
        @test updated_ob_partial.bids[1][1] == Price(10.0) # Price
        @test updated_ob_partial.bids[1][2] == 1.0       # Remaining size
        @test isempty(updated_ob_partial.asks) # Seller's order filled
    end
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
        0.0
    )
    opt = OptionContract("OPT", 100.0, 0.25, false, true)
    config = SimConfig(1000, UInt(42), 0.05, 0.005, 0.01, 0.005, 0.1, 0.0001, 20, Dict(:behavioral_exercise => true))
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
    config = SimConfig(100, UInt(123), 0.05, 0.005, 0.01, 0.005, 0.1, 0.0001, 20, Dict(:behavioral_exercise => true))
    n_steps_test = 10
    final_agents, stock_price_history, option_price_history, delta_history, spread_history, volume_history, agent_capital_history = run_simulation(config, n_steps_test)
    
    # Test results structure
    @test isa(final_agents, Vector{<:CatallaxyModel.Types.Entrepreneur})
    @test isa(stock_price_history, Vector{Float64})
    @test isa(option_price_history, Vector{Float64})
    @test isa(delta_history, Vector{Float64})
    @test isa(spread_history, Vector{Union{Float64, Missing}})
    @test isa(volume_history, Vector{Float64})
    @test isa(agent_capital_history, Dict{Int, Vector{Float64}})

    @test length(stock_price_history) == n_steps_test + 1
    @test length(option_price_history) == n_steps_test
    @test length(delta_history) == n_steps_test
    @test length(spread_history) == n_steps_test
    @test length(volume_history) == n_steps_test
    @test all(length(cap_hist) == n_steps_test + 1 for cap_hist in values(agent_capital_history))

    # Test values
    @test all(price -> price > 0, stock_price_history)
    @test all(price -> price > 0, option_price_history)
    @test all(vol -> vol >= 0, volume_history)
end

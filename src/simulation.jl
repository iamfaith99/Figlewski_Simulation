"""
Simulation loop and diagnostics for Catallaxy Model.
Coordinates agent actions, market evolution, and output.
References: Brandimarte, Glasserman, Figlewski.
"""

module Simulation
# Import the Types module to access its exported types
using ..Types: Price, MarketState, OptionContract, OrderBook, Entrepreneur, SimConfig, Position
using ..Pricing # Import pricing engines
# Import functions from other modules
using ..Agents: decide_action, calculate_volatility
using ..Market: update_orderbook
using ..Market: evolve_market # Import market evolution function

# Import only what's needed from standard libraries and packages
using Random: MersenneTwister
using Statistics: mean, std
using Printf: @sprintf # Keep this specific import for clarity
using Printf # Add the general using statement for the module
using Plots: plot, savefig # For plotting results
using DataFrames: DataFrame

"""
run_simulation(config::SimConfig) -> DataFrame

Coordinates the Catallaxy Model simulation:
- Initializes agents, market state, contracts
- Loops: agents observe → decide → act → market updates → diagnostics
- Returns structured DataFrame with diagnostics (prices, deltas, risk premia)

References: Figlewski (1989, 2017), Brandimarte (Ch8, Ch10).
"""
function run_simulation(config::SimConfig, n_steps::Int)
    # Initialize reproducible RNG with explicit seed
    rng = MersenneTwister(config.seed)                # Pure functional RNG
    
    # Define option contract
    opt = OptionContract(
        "STOCK",                                      # Underlying asset
        100.0,                                        # Strike
        0.25,                                         # Maturity (3 months)
        false,                                        # Put option
        true                                          # American style
    )

    # Initialize market state
    state = MarketState(
        Dict(opt.underlying => Price(100.0)),        # Initial underlying price
        Dict(opt.underlying => 0.2),                # Initial underlying volatility
        Dict(opt.underlying => 1000.0),             # Initial underlying liquidity (depth)
        0.0                                           # Initial time
    )
    
    # Initialize order book
    ob = OrderBook()                         # Use default empty book constructor
    
    # Initialize agents
    agents = initialize_agents(config, opt, state)

    # Create a map for quick agent lookup by ID
    agent_map = Dict(agent.id => agent for agent in agents)

    # --- Data Logging Initialization ---
    price_history = [state.prices[opt.underlying].value]
    # New Histories:
    delta_history = Float64[]
    spread_history = Float64[]
    volume_history = Float64[]
    # Store capital history per agent {AgentID => [Capital]}
    agent_capital_history = Dict{Int, Vector{Float64}}(
        agent.id => [agent.capital] for agent in agents
    )
    option_price_history = Float64[] # Also track option price for plotting

    println("Starting simulation...")
    # Main simulation loop
    for step in 1:n_steps
        # --- Update All Agents' Volatility Estimates --- 
        # Before agents decide actions, update their view of market vol based on current price
        current_stock_price = state.prices[opt.underlying].value
        for agent in agents
            asset_key = opt.underlying
            # Ensure history vector exists
            if !haskey(agent.price_history, asset_key); agent.price_history[asset_key] = Float64[]; end
            # Append current market price
            push!(agent.price_history[asset_key], current_stock_price)
            # Keep history within lookback window
            if length(agent.price_history[asset_key]) > config.vol_lookback + 1
                popfirst!(agent.price_history[asset_key])
            end
            # Recalculate estimated volatility
            agent.estimated_volatility[asset_key] = calculate_volatility(agent.price_history[asset_key])
            # Handle potential NaN from calculate_volatility (e.g., if history is still too short)
            if isnan(agent.estimated_volatility[asset_key])
                agent.estimated_volatility[asset_key] = state.volatility[asset_key] # Fallback to market state vol if NaN
            end
        end
        
        # --- Agent Actions Phase ---
        orders = Dict{Symbol, Any}[] # Initialize with specific type
        for agent in agents
            # Volatility is now updated once for all agents before this loop.
            # Agent decides action based on current state and their estimated vol
            agent_rng = MersenneTwister(agent.rng_seed + step) # Agent-specific RNG for this step
            action = decide_action(agent, state, config, agent_rng, opt) # Use agent's vol via agent struct
            
            # 3. Append valid orders to the main list
            if action isa Vector{Dict{Symbol, Any}} # Check if it's a list of orders
                append!(orders, action) # Append all orders from the list
            elseif action isa Dict # Check if it's *any* dictionary
                # If it's a Dict, check if it's specifically the :none action
                if get(action, :action, nothing) != :none
                    # This case shouldn't happen with current decide_action logic
                    # but good to handle defensively
                    println("Warning: Agent $(agent.id) returned unexpected Dict action: $action")
                end
                # If action[:action] == :none, successfully do nothing
            else
                 println("Warning: Agent $(agent.id) returned unexpected action type: $(typeof(action))")
            end
        end
        
        # --- Market Update Phase ---
        ob, trades = update_orderbook(ob, orders, state, config, opt) 

        # --- Collect Market Data (Post-Trade) ---
        # Volume
        step_volume = sum(trade[:trade_size] for trade in trades; init=0.0)
        push!(volume_history, step_volume)

        # Spread (Handle empty book case)
        best_bid = isempty(ob.bids) ? -Inf : first(ob.bids)[1] # price is first element
        best_ask = isempty(ob.asks) ? Inf : first(ob.asks)[1] # price is first element
        step_spread = if !ismissing(best_bid) && !ismissing(best_ask)
            # Extract values before subtraction, handling potential Inf/-Inf
            ask_val = isa(best_ask, Price) ? best_ask.value : best_ask # Get Float64 value
            bid_val = isa(best_bid, Price) ? best_bid.value : best_bid # Get Float64 value
            max(0.0, ask_val - bid_val) # Ensure spread is non-negative
        else
            missing # Store missing if either is missing
        end
        push!(spread_history, step_spread)

        # --- Option Price History (Theoretical Fair Value) ---
        # Record the fair value calculated by a reference agent (MM1) 
        # using the current market state and their estimated volatility.
        # This provides a smoother view than post-trade mid-price or last trade.
        mm1 = agent_map[1] # Assume Agent 1 is the reference market maker
        mm1_est_vol = mm1.estimated_volatility[opt.underlying] # Use MM1's current vol estimate for the underlying
        # Ensure volatility is not NaN before pricing
        if isnan(mm1_est_vol)
            # Fallback strategy if vol is NaN (e.g., use market state vol or a default)
            println("Warning: MM1 estimated volatility is NaN at step $step. Using market state volatility.")
            mm1_est_vol = state.volatility[opt.underlying]
        end
        # Create a local RNG for this pricing call using the agent's seed
        agent_rng = MersenneTwister(mm1.rng_seed)
        # Calculate theoretical fair value using the current state (S, t) and MM1's vol
        theoretical_fair_value = Pricing.price_american_mc(opt, state, config, agent_rng, mm1_est_vol)
        push!(option_price_history, theoretical_fair_value)
        
        # --- Process Trades and Update Agents --- 
        for trade in trades
            buyer = agent_map[trade[:buyer_id]]
            seller = agent_map[trade[:seller_id]]
            trade_size = trade[:trade_size]
            trade_price = trade[:trade_price]
            buyer_cost = trade[:buyer_cost]
            seller_cost = trade[:seller_cost]
            trade_type = trade[:type]
            option_key = trade[:option_key] # Will be nothing for stock trades

            # Determine the asset key for position update
            asset_key = (trade_type == :stock) ? opt.underlying : option_key
            if isnothing(asset_key)
                println("Warning: Trade has unknown type or missing key: $trade")
                continue
            end

            # Update Capital
            buyer.capital -= (trade_price * trade_size + buyer_cost)
            seller.capital += (trade_price * trade_size - seller_cost)

            # Update Accumulated Costs
            buyer.accumulated_costs += buyer_cost
            seller.accumulated_costs += seller_cost

            # Update Positions
            buyer.position[asset_key] = get(buyer.position, asset_key, 0.0) + trade_size
            seller.position[asset_key] = get(seller.position, asset_key, 0.0) - trade_size
            
            # Clean up zero positions (optional)
            if haskey(buyer.position, asset_key) && abs(buyer.position[asset_key]) < 1e-9; delete!(buyer.position, asset_key); end
            if haskey(seller.position, asset_key) && abs(seller.position[asset_key]) < 1e-9; delete!(seller.position, asset_key); end

            # Update market state price using the correct Price constructor
            state.prices[asset_key] = Price(trade_price)

            # --- REMOVED: Redundant Price History / Vol Update based on Trade --- 
            # This logic is now handled for all agents at the start of the step
            # based on the market price, ensuring consistent vol estimates.

            # --- Update Log Counters --- 
            step_volume += trade_size
        end

        # --- Market State Evolution --- 
        state = evolve_market(state, config, rng, opt.underlying)  # Evolve underlying price/vol
        push!(price_history, state.prices[opt.underlying].value) # Log new stock price

        # --- Calculate Delta (End of Step) ---
        S = state.prices[opt.underlying].value
        K = opt.strike
        T = max(1e-9, (n_steps - step) / n_steps) # Time to expiry in years, ensure > 0
        # Use risk_premium as r? Or 0? Using 0 for theoretical delta for now.
        r = config.risk_premium 
        # Which sigma? Use market state's sigma for now.
        sigma = state.volatility[opt.underlying]
        # Handle T=0 case (expiration) - already handled by Pricing.calculate_delta if T is small but positive
        step_delta = Pricing.calculate_bs_delta(S, K, T, r, sigma, opt.is_call) # Use calculate_bs_delta
        push!(delta_history, step_delta)

        # --- Update Agent Portfolio & Capital History (End of Step) ---
        for agent in agents
            # Calculate current portfolio value: Capital + MTM value of stock position
            # Use get() for agent.position to handle cases where agent has no stock position
            mtm_position_value = get(agent.position, opt.underlying, 0.0) * state.prices[opt.underlying].value
            current_portfolio_value = agent.capital + mtm_position_value
            push!(agent.portfolio_history, current_portfolio_value)

            # Capital History (Log current capital)
            push!(agent_capital_history[agent.id], agent.capital)
        end

        # Optional: Print progress
        if step % 50 == 0 || step == n_steps
            println("Step $step/$n_steps completed. Price: $(round(state.prices[opt.underlying].value, digits=2))")
        end
    end
    
    println("Simulation finished.")
    
    # Return final agent states and ALL collected histories
    return agents, price_history, option_price_history, delta_history, spread_history, volume_history, agent_capital_history
end

function initialize_agents(config::SimConfig, option::OptionContract, initial_state::MarketState) :: Vector{Entrepreneur}
    agents = Vector{Entrepreneur}()
    # Use the correct field name: config.seed
    rng = MersenneTwister(config.seed) # Master RNG for agent seeds
    
    # Generate unique option key
    option_key = "OPTION_$(option.strike)_$(option.is_call ? 'C' : 'P')"
    
    # Get initial volatility from the market state
    initial_vol = initial_state.volatility[option.underlying]

    # 1. Market Maker - Use hardcoded initial capital
    mm_initial_capital = 10_000_000.0  # Increased x10
    # Initialize using keyword arguments
    push!(agents, Entrepreneur(
        id = 1,
        initial_capital = mm_initial_capital,
        capital = mm_initial_capital,
        risk_aversion = 0.2, # low for market makers
        role = :market_maker,
        position = Dict{String, Float64}(), # empty to start
        estimated_volatility = Dict(option.underlying => initial_vol),
        price_history = Dict(option.underlying => [initial_state.prices[option.underlying].value]),
        accumulated_costs = 0.0,
        portfolio_history = Float64[], # empty vector of Float64
        stop_loss = missing,
        margin_limit = missing,
        rng_seed = rand(rng, 1:10000)
    ))

    # 2. Portfolio Insurer 
    # EXAMPLE: Start short 1 unit of the option
    insurer_initial_pos = Dict{String, Float64}() # Use Dict constructor
    insurer_initial_pos[option_key] = -1.0 # Short 1 option
    insurer_initial_capital = 5_000_000.0  # Increased x10
    # Initialize using keyword arguments
    push!(agents, Entrepreneur(
        id = 2,
        initial_capital = insurer_initial_capital,
        capital = insurer_initial_capital,
        risk_aversion = 0.5, # medium
        role = :insurer,
        position = insurer_initial_pos, # short 1 option
        estimated_volatility = Dict(option.underlying => initial_vol),
        price_history = Dict(option.underlying => [initial_state.prices[option.underlying].value]),
        accumulated_costs = 0.0,
        portfolio_history = Float64[], # empty vector of Float64
        stop_loss = missing,
        margin_limit = missing,
        rng_seed = rand(rng, 1:10000)
    ))

    # 3. Retail Trader
    retail_initial_capital = 1_000_000.0 # Increased x10
    # Initialize using keyword arguments
    push!(agents, Entrepreneur(
        id = 3,
        initial_capital = retail_initial_capital,
        capital = retail_initial_capital,
        risk_aversion = 0.5, # higher than MM
        role = :retail,
        position = Dict{String, Float64}(),
        estimated_volatility = Dict(option.underlying => initial_vol),
        price_history = Dict(option.underlying => [initial_state.prices[option.underlying].value]),
        accumulated_costs = 0.0,
        portfolio_history = Float64[],
        stop_loss = missing,
        margin_limit = missing,
        rng_seed = rand(rng, 1:10000)
    ))

    # 4. Retail Trader 2
    retail2_initial_capital = 900_000.0 # Increased x10
    # Initialize using keyword arguments
    push!(agents, Entrepreneur(
        id = 4,
        initial_capital = retail2_initial_capital,
        capital = retail2_initial_capital,
        risk_aversion = 0.6, # slightly higher
        role = :retail,
        position = Dict{String, Float64}(),
        estimated_volatility = Dict(option.underlying => initial_vol),
        price_history = Dict(option.underlying => [initial_state.prices[option.underlying].value]),
        accumulated_costs = 0.0,
        portfolio_history = Float64[],
        stop_loss = missing,
        margin_limit = missing,
        rng_seed = rand(rng, 1:10000)
    ))

    # 5. Retail Trader 3
    retail3_initial_capital = 1_100_000.0 # Increased x10
    # Initialize using keyword arguments
    push!(agents, Entrepreneur(
        id = 5,
        initial_capital = retail3_initial_capital,
        capital = retail3_initial_capital,
        risk_aversion = 0.4, # slightly lower
        role = :retail,
        position = Dict{String, Float64}(),
        estimated_volatility = Dict(option.underlying => initial_vol),
        price_history = Dict(option.underlying => [initial_state.prices[option.underlying].value]),
        accumulated_costs = 0.0,
        portfolio_history = Float64[],
        stop_loss = missing,
        margin_limit = missing,
        rng_seed = rand(rng, 1:10000)
    ))

    # 6. Market Maker 2 (Competing)
    mm2_initial_capital = 9_500_000.0 # Increased x10
    # Initialize using keyword arguments
    push!(agents, Entrepreneur(
        id = 6,
        initial_capital = mm2_initial_capital,
        capital = mm2_initial_capital,
        risk_aversion = 0.25, # slightly higher than MM1
        role = :market_maker,
        position = Dict{String, Float64}(),
        estimated_volatility = Dict(option.underlying => initial_vol),
        price_history = Dict(option.underlying => [initial_state.prices[option.underlying].value]),
        accumulated_costs = 0.0,
        portfolio_history = Float64[],
        stop_loss = missing,
        margin_limit = missing,
        rng_seed = rand(rng, 1:10000)
    ))

    # 7. Arbitrageur
    arb_initial_capital = 2_000_000.0 # Increased x10
    # Initialize using keyword arguments
    push!(agents, Entrepreneur(
        id = 7,
        initial_capital = arb_initial_capital,
        capital = arb_initial_capital,
        risk_aversion = 0.1, # low
        role = :arbitrageur,
        position = Dict{String, Float64}(),
        estimated_volatility = Dict(option.underlying => initial_vol),
        price_history = Dict(option.underlying => [initial_state.prices[option.underlying].value]),
        accumulated_costs = 0.0,
        portfolio_history = Float64[],
        stop_loss = missing,
        margin_limit = missing,
        rng_seed = rand(rng, 1:10000)
    ))

    return agents
end

"""
plot_simulation_results(results::DataFrame) -> Plots.Plot

Visualizes key metrics from a Catallaxy Model simulation.
- Spot and option prices
- Option delta
- Bid-ask spread
- Trade volume

Returns a multi-panel plot object.
"""
function plot_simulation_results(results::DataFrame) :: Plots.Plot
    # Create multi-panel plot
    p1 = plot(
        results.time,
        [results.spot_price, results.option_price],
        label = ["Spot Price" "Option Price"],
        title = "Prices",
        xlabel = "Time (years)",
        ylabel = "Price",
        linewidth = 2
    )
    
    p2 = plot(
        results.time,
        results.option_delta,
        label = "Delta",
        title = "Option Delta",
        xlabel = "Time (years)",
        ylabel = "Delta",
        linewidth = 2,
        color = :red
    )
    
    p3 = plot(
        results.time,
        results.bid_ask_spread,
        label = "Bid-Ask Spread",
        title = "Market Liquidity",
        xlabel = "Time (years)",
        ylabel = "Spread",
        linewidth = 2,
        color = :green
    )
    
    p4 = bar(
        results.time,
        results.trade_volume,
        label = "Volume",
        title = "Trading Activity",
        xlabel = "Time (years)",
        ylabel = "# Trades",
        color = :purple
    )
    
    # Combine plots
    plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 600))
end

end # module

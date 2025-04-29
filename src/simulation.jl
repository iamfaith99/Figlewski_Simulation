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
    
    # Initialize market state (using "STOCK" as the underlying key)
    state = MarketState(
        Dict("STOCK" => Price(100.0)),               # Initial underlying price
        Dict("STOCK" => 0.2),                       # Initial underlying volatility
        Dict("STOCK" => 1000.0),                    # Initial underlying liquidity (depth)
        0.0                                           # Initial time
    )
    
    # Initialize order book
    ob = OrderBook([], [], 5)                         # Empty book with depth 5
    
    # Define option contract
    opt = OptionContract(
        "STOCK",                                      # Underlying asset
        100.0,                                        # Strike
        0.25,                                         # Maturity (3 months)
        false,                                        # Put option
        true                                          # American style
    )
    
    # Initialize agents
    agents = initialize_agents(config, opt, state)

    # Create a map for quick agent lookup by ID
    agent_map = Dict(agent.id => agent for agent in agents)

    # --- Data Logging Initialization ---
    price_history = [state.prices["STOCK"].value]
    # Add other histories if needed, e.g., agent-specific data
    
    println("Starting simulation...")
    # Main simulation loop
    for step in 1:n_steps
        # --- Agent Actions Phase ---
        orders = Dict{Symbol, Any}[] # Initialize with specific type
        for agent in agents
            # 1. Update Agent's Price History & Volatility Estimate
            current_price = state.prices["STOCK"].value
            push!(agent.price_history["STOCK"], current_price)
            # Keep history within lookback window
            if length(agent.price_history["STOCK"]) > config.vol_lookback + 1 # Need N+1 points for N returns
                popfirst!(agent.price_history["STOCK"])
            end
            # Calculate new volatility estimate
            new_vol_estimate = calculate_volatility(agent.price_history["STOCK"])
            if !isnan(new_vol_estimate)
                agent.estimated_volatility["STOCK"] = new_vol_estimate
            end # Otherwise, keep the previous estimate

            # 2. Agent decides action based on current state and their estimated vol
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
            if abs(buyer.position[asset_key]) < 1e-9; delete!(buyer.position, asset_key); end
            if abs(seller.position[asset_key]) < 1e-9; delete!(seller.position, asset_key); end

            # Update market state price using the correct Price constructor
            state.prices[asset_key] = Price(trade_price)

            # Update agent price history and vol estimate based on the traded asset
            current_price = trade_price
            
            # Ensure history vector exists before pushing
            if !haskey(buyer.price_history, asset_key); buyer.price_history[asset_key] = Float64[]; end
            push!(buyer.price_history[asset_key], current_price)
            if !haskey(seller.price_history, asset_key); seller.price_history[asset_key] = Float64[]; end
            push!(seller.price_history[asset_key], current_price)
            
            # Keep history within lookback window
            if length(buyer.price_history[asset_key]) > config.vol_lookback + 1 # Need N+1 points for N returns
                popfirst!(buyer.price_history[asset_key])
            end
            if length(seller.price_history[asset_key]) > config.vol_lookback + 1 # Need N+1 points for N returns
                popfirst!(seller.price_history[asset_key])
            end
            
            # Calculate new volatility estimate
            # Ensure volatility entry exists before potentially updating
            initial_stock_vol = state.volatility["STOCK"] # Get initial stock vol (might need adjustment if state vol changes)
            if !haskey(buyer.estimated_volatility, asset_key); buyer.estimated_volatility[asset_key] = initial_stock_vol; end
            new_vol_estimate_buyer = calculate_volatility(buyer.price_history[asset_key])
            if !isnan(new_vol_estimate_buyer)
                buyer.estimated_volatility[asset_key] = new_vol_estimate_buyer
            end # Otherwise, keep the existing estimate
            
            if !haskey(seller.estimated_volatility, asset_key); seller.estimated_volatility[asset_key] = initial_stock_vol; end
            new_vol_estimate_seller = calculate_volatility(seller.price_history[asset_key])
            if !isnan(new_vol_estimate_seller)
                seller.estimated_volatility[asset_key] = new_vol_estimate_seller
            end # Otherwise, keep the existing estimate
        end

        # --- Market State Evolution --- 
        state = evolve_market(state, config, rng)  # Evolve underlying price/vol
        push!(price_history, state.prices["STOCK"].value) # Log new price

        # --- Update Agent Portfolio History (End of Step) ---
        for agent in agents
            # Calculate current portfolio value: Capital + MTM value of stock position
            # Use get() for agent.position to handle cases where agent has no stock position
            mtm_position_value = get(agent.position, "STOCK", 0.0) * state.prices["STOCK"].value
            current_portfolio_value = agent.capital + mtm_position_value
            push!(agent.portfolio_history, current_portfolio_value)
        end

        # --- Diagnostics (Optional) ---
        if step % 50 == 0 || step == n_steps
            println("Step $step/$n_steps completed. Price: $(round(state.prices["STOCK"].value, digits=2))")
        end
    end
    
    println("Simulation finished.")
    
    # Return final agent states and price history
    return agents, price_history # Return agents for detailed analysis
end

function initialize_agents(config::SimConfig, option::OptionContract, initial_state::MarketState) :: Vector{Entrepreneur}
    agents = Vector{Entrepreneur}()
    # Use the correct field name: config.seed
    rng = MersenneTwister(config.seed) # Master RNG for agent seeds
    
    # Generate unique option key
    option_key = "OPTION_$(option.strike)_$(option.is_call ? 'C' : 'P')"
    
    # Get initial volatility from the market state
    initial_vol = initial_state.volatility["STOCK"]

    # 1. Market Maker - Use hardcoded initial capital
    mm_initial_capital = 10_000_000.0  # Increased x10
    # Initialize using positional arguments matching the struct definition
    # id, initial_capital, capital, risk_aversion, role, position, estimated_volatility, price_history, accumulated_costs, portfolio_history, stop_loss, margin_limit, rng_seed
    push!(agents, Entrepreneur(
        1, # id
        mm_initial_capital, # initial_capital
        mm_initial_capital, # capital (starts equal to initial)
        0.2, # risk_aversion (low for market makers)
        :market_maker, # role
        Dict{String, Float64}(), # position (empty to start)
        Dict(option.underlying => initial_vol), # estimated_volatility
        Dict(option.underlying => [initial_state.prices[option.underlying].value]), # price_history
        0.0, # accumulated_costs
        Float64[], # portfolio_history (empty vector of Float64)
        missing, # stop_loss
        missing, # margin_limit
        rand(rng, 1:10000) # rng_seed as Int
    ))

    # 2. Portfolio Insurer 
    # EXAMPLE: Start short 1 unit of the option
    insurer_initial_pos = Dict{String, Float64}() # Use Dict constructor
    insurer_initial_pos[option_key] = -1.0 # Short 1 option
    insurer_initial_capital = 5_000_000.0  # Increased x10
    # Initialize using positional arguments matching the struct definition
    # id, initial_capital, capital, risk_aversion, role, position, estimated_volatility, price_history, accumulated_costs, portfolio_history, stop_loss, margin_limit, rng_seed
    push!(agents, Entrepreneur(
        2, # id
        insurer_initial_capital, # initial_capital
        insurer_initial_capital, # capital (starts equal to initial)
        0.5, # risk_aversion (medium)
        :insurer, # role
        insurer_initial_pos, # position (short 1 option)
        Dict(option.underlying => initial_vol), # estimated_volatility
        Dict(option.underlying => [initial_state.prices[option.underlying].value]), # price_history
        0.0, # accumulated_costs
        Float64[], # portfolio_history (empty vector of Float64)
        missing, # stop_loss
        missing, # margin_limit
        rand(rng, 1:10000) # rng_seed as Int
    ))

    # 3. Retail Trader
    retail_initial_capital = 1_000_000.0 # Increased x10
    # Initialize using positional arguments matching the struct definition
    push!(agents, Entrepreneur(
        3, # id
        retail_initial_capital, # initial_capital
        retail_initial_capital, # capital
        0.5, # risk_aversion (higher than MM)
        :retail, # role
        Dict{String, Float64}(), # position
        Dict(option.underlying => initial_vol), # estimated_volatility
        Dict(option.underlying => [initial_state.prices[option.underlying].value]), # price_history
        0.0, # accumulated_costs
        Float64[], # portfolio_history
        missing, # stop_loss
        missing, # margin_limit
        rand(rng, 1:10000) # rng_seed
    ))

    # 4. Retail Trader 2
    retail2_initial_capital = 900_000.0 # Increased x10
    push!(agents, Entrepreneur(
        4, # id
        retail2_initial_capital, # initial_capital
        retail2_initial_capital, # capital
        0.6, # risk_aversion (slightly higher)
        :retail, # role
        Dict{String, Float64}(), # position
        Dict(option.underlying => initial_vol), # estimated_volatility
        Dict(option.underlying => [initial_state.prices[option.underlying].value]), # price_history
        0.0, # accumulated_costs
        Float64[], # portfolio_history
        missing, # stop_loss
        missing, # margin_limit
        rand(rng, 1:10000) # rng_seed
    ))

    # 5. Retail Trader 3
    retail3_initial_capital = 1_100_000.0 # Increased x10
    push!(agents, Entrepreneur(
        5, # id
        retail3_initial_capital, # initial_capital
        retail3_initial_capital, # capital
        0.4, # risk_aversion (slightly lower)
        :retail, # role
        Dict{String, Float64}(), # position
        Dict(option.underlying => initial_vol), # estimated_volatility
        Dict(option.underlying => [initial_state.prices[option.underlying].value]), # price_history
        0.0, # accumulated_costs
        Float64[], # portfolio_history
        missing, # stop_loss
        missing, # margin_limit
        rand(rng, 1:10000) # rng_seed
    ))

    # 6. Market Maker 2 (Competing)
    mm2_initial_capital = 9_500_000.0 # Increased x10
    push!(agents, Entrepreneur(
        6, # id
        mm2_initial_capital, # initial_capital
        mm2_initial_capital, # capital
        0.25, # risk_aversion (slightly higher than MM1)
        :market_maker, # role
        Dict{String, Float64}(), # position
        Dict(option.underlying => initial_vol), # estimated_volatility
        Dict(option.underlying => [initial_state.prices[option.underlying].value]), # price_history
        0.0, # accumulated_costs
        Float64[], # portfolio_history
        missing, # stop_loss
        missing, # margin_limit
        rand(rng, 1:10000) # rng_seed
    ))

    # 7. Arbitrageur
    arb_initial_capital = 2_000_000.0 # Increased x10
    push!(agents, Entrepreneur(
        7, # id
        arb_initial_capital, # initial_capital
        arb_initial_capital, # capital
        0.1, # risk_aversion (low)
        :arbitrageur, # role
        Dict{String, Float64}(), # position
        Dict(option.underlying => initial_vol), # estimated_volatility
        Dict(option.underlying => [initial_state.prices[option.underlying].value]), # price_history
        0.0, # accumulated_costs
        Float64[], # portfolio_history
        missing, # stop_loss
        missing, # margin_limit
        rand(rng, 1:10000) # rng_seed
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

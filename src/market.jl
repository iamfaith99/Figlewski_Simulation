"""
Market/order book mechanics for Catallaxy Model.
Handles decentralized matching, price discovery, and arbitrage limits.
References: Figlewski, market microstructure.
"""

module Market
# Import the Types module to access its exported types
using ..Types: Price, OrderBook, SimConfig, MarketState, OptionContract, Agent
using DataStructures: SortedDict
using Random: AbstractRNG

# Export functions provided by this module
export update_orderbook, evolve_market

"""
update_orderbook(ob::OrderBook, orders::Vector{Dict{Symbol, Any}}, state::MarketState, config::SimConfig, option::OptionContract) 
    
    Updates the order book by processing incoming orders, matching trades, and calculating costs.
    Handles separate matching for stock and option orders.

    Returns: 
        - trades: Vector{Dict} - List of executed trades with details.
        - updated_ob: OrderBook - Updated order book.
"""
function update_orderbook(ob::OrderBook, orders::Vector{Dict{Symbol, Any}}, state::MarketState, config::SimConfig, option::OptionContract)
    trades = Dict{Symbol, Any}[] # Initialize list for trades
    # Use deep copies to avoid modifying the original order book state unintentionally
    updated_bids = deepcopy(ob.bids)
    updated_asks = deepcopy(ob.asks)
    
    # Separate orders by type (stock vs option) and option key
    stock_orders = filter(o -> o[:type] == :stock, orders)
    option_orders = filter(o -> o[:type] == :option, orders)

    # Process Stock Orders
    stock_bids = filter(o -> o[:side] == :buy, stock_orders)
    stock_asks = filter(o -> o[:side] == :sell, stock_orders)
    sort!(stock_bids, by=x->x[:price], rev=true) # Highest bid first
    sort!(stock_asks, by=x->x[:price])          # Lowest ask first

    # --- Stock Matching Logic (Simplified Cross) ---
    # For simplicity, match best bid vs best ask sequentially
    # A full price-time priority matching is more complex
    bid_idx, ask_idx = 1, 1
    while bid_idx <= length(stock_bids) && ask_idx <= length(stock_asks)
        best_bid = stock_bids[bid_idx]
        best_ask = stock_asks[ask_idx]
        
        if best_bid[:price] >= best_ask[:price]
            trade_price = (best_bid[:price] + best_ask[:price]) / 2 # Mid-price execution
            trade_size = min(best_bid[:size], best_ask[:size])

            # Apply costs (assuming apply_costs handles None option_key for stocks)
            buyer_id = best_bid[:agent_id]
            seller_id = best_ask[:agent_id]
            buyer_role = best_bid[:agent_role]
            seller_role = best_ask[:agent_role]
            buyer_cost_total = apply_costs(trade_price, trade_size, :buy, buyer_role, config, state, option)
            seller_cost_total = apply_costs(trade_price, trade_size, :sell, seller_role, config, state, option)

            push!(trades, Dict(
                :trade_price => trade_price,
                :trade_size => trade_size,
                :buyer_id => buyer_id,
                :seller_id => seller_id,
                :buyer_cost => buyer_cost_total, 
                :seller_cost => seller_cost_total,
                :type => :stock, # Identify trade type
                :option_key => nothing # Stock trades have no option key
            ))

            # Update remaining order sizes
            best_bid[:size] -= trade_size
            best_ask[:size] -= trade_size

            # Remove filled orders
            if best_bid[:size] < 1e-9; bid_idx += 1; end
            if best_ask[:size] < 1e-9; ask_idx += 1; end
        else
            break # Bids and asks don't cross
        end
    end
    # Note: Remaining unmatched stock orders are discarded in this simple model.
    # A real order book would add them to updated_bids/updated_asks.

    # --- Process Option Orders (Group by Option Key) ---
    grouped_option_orders = Dict{String, Vector{Dict{Symbol, Any}}}()
    for order in option_orders
        key = order[:option_key]
        if !haskey(grouped_option_orders, key)
            grouped_option_orders[key] = []
        end
        push!(grouped_option_orders[key], order)
    end

    # Match orders for each option key separately
    for (option_key, key_orders) in grouped_option_orders
        option_bids = filter(o -> o[:side] == :buy, key_orders)
        option_asks = filter(o -> o[:side] == :sell, key_orders)
        sort!(option_bids, by=x->x[:price], rev=true)
        sort!(option_asks, by=x->x[:price])

        # --- Option Matching Logic (Similar to Stock) ---
        bid_idx, ask_idx = 1, 1
        while bid_idx <= length(option_bids) && ask_idx <= length(option_asks)
            best_bid = option_bids[bid_idx]
            best_ask = option_asks[ask_idx]

            if best_bid[:price] >= best_ask[:price]
                trade_price = (best_bid[:price] + best_ask[:price]) / 2
                trade_size = min(best_bid[:size], best_ask[:size])

                buyer_id = best_bid[:agent_id]
                seller_id = best_ask[:agent_id]
                buyer_role = best_bid[:agent_role]
                seller_role = best_ask[:agent_role]
                buyer_cost_total = apply_costs(trade_price, trade_size, :buy, buyer_role, config, state, option)
                seller_cost_total = apply_costs(trade_price, trade_size, :sell, seller_role, config, state, option)

                push!(trades, Dict(
                    :trade_price => trade_price,
                    :trade_size => trade_size,
                    :buyer_id => buyer_id,
                    :seller_id => seller_id,
                    :buyer_cost => buyer_cost_total, 
                    :seller_cost => seller_cost_total,
                    :type => :option, # Identify trade type
                    :option_key => option_key # Include the specific option key
                ))

                best_bid[:size] -= trade_size
                best_ask[:size] -= trade_size

                if best_bid[:size] < 1e-9; bid_idx += 1; end
                if best_ask[:size] < 1e-9; ask_idx += 1; end
            else
                break
            end
        end
        # Note: Remaining unmatched option orders are also discarded here.
    end

    # Convert SortedDicts to Vector{Tuple{Price, Float64}} for the constructor
    bids_vector = [(Price(k.value), v) for (k, v) in updated_bids]
    asks_vector = [(Price(k.value), v) for (k, v) in updated_asks]
    
    # Construct the updated OrderBook object with correct types and depth argument
    updated_ob = OrderBook(bids_vector, asks_vector, 0) # Using 0 as placeholder depth

    # Return the updated order book and the list of trades
    return updated_ob, trades 
end

"""
apply_costs(price::Float64, size::Float64, side::Symbol, role::Symbol, config::SimConfig, state::MarketState, option::OptionContract) -> Float64

Calculates the total transaction cost for a trade based on role, price, size, and config.
Incorporates fixed costs and slippage.
"""
function apply_costs(price::Float64, size::Float64, side::Symbol, role::Symbol, config::SimConfig, state::MarketState, option::OptionContract) :: Float64
    # 1. Role-based fixed cost per unit
    cost_per_unit = 0.0
    if role == :market_maker
        cost_per_unit = config.mm_transaction_cost
    elseif role == :retail
        cost_per_unit = config.retail_transaction_cost
    elseif role == :insurer
        cost_per_unit = config.insurer_transaction_cost
    end
    fixed_cost = cost_per_unit * price # Assume cost is proportional to price

    # 2. Slippage cost (simplified: assumes impact on execution price)
    # We'll treat slippage as an additional cost component here.
    # A more realistic model might adjust the trade_price itself.
    liquidity = get(state.liquidity, option.underlying, 1e9) # Use underlying liquidity
    slippage_impact_per_unit = config.slippage_factor * (size / max(1.0, liquidity))
    
    # Slippage effectively worsens the execution price
    # Buy side: pays more -> positive cost impact
    # Sell side: receives less -> positive cost impact
    slippage_cost = slippage_impact_per_unit * price 

    total_cost_per_unit = fixed_cost + slippage_cost
    
    return total_cost_per_unit * size # Total cost for the trade size
end

"""
evolve_market(state::MarketState, config::SimConfig, rng::AbstractRNG)

Evolves the underlying market state (e.g., stock price) using GBM.
- Assumes a single underlying asset "STOCK".
- Uses risk premium from config as the drift `mu`.
- Returns a new `MarketState`.
"""
function evolve_market(state::MarketState, config::SimConfig, rng::AbstractRNG)
    # --- Evolve Underlying Asset Price (GBM) --- 
    current_price = state.prices["STOCK"].value
    sigma = state.volatility["STOCK"] # Use current volatility
    mu = config.risk_premium          # Use risk premium as drift 
    dt = 1.0 / 252.0                 # Time step (daily)
    Z = randn(rng)                   # Standard normal random variable
    
    # GBM formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    next_price_val = current_price * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    next_price_val = max(0.01, next_price_val) # Ensure price doesn't go below a floor
    
    new_prices = Dict("STOCK" => Price(next_price_val))
    
    # --- Update Volatility (Could be more complex, e.g., GARCH) ---
    # For now, keep volatility constant, but this is where it could change
    new_volatility = state.volatility 
    
    # --- Update Liquidity (Could be dynamic) ---
    # For now, keep liquidity constant
    new_liquidity = state.liquidity
    
    # --- Update Time ---
    new_time = state.time + dt
    
    return MarketState(new_prices, new_volatility, new_liquidity, new_time)
end

end # module

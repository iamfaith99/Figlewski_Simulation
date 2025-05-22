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
    # Deep copy needs to handle the new tuple structure
    updated_bids = deepcopy(ob.bids) # Now Vector{Tuple{Price, Size, AgentID, Role}}
    updated_asks = deepcopy(ob.asks) # Now Vector{Tuple{Price, Size, AgentID, Role}}
    
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
        # --- Combine Incoming and Resting Orders --- 
        for order in key_orders
            price = Price(order[:price])
            size = order[:size]
            agent_id = order[:agent_id]
            agent_role = order[:agent_role]
            target_list = order[:side] == :buy ? updated_bids : updated_asks
            
            # Find if price level exists (matching on Price only)
            found_idx = findfirst(item -> item[1] == price, target_list)
            if !isnothing(found_idx)
                # TODO MAJOR: Current order aggregation at a price level is a simplification.
                # It sums total size at a price level but only stores the (AgentID, AgentRole) 
                # of the *last* incoming order that contributed to (or created) this specific price level. 
                # This means:
                #   1. If multiple agents have orders at the same best price, only one agent's
                #      details are associated with that price level for matching purposes.
                #   2. Cost attribution for trades (see cost calculation note below) might use
                #      this simplified agent info, potentially leading to inaccuracies if the
                #      matched agent info at the BBO isn't the one whose order portion is filled.
                #   3. Market analysis relying on specific agent contributions at various price
                #      levels will not be accurate with this simplification.
                # A more robust solution would involve changing the OrderBook structure 
                # (e.g., `bids::Vector{Tuple{Price, Vector{Tuple{Float64, Int, Symbol}}}}` 
                # to store Price -> List of (Size, AgentID, AgentRole))
                # to track individual orders or detailed agent contributions within an aggregated price level.
                # This would significantly impact matching logic, deepcopy, and data handling.
                # For now, accepting this limitation: update total size and overwrite agent info with the last incoming order's details.
                existing_price, existing_size, _, _ = target_list[found_idx]
                target_list[found_idx] = (existing_price, existing_size + size, agent_id, agent_role)
            else
                # Append new price level with agent info
                push!(target_list, (price, size, agent_id, agent_role))
            end
        end
        
        # --- Sort Combined Book --- 
        # Sort by price (primary), potentially add secondary sort later if needed
        sort!(updated_bids, by=item->item[1], rev=true) # Highest bid first
        sort!(updated_asks, by=item->item[1])          # Lowest ask first

        # --- Option Matching Logic (on combined book) ---
        while !isempty(updated_bids) && !isempty(updated_asks)
            best_bid_price, best_bid_size, buyer_id, buyer_role = first(updated_bids)
            best_ask_price, best_ask_size, seller_id, seller_role = first(updated_asks)

            if best_bid_price >= best_ask_price # Market crossed or touched
                trade_price = (best_bid_price.value + best_ask_price.value) / 2 # Mid-point execution
                trade_size = min(best_bid_size, best_ask_size)

                # --- Calculate Costs using Agent Info from Book --- 
                # NOTE ON COST ATTRIBUTION: Cost calculation uses the AgentID and AgentRole 
                # retrieved from the best bid/ask price level data in the `updated_bids`/`updated_asks`
                # lists (i.e., `buyer_id, buyer_role` and `seller_id, seller_role`). 
                # Due to the current order aggregation simplification (see TODO MAJOR above on aggregation), 
                # this agent information represents the *last* order that formed or added to this 
                # specific price level. It does not necessarily reflect all individual orders that might 
                # constitute that price level, nor does it guarantee that this specific agent's order is 
                # the one being matched if the level represents aggregated volume. This can lead to 
                # misattribution of transaction costs if precise per-agent, per-order cost tracking is critical.
                buyer_cost_total = apply_costs(trade_price, trade_size, :buy, buyer_role, config, state, option) 
                seller_cost_total = apply_costs(trade_price, trade_size, :sell, seller_role, config, state, option)

                push!(trades, Dict(
                    :trade_price => trade_price,
                    :trade_size => trade_size,
                    :buyer_id => buyer_id, # Use ID from book tuple
                    :seller_id => seller_id, # Use ID from book tuple
                    :buyer_cost => buyer_cost_total,
                    :seller_cost => seller_cost_total, 
                    :type => :option,
                    :option_key => option_key
                ))

                # Update remaining sizes in the book state
                new_bid_size = best_bid_size - trade_size
                new_ask_size = best_ask_size - trade_size

                # Remove filled levels or update remaining size
                if new_bid_size < 1e-9
                    popfirst!(updated_bids)
                else
                    # Update size, keep other info
                    updated_bids[1] = (best_bid_price, new_bid_size, buyer_id, buyer_role) 
                end

                if new_ask_size < 1e-9
                    popfirst!(updated_asks)
                else
                     # Update size, keep other info
                    updated_asks[1] = (best_ask_price, new_ask_size, seller_id, seller_role)
                end
            else
                break # Bids and asks do not cross
            end
        end
        # --- Combined book state is now updated after matching --- 
    end # End loop over option keys

    # Ensure book depth is maintained (truncate if needed) - Not implemented yet
    
    # Return the updated order book (containing agent info) and trades
    updated_ob = OrderBook(updated_bids, updated_asks) # Use updated lists directly
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
evolve_market(state::MarketState, config::SimConfig, rng::AbstractRNG, underlying_asset_key::String)

Evolves the underlying market state (e.g., stock price) using GBM.
- Uses the provided `underlying_asset_key`.
- Uses risk premium from config as the drift `mu`.
- Returns a new `MarketState`.
"""
function evolve_market(state::MarketState, config::SimConfig, rng::AbstractRNG, underlying_asset_key::String)
    # --- Evolve Underlying Asset Price (GBM) --- 
    current_price = state.prices[underlying_asset_key].value
    sigma = state.volatility[underlying_asset_key] # Use current volatility
    mu = config.risk_premium          # Use risk premium as drift 
    dt = 1.0 / 252.0                 # Time step (daily)
    Z = randn(rng)                   # Standard normal random variable
    
    # GBM formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    next_price_val = current_price * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    next_price_val = max(0.01, next_price_val) # Ensure price doesn't go below a floor
    
    new_prices = Dict(underlying_asset_key => Price(next_price_val))
    
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

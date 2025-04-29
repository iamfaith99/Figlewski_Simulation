"""
Behavioral agent logic for Catallaxy Model.
Implements empirical rules and bounded rationality.
References: Figlewski, market microstructure, behavioral finance.
"""

module Agents
# Import the Types module to access its exported types
using ..Types: Price, MarketState, SimConfig, Entrepreneur, MaybeFloat, OptionContract, Position, Agent
using ..Pricing # Import the Pricing module

# Import only what's needed from standard libraries
using Random: AbstractRNG, rand, randn
using Statistics: mean, std
using Distributions: Normal

"""
calculate_volatility(history::Vector{Float64}) -> Float64

Calculates annualized standard deviation of log returns from price history.
Returns NaN if history is too short (< 2 points).
"""
function calculate_volatility(history::Vector{Float64})
    if length(history) < 2
        return NaN # Cannot calculate returns
    end
    # Calculate log returns
    log_returns = log.(history[2:end] ./ history[1:end-1])
    # Calculate annualized standard deviation
    # Use sample standard deviation (std); default assumes n-1 denominator
    vol = std(log_returns) * sqrt(252) 
    return vol
end

"""
decide_action(agent::Entrepreneur, state::MarketState, config::SimConfig, rng::AbstractRNG, option::OptionContract) -> Dict{Symbol,Any}

Pure function: decides action based on role, state, and behavioral rules.
- Market maker: sets quotes, manages inventory, applies spread.
- Insurer: hedges dynamically, applies stop-loss.
- Retail: submits orders, may display behavioral bias.
- Returns: EITHER a Dict(:action => :none) OR a List{Dict} of orders.
"""
function decide_action(agent::Entrepreneur, state::MarketState, config::SimConfig, rng::AbstractRNG, option::OptionContract) # Return type hint removed for flexibility
    # Extract agent's estimated vol for the underlying
    est_vol = agent.estimated_volatility[option.underlying]
    underlying_price = state.prices[option.underlying].value
    # Generate a unique key for the specific option contract
    option_key = "OPTION_$(option.strike)_$(option.is_call ? 'C' : 'P')"

    orders = Dict{Symbol, Any}[] # Initialize list to store potential orders for this agent
    
    # === Market Maker Logic ===
    if agent.role == :market_maker
        # 1. Quote Option Spread
        fair_value = Pricing.price_american_mc(option, state, config, rng, est_vol)
        spread_factor = config.mm_transaction_cost + (agent.risk_aversion * est_vol)
        spread = fair_value * spread_factor
        bid = fair_value - spread/2
        ask = fair_value + spread/2
        
        # Add option quotes to orders list
        order_size = 1.0 # Standard quote size
        append!(orders, [
            Dict(:type => :option, :side => :buy, :price => bid, :size => order_size, :agent_role => agent.role, :agent_id => agent.id, :option_key => option_key), 
            Dict(:type => :option, :side => :sell, :price => ask, :size => order_size, :agent_role => agent.role, :agent_id => agent.id, :option_key => option_key)
        ])

        # 2. Delta Hedge Option Inventory Risk
        current_opt_pos = get(agent.position, option_key, 0.0)
        if abs(current_opt_pos) > 1e-6 # Only hedge if holding options
            greeks = Pricing.calculate_option_greeks(option, state, config, rng, est_vol)
            delta = greeks[:delta]
            target_underlying_pos = -delta * current_opt_pos # Target stock pos to neutralize option delta
            current_underlying_pos = get(agent.position, option.underlying, 0.0)
            hedge_amount = target_underlying_pos - current_underlying_pos

            if abs(hedge_amount) > config.hedge_threshold
                hedge_side = hedge_amount > 0 ? :buy : :sell
                hedge_size = abs(hedge_amount)
                # Place stock hedge order
                push!(orders, Dict(:type => :stock, :side => hedge_side, :price => underlying_price, :size => hedge_size, :agent_role => agent.role, :agent_id => agent.id, :option_key => nothing))
            end
        end
    
    # === Portfolio Insurer Logic ===
    elseif agent.role == :insurer
        # Check stop-loss on underlying first
        if !ismissing(agent.stop_loss) && underlying_price < agent.stop_loss
            current_underlying_pos = get(agent.position, option.underlying, 0.0)
            if abs(current_underlying_pos) > 1e-6
                side = current_underlying_pos > 0 ? :sell : :buy
                size = abs(current_underlying_pos)
                push!(orders, Dict(:type => :stock, :side => side, :price => underlying_price, :size => size, :agent_role => agent.role, :agent_id => agent.id, :option_key => nothing))
            end
        else 
            # Hedge delta of existing option position
            current_opt_pos = get(agent.position, option_key, 0.0) # Assumes insurer holds the option
            if abs(current_opt_pos) > 1e-6 # Only hedge if holding options
                 greeks = Pricing.calculate_option_greeks(option, state, config, rng, est_vol)
                 delta = greeks[:delta]
                 target_underlying_pos = -delta * current_opt_pos # Target stock hedge
                 current_underlying_pos = get(agent.position, option.underlying, 0.0)
                 hedge_amount = target_underlying_pos - current_underlying_pos
                 
                 if abs(hedge_amount) > config.hedge_threshold
                    hedge_side = hedge_amount > 0 ? :buy : :sell
                    hedge_size = abs(hedge_amount)
                    push!(orders, Dict(:type => :stock, :side => hedge_side, :price => underlying_price, :size => hedge_size, :agent_role => agent.role, :agent_id => agent.id, :option_key => nothing))
                 end
            else
                 # Initial trade logic: If insurer doesn't hold the option, try to buy it.
                 # Calculate insurer's perceived fair value
                 insurer_fair_value = Pricing.price_american_mc(option, state, config, rng, est_vol)
                 # Place a limit buy order slightly above fair value
                 buy_price = insurer_fair_value * 1.01 # Pay a small premium to ensure purchase
                 buy_size = 1.0
                 push!(orders, Dict(:type => :option, :side => :buy, :price => buy_price, :size => buy_size, :agent_role => agent.role, :agent_id => agent.id, :option_key => option_key))
            end
        end

    # === Arbitrageur Logic ===
    elseif agent.role == :arbitrageur
        # Basic No-Arbitrage Bounds Check (Simplified for American Put)
        S = underlying_price
        K = option.strike
        T = option.maturity
        # Use risk_premium as proxy for risk-free rate 'r' for discounting
        r_proxy = config.risk_premium 
        discount_factor = exp(-r_proxy * T)
        
        # Calculate bounds for Put (assuming option is a Put)
        # TODO: Add logic to handle calls if necessary
        lower_bound = max(0.0, K * discount_factor - S)
        upper_bound = K * discount_factor # Tighter upper bound

        # Simulate MM1 quotes to represent market BBO
        # Needs MM1's vol estimate - use agent's own for simplicity here
        arb_fair_value = Pricing.price_american_mc(option, state, config, rng, est_vol)
        mm_spread_factor = config.mm_transaction_cost + (0.1 * est_vol) # MM1 risk aversion = 0.1 (fixed)
        mm_spread = arb_fair_value * mm_spread_factor
        mm_bid_proxy = arb_fair_value - mm_spread / 2
        mm_ask_proxy = arb_fair_value + mm_spread / 2

        # Scale order size based on capital relative to base (100k)
        base_capital = 100_000.0
        capital_scale = max(1.0, agent.capital / base_capital)
        arb_order_size = round(capital_scale) # Round to integer size

        # Check for arbitrage opportunities
        if mm_ask_proxy < lower_bound * 0.999 # Ask is significantly below lower bound -> BUY
            # Price is too low, buy the option
            push!(orders, Dict(:type => :option, :side => :buy, :price => mm_ask_proxy, :size => arb_order_size, :agent_role => agent.role, :agent_id => agent.id, :option_key => option_key))
            # TODO: Potentially hedge by buying underlying stock
        elseif mm_bid_proxy > upper_bound * 1.001 # Bid is significantly above upper bound -> SELL
            # Price is too high, sell the option
            push!(orders, Dict(:type => :option, :side => :sell, :price => mm_bid_proxy, :size => arb_order_size, :agent_role => agent.role, :agent_id => agent.id, :option_key => option_key))
            # TODO: Potentially hedge by selling underlying stock
        end
        # Otherwise, do nothing if price is within bounds

    # === Retail Logic ===
    elseif agent.role == :retail
        # 1. Option Trading Logic (Simplified)
        # Calculate retail's perceived fair value (e.g., naive BSM or MC with own vol)
        # For simplicity: use MM's fair value from previous step + noise
        # NOTE: This requires passing MM's value or order book state. 
        # Alternative: Use own MC pricing. Let's do that.
        retail_fair_value = Pricing.price_american_mc(option, state, config, rng, est_vol) 
        # Add some random noise/bias to retail valuation
        valuation_noise = retail_fair_value * 0.01 * randn(rng) # +/- 1% noise
        perceived_value = retail_fair_value + valuation_noise

        # SIMPLIFICATION: Assume access to MM's bid/ask (e.g., use MM logic internally)
        # In reality, need to see OrderBook. Let's recalculate MM spread for comparison.
        mm_spread_factor = config.mm_transaction_cost + (0.1 * est_vol) # Assume MM risk aversion = 0.1
        mm_spread = retail_fair_value * mm_spread_factor # Use retail's FV as proxy for MM's FV center
        mm_bid_proxy = retail_fair_value - mm_spread / 2
        mm_ask_proxy = retail_fair_value + mm_spread / 2

        # Scale order size based on capital relative to base (100k)
        base_capital = 100_000.0
        capital_scale = max(1.0, agent.capital / base_capital)
        option_order_size = round(capital_scale) # Round to integer size

        if perceived_value > mm_ask_proxy * 1.001 # Buy if perceived value > ask (with buffer)
            # Place BUY order for the OPTION
            push!(orders, Dict(:type => :option, :side => :buy, :price => mm_ask_proxy, :size => option_order_size, :agent_role => agent.role, :agent_id => agent.id, :option_key => option_key))
        elseif perceived_value < mm_bid_proxy * 0.999 # Sell if perceived value < bid (with buffer)
            # Place SELL order for the OPTION
            push!(orders, Dict(:type => :option, :side => :sell, :price => mm_bid_proxy, :size => option_order_size, :agent_role => agent.role, :agent_id => agent.id, :option_key => option_key))
        end

        # 2. Underlying Stock Trading Logic (Keep simple directional)
        entry_price = 100.0 # Static entry price for stock
        current_stock_pos = get(agent.position, option.underlying, 0.0)
        if current_stock_pos > 0 && underlying_price > entry_price * 1.05 # Sell stock on profit
            sell_size = min(current_stock_pos, 10.0)
            push!(orders, Dict(:type => :stock, :side => :sell, :price => underlying_price, :size => sell_size, :agent_role => agent.role, :agent_id => agent.id, :option_key => nothing))
        end 
    end 
    
    # Return list of orders (potentially empty) or :none if no orders
    if isempty(orders)
        return Dict(:action => :none)
    else
        return orders
    end
end

end # module

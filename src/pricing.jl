"""
Option pricing engines for Catallaxy Model.
Monte Carlo and Dynamic Programming for American options.
References: Brandimarte Ch8, Ch10; Glasserman.
"""

module Pricing
# Import the Types module to access its exported types
using ..Types: Price, OptionContract, MarketState, SimConfig

# Import only what's needed from standard libraries and packages
using Random: AbstractRNG, rand, MersenneTwister
using Distributions: Normal
using Statistics: mean, std
using GLM: lm, predict, @formula  # Import GLM for regression
using DataFrames: DataFrame # Needed for GLM

"""
price_american_mc(opt::OptionContract, state::MarketState, config::SimConfig, rng::AbstractRNG, estimated_volatility::Float64) -> Float64

Monte Carlo pricing for American options using Least Squares Monte Carlo (LSMC).
- Simulates price paths using GBM.
- Uses regression to estimate continuation value at each step.
- Determines optimal early exercise strategy.

References: Brandimarte Ch10; Longstaff & Schwartz (2001).
"""
function price_american_mc(opt::OptionContract, state::MarketState, config::SimConfig, rng::AbstractRNG, estimated_volatility::Float64) :: Float64
    # Extract parameters
    S0 = state.prices[opt.underlying].value
    K = opt.strike
    T = opt.maturity
    σ = estimated_volatility
    r = config.risk_premium # Using risk premium as the drift/discount rate
    npaths = config.npaths
    dt = 1.0/252
    nsteps = Int(round(T/dt))

    # Simulate stock price paths
    S_paths = zeros(nsteps + 1, npaths)
    S_paths[1, :] .= S0
    normal_dist = Normal(0, 1)
    for i in 1:npaths
        for t in 1:nsteps
            dW = sqrt(dt) * rand(rng, normal_dist)
            S_paths[t+1, i] = S_paths[t, i] * exp((r - 0.5*σ^2)*dt + σ*dW)
        end
    end

    # Initialize cash flows at maturity (undiscounted)
    if opt.is_call
        cash_flows = max.(0.0, S_paths[nsteps+1, :] .- K) # Size npaths
    else
        cash_flows = max.(0.0, K .- S_paths[nsteps+1, :]) # Size npaths
    end

    # Exercise decision array (holds the optimal cash flow value for each path at each time)
    exercise_policy = zeros(nsteps + 1, npaths)
    exercise_policy[nsteps + 1, :] = cash_flows # Initialize with maturity payoff

    # Backward induction loop (LSMC)
    for t in nsteps-1:-1:1
        # Expected future cash flow, discounted to time t
        expected_future_cf = exercise_policy[t+2, :] .* exp(-r * dt)

        # Identify ITM paths at time t (using prices S[t+1])
        current_prices = S_paths[t+1, :]
        if opt.is_call
            itm_indices = findall(current_prices .> K)
            immediate_exercise_value = max.(0.0, current_prices[itm_indices] .- K)
        else
            itm_indices = findall(current_prices .< K)
            immediate_exercise_value = max.(0.0, K .- current_prices[itm_indices])
        end

        # Default continuation value if no ITM paths or regression fails
        continuation_value_itm = zeros(length(itm_indices))

        if !isempty(itm_indices)
            # Regress discounted future payoffs onto basis functions of current price for ITM paths
            X_itm = current_prices[itm_indices]
            Y_itm = expected_future_cf[itm_indices] # Use discounted future CF for ITM paths as Y

            df_itm = DataFrame(Y = Y_itm, X1 = X_itm, X2 = X_itm.^2)
            try
                ols = lm(@formula(Y ~ X1 + X2), df_itm)
                continuation_value_itm = predict(ols) # Estimated continuation value for ITM paths
            catch e
                println("LSMC Regression Warning at step $t: $e")
                # Keep continuation_value_itm as 0 if regression fails (prudent exercise)
            end
        end

        # Make exercise decision for ITM paths
        exercise_now_mask = immediate_exercise_value .> continuation_value_itm # Boolean mask for ITM indices

        # Update the exercise policy for time t (index t+1)
        # Start with the continuation value (discounted future payoff) for ALL paths
        exercise_policy[t+1, :] = expected_future_cf
        # For ITM paths where exercise is optimal, overwrite with immediate exercise value
        exercise_policy[t+1, itm_indices[exercise_now_mask]] = immediate_exercise_value[exercise_now_mask]
    end # End backward loop

    # Price at time 0 is the expected value of the cash flow realized at time 1 (index 2), discounted back to time 0
    price = mean(exercise_policy[2, :] .* exp(-r*dt))

    return price
end

"""
calculate_option_greeks(opt::OptionContract, state::MarketState, config::SimConfig, rng::AbstractRNG, estimated_volatility::Float64) -> Dict{Symbol,Float64}

Calculates option Greeks using finite difference approximations based on LSMC pricing.
"""
function calculate_option_greeks(opt::OptionContract, state::MarketState, config::SimConfig, rng::AbstractRNG, estimated_volatility::Float64) :: Dict{Symbol,Float64}
    # Base price using estimated volatility (now uses LSMC)
    price = price_american_mc(opt, state, config, rng, estimated_volatility)
    
    # Parameters for finite differences
    S = state.prices[opt.underlying].value
    h_s = 0.01 * S  # 1% price bump
    h_v = 0.01      # 1% vol bump (applied to estimated_volatility)
    h_t = 1.0/252   # 1 day
    
    # Delta: dV/dS
    state_up = deepcopy(state)
    state_up.prices[opt.underlying] = Price(S + h_s)
    price_up = price_american_mc(opt, state_up, config, rng, estimated_volatility)
    
    state_down = deepcopy(state)
    state_down.prices[opt.underlying] = Price(max(0.001, S - h_s))
    price_down = price_american_mc(opt, state_down, config, rng, estimated_volatility)
    
    delta = (price_up - price_down) / (2 * h_s)
    
    # Gamma: d²V/dS²
    gamma = (price_up - 2*price + price_down) / (h_s^2)
    
    # Vega: dV/dσ (sensitivity to agent's estimated volatility)
    price_vega_up = price_american_mc(opt, state, config, rng, estimated_volatility + h_v)
    price_vega_down = price_american_mc(opt, state, config, rng, max(0.001, estimated_volatility - h_v))
    vega = (price_vega_up - price_vega_down) / (2 * h_v)
    
    # Theta: -dV/dt (create new instance instead of modifying)
    opt_theta = OptionContract(
        opt.underlying,
        opt.strike,
        opt.maturity - h_t,  # Reduced maturity
        opt.is_call,
        opt.is_american
    )
    price_theta = price_american_mc(opt_theta, state, config, rng, estimated_volatility)
    theta = -(price_theta - price) / h_t
    
    return Dict(:delta => delta, :gamma => gamma, :vega => vega, :theta => theta)
end

end # module

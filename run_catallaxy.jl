#!/usr/bin/env julia
"""
Catallaxy Model Runner
Demonstrates the decentralized Catallaxy Model with Entrepreneurship, Private Property,
and Monetary Prices driving Market Calculation and Coordinated Plans.

References: Figlewski (1989, 2017), Brandimarte (Ch8, Ch10).
"""

# Activate the project environment
using Pkg
Pkg.activate(".")

# Import only what's needed with explicit namespaces
using CatallaxyModel
using .Simulation: run_simulation
using .Types: SimConfig
using Plots
using Statistics
using Printf
using Random: MersenneTwister

# Set reproducible seed for the entire simulation
# const SEED = 42 # REMOVED - Let RNG seed itself for different runs

# Create configuration with behavioral flags
config = SimConfig(
    10000,                                  # Number of Monte Carlo paths
    rand(UInt),                             # RNG seed for reproducibility (now random)
    0.05,                                   # Risk premium (not risk-free rate)
    0.005,                                  # mm_transaction_cost: Increased for wider MM spread
    0.01,                                   # retail_transaction_cost: High for retail
    0.005,                                  # insurer_transaction_cost: Medium for insurers
    0.1,                                    # hedge_threshold: Minimum hedge adjustment size
    0.0001,                                 # slippage_factor: Controls price impact
    20,                                     # vol_lookback: Days for vol estimation window
    Dict(                                   # Behavioral flags
        :behavioral_exercise => true,       # Use behavioral exercise boundaries
        :disposition_effect => true,        # Retail traders exhibit disposition effect
        :inventory_control => true          # Market makers control inventory
    )
)

n_steps = 252 # Define number of steps (e.g., 1 year)

println("Running Catallaxy Model simulation...")

# --- Run Simulation ---
# Capture all returned histories
final_agents, stock_price_history, option_price_history, delta_history, spread_history, volume_history, agent_capital_history = run_simulation(config, n_steps)

# --- Extract Data for Analysis ---
agent_ids = [agent.id for agent in final_agents]
agent_roles = [agent.role for agent in final_agents]
# agent_capital_history is already a Dict {ID -> History}

# --- Calculate P&L History ---
agent_pnl_history = Dict{Int, Vector{Float64}}()
for agent_id in agent_ids
    initial_capital = agent_capital_history[agent_id][1] # First element is initial capital
    agent_pnl_history[agent_id] = [cap - initial_capital for cap in agent_capital_history[agent_id]]
end

# --- Summary Statistics ---
println("\nSummary Statistics:")
println("==================")
# Calculate mean price from step 1 onwards, excluding initial price if desired
mean_stock_price = mean(stock_price_history[2:end])
@printf("Mean Spot Price (Steps 1-%d): %.8f\n", n_steps, mean_stock_price)

# Add agent-specific summaries
println("\nAgent Final States:")
println("-------------------")
for i in 1:length(final_agents)
    agent_id = agent_ids[i]
    final_capital = agent_capital_history[agent_id][end]
    initial_capital = agent_capital_history[agent_id][1]
    accumulated_costs = final_agents[i].accumulated_costs # Get from agent struct
    final_pnl = final_capital - initial_capital
    @printf("Agent %d (%s):\n", agent_id, agent_roles[i])
    @printf("  Final Capital:      %.2f\n", final_capital)
    @printf("  Accumulated Costs:  %.2f\n", accumulated_costs)
    @printf("  Total P&L:          %.2f\n", final_pnl)
end

# --- Generate Plots ---
println("\nGenerating plots...")

# Define time steps for plotting
steps_state = 0:n_steps  # For stock price, agent capital/pnl (n_steps + 1 points)
steps_market = 1:n_steps # For option price, delta, spread, volume (n_steps points)

# 1. Spot and Option Prices (Option on secondary axis)
p_prices = plot(steps_state, stock_price_history, title="Spot & Option Prices", xlabel="Step", ylabel="Spot Price", label="Spot Price (LHS)", legend=:outertopright, color=:blue)
plot!(twinx(), steps_market, option_price_history, ylabel="Option Price", label="Option Price (RHS)", color=:red) # Use steps_market for option price

# 2. Option Delta
p_delta = plot(steps_market, delta_history, title="Option Delta (BS)", xlabel="Step", ylabel="Delta", label="Delta", legend=false, color=:green) # Use steps_market for delta

# 3. Bid-Ask Spread
# Filter out NaNs for plotting if necessary, or Plots.jl might handle them
p_spread = plot(steps_market, spread_history, title="Bid-Ask Spread", xlabel="Step", ylabel="Spread", label="Spread", color=:green)

# 4. Trade Volume
p_volume = plot(steps_market, volume_history, title="Trade Volume", xlabel="Step", ylabel="Volume", label="Volume", color=:purple, linetype=:steppre) # Use step plot for volume

# 6. Agent P&L
p_pnl = plot(title="Agent P&L", xlabel="Step", ylabel="P&L", legend=:outertopright)
for agent_id in agent_ids
    role = agent_roles[findfirst(id -> id == agent_id, agent_ids)]
    plot!(p_pnl, steps_state, agent_pnl_history[agent_id], label="Agent $agent_id ($role)")
end

# Combine plots
combined_plot = plot(p_prices, p_delta, p_spread, p_volume, p_pnl,
                     layout=(3, 2), size=(1200, 900), margin=5Plots.mm) # Layout remains 3x2, last slot empty

# Save plot
plot_filename = "catallaxy_agent_results.png"
savefig(combined_plot, plot_filename)
println("Plot saved to $(plot_filename)")

println("\nCatallaxy Model simulation complete.")

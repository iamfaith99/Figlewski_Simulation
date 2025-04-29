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
    0.001,                                  # mm_transaction_cost: Low for market makers
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
# Simulation now returns the list of agents and price history
final_agents, price_history = run_simulation(config, n_steps)

# --- Extract Data for Analysis ---
agent_ids = [agent.id for agent in final_agents]
agent_roles = [agent.role for agent in final_agents]
final_capitals = [agent.capital for agent in final_agents]
initial_capitals = [agent.initial_capital for agent in final_agents]
accumulated_costs = [agent.accumulated_costs for agent in final_agents]
portfolio_histories = [agent.portfolio_history for agent in final_agents]

# Calculate P&L
final_pnl = [hist[end] - hist[1] for hist in portfolio_histories]

# --- Summary Statistics ---
println("\nSummary Statistics:")
println("==================")
@printf("Mean Spot Price: %.8f\n", mean(price_history))
# Add agent-specific summaries
println("\nAgent Final States:")
println("-------------------")
for i in 1:length(final_agents)
    @printf("Agent %d (%s):\n", agent_ids[i], agent_roles[i])
    @printf("  Final Capital:      %.2f\n", final_capitals[i])
    @printf("  Accumulated Costs:  %.2f\n", accumulated_costs[i])
    @printf("  Total P&L:          %.2f\n", final_pnl[i])
end

# --- Generate Plots ---
println("\nGenerating plots...")

# 1. Spot Price History
p1 = plot(1:length(price_history), price_history, title="Spot Price Evolution", xlabel="Step", ylabel="Price", legend=false)

# 2. Agent Capital History
p2 = plot(title="Agent Capital Over Time", xlabel="Step", ylabel="Capital")
for i in 1:length(final_agents)
    # Need to extract capital history (requires modification to simulation.jl or calculate from portfolio history)
    # For now, let's plot portfolio value as a proxy
    plot!(p2, 1:length(portfolio_histories[i]), portfolio_histories[i], label="Agent $(agent_ids[i]) ($(agent_roles[i]))")
end

# Combine plots
combined_plot = plot(p1, p2, layout=(2, 1), size=(800, 600))

# Save plot
plot_filename = "catallaxy_agent_results.png"
savefig(combined_plot, plot_filename)
println("Plot saved to $(plot_filename)")

println("\nCatallaxy Model simulation complete.")

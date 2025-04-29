module CatallaxyModel
"""
Main entry point for the Catallaxy Model prototype.
Includes all submodules: types, agents, market, pricing, simulation.

Implements a decentralized Catallaxy Model characterized by Entrepreneurship,
Private Property, and Monetary Prices driving Market Calculation and Coordinated Plans.
References: Figlewski (1989, 2017), Brandimarte (Ch8, Ch10).
"""

# Include submodules in correct dependency order
include("types.jl")       # Core types and data structures first
include("pricing.jl")     # Pricing engines (needed by agents, simulation)
include("agents.jl")      # Agent behaviors (needs types, pricing)
include("market.jl")      # Market mechanics (needs types)
include("simulation.jl")  # Simulation orchestration (needs all)

# Export submodules for external use
export Types, Agents, Market, Pricing, Simulation

end # module

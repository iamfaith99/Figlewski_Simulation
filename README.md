# Catallaxy Model Prototype (Julia)

A decentralized, behavioral option market model integrating:
- Entrepreneurship, private property, monetary prices (Catallaxy)
- Multiple interacting agent types with distinct behavioral logic:
  - Market Makers (competing)
  - Portfolio Insurer (delta-hedging, initial purchase)
  - Retail Traders (heterogeneous)
  - Arbitrageur (basic bounds checking)
- Empirical market dynamics (Figlewski 1989, 2017)
- Monte Carlo option pricing (Brandimarte Ch8, Ch10; Glasserman)
- Explicit market frictions and transaction costs

## Key Features & Findings
- Simulates price discovery through decentralized order matching.
- Models agent heterogeneity in roles, capital, and risk aversion.
- Demonstrates the impact of competing market makers on liquidity.
- Highlights **limits to arbitrage**, showing how even simple arbitrage strategies may remain inactive due to market structure and frictions, consistent with real-world observations.

## Architecture
- `/src/types.jl`: Core types (SimConfig, MarketState, Agent, OptionContract, OrderBook, Entrepreneur)
- `/src/agents.jl`: Behavioral agent logic (Market Maker, Insurer, Retail, Arbitrageur)
- `/src/market.jl`: Market/order book matching logic
- `/src/pricing.jl`: Option pricing engines (Monte Carlo)
- `/src/simulation.jl`: Simulation loop orchestration and agent initialization
- `/run_catallaxy.jl`: Main script to configure and run the simulation.

## References
- Figlewski, S. (1989, 2017)
- Brandimarte, P. (Ch8, Ch10)
- Glasserman, P. (Monte Carlo Methods in Financial Engineering)

## Usage
1. Ensure Julia is installed.
2. Navigate to the project directory.
3. Activate the environment: `using Pkg; Pkg.activate(".")`
4. Instantiate dependencies (if first time): `Pkg.instantiate()`
5. Run the simulation: `julia run_catallaxy.jl`
6. View results plot: `catallaxy_agent_results.png`

---
This project sets a new standard for behavioral realism and strategic adaptation in option market modeling for 2025 and beyond.

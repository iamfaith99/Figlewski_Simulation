"""
Core types for Catallaxy Model.
References: Figlewski (1989, 2017), Brandimarte (Ch8, Ch10), Glasserman.
"""

module Types

# Import only what's needed for this module
using Random: AbstractRNG

# Export types for use in other modules
export Price, Position, MaybeFloat, SimConfig, Agent, MarketState, OptionContract, OrderBook, Entrepreneur

# --- Strong types for prices, positions, and risk metrics ---
struct Price{T<:Real} <: Number
    value::T # Enforces nonnegative at construction
    function Price(val::T) where T<:Real
        if val < 0
            throw(ArgumentError("Negative price not allowed"))
        end
        new{T}(val)
    end
end

struct Position{T<:Real}
    asset::String
    quantity::T
end

# --- Explicit missing data representation ---
const MaybeFloat = Union{Float64,Missing}

# --- Simulation configuration ---
struct SimConfig
    npaths::Int                             # Number of MC paths for pricing
    seed::Int                               # Master RNG seed
    risk_premium::Float64                 # Risk premium used in MC drift
    mm_transaction_cost::Float64          # Market maker transaction cost
    retail_transaction_cost::Float64      # Retail trader transaction cost
    insurer_transaction_cost::Float64     # Insurer transaction cost
    hedge_threshold::Float64              # Minimum hedge adjustment size for insurer
    slippage_factor::Float64              # Parameter controlling price impact of trades
    vol_lookback::Int                     # Lookback window (days) for vol estimation
    behavioral_flags::Dict{Symbol, Bool}  # Flags for behavioral features
end

abstract type Agent end  # Any market participant

struct MarketState
    prices::Dict{String, Price{Float64}}          # Asset prices (symbol => price)
    volatility::Dict{String, Float64}      # Asset volatilities
    liquidity::Dict{String, Float64}       # Available depth/volume at BBO
    # transaction_costs::Dict{String, Float64} # Per-asset cost (Removed - now in SimConfig)
    time::Float64                         # Current simulation time
end

struct OptionContract
    underlying::String       # Symbol
    strike::Float64          # Strike price
    maturity::Float64        # Time to maturity
    is_call::Bool            # Call or put
    is_american::Bool        # American or European
end

struct OrderBook
    bids::Vector{Tuple{Price{Float64}, Float64}} # (price, size)
    asks::Vector{Tuple{Price{Float64}, Float64}} # (price, size)
    depth::Int                            # Number of levels
end

mutable struct Entrepreneur <: Agent
    id::Int
    initial_capital::Float64             # Starting capital for P&L calc
    capital::Float64                     # Current liquid capital
    risk_aversion::Float64
    role::Symbol                           # :market_maker, :retail, :insurer, etc.
    position::Dict{String, Float64}        # Current holdings (asset => quantity)
    estimated_volatility::Dict{String, Float64} # Agent's belief about volatility
    price_history::Dict{String, Vector{Float64}} # History of observed prices for vol est.
    accumulated_costs::Float64           # Sum of transaction costs paid
    portfolio_history::Vector{Float64}     # History of total portfolio value (capital + MTM)
    stop_loss::MaybeFloat          # Price level to trigger liquidation
    margin_limit::MaybeFloat               # Max leverage or position size constraint
    rng_seed::Int                          # Per-agent RNG seed for reproducibility
    # Internal state (optional, e.g., for learning models)
end

end # module

# Import projects
import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

# References I am using:
# https://www.youtube.com/watch?v=58jVYGnsE60


# Based on the newsvendor supply chain problem
# You stock to little, you miss your sales (i.e shortage cost or (Cu) per unit.
# You stock to much you overstock have to give discounts or hold overage costs per unit
# The critical fractile tells you the probability that you should cover demand up to



def critical_fractile(shortage_cost: float, overage_cost: float) -> float:
    """Return the newsvendor critical fractile Cu/(Cu+Co); clipped to (0.0001, 0.9999)."""
    denom = shortage_cost + overage_cost # This sets the denominator value
    if denom <= 0:
        return 0.5 # A neutral guess for outliers, to keep the model working correctly
    frac = shortage_cost / denom # builds the critical fractile formula Cu / (Cu + Co)
    return float(np.clip(frac, 1e-4, 1 - 1e-4)) # Makes sure the fraction never equals exactly 0 or 1


# This is computing the optimal order quantity under a normal demand distribution
# mus is the mean for demand
# critical_fractile → target service level (probability to cover demand).
# optimal_Q → translates that service level into an actual order quantity.
def optimal_Q(mu: float, sigma: float, cf: float) -> float:
    """Return optimal order quantity for Normal demand: Q* = mu + sigma * Phi^{-1}(cf)."""
    z = norm.ppf(cf)
    return float(mu + sigma * z)

def apply_inflation(value: float, annual_rate_pct: float, months: int) -> float:
    """Compound inflate a value over N months at an annual rate (percent)."""
    r = annual_rate_pct / 100.0         # Converts the annual percentage rate into a decimal
    factor = (1 + r) ** (months / 12.0)   # Compunds this inflation monthly
    return value * factor   # Gives the inflation value.

def simulate_once(
    demand_mu, demand_sigma, purchase_cost, sell_price, salvage_value,
    shortage_penalty, manual_Q, lead_time_days, extra_lead_time_days,
    use_optimal_Q, optimal_Q_value, demand_spike_mult,
    cost_jump_mult, price_inflation_months, annual_inflation_rate_pct,
    rng: np.random.Generator
) -> dict:
    """
    Simulate one period outcome, returning dict with demand, order_qty, sales, leftover, lost_sales,
    total_cost, revenue, penalty_cost, salvage_revenue, profit, plus debug fields:
    arrival_fraction, effective_lead_time, late_qty.
    """
    # Inflation applied to price and costs over the horizon (months)
    eff_purchase_cost = apply_inflation(purchase_cost, annual_inflation_rate_pct, price_inflation_months)
    eff_sell_price    = apply_inflation(sell_price,    annual_inflation_rate_pct, price_inflation_months)
    eff_salvage_value = apply_inflation(salvage_value, annual_inflation_rate_pct, price_inflation_months)

    # Apply random shocks (already sampled outside): demand_spike_mult, extra lead time, cost jump
    eff_purchase_cost *= cost_jump_mult

    # Choose order policy (non-negative)
    order_qty = max(0.0, optimal_Q_value if use_optimal_Q else manual_Q)

    # Lead time effect: if extra lead time pushes receipt beyond period, only a fraction arrives
    period_days = 30.0
    effective_lead_time = max(0.0, lead_time_days + extra_lead_time_days)
    late_days = max(0.0, effective_lead_time - period_days)
    arrival_fraction = max(0.0, 1.0 - late_days / period_days)
    effective_supply = order_qty * arrival_fraction
    late_qty = max(0.0, order_qty - effective_supply)  # paid for, not usable this period

    # Realized demand (Normal) with mean spike applied
    demand = max(0.0, rng.normal(demand_mu * demand_spike_mult, demand_sigma))

    # Material flows
    sales = min(effective_supply, demand)
    leftover = max(0.0, effective_supply - sales)
    lost_sales = max(0.0, demand - sales)

    # P&L
    revenue = sales * eff_sell_price
    purchase_total = order_qty * eff_purchase_cost  # pay for the full order regardless of arrival timing
    salvage_rev = leftover * eff_salvage_value
    penalty_cost = lost_sales * shortage_penalty

    profit = revenue + salvage_rev - purchase_total - penalty_cost

    return {
        "demand": demand,
        "order_qty": order_qty,
        "effective_supply": effective_supply,
        "sales": sales,
        "leftover": leftover,
        "lost_sales": lost_sales,
        "revenue": revenue,
        "purchase_cost": purchase_total,
        "salvage_revenue": salvage_rev,
        "penalty_cost": penalty_cost,
        "profit": profit,

        # --- Debug fields ---
        "arrival_fraction": arrival_fraction,
        "effective_lead_time": effective_lead_time,
        "late_qty": late_qty,
    }


if __name__ == "__main__":
    from numpy.random import default_rng
    rng = default_rng(42)

    # Example test
    result = simulate_once(
        demand_mu=1200, demand_sigma=250,
        purchase_cost=30, sell_price=50, salvage_value=10,
        shortage_penalty=5,
        manual_Q=1200, lead_time_days=20, extra_lead_time_days=0,
        use_optimal_Q=True, optimal_Q_value=1250,
        demand_spike_mult=1.0, cost_jump_mult=1.0,
        price_inflation_months=3, annual_inflation_rate_pct=3,
        rng=rng
    )
    print(result)




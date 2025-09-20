import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from core_simulator import critical_fractile, optimal_Q, apply_inflation, simulate_once

st.set_page_config(page_title="Demand / Profit Simulator", layout="wide")

st.title("Demand / Profit Simulator")

# ---- Session state to hold last results ----
if "results" not in st.session_state:
    st.session_state.results = None

# ================= SIDEBAR (all controls) =================
with st.sidebar:
    st.header("Demand & Economics")
    demand_mu = st.number_input("Average demand (units)", 1.0, 1e9, 1200.0, 10.0)
    sell_price = st.number_input("Sell price ($/unit)", 0.0, 1e9, 50.0, 1.0)
    purchase_cost = st.number_input("Purchase cost ($/unit)", 0.0, 1e9, 30.0, 1.0)
    salvage_value = st.number_input("Salvage value ($/unit)", 0.0, 1e9, 10.0, 1.0)
    shortage_penalty = st.number_input("Shortage penalty ($/unit)", 0.0, 1e9, 5.0, 1.0)

    st.header("Logistics & Price Dynamics")
    lead_time_days = st.number_input("Lead time (days)", 0.0, 365.0, 20.0, 1.0)
    annual_infl = st.slider("Annual inflation (%)", 0.0, 25.0, 3.0, 0.5)
    months_until_sale = st.slider("Months until sale", 0, 24, 3)

    st.header("Order Policy")
    policy = st.radio("Choose policy", ["Optimal Q* (economic)", "Manual Q"], index=0)
    manual_Q = st.number_input("Manual Q (if chosen)", 0.0, 1e9, 1200.0, 10.0)

    st.header("Monte Carlo")
    demand_sigma = st.number_input(
        "Demand std dev (units)",
        0.0, 1e9, 250.0, 10.0,
        help="Used for both Optimal Q and Monte Carlo"
    )
    n_sims = st.slider("Number of simulations", 500, 50000, 5000, 500)
    prob_demand_spike = st.slider("Demand spike probability", 0.0, 1.0, 0.10, 0.01)
    demand_spike_multiplier = st.slider("Demand spike multiplier (×)", 1.0, 3.0, 1.5, 0.1)
    prob_supplier_delay = st.slider("Supplier delay probability", 0.0, 1.0, 0.10, 0.01)
    extra_delay_days = st.slider("Extra delay if occurs (days)", 0, 60, 15)
    prob_cost_jump = st.slider("Unit cost jump probability", 0.0, 1.0, 0.05, 0.01)
    cost_jump_multiplier = st.slider("Cost jump multiplier (×)", 1.0, 2.0, 1.2, 0.05)

# ================= Helper: run one full simulation pass =================
def run_simulation():
    # Economic fractile & chosen Q (NOW uses demand_sigma)
    underage = max(0.0, (sell_price - purchase_cost)) + shortage_penalty
    overage = max(0.0, purchase_cost - salvage_value)
    cf_econ = critical_fractile(underage, overage)

    # >>> change here: use sigma in optimal_Q <<<
    Q_econ = optimal_Q(demand_mu, demand_sigma, cf_econ)
    Q = Q_econ if policy == "Optimal Q* (economic)" else manual_Q

    # Monte Carlo
    rng = np.random.default_rng(42)   # fixed seed for reproducibility; remove/parameterize if desired
    sim_count = int(n_sims)

    shock_demand = rng.random(sim_count) < prob_demand_spike
    shock_delay  = rng.random(sim_count) < prob_supplier_delay
    shock_cost   = rng.random(sim_count) < prob_cost_jump

    order_qty = np.full(sim_count, max(0.0, Q))
    period_days = 30.0
    extra_delay = np.where(shock_delay, float(extra_delay_days), 0.0)
    arrival_frac = 1.0 - np.maximum(0.0, (lead_time_days + extra_delay - period_days)) / period_days
    arrival_frac = np.clip(arrival_frac, 0.0, 1.0)

    mult = np.where(shock_demand, demand_spike_multiplier, 1.0)
    demand = rng.normal(demand_mu * mult, demand_sigma)
    demand = np.maximum(0.0, demand)

    effective_supply = order_qty * arrival_frac

    eff_c = apply_inflation(purchase_cost, annual_infl, months_until_sale)
    eff_p = apply_inflation(sell_price,     annual_infl, months_until_sale)
    eff_s = apply_inflation(salvage_value,  annual_infl, months_until_sale)
    purchase_cost_unit = np.where(shock_cost, eff_c * cost_jump_multiplier, eff_c)

    sales = np.minimum(effective_supply, demand)
    leftover = np.maximum(0.0, effective_supply - sales)
    lost = np.maximum(0.0, demand - sales)

    revenue = sales * eff_p
    purchase_total = order_qty * purchase_cost_unit
    salvage_rev = leftover * eff_s
    penalty = lost * shortage_penalty
    profit = revenue + salvage_rev - purchase_total - penalty

    # KPIs
    mean_profit = float(np.mean(profit))
    median_profit = float(np.median(profit))
    p5 = float(np.percentile(profit, 5))
    p95 = float(np.percentile(profit, 95))
    fill_rate = float(np.mean(sales / np.maximum(1e-9, demand)))
    stockout_prob = float(np.mean(lost > 0))

    # Deterministic snapshot (averages only)
    det = simulate_once(
        demand_mu=demand_mu, demand_sigma=0.0,
        purchase_cost=purchase_cost, sell_price=sell_price, salvage_value=salvage_value,
        shortage_penalty=shortage_penalty, manual_Q=Q, lead_time_days=lead_time_days,
        extra_lead_time_days=0, use_optimal_Q=False, optimal_Q_value=Q,
        demand_spike_mult=1.0, cost_jump_mult=1.0,
        price_inflation_months=months_until_sale, annual_inflation_rate_pct=annual_infl,
        rng=np.random.default_rng(123)
    )

    return {
        "cf_econ": cf_econ,
        "Q_econ": Q_econ,
        "Q": Q,
        "eff_p_det": apply_inflation(sell_price, annual_infl, months_until_sale),
        "eff_c_det": apply_inflation(purchase_cost, annual_infl, months_until_sale),
        "eff_s_det": apply_inflation(salvage_value, annual_infl, months_until_sale),
        "det": det,
        "profit": profit,
        "sales": sales,
        "demand": demand,
        "arrival_frac": arrival_frac,
        "order_qty": order_qty,
        "purchase_cost_unit": purchase_cost_unit,
        "kpis": {
            "mean_profit": mean_profit,
            "median_profit": median_profit,
            "p5": p5,
            "p95": p95,
            "fill_rate": fill_rate,
            "stockout_prob": stockout_prob,
        },
    }

# ================= Run button =================
run_clicked = st.button("Run simulation", type="primary")
if run_clicked:
    st.session_state.results = run_simulation()

# ================= Display =================
if st.session_state.results is None:
    st.info("Adjust inputs in the sidebar, then click **Run simulation**.")
else:
    r = st.session_state.results

    # KPIs at top
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Mean profit", f"${r['kpis']['mean_profit']:,.0f}")
    k2.metric("Median profit", f"${r['kpis']['median_profit']:,.0f}")
    k3.metric("5th percentile", f"${r['kpis']['p5']:,.0f}")
    k4.metric("95th percentile", f"${r['kpis']['p95']:,.0f}")
    k5.metric("Fill rate", f"{r['kpis']['fill_rate']:.2%}")
    k6.metric("Stockout probability", f"{r['kpis']['stockout_prob']:.2%}")

    # Histogram (matplotlib; currency x-axis, labeled, larger fonts; dark-friendly)
    st.subheader("Profit distribution (Monte Carlo)")

    profits = np.asarray(r["profit"], dtype=float)
    if profits.size == 0:
        st.info("No profit samples yet. Click Run simulation.")
    else:
        bins = 30
        hist, edges = np.histogram(profits, bins=bins)

        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = (edges[1:] - edges[:-1]) * 0.98  # slight gap

        # Dark-friendly colors (works regardless of Streamlit theme)
        BG = "#0f1117"
        FG = "#e5e7eb"
        BAR = "#ffa07a"
        OUT = "#9e6149"

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        ax.bar(centers, hist, width=widths, align="center", color=BAR, edgecolor=OUT)

        ax.set_title("Profit Distribution (Monte Carlo)", fontsize=14, color=FG)
        ax.set_xlabel("Profit ($)", fontsize=13, color=FG)
        ax.set_ylabel("Count", fontsize=13, color=FG)

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x:,.2f}"))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.tick_params(axis="both", labelsize=11, colors=FG)

        for spine in ax.spines.values():
            spine.set_color(FG)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, color=FG, alpha=0.25)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Deterministic snapshots underneath
    st.subheader("Deterministic snapshots")
    c1, c2, c3 = st.columns(3)
    c1.metric("Economic fractile", f"{r['cf_econ']:.3f}")
    c2.metric("Q* economic", f"{r['Q_econ']:,.0f}")
    c3.metric("Chosen Q", f"{r['Q']:,.0f}")

    d1, d2, d3 = st.columns(3)
    d1.metric("Price (adj.)", f"${r['eff_p_det']:,.2f}")
    d2.metric("Cost (adj.)", f"${r['eff_c_det']:,.2f}")
    d3.metric("Salvage (adj.)", f"${r['eff_s_det']:,.2f}")
# Commiting my code
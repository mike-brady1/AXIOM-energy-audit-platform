"""AXIOM Benchmarking Engine — scores buildings against ADEME sector data."""

CARBON_FACTORS = {
    "Electricite": 0.052, "Gaz": 0.227, "Fioul": 0.324,
    "Reseau de chaleur": 0.110, "Reseau de froid": 0.025,
    "Bois": 0.030, "Gazole non routier": 0.324, "ND": 0.150,
}
ENERGY_PRICE = {
    "Electricite": 0.20, "Gaz": 0.08, "Fioul": 0.12,
    "Reseau de chaleur": 0.09, "Reseau de froid": 0.06,
    "Bois": 0.04, "Gazole non routier": 0.12, "ND": 0.10,
}

def benchmark_building(benchmark_df, activity, floor_area_m2,
                        annual_consumption_kwh, energy_carrier="Electricite"):
    match = benchmark_df[benchmark_df["activity"].str.contains(activity, case=False, na=False)]
    if match.empty:
        return {"error": f"Activity not found: {activity}"}
    row = match.iloc[0]
    actual_eui = annual_consumption_kwh / floor_area_m2
    if actual_eui <= row["eui_best"]:   tier = "Best Practice (Top 10%)"
    elif actual_eui <= row["eui_p25"]:  tier = "Good (Top 25%)"
    elif actual_eui <= row["eui_median"]: tier = "Average (Top 50%)"
    elif actual_eui <= row["eui_p75"]:  tier = "Below Average"
    else:                               tier = "Poor (Bottom 25%)"
    gap_eui     = max(0, actual_eui - row["eui_best"])
    savings_kwh = gap_eui * floor_area_m2
    cf          = CARBON_FACTORS.get(energy_carrier, 0.15)
    price       = ENERGY_PRICE.get(energy_carrier, 0.15)
    return {
        "activity": row["activity"], "floor_area_m2": floor_area_m2,
        "actual_eui": round(actual_eui, 1),
        "benchmark_best": row["eui_best"], "benchmark_median": row["eui_median"],
        "benchmark_p75": row["eui_p75"], "performance_tier": tier,
        "gap_to_best_eui": round(gap_eui, 1),
        "savings_potential_kwh": round(savings_kwh),
        "savings_potential_co2_kgCO2e": round(savings_kwh * cf),
        "savings_potential_eur_year": round(savings_kwh * price),
        "energy_carrier": energy_carrier,
    }

def identify_ecms(benchmark_result):
    actual_eui = benchmark_result["actual_eui"]
    median_eui = benchmark_result["benchmark_median"]
    area       = benchmark_result["floor_area_m2"]
    carrier    = benchmark_result["energy_carrier"]
    price      = ENERGY_PRICE.get(carrier, 0.15)
    cf         = CARBON_FACTORS.get(carrier, 0.15)
    ecms = []
    def add(name, share, saving_pct, cost_per_m2, priority):
        kwh  = actual_eui * area * share * saving_pct
        eur  = kwh * price
        cost = area * cost_per_m2
        ecms.append({"ecm": name,
                     "saving_kwh_year": round(kwh),
                     "saving_eur_year": round(eur),
                     "saving_co2_kgCO2e_year": round(kwh * cf),
                     "capex_eur": round(cost),
                     "payback_years": round(cost / eur, 1),
                     "priority": priority})
    add("LED Lighting Retrofit + Occupancy Controls", 0.20, 0.60, 15, "High")
    add("BMS Optimisation (Setpoints + Scheduling + VFDs)", 1.00, 0.15, 8, "High")
    if actual_eui > median_eui:
        add("HVAC Heat Pump Upgrade", 1.00, 0.25, 80, "Medium")
    if actual_eui > median_eui * 1.3:
        add("Building Envelope (Glazing + Insulation Upgrade)", 1.00, 0.10, 120, "Low")
    ecms.sort(key=lambda x: x["payback_years"])
    return ecms

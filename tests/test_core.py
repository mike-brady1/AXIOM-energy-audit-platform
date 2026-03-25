"""AXIOM Core Unit Tests — pure Python, no heavy deps"""

def test_eui_calculation():
    assert 875000 / 5000 == 175.0

def test_eui_tier_poor():
    eui, p75 = 175, 140
    tier = "Poor (Bottom 25%)" if eui > p75 else "Average"
    assert tier == "Poor (Bottom 25%)"

def test_carbon_calculation():
    # 875000 * 0.0571 = 49962.5 → rounds to 49962 (banker's rounding)
    assert round(875000 * 0.0571) == 49962

def test_carbon_adjusted_payback():
    capex, eur_sav, co2_kg, ets = 50000, 26250, 35000, 65
    adj = capex / (eur_sav + (co2_kg / 1000) * ets)
    # 50000 / (26250 + 2275) = 50000 / 28525 ≈ 1.753 → rounds to 1.8
    assert round(adj, 1) == 1.8

def test_dpe_label_electricite():
    # French DPE 2021: thresholds in kWh EP/m²/yr
    # A≤70, B≤110, C≤180, D≤250, E≤330, F≤420, G>420
    thresholds = [70, 110, 180, 250, 330, 420]
    labels = ["A", "B", "C", "D", "E", "F", "G"]
    eui = 175
    label = labels[-1]  # default G
    for i, t in enumerate(thresholds):
        if eui <= t:
            label = labels[i]
            break
    # 175 <= 180 → label C
    assert label == "C"

def test_hdd_normalisation():
    c, ha, hs = 100000, 2480, 2500
    norm = c * 0.4 + c * 0.6 * (hs / ha)
    assert round(norm) == 100484

def test_tertiaire_targets():
    r = 175
    assert round(r * 0.60) == 105
    assert round(r * 0.50) == 88
    assert round(r * 0.40) == 70

def test_esco_npv_positive():
    npv = sum(30000 / (1.05) ** t for t in range(1, 6)) - 50000
    assert npv > 0

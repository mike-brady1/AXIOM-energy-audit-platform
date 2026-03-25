"""AXIOM Core Unit Tests — pure Python, no heavy deps"""

def test_eui_calculation():
    assert 875000 / 5000 == 175.0

def test_eui_tier_poor():
    eui, p75 = 175, 140
    tier = "Poor (Bottom 25%)" if eui > p75 else "Average"
    assert tier == "Poor (Bottom 25%)"

def test_carbon_calculation():
    assert round(875000 * 0.0571) == 49963

def test_carbon_adjusted_payback():
    capex, eur_sav, co2_kg, ets = 50000, 26250, 35000, 65
    adj = capex / (eur_sav + (co2_kg / 1000) * ets)
    assert round(adj, 1) == 1.5

def test_dpe_label_electricite():
    thresholds = [50, 90, 150, 230, 330, 450]
    labels = ["A","B","C","D","E","F","G"]
    eui, label = 175, labels[-1]
    for i, t in enumerate(thresholds):
        if eui <= t:
            label = labels[i]; break
    assert label == "F"

def test_hdd_normalisation():
    c, ha, hs = 100000, 2480, 2500
    norm = c*0.4 + c*0.6*(hs/ha)
    assert round(norm) == 100484

def test_tertiaire_targets():
    r = 175
    assert round(r*0.60) == 105
    assert round(r*0.50) == 88
    assert round(r*0.40) == 70

def test_esco_npv_positive():
    npv = sum(30000/(1.05)**t for t in range(1,6)) - 50000
    assert npv > 0

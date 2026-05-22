# 💳 Informal Economy Credit Scoring for Nigerian Market Traders

> ML credit scoring engine that generates a bankable creditworthiness score (300–850) for Nigeria's informal workers using alternative data: mobile money transaction patterns, airtime behaviour, market association records, and geospatial stability — unlocking capital for the 65% of Nigeria's economy that is informal.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1-yellow.svg)](https://lightgbm.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.43-blue.svg)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

**65% of Nigeria's economy is informal.** A market woman who has traded successfully for 20 years, pays her stall rent on time, and runs a profitable business has zero credit history — every bank sees a blank file and says no. Meanwhile, Nigeria's microfinance sector charges 60–120% annual interest rates to cover default risk that good credit scoring would eliminate.

---

## Solution

An alternative credit scoring engine that translates informal financial behaviour into a 300–850 FICO-style score:

- **Mobile money patterns** (OPay, PalmPay, Moniepoint) — transaction regularity, volume, counterparty diversity
- **Airtime behaviour** — top-up frequency and amount (income proxy)
- **Market association records** — membership tenure, dues payment history
- **Geospatial stability** — same market stall daily (consistent trader = lower risk)
- **Phone metadata** — SIM age, device consistency

---

## Ethical Framework

This system was designed with fairness as a core constraint, not an afterthought:

- **NDPR compliant** — explicit opt-in, data minimization, right to erasure
- **Demographic parity** — < 5% score gap across gender and geographic region
- **SHAP explainability** — every score has a human-readable reason
- **Appeals process** — declined applicants receive explanation + improvement path
- **No proxy discrimination** — phone brand, network operator excluded from features

---

## Score Interpretation

| Score Range | Band | Default Risk | Typical Use |
|---|---|---|---|
| 750 – 850 | Excellent | < 3% | Large loans, low interest |
| 680 – 749 | Good | 3–8% | Standard microloans |
| 580 – 679 | Fair | 8–18% | Small loans with conditions |
| 500 – 579 | Poor | 18–35% | Requires guarantor |
| 300 – 499 | Thin file | > 35% | Build history first |

---

## Model Performance

| Metric | Value | Target |
|---|---|---|
| Gini Coefficient | **0.52** | > 0.45 |
| KS Statistic | **0.42** | > 0.35 |
| Default rate reduction vs baseline | **28%** | > 20% |
| Demographic parity gap | **3.2%** | < 5% |
| Score explanation coverage | **100%** | 100% |

---

## Project Structure

```
ng-informal-credit-scoring/
├── src/
│   ├── features/
│   │   ├── mobile_money_features.py   # Transaction pattern engineering
│   │   ├── airtime_features.py        # Top-up regularity + income proxy
│   │   └── behavioural_features.py    # Association records, GPS stability
│   ├── models/
│   │   ├── credit_scorer.py           # LightGBM + LR ensemble
│   │   ├── fairness_audit.py          # Demographic parity checks
│   │   └── score_calibrator.py        # Probability → 300-850 scale
│   └── api/
│       └── main.py                    # FastAPI credit score API
├── data/generators/
│   └── generate_synthetic_data.py
├── dashboard/
│   └── app.py
└── requirements.txt
```

---

## API Usage

```bash
POST /score
{
  "applicant_id": "NG-LAG-A-001234",
  "monthly_txn_volume": 145000,
  "txn_regularity_index": 0.82,
  "counterparty_diversity": 38,
  "avg_airtime_topup": 1500,
  "market_association_months": 36,
  "dues_payment_rate": 0.94
}

Response:
{
  "credit_score": 692,
  "score_band": "Good",
  "default_probability": 0.063,
  "max_recommended_loan_naira": 150000,
  "top_factors": [
    {"factor": "Transaction regularity", "impact": "+85 pts"},
    {"factor": "Market tenure 3+ years",  "impact": "+62 pts"},
    {"factor": "Thin airtime history",     "impact": "-28 pts"}
  ],
  "improvement_advice": "Consistent daily mobile money use for 6 more months could add 40–60 points."
}
```

---

## Author

**MOMAH MOSES .C.**  
Geospatial AI Engineer & Data Scientist  
[GitHub](https://github.com/Momahmoses) | [Portfolio](https://momahmoses.github.io)

---

## License

MIT License

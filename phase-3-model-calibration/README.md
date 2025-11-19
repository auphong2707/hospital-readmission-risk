# Phase 3: Model Calibration

## Purpose
Ensure predicted probabilities accurately reflect actual readmission risk for reliable clinical decision-making.

## Methodology

### Calibration Techniques
1. **Platt Scaling**: Logistic regression transformation of raw predictions
2. **Isotonic Regression**: Non-parametric piecewise-constant calibration
3. **Group-Specific Calibration**: Separate calibration by demographics (age, diagnosis)

### Validation Methods
- **Reliability Diagrams**: Plot predicted vs. observed probabilities
- **Brier Score**: Measure calibration quality (target: < 0.15)
- **Hosmer-Lemeshow Test**: Statistical calibration assessment (p-value > 0.05)

### Risk Score Mapping
| Risk Category | Probability | Clinical Action |
|--------------|-------------|-----------------|
| Low | 0-5% | Standard discharge |
| Medium | 5-15% | Enhanced education, 1-week follow-up |
| High | 15%+ | Intensive case management, home visit |

## Input
- Best-performing model from Phase 2
- Raw probability predictions on validation set
- True readmission outcomes
- Patient demographic data

## Output
1. **Calibrated Model**: Deployment-ready with adjusted probabilities
2. **Calibration Report**: Reliability diagrams, Brier scores, H-L test results
3. **Risk Score System**: Validated probability thresholds and categories
4. **Validation Table**: Actual readmission rates per risk category

## Success Criteria
- Brier score < 0.15
- Predictions within ±5% of diagonal on reliability plot
- Hosmer-Lemeshow p-value > 0.05
- Clinically validated risk categories
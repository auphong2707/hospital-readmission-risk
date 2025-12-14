#!/bin/bash

################################################################################
# Phase 5: Fairness Assessment & Mitigation - Master Orchestration Script
#
# This script orchestrates the complete fairness assessment and mitigation
# workflow by running both evaluation and (conditionally) mitigation scripts.
#
# Workflow:
#   1. Run fairness evaluation (evaluate_fairness_*.py)
#   2. Check if mitigation is required (reads phase5_summary_for_phase6.json)
#   3. If required, run mitigation script (calculate_group_thresholds_*.py)
#   4. Generate final summary combining both results
#
# Usage:
#   ./run_fairness_assessment_and_mitigation.sh [METHOD]
#
#   METHOD: gradient_boosting | random_forest | logistic_regression
#           (default: gradient_boosting)
#
# Examples:
#   ./run_fairness_assessment_and_mitigation.sh gradient_boosting
#   ./run_fairness_assessment_and_mitigation.sh random_forest
#   ./run_fairness_assessment_and_mitigation.sh logistic_regression
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default method
METHOD=${1:-gradient_boosting}

# Validate method
if [[ ! "$METHOD" =~ ^(gradient_boosting|random_forest|logistic_regression)$ ]]; then
    echo -e "${RED}❌ Error: Invalid method '$METHOD'${NC}"
    echo "   Valid options: gradient_boosting, random_forest, logistic_regression"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Output directory
OUTPUT_DIR="$SCRIPT_DIR/outputs/$METHOD"
EVALUATION_OUTPUT="$OUTPUT_DIR/evaluation"
MITIGATION_OUTPUT="$OUTPUT_DIR/mitigation"

echo "================================================================================"
echo "PHASE 5: FAIRNESS ASSESSMENT & MITIGATION"
echo "Hospital Readmission Risk Prediction - $METHOD"
echo "================================================================================"
echo ""
echo "📁 Output directory: $OUTPUT_DIR"
echo "   Evaluation: $EVALUATION_OUTPUT"
echo "   Mitigation: $MITIGATION_OUTPUT"
echo ""

################################################################################
# STEP 1: Run Fairness Evaluation
################################################################################

echo "================================================================================"
echo "STEP 1: Fairness Evaluation (Always Runs)"
echo "================================================================================"
echo ""

echo -e "${BLUE}🔍 Running fairness evaluation for $METHOD...${NC}"

# Create output directories
mkdir -p "$EVALUATION_OUTPUT"

# Run evaluation script
if [[ "$METHOD" == "gradient_boosting" ]]; then
    python3 "$SCRIPT_DIR/evaluate_fairness_gradient_boosting.py" \
        --output-dir "$EVALUATION_OUTPUT"
elif [[ "$METHOD" == "random_forest" ]]; then
    python3 "$SCRIPT_DIR/evaluate_fairness_random_forest.py" \
        --output-dir "$EVALUATION_OUTPUT"
else
    python3 "$SCRIPT_DIR/evaluate_fairness_logistic_regression.py" \
        --output-dir "$EVALUATION_OUTPUT"
fi

EVALUATION_STATUS=$?

if [ $EVALUATION_STATUS -ne 0 ]; then
    echo -e "${RED}❌ Fairness evaluation failed with exit code $EVALUATION_STATUS${NC}"
    exit $EVALUATION_STATUS
fi

echo -e "${GREEN}✅ Fairness evaluation completed successfully${NC}"
echo ""

################################################################################
# STEP 2: Check if Mitigation is Required
################################################################################

echo "================================================================================"
echo "STEP 2: Check Mitigation Requirement"
echo "================================================================================"
echo ""

SUMMARY_FILE="$EVALUATION_OUTPUT/phase5_summary_for_phase6.json"

if [ ! -f "$SUMMARY_FILE" ]; then
    echo -e "${RED}❌ Error: Summary file not found: $SUMMARY_FILE${NC}"
    exit 1
fi

# Check if mitigation is required using Python
REQUIRES_MITIGATION=$(python3 -c "
import json
with open('$SUMMARY_FILE', 'r') as f:
    data = json.load(f)
    print('true' if data.get('requires_mitigation', False) else 'false')
")

if [ "$REQUIRES_MITIGATION" = "true" ]; then
    PRIORITY=$(python3 -c "
import json
with open('$SUMMARY_FILE', 'r') as f:
    data = json.load(f)
    print(data.get('mitigation_priority', 'unknown').upper())
")
    
    echo -e "${YELLOW}⚠️  FAIRNESS VIOLATIONS DETECTED${NC}"
    echo "   Priority: $PRIORITY"
    echo "   Action: Proceeding to mitigation step"
    echo ""
    
    ############################################################################
    # STEP 3: Run Fairness Mitigation
    ############################################################################
    
    echo "================================================================================"
    echo "STEP 3: Fairness Mitigation (Conditional - Running)"
    echo "================================================================================"
    echo ""
    
    echo -e "${BLUE}🔧 Calculating group-specific thresholds for $METHOD...${NC}"
    
    # Create mitigation output directory
    mkdir -p "$MITIGATION_OUTPUT"
    
    # Run mitigation script
    if [[ "$METHOD" == "gradient_boosting" ]]; then
        python3 "$SCRIPT_DIR/calculate_group_thresholds_gradient_boosting.py" \
            --phase5-summary "$SUMMARY_FILE" \
            --output-dir "$MITIGATION_OUTPUT"
    elif [[ "$METHOD" == "random_forest" ]]; then
        python3 "$SCRIPT_DIR/calculate_group_thresholds_random_forest.py" \
            --phase5-summary "$SUMMARY_FILE" \
            --output-dir "$MITIGATION_OUTPUT"
    else
        python3 "$SCRIPT_DIR/calculate_group_thresholds_logistic_regression.py" \
            --phase5-summary "$SUMMARY_FILE" \
            --output-dir "$MITIGATION_OUTPUT"
    fi
    
    MITIGATION_STATUS=$?
    
    if [ $MITIGATION_STATUS -ne 0 ]; then
        echo -e "${RED}❌ Fairness mitigation failed with exit code $MITIGATION_STATUS${NC}"
        exit $MITIGATION_STATUS
    fi
    
    echo -e "${GREEN}✅ Fairness mitigation completed successfully${NC}"
    echo ""
    
    MITIGATION_APPLIED=true
else
    echo -e "${GREEN}✅ NO FAIRNESS VIOLATIONS DETECTED${NC}"
    echo "   No mitigation required"
    echo "   Skipping mitigation step"
    echo ""
    
    echo "================================================================================"
    echo "STEP 3: Fairness Mitigation (Conditional - Skipped)"
    echo "================================================================================"
    echo ""
    echo -e "${GREEN}✅ Mitigation step skipped - no fairness violations${NC}"
    echo ""
    
    MITIGATION_APPLIED=false
fi

################################################################################
# STEP 4: Generate Combined Summary
################################################################################

echo "================================================================================"
echo "STEP 4: Generate Combined Summary"
echo "================================================================================"
echo ""

# Create combined summary using Python
python3 << EOF
import json
from pathlib import Path

# Load evaluation results
eval_dir = Path("$EVALUATION_OUTPUT")
with open(eval_dir / "fairness_report.json", 'r') as f:
    fairness_report = json.load(f)

# Initialize combined summary
combined_summary = {
    "method": "$METHOD",
    "evaluation": {
        "output_dir": "$EVALUATION_OUTPUT",
        "bias_detected": fairness_report.get("bias_detected", False),
        "overall_performance": fairness_report.get("overall_performance", {}),
        "fairness_summary": fairness_report
    },
    "mitigation_applied": $MITIGATION_APPLIED
}

# Add mitigation results if applied
if $MITIGATION_APPLIED:
    mitigation_dir = Path("$MITIGATION_OUTPUT")
    
    # Load group thresholds
    if (mitigation_dir / "group_thresholds.json").exists():
        with open(mitigation_dir / "group_thresholds.json", 'r') as f:
            combined_summary["group_thresholds"] = json.load(f)
    
    # Load mitigation impact
    if (mitigation_dir / "mitigation_impact.json").exists():
        with open(mitigation_dir / "mitigation_impact.json", 'r') as f:
            mitigation_impact = json.load(f)
            combined_summary["mitigation"] = {
                "output_dir": "$MITIGATION_OUTPUT",
                "impact": mitigation_impact,
                "recommended_for_deployment": mitigation_impact.get("summary", {}).get("recommended_for_deployment", False)
            }

# Add deployment recommendation
if combined_summary["mitigation_applied"]:
    recommended = combined_summary.get("mitigation", {}).get("recommended_for_deployment", False)
    combined_summary["deployment_recommendation"] = {
        "use_group_thresholds": recommended,
        "threshold_configuration": "group_specific" if recommended else "global",
        "ready_for_deployment": True,
        "reason": "Fairness mitigation recommended" if recommended else "Fairness mitigation attempted but not recommended"
    }
else:
    combined_summary["deployment_recommendation"] = {
        "use_group_thresholds": False,
        "threshold_configuration": "global",
        "ready_for_deployment": True,
        "reason": "No fairness violations detected"
    }

# Save combined summary
output_dir = Path("$OUTPUT_DIR")
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "phase5_complete_summary.json", 'w') as f:
    json.dump(combined_summary, f, indent=2)

print("✅ Combined summary saved to: $OUTPUT_DIR/phase5_complete_summary.json")
EOF

echo ""

################################################################################
# FINAL SUMMARY
################################################################################

echo "================================================================================"
echo "✅ PHASE 5 COMPLETE: FAIRNESS ASSESSMENT & MITIGATION"
echo "================================================================================"
echo ""
echo "📊 Summary:"
echo "   Method: $METHOD"
echo "   Mitigation Applied: $([ "$MITIGATION_APPLIED" = true ] && echo "YES" || echo "NO")"
echo ""
echo "📁 Outputs:"
echo "   Evaluation results: $EVALUATION_OUTPUT"
if [ "$MITIGATION_APPLIED" = true ]; then
    echo "   Mitigation results: $MITIGATION_OUTPUT"
fi
echo "   Combined summary: $OUTPUT_DIR/phase5_complete_summary.json"
echo ""
echo "🎯 Next Steps:"
if [ "$MITIGATION_APPLIED" = true ]; then
    echo "   1. Review mitigation impact report"
    echo "   2. Consult with clinical and ethics teams"
    echo "   3. Proceed to Phase 6 (Final System Evaluation)"
else
    echo "   1. Proceed to Phase 6 (Final System Evaluation) with global threshold"
    echo "   2. Implement production monitoring for fairness metrics"
fi
echo ""
echo "================================================================================"

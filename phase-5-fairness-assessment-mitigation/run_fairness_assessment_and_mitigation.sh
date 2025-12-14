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

# Output directory - ALWAYS use consistent subdirectory structure
OUTPUT_DIR="$SCRIPT_DIR/outputs/$METHOD"
EVALUATION_OUTPUT="$OUTPUT_DIR/evaluation"
MITIGATION_OUTPUT="$OUTPUT_DIR/mitigation"

# Create output directories upfront
mkdir -p "$EVALUATION_OUTPUT"
mkdir -p "$MITIGATION_OUTPUT"

echo "================================================================================"
echo "PHASE 5: FAIRNESS ASSESSMENT & MITIGATION"
echo "Hospital Readmission Risk Prediction - $METHOD"
echo "================================================================================"
echo ""
echo "📁 Output directory: $OUTPUT_DIR"
echo "   Evaluation: $EVALUATION_OUTPUT (always)"
echo "   Mitigation: $MITIGATION_OUTPUT (conditional)"
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
    echo "   Creating placeholder for consistent structure"
    echo ""
    
    echo "================================================================================"
    echo "STEP 3: Fairness Mitigation (Conditional - Skipped)"
    echo "================================================================================"
    echo ""
    echo -e "${GREEN}✅ Mitigation step skipped - no fairness violations${NC}"
    
    # Create a mitigation summary indicating no mitigation was needed
    python3 << EOF
import json
from pathlib import Path

mitigation_dir = Path("$MITIGATION_OUTPUT")
mitigation_dir.mkdir(parents=True, exist_ok=True)

# Create a summary file indicating mitigation was not needed
no_mitigation_summary = {
    "mitigation_applied": False,
    "reason": "No fairness violations detected",
    "message": "Model meets fairness criteria with global threshold",
    "recommendation": {
        "use_group_thresholds": False,
        "threshold_configuration": "global",
        "deployment_ready": True
    }
}

with open(mitigation_dir / "no_mitigation_needed.json", 'w') as f:
    json.dump(no_mitigation_summary, f, indent=2)

print(f"✅ Created no-mitigation placeholder: $MITIGATION_OUTPUT/no_mitigation_needed.json")
EOF
    
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

# Create unified deployment configuration for Phase 6
python3 << EOF
import json
from pathlib import Path

# Load evaluation results
eval_dir = Path("$EVALUATION_OUTPUT")
with open(eval_dir / "fairness_report.json", 'r') as f:
    fairness_report = json.load(f)

mitigation_dir = Path("$MITIGATION_OUTPUT")

# ============================================================================
# UNIFIED OUTPUT FORMAT - Same structure for both cases
# ============================================================================

# Determine deployment configuration
# Convert bash boolean to Python boolean
is_mitigated = True if "$MITIGATION_APPLIED" == "true" else False
use_group_thresholds = False
threshold_source = None
group_thresholds = None

if is_mitigated:
    # Load mitigation results
    if (mitigation_dir / "mitigation_impact.json").exists():
        with open(mitigation_dir / "mitigation_impact.json", 'r') as f:
            mitigation_impact = json.load(f)
            use_group_thresholds = mitigation_impact.get("summary", {}).get("recommended_for_deployment", False)
    
    if use_group_thresholds and (mitigation_dir / "group_thresholds.json").exists():
        with open(mitigation_dir / "group_thresholds.json", 'r') as f:
            group_thresholds = json.load(f)
        threshold_source = "phase5_mitigation"

# Standard deployment config for Phase 6 (ALWAYS same format)
deployment_config = {
    "method": "$METHOD",
    "is_mitigated": is_mitigated,
    "use_group_thresholds": use_group_thresholds,
    "threshold_configuration": {
        "type": "group_specific" if use_group_thresholds else "global",
        "source": threshold_source if use_group_thresholds else "phase4_roi_optimization",
        "group_thresholds": group_thresholds if use_group_thresholds else None
    },
    "fairness_status": {
        "bias_detected": fairness_report.get("bias_detected", False),
        "mitigation_applied": is_mitigated,
        "fairness_compliant": not fairness_report.get("bias_detected", False) or is_mitigated
    },
    "phase6_instructions": {
        "load_thresholds_from": threshold_source if use_group_thresholds else "phase4",
        "apply_group_specific": use_group_thresholds,
        "evaluation_type": "mitigated_system" if use_group_thresholds else "global_threshold_system"
    }
}

# Save deployment config (standardized for Phase 6)
output_dir = Path("$OUTPUT_DIR")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "deployment_config.json", 'w') as f:
    json.dump(deployment_config, f, indent=2)

# Also save complete summary for human review
complete_summary = {
    "method": "$METHOD",
    "phase": "Phase 5: Fairness Assessment & Mitigation",
    "deployment_config": deployment_config,
    "evaluation": {
        "output_dir": "$EVALUATION_OUTPUT",
        "bias_detected": fairness_report.get("bias_detected", False),
        "overall_performance": fairness_report.get("overall_performance", {}),
        "fairness_summary": fairness_report
    },
    "mitigation": {}
}

if is_mitigated:
    complete_summary["mitigation"]["status"] = "applied"
    complete_summary["mitigation"]["output_dir"] = "$MITIGATION_OUTPUT"
    if (mitigation_dir / "mitigation_impact.json").exists():
        with open(mitigation_dir / "mitigation_impact.json", 'r') as f:
            complete_summary["mitigation"]["impact"] = json.load(f)
else:
    complete_summary["mitigation"]["status"] = "not_needed"
    complete_summary["mitigation"]["reason"] = "No fairness violations detected"

with open(output_dir / "phase5_complete_summary.json", 'w') as f:
    json.dump(complete_summary, f, indent=2)

print("✅ Deployment configuration saved:")
print(f"   • $OUTPUT_DIR/deployment_config.json (standardized for Phase 6)")
print(f"   • $OUTPUT_DIR/phase5_complete_summary.json (complete results)")
print(f"   Config: is_mitigated={is_mitigated}, use_group_thresholds={use_group_thresholds}")
EOF

echo ""

################################################################################
# STEP 5: Upload to HuggingFace Hub (Once, with All Results)
################################################################################

echo "================================================================================"
echo "STEP 5: Upload Results to HuggingFace Hub"
echo "================================================================================"
echo ""

# Determine repo name based on method
if [[ "$METHOD" == "gradient_boosting" ]]; then
    REPO_ID="auphong2707/hospital-readmission-gradient-boosting-fairness-assessment-mitigation"
elif [[ "$METHOD" == "random_forest" ]]; then
    REPO_ID="auphong2707/hospital-readmission-random-forest-fairness-assessment-mitigation"
else
    REPO_ID="auphong2707/hospital-readmission-logistic-regression-fairness-assessment-mitigation"
fi

echo -e "${BLUE}📤 Uploading combined results to HuggingFace Hub...${NC}"
echo "   Repository: $REPO_ID"
echo ""

# Run Python upload script
python3 << EOF
import sys
sys.path.insert(0, "$SCRIPT_DIR")

from utilities import upload_results_to_hf

try:
    repo_url = upload_results_to_hf(
        output_dir="$OUTPUT_DIR",
        repo_id="$REPO_ID",
        commit_message="Phase 5: Fairness Assessment & Mitigation - $METHOD (Combined Results)",
        include_visualizations=True
    )
    print(f"✅ Successfully uploaded to HuggingFace Hub!")
    print(f"🌐 View results at: {repo_url}")
except Exception as e:
    print(f"⚠️  Upload to HuggingFace failed: {e}")
    print(f"💡 Results are still saved locally in $OUTPUT_DIR")
    print(f"   You can upload manually later or set HF_TOKEN environment variable")
EOF

echo ""

################################################################################
# FINAL SUMMARY
################################################################################

echo "================================================================================"
echo "✅ PHASE 5 COMPLETE: FAIRNESS ASSESSMENT & MITIGATION"
echo "================================================================================"
echo ""
echo "📊 Status:"
echo "   Method: $METHOD"
echo "   Mitigation Applied: $([ "$MITIGATION_APPLIED" = true ] && echo "YES" || echo "NO (not needed)")"
echo ""
echo "📁 Output Structure (Consistent):"
echo "   $OUTPUT_DIR/"
echo "   ├── deployment_config.json         ← Standardized for Phase 6"
echo "   ├── phase5_complete_summary.json   ← Full results"
echo "   ├── evaluation/                    ← Always present"
if [ "$MITIGATION_APPLIED" = true ]; then
    echo "   └── mitigation/                    ← Applied (with thresholds)"
else
    echo "   └── mitigation/                    ← Placeholder (not needed)"
fi
echo ""
echo "🎯 For Phase 6 (Next Phase):"
echo "   Read: $OUTPUT_DIR/deployment_config.json"
echo "   Fields:"
echo "      • is_mitigated: $([ "$MITIGATION_APPLIED" = true ] && echo "true" || echo "false")"
echo "      • use_group_thresholds: (check config file)"
echo "      • threshold_configuration.type: global | group_specific"
echo "      • threshold_configuration.source: phase4 | phase5_mitigation"
echo ""
echo "📄 Output Files:"
echo "   • deployment_config.json          (standardized, same format always)"
echo "   • phase5_complete_summary.json    (detailed results)"
echo "   • evaluation/fairness_report.json"
if [ "$MITIGATION_APPLIED" = true ]; then
    echo "   • mitigation/group_thresholds.json"
    echo "   • mitigation/mitigation_impact.json"
else
    echo "   • mitigation/no_mitigation_needed.json"
fi
echo ""
echo "🌐 HuggingFace: $REPO_ID"
echo ""
echo "📋 Next Steps:"
echo "   Phase 6: Use deployment_config.json to load correct threshold configuration"
if [ "$MITIGATION_APPLIED" = true ]; then
    echo "            Review if group-specific thresholds are recommended"
else
    echo "            Use global threshold from Phase 4"
fi
echo "   Phase 7: Collect Phase 6 final system metrics"
echo ""
echo "================================================================================"

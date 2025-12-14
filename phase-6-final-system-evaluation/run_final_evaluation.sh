#!/bin/bash

################################################################################
# Phase 6: Final System Evaluation Orchestrator
#
# This script runs comprehensive final evaluation for all three models:
# - Gradient Boosting (LightGBM)
# - Random Forest
# - Logistic Regression
#
# It applies the deployed threshold configuration from Phase 5 (either global
# thresholds from Phase 4 or group-specific thresholds from Phase 5 mitigation)
# and generates comprehensive evaluation reports and visualizations.
#
# Outputs (per method):
#   - final_system_metrics.json (single source of truth for Phase 7)
#   - deployment_report.json (stakeholder-friendly summary)
#   - visualizations/ (9 comprehensive plots)
#
# Usage:
#   ./run_final_evaluation.sh [OPTIONS]
#
# Options:
#   --skip-gradient-boosting   Skip Gradient Boosting evaluation
#   --skip-random-forest       Skip Random Forest evaluation
#   --skip-logistic-regression Skip Logistic Regression evaluation
#   --readmission-cost VALUE   Cost of readmission in dollars (default: 15000)
#   --intervention-cost VALUE  Cost of intervention in dollars (default: 500)
#
# Examples:
#   # Run all three methods with default settings
#   ./run_final_evaluation.sh
#
#   # Run only Gradient Boosting
#   ./run_final_evaluation.sh --skip-random-forest --skip-logistic-regression
#
#   # Run with custom cost parameters
#   ./run_final_evaluation.sh --readmission-cost 20000 --intervention-cost 1000
#
# Requirements:
#   - Python 3.8+
#   - Dependencies installed: pip install -r requirements.txt
#   - Phase 5 outputs available (deployment_config.json for each method)
#   - HuggingFace authentication configured
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
RUN_GRADIENT_BOOSTING=true
RUN_RANDOM_FOREST=true
RUN_LOGISTIC_REGRESSION=true
READMISSION_COST=15000
INTERVENTION_COST=500

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-gradient-boosting)
            RUN_GRADIENT_BOOSTING=false
            shift
            ;;
        --skip-random-forest)
            RUN_RANDOM_FOREST=false
            shift
            ;;
        --skip-logistic-regression)
            RUN_LOGISTIC_REGRESSION=false
            shift
            ;;
        --readmission-cost)
            READMISSION_COST="$2"
            shift 2
            ;;
        --intervention-cost)
            INTERVENTION_COST="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-gradient-boosting   Skip Gradient Boosting evaluation"
            echo "  --skip-random-forest       Skip Random Forest evaluation"
            echo "  --skip-logistic-regression Skip Logistic Regression evaluation"
            echo "  --readmission-cost VALUE   Cost of readmission (default: 15000)"
            echo "  --intervention-cost VALUE  Cost of intervention (default: 500)"
            echo "  -h, --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         PHASE 6: FINAL SYSTEM EVALUATION - ORCHESTRATOR                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}📋 Configuration:${NC}"
echo "  - Readmission Cost: \$$READMISSION_COST"
echo "  - Intervention Cost: \$$INTERVENTION_COST"
echo ""
echo -e "${YELLOW}🎯 Methods to evaluate:${NC}"
echo "  - Gradient Boosting: $RUN_GRADIENT_BOOSTING"
echo "  - Random Forest: $RUN_RANDOM_FOREST"
echo "  - Logistic Regression: $RUN_LOGISTIC_REGRESSION"
echo ""

# Track results
SUCCESSFUL_METHODS=()
FAILED_METHODS=()

################################################################################
# Function: Run evaluation for a method
################################################################################
run_evaluation() {
    local method_name=$1
    local script_name=$2
    
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ METHOD: ${method_name}${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Run evaluation script
    if python "$SCRIPT_DIR/$script_name" \
        --readmission-cost "$READMISSION_COST" \
        --intervention-cost "$INTERVENTION_COST"; then
        
        echo -e "\n${GREEN}✅ ${method_name} evaluation completed successfully${NC}\n"
        SUCCESSFUL_METHODS+=("$method_name")
        return 0
    else
        echo -e "\n${RED}❌ ${method_name} evaluation failed${NC}\n"
        FAILED_METHODS+=("$method_name")
        return 1
    fi
}

################################################################################
# Main Execution
################################################################################

START_TIME=$(date +%s)

# Run evaluations
if [ "$RUN_GRADIENT_BOOSTING" = true ]; then
    run_evaluation "Gradient Boosting" "final_evaluation_gradient_boosting.py" || true
    echo ""
fi

if [ "$RUN_RANDOM_FOREST" = true ]; then
    run_evaluation "Random Forest" "final_evaluation_random_forest.py" || true
    echo ""
fi

if [ "$RUN_LOGISTIC_REGRESSION" = true ]; then
    run_evaluation "Logistic Regression" "final_evaluation_logistic_regression.py" || true
    echo ""
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

################################################################################
# Summary
################################################################################

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                         FINAL SUMMARY                                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Count results
TOTAL_METHODS=0
[ "$RUN_GRADIENT_BOOSTING" = true ] && ((TOTAL_METHODS++))
[ "$RUN_RANDOM_FOREST" = true ] && ((TOTAL_METHODS++))
[ "$RUN_LOGISTIC_REGRESSION" = true ] && ((TOTAL_METHODS++))

echo -e "${YELLOW}⏱️  Total Execution Time: ${DURATION}s${NC}"
echo ""

if [ ${#SUCCESSFUL_METHODS[@]} -gt 0 ]; then
    echo -e "${GREEN}✅ Successful Evaluations (${#SUCCESSFUL_METHODS[@]}/$TOTAL_METHODS):${NC}"
    for method in "${SUCCESSFUL_METHODS[@]}"; do
        echo "  - $method"
    done
    echo ""
fi

if [ ${#FAILED_METHODS[@]} -gt 0 ]; then
    echo -e "${RED}❌ Failed Evaluations (${#FAILED_METHODS[@]}/$TOTAL_METHODS):${NC}"
    for method in "${FAILED_METHODS[@]}"; do
        echo "  - $method"
    done
    echo ""
fi

# Output locations
echo -e "${YELLOW}📁 Output Locations:${NC}"
if [ "$RUN_GRADIENT_BOOSTING" = true ]; then
    echo "  - Gradient Boosting: ./outputs/gradient_boosting/final_evaluation/"
fi
if [ "$RUN_RANDOM_FOREST" = true ]; then
    echo "  - Random Forest: ./outputs/random_forest/final_evaluation/"
fi
if [ "$RUN_LOGISTIC_REGRESSION" = true ]; then
    echo "  - Logistic Regression: ./outputs/logistic_regression/final_evaluation/"
fi
echo ""

# Key output files
echo -e "${YELLOW}📄 Key Output Files (per method):${NC}"
echo "  - final_system_metrics.json   (single source of truth for Phase 7)"
echo "  - deployment_report.json      (stakeholder-friendly summary)"
echo "  - visualizations/              (9 comprehensive plots)"
echo ""

# Exit status
if [ ${#FAILED_METHODS[@]} -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║            ALL EVALUATIONS COMPLETED SUCCESSFULLY! 🎉                      ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Review final_system_metrics.json files for comprehensive results"
    echo "  2. Share deployment_report.json with clinical stakeholders"
    echo "  3. Examine visualizations for insights"
    echo "  4. Proceed to Phase 7 (Results Collection & Publication)"
    echo ""
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║               SOME EVALUATIONS FAILED                                      ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "  - Check that Phase 5 has been completed for all methods"
    echo "  - Verify deployment_config.json exists for failed methods"
    echo "  - Check HuggingFace authentication and repository access"
    echo "  - Review error messages above for specific issues"
    echo ""
    exit 1
fi

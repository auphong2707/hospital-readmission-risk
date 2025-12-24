/**
 * Manager Dashboard JavaScript
 * 
 * Loads and visualizes savings analysis, resource planning, cost breakdown, and model comparison data.
 */

// ============================================================================
// Data Loading Functions
// ============================================================================

let savingsData = null;
let resourceData = null;
let costData = null;
let comparisonData = null;

/**
 * Load all dashboard data
 */
async function loadDashboardData() {
    try {
        console.log("Loading manager dashboard data...");
        
        // Use Random Forest as the recommended model
        const method = "random_forest";
        
        // Load all data in parallel
        const [savings, cost, comparison] = await Promise.all([
            fetch(`/api/v1/manager/models/${method}/savings-summary`).then(r => r.json()),
            fetch(`/api/v1/manager/models/${method}/cost-breakdown`).then(r => r.json()),
            fetch(`/api/v1/manager/models/comparison`).then(r => r.json())
        ]);
        
        savingsData = savings;
        costData = cost;
        comparisonData = comparison;
        
        console.log("All data loaded successfully");
        console.log("Savings Data:", savingsData);
        console.log("Resource Data:", resourceData);
        
        // Render all sections
        renderExecutiveSummary();
        renderSavingsMetrics();
        renderImpactMetrics();
        renderWaterfallChart();
        renderCostComponents();
        renderBenefitComponents();
        renderModelComparisonTable();
        
    } catch (error) {
        console.error("Error loading dashboard data:", error);
        showError("Failed to load dashboard data. Please try again later.");
    }
}

/**
 * Show error message
 */
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-error';
    errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${message}`;
    document.querySelector('.dashboard-container').prepend(errorDiv);
}


// ============================================================================
// ROI Summary Section
// ============================================================================

/**
 * Render Executive Summary Card
 */
function renderExecutiveSummary() {
    const container = document.getElementById('executive-summary-card');
    
    let savingsRatioHtml = '';
    if (savingsData.savings_ratio !== undefined) {
        savingsRatioHtml = `
                <div class="summary-row">
                    <div class="summary-label">Savings per $1 Spent</div>
                    <div class="summary-value" style="font-size: 2em; font-weight: 700; color: #2563eb;">$${savingsData.savings_ratio.toFixed(2)}</div>
                </div>`;
    }
    
    const html = `
        <div class="executive-summary">
            <div class="summary-body">
                <div class="summary-row">
                    <div class="summary-label">Cost Saving</div>
                    <div class="summary-value" style="font-size: 2em; font-weight: 700; color: #10b981;">${formatCurrency(savingsData.cost_savings)}</div>
                </div>
                ${savingsRatioHtml}
                <hr class="summary-divider">
                <div class="summary-row" style="margin-top: 1em;">
                    <div class="summary-label" style="font-size: 0.9em; opacity: 0.8;">Model Performance (${savingsData.total_patients.toLocaleString()} patients)</div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5em; margin-top: 0.5em;">
                    <div class="summary-row">
                        <div class="summary-label">Readmissions Prevented</div>
                        <div class="summary-value">${savingsData.tp.toLocaleString()}</div>
                    </div>
                    <div class="summary-row">
                        <div class="summary-label">Unnecessary Interventions</div>
                        <div class="summary-value">${savingsData.fp.toLocaleString()}</div>
                    </div>
                    <div class="summary-row">
                        <div class="summary-label">No Action Needed</div>
                        <div class="summary-value">${savingsData.tn.toLocaleString()}</div>
                    </div>
                    <div class="summary-row">
                        <div class="summary-label">Readmissions Missed</div>
                        <div class="summary-value">${savingsData.fn.toLocaleString()}</div>
                    </div>
                </div>
                <hr class="summary-divider">
                <div class="summary-row" style="margin-top: 1em;">
                    <div class="summary-label" style="font-size: 0.9em; opacity: 0.8;">Financial Impact Breakdown</div>
                </div>
                <div class="summary-row">
                    <div class="summary-label">Prevented Readmissions Value (${savingsData.tp} × $14.5K)</div>
                    <div class="summary-value" style="color: #10b981;">${formatCurrency(savingsData.tp_value)}</div>
                </div>
                <div class="summary-row">
                    <div class="summary-label">Unnecessary Interventions Cost (${savingsData.fp} × $500)</div>
                    <div class="summary-value" style="color: #ef4444;">-${formatCurrency(savingsData.fp_cost)}</div>
                </div>
                <div class="summary-row">
                    <div class="summary-label">Missed Readmissions Cost (${savingsData.fn} × $15K)</div>
                    <div class="summary-value" style="color: #ef4444;">-${formatCurrency(savingsData.fn_cost)}</div>
                </div>
                <hr class="summary-divider">
                <div class="summary-row">
                    <div class="summary-label">Baseline Cost (do nothing)</div>
                    <div class="summary-value">${formatCurrency(savingsData.baseline_cost)}</div>
                </div>
                ${savingsData.intervention_costs !== undefined ? `
                <div class="summary-row">
                    <div class="summary-label">Intervention Costs</div>
                    <div class="summary-value">${formatCurrency(savingsData.intervention_costs)}</div>
                </div>` : ''}
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

/**
 * Render Savings Metrics
 */
function renderSavingsMetrics() {
    const container = document.getElementById('savings-metrics-stats');
    
    const stats = [
        {
            icon: 'fas fa-dollar-sign',
            label: 'Cost Saving',
            value: formatCurrency(savingsData.cost_savings),
            color: 'success'
        },
        {
            icon: 'fas fa-percentage',
            label: 'Intervention Rate',
            value: `${savingsData.intervention_rate.toFixed(1)}%`,
            color: 'info'
        }
    ];
    
    // Only add savings ratio if it exists
    if (savingsData.savings_ratio !== undefined) {
        stats.splice(1, 0, {
            icon: 'fas fa-chart-line',
            label: 'Savings per $1',
            value: `$${savingsData.savings_ratio.toFixed(2)}`,
            color: 'success'
        });
    }
    
    container.innerHTML = createStatsGrid(stats);
}

/**
 * Render Impact Metrics
 */
function renderImpactMetrics() {
    const container = document.getElementById('impact-metrics-stats');
    
    const stats = [
        {
            icon: 'fas fa-hospital',
            label: 'Readmissions Prevented',
            value: savingsData.tp.toLocaleString(),
            color: 'primary'
        },
        {
            icon: 'fas fa-exclamation-triangle',
            label: 'Readmissions Missed',
            value: savingsData.fn.toLocaleString(),
            color: 'warning'
        }
    ];
    
    // Only add intervention costs if it exists
    if (savingsData.intervention_costs !== undefined) {
        stats.push({
            icon: 'fas fa-money-bill-wave',
            label: 'Intervention Costs',
            value: formatCurrency(savingsData.intervention_costs),
            color: 'info'
        });
    }
    
    container.innerHTML = createStatsGrid(stats);
}


// ============================================================================
// Resource Planning Section
// ============================================================================

/**
 * Render Resource Planning Overview (Combined)
 */
function renderResourcePlanningOverview() {
    const container = document.getElementById('resource-planning-overview');
    
    const staffing = resourceData.staffing;
    
    const html = `
        <div class="resource-planning-grid">
            <!-- Intervention Volume Section -->
            <div class="resource-section">
                <h4 class="resource-section-title"><i class="fas fa-calendar-alt"></i> Intervention Volume</h4>
                <p class="resource-section-desc">Patients flagged for intervention at recommended threshold (${resourceData.threshold.toFixed(3)})</p>
                <div class="stats-grid">
                    <div class="stat-card stat-primary">
                        <div class="stat-icon"><i class="fas fa-users"></i></div>
                        <div class="stat-content">
                            <div class="stat-label">Patients Flagged (Annual)</div>
                            <div class="stat-value">${resourceData.patients_flagged.toLocaleString()}</div>
                        </div>
                    </div>
                    <div class="stat-card stat-info">
                        <div class="stat-icon"><i class="fas fa-percentage"></i></div>
                        <div class="stat-content">
                            <div class="stat-label">Intervention Rate</div>
                            <div class="stat-value">${resourceData.intervention_rate.toFixed(1)}%</div>
                        </div>
                    </div>
                    <div class="stat-card stat-secondary">
                        <div class="stat-icon"><i class="fas fa-calendar-alt"></i></div>
                        <div class="stat-content">
                            <div class="stat-label">Monthly Volume</div>
                            <div class="stat-value">${resourceData.monthly_volume.toLocaleString()}</div>
                        </div>
                    </div>
                    <div class="stat-card stat-secondary">
                        <div class="stat-icon"><i class="fas fa-calendar-week"></i></div>
                        <div class="stat-content">
                            <div class="stat-label">Weekly Volume</div>
                            <div class="stat-value">${resourceData.weekly_volume.toLocaleString()}</div>
                        </div>
                    </div>
                    <div class="stat-card stat-secondary">
                        <div class="stat-icon"><i class="fas fa-calendar-day"></i></div>
                        <div class="stat-content">
                            <div class="stat-label">Daily Volume</div>
                            <div class="stat-value">${resourceData.daily_volume.toLocaleString()}</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <hr class="resource-divider">
            
            <!-- Staffing Requirements Section -->
            <div class="resource-section">
                <h4 class="resource-section-title"><i class="fas fa-user-friends"></i> Staffing Requirements</h4>
                <p class="resource-section-desc">Recommended full-time equivalent (FTE) positions</p>
                <div class="stats-grid">
                    <div class="stat-card stat-primary">
                        <div class="stat-icon"><i class="fas fa-user-nurse"></i></div>
                        <div class="stat-content">
                            <div class="stat-label">Care Coordinators</div>
                            <div class="stat-value">${staffing.care_coordinators} FTE</div>
                        </div>
                    </div>
                    <div class="stat-card stat-primary">
                        <div class="stat-icon"><i class="fas fa-user-md"></i></div>
                        <div class="stat-content">
                            <div class="stat-label">Nurse Case Managers</div>
                            <div class="stat-value">${staffing.nurse_case_managers} FTE</div>
                        </div>
                    </div>
                    <div class="stat-card stat-primary">
                        <div class="stat-icon"><i class="fas fa-hands-helping"></i></div>
                        <div class="stat-content">
                            <div class="stat-label">Social Workers</div>
                            <div class="stat-value">${staffing.social_workers} FTE</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <hr class="resource-divider">
            
            <!-- Personnel Costs Section -->
            <div class="resource-section">
                <h4 class="resource-section-title"><i class="fas fa-money-bill-wave"></i> Personnel Costs</h4>
                <div class="cost-summary-grid">
                    <div class="cost-summary-item">
                        <div class="cost-label">Annual Personnel Cost</div>
                        <div class="cost-value">${formatCurrency(resourceData.costs.annual_personnel_cost)}</div>
                    </div>
                    <div class="cost-summary-item">
                        <div class="cost-label">Cost per Patient Served</div>
                        <div class="cost-value">${formatCurrency(resourceData.costs.cost_per_patient)}</div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}


// ============================================================================
// Cost Breakdown Section
// ============================================================================


// ============================================================================
// Cost Breakdown Section
// ============================================================================

/**
 * Render Waterfall Chart
 */
function renderWaterfallChart() {
    const container = document.getElementById('plot-waterfall');
    
    const breakdown = costData.financial_breakdown;
    const baseline = costData.baseline;
    const summary = costData.summary;
    
    // Waterfall data showing cost matrix calculation flow
    const x = [
        'Baseline Cost',
        'Prevented Readmissions<br>Value',
        'Unnecessary<br>Interventions Cost',
        'Missed Readmissions<br>Cost',
        'Cost Saving'
    ];
    
    const measure = ['absolute', 'relative', 'relative', 'relative', 'total'];
    const y = [
        baseline.baseline_cost,
        breakdown.tp_value,
        -breakdown.fp_cost,
        -breakdown.fn_cost,
        summary.cost_savings
    ];
    
    const text = y.map((val, idx) => {
        return formatCurrency(Math.abs(val));
    });
    
    const trace = {
        type: 'waterfall',
        orientation: 'v',
        measure: measure,
        x: x,
        y: y,
        text: text,
        textposition: 'auto',
        textangle: 0,
        textfont: {
            size: 12,
            color: 'black'
        },
        insidetextfont: {
            color: 'white',
            size: 12
        },
        outsidetextfont: {
            color: 'black',
            size: 12
        },
        connector: {
            line: { color: 'rgb(63, 63, 63)' }
        },
        increasing: { marker: { color: '#10b981' } },
        decreasing: { marker: { color: '#ef4444' } },
        totals: { marker: { color: '#3b82f6' } }
    };
    
    const layout = {
        title: '',
        showlegend: false,
        xaxis: {
            title: '',
            type: 'category',
            tickangle: 0
        },
        yaxis: {
            title: 'Amount ($)',
            tickformat: '$,.0f'
        },
        margin: { l: 100, r: 50, t: 50, b: 120 },
        height: 450,
        plot_bgcolor: '#f9fafb',
        paper_bgcolor: 'white'
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
    };
    
    Plotly.newPlot(container, [trace], layout, config);
}

/**
 * Render Cost Components Table
 */
function renderCostComponents() {
    const container = document.getElementById('cost-components-table');
    
    const matrix = costData.cost_matrix;
    const confusion = costData.confusion_matrix;
    const breakdown = costData.financial_breakdown;
    
    const rows = [
        { label: 'Cost Matrix (per patient)', value: '', bold: true, header: true },
        { label: 'Prevented Readmission (TP)', value: `+$${matrix.tp_unit_value.toLocaleString()}` },
        { label: 'Unnecessary Intervention (FP)', value: `-$${matrix.fp_unit_cost.toLocaleString()}` },
        { label: 'No Action Needed (TN)', value: `$${matrix.tn_unit_value.toLocaleString()}` },
        { label: 'Missed Readmission (FN)', value: `-$${matrix.fn_unit_cost.toLocaleString()}` },
        { separator: true },
        { label: 'Financial Breakdown', value: '', bold: true, header: true },
        { label: `Prevented Readmissions Value (${confusion.tp} × $${matrix.tp_unit_value.toLocaleString()})`, value: breakdown.tp_value },
        { label: `Unnecessary Interventions Cost (${confusion.fp} × $${matrix.fp_unit_cost.toLocaleString()})`, value: -breakdown.fp_cost, isNegative: true },
        { label: `Missed Readmissions Cost (${confusion.fn} × $${matrix.fn_unit_cost.toLocaleString()})`, value: -breakdown.fn_cost, isNegative: true },
        { separator: true },
        { label: 'Net Program Value', value: breakdown.net_program_value, bold: true, isNegative: breakdown.net_program_value < 0 }
    ];
    
    let html = '<table class="data-table">';
    
    rows.forEach(row => {
        if (row.separator) {
            html += '<tr class="table-separator"><td colspan="2"></td></tr>';
        } else if (row.header) {
            html += `
                <tr style="background: #f3f4f6;">
                    <td colspan="2" style="font-weight: bold; padding: 0.5em;">${row.label}</td>
                </tr>
            `;
        } else {
            const rowClass = row.highlight ? 'table-total' : '';
            const labelStyle = row.bold ? 'font-weight: bold;' : '';
            const valueStyle = row.bold ? 'font-weight: bold;' : '';
            const valueColor = row.isNegative ? 'color: #ef4444;' : '';
            let displayValue;
            if (row.isRatio) {
                displayValue = '$' + row.value;
            } else if (row.isCount) {
                displayValue = row.value.toLocaleString();
            } else if (typeof row.value === 'string') {
                displayValue = row.value;
            } else {
                displayValue = formatCurrency(row.value);
            }
            html += `
                <tr class="${rowClass}">
                    <td style="${labelStyle}">${row.label}</td>
                    <td class="text-right" style="${valueStyle}${valueColor}">${displayValue}</td>
                </tr>
            `;
        }
    });
    
    html += '</table>';
    container.innerHTML = html;
}

/**
 * Render Benefit Components Table
 */
function renderBenefitComponents() {
    const container = document.getElementById('benefit-components-table');
    
    const baseline = costData.baseline;
    const summary = costData.summary;
    const confusion = costData.confusion_matrix;
    
    const rows = [
        { label: 'Baseline (Do Nothing)', value: '', bold: true, header: true },
        { label: 'Potential Readmissions', value: baseline.potential_readmissions, isCount: true },
        { label: 'Baseline Cost', value: baseline.baseline_cost },
        { separator: true },
        { label: 'With Intervention', value: '', bold: true, header: true },
        { label: 'Readmissions Prevented (TP)', value: confusion.tp, isCount: true },
        { label: 'Readmissions Missed (FN)', value: confusion.fn, isCount: true },
        { separator: true },
        { label: 'Financial Summary', value: '', bold: true, header: true }
    ];
    
    // Only add intervention costs if it exists
    if (summary.intervention_costs !== undefined) {
        rows.push({ label: 'Intervention Costs', value: summary.intervention_costs });
    }
    
    rows.push({ label: 'Cost Saving', value: summary.cost_savings, bold: true, highlight: true });
    
    // Only add savings ratio if it exists
    if (summary.savings_ratio !== undefined) {
        rows.push({ label: 'Savings per $1 Spent', value: `${summary.savings_ratio.toFixed(2)}`, bold: true, isRatio: true });
    }
    
    let html = '<table class="data-table">';
    
    rows.forEach(row => {
        if (row.separator) {
            html += '<tr class="table-separator"><td colspan="2"></td></tr>';
        } else {
            const rowClass = row.highlight ? 'table-total' : '';
            const labelStyle = row.bold ? 'font-weight: bold;' : '';
            const valueStyle = row.bold ? 'font-weight: bold;' : '';
            const valueColor = row.isNegative ? 'color: #ef4444;' : '';
            const displayValue = row.isCount ? row.value.toLocaleString() : formatCurrency(row.value);
            html += `
                <tr class="${rowClass}">
                    <td style="${labelStyle}">${row.label}</td>
                    <td class="text-right" style="${valueStyle}${valueColor}">${displayValue}</td>
                </tr>
            `;
        }
    });
    
    html += '</table>';
    container.innerHTML = html;
}


// ============================================================================
// Model Comparison Section
// ============================================================================

/**
 * Render Model Comparison Table
 */
function renderModelComparisonTable() {
    const container = document.getElementById('model-comparison-table');
    
    const comparison = comparisonData.comparison;
    
    // Check if any model has savings_ratio
    const hasSavingsRatio = comparison.some(model => model.savings_ratio !== undefined);
    
    let html = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th class="text-right">TP</th>
                    <th class="text-right">FP</th>
                    <th class="text-right">FN</th>
                    <th class="text-right">Cost Saving</th>
                    ${hasSavingsRatio ? '<th class="text-right">Savings per $1</th>' : ''}
                    <th class="text-right">Intervention Rate</th>
                    <th class="text-right">ROC-AUC</th>
                    <th class="text-center">Recommended</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    comparison.forEach(model => {
        const modelName = formatModelName(model.method);
        const badge = model.recommended ? '<span class="badge-success">✓ Recommended</span>' : '';
        const rowClass = model.recommended ? 'table-highlight' : '';
        
        html += `
            <tr class="${rowClass}">
                <td><strong>${modelName}</strong></td>
                <td class="text-right">${model.tp.toLocaleString()}</td>
                <td class="text-right">${model.fp.toLocaleString()}</td>
                <td class="text-right">${model.fn.toLocaleString()}</td>
                <td class="text-right">${formatCurrency(model.cost_savings)}</td>
                ${hasSavingsRatio ? `<td class="text-right">${model.savings_ratio !== undefined ? '$' + model.savings_ratio.toFixed(2) : 'N/A'}</td>` : ''}
                <td class="text-right">${model.intervention_rate.toFixed(1)}%</td>
                <td class="text-right">${model.roc_auc.toFixed(3)}</td>
                <td class="text-center">${badge}</td>
            </tr>
        `;
    });
    
    html += `
            </tbody>
        </table>
    `;
    
    container.innerHTML = html;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Create stats grid HTML
 */
function createStatsGrid(stats) {
    let html = '<div class="stats-grid">';
    
    stats.forEach(stat => {
        html += `
            <div class="stat-card stat-${stat.color}">
                <div class="stat-icon">
                    <i class="${stat.icon}"></i>
                </div>
                <div class="stat-content">
                    <div class="stat-label">${stat.label}</div>
                    <div class="stat-value">${stat.value}</div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    return html;
}

/**
 * Format currency
 */
function formatCurrency(value) {
    if (value === null || value === undefined) return 'N/A';
    return '$' + value.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
}

/**
 * Format percentage
 */
function formatPercent(value, decimals = 2) {
    if (value === null || value === undefined) return 'N/A';
    return value.toFixed(decimals) + '%';
}

/**
 * Format model name
 */
function formatModelName(method) {
    const names = {
        'gradient_boosting': 'Gradient Boosting',
        'random_forest': 'Random Forest',
        'logistic_regression': 'Logistic Regression'
    };
    return names[method] || method;
}


// ============================================================================
// Initialize Dashboard
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log("Manager dashboard initialized");
    loadDashboardData();
});

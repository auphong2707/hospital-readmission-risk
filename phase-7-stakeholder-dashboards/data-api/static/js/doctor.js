// Doctor Dashboard JavaScript
// Handles clinician-focused data fetching and visualization

// Current selected model (for fairness section only)
let currentModel = 'gradient_boosting';

// Initialize dashboard on page load
document.addEventListener('DOMContentLoaded', function() {
    // Set up model selector (only exists in fairness section now)
    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        modelSelect.addEventListener('change', handleModelChange);
        
        // Get model from URL parameter if present
        const urlParams = new URLSearchParams(window.location.search);
        const modelParam = urlParams.get('model');
        if (modelParam) {
            currentModel = modelParam;
            modelSelect.value = currentModel;
        }
    }
    
    // Initialize dashboard
    initializeDashboard();
    
    // Set up auto-refresh every 5 minutes (300000ms)
    setInterval(initializeDashboard, 300000);
});

/**
 * Handle model selection change
 */
function handleModelChange(event) {
    currentModel = event.target.value;
    
    // Update URL without reloading
    const url = new URL(window.location);
    url.searchParams.set('model', currentModel);
    window.history.pushState({}, '', url);
    
    // Reload dashboard with new model
    initializeDashboard();
}

/**
 * Initialize all dashboard panels
 */
async function initializeDashboard() {
    try {
        // Load all panels in parallel
        await Promise.all([
            loadRiskFactors(), // Uses ensemble endpoint (no model selector)
            loadRiskDistribution(), // Risk score distribution chart
            loadLatestPatients() // Uses ensemble endpoint for patient list
        ]);
    } catch (error) {
        console.error('Error initializing dashboard:', error);
    }
}

// Store risk factors data globally for click handling
let globalRiskFactorsData = null;

/**
 * Panel 2: Top 10 Risk Factors - Ensemble Visualization (2-Column Layout)
 * Shows performance-weighted ensemble from all 3 models
 */
async function loadRiskFactors() {
    try {
        showLoading('table-risk-factors');
        
        // Use ensemble endpoint instead of single model
        const data = await fetchJSON(`/api/v1/models/ensemble/risk-factors?top_n=10`);
        
        // Store data globally for click handling
        globalRiskFactorsData = data;
        
        // Create container for visualization
        const container = document.getElementById('table-risk-factors');
        container.innerHTML = '';
        
        // Create horizontal bar chart
        const chartDiv = document.createElement('div');
        chartDiv.id = 'risk-factors-chart';
        chartDiv.className = 'risk-factors-chart';
        container.appendChild(chartDiv);
        
        // Prepare data for horizontal bar chart
        const features = data.risk_factors.map(f => f.feature);
        const importances = data.risk_factors.map(f => f.importance);
        const agreements = data.risk_factors.map(f => f.agreement);
        
        // Create bar chart with clinical labels and click handling
        createHorizontalBarChart(chartDiv.id, features, importances, agreements, data.risk_factors);
        
        // Display first feature by default
        if (data.risk_factors && data.risk_factors.length > 0) {
            displayFeatureDetail(data.risk_factors[0], 0);
        }
        
    } catch (error) {
        console.error('Error loading risk factors:', error);
        showError('table-risk-factors', 'Failed to load risk factors');
    }
}

/**
 * Display detailed information for a selected feature in the right panel
 * @param {Object} feature - The selected risk factor
 * @param {number} rank - The rank (0-indexed)
 */
function displayFeatureDetail(feature, rank) {
    const detailCard = document.getElementById('feature-detail-card');
    
    detailCard.innerHTML = `
        <div class="feature-detail-header">
            <div class="feature-rank-badge">${rank + 1}</div>
            <div class="feature-title">
                <div class="feature-name">${feature.clinical_meaning}</div>
                <div class="feature-importance">Importance Score: ${feature.importance.toFixed(3)}</div>
            </div>
        </div>
        
        <div class="feature-section">
            <div class="feature-section-title">What This Means</div>
            <div class="feature-section-content">
                ${feature.clinical_meaning} is a significant predictor of hospital readmission. 
                Patients with concerning patterns in this factor should receive targeted interventions.
            </div>
        </div>
        
        <div class="feature-section">
            <div class="feature-section-title">Clinical Recommendation</div>
            <div class="recommendation-box">
                <i class="fas fa-check-circle"></i>
                <span>${feature.recommendation}</span>
            </div>
        </div>
        
        <div class="feature-section">
            <div class="feature-section-title">Technical Feature Name</div>
            <div class="feature-section-content">
                <code style="background: #f5f5f5; padding: 4px 8px; border-radius: 4px; font-size: 0.875rem;">${feature.feature}</code>
            </div>
        </div>
    `;
}

/**
 * Create horizontal bar chart for risk factors
 * @param {string} elementId - ID of container element
 * @param {Array} labels - Clinical meanings (y-axis)
 * @param {Array} values - Importance scores
 * @param {Array} agreements - Agreement levels (high/medium/low)
 * @param {Array} riskFactors - Full risk factor objects for click handling
 */
function createHorizontalBarChart(elementId, labels, values, agreements, riskFactors) {
    const colors = agreements.map(a => {
        if (a === 'high') return '#2ecc71'; // Solid green
        if (a === 'medium') return '#f39c12'; // Orange
        return '#95a5a6'; // Gray
    });
    
    const trace = {
        type: 'bar',
        orientation: 'h',
        x: values,
        y: labels.map((l, i) => labels.length - i), // Reverse order for top-to-bottom
        text: labels.map(() => ''), // No text labels on bars
        hovertemplate: '<b>%{customdata}</b><br>Relative Importance<br><i>Click to view details</i><extra></extra>',
        customdata: labels,
        marker: {
            color: colors,
            line: {
                color: agreements.map(a => a === 'medium' ? '#d68910' : 'transparent'),
                width: agreements.map(a => a === 'medium' ? 2 : 0)
            },
            pattern: {
                shape: agreements.map(a => a === 'medium' ? '/' : ''),
                solidity: 0.3
            }
        }
    };
    
    const layout = {
        margin: { l: 250, r: 40, t: 30, b: 50 },
        height: 500,
        xaxis: {
            title: 'Relative Importance',
            showticklabels: false, // Hide numeric values
            showgrid: true,
            gridcolor: '#e0e0e0'
        },
        yaxis: {
            title: '',
            autorange: 'reversed',
            ticktext: labels.slice().reverse(),
            tickvals: labels.map((l, i) => i + 1)
        },
        showlegend: false,
        plot_bgcolor: '#fafafa',
        paper_bgcolor: 'white'
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot(elementId, [trace], layout, config);
    
    // Add click handler
    const plotDiv = document.getElementById(elementId);
    plotDiv.on('plotly_click', function(data) {
        const pointIndex = labels.length - 1 - data.points[0].pointNumber; // Reverse the index
        const feature = riskFactors[pointIndex];
        displayFeatureDetail(feature, pointIndex);
    });
}

/**
 * Panel 3: Understanding the Metrics (Generic Explanation)
 */
async function loadPerformanceClinical() {
    try {
        showLoading('table-performance-clinical');
        
        // Generic metrics explanation - not model-specific
        const columns = ['Metric', 'What It Means', 'Clinical Application'];
        const rows = [
            [
                'Sensitivity (Recall)',
                'Percentage of actual readmissions correctly identified',
                'High sensitivity means fewer readmissions are missed. Important for patient safety.'
            ],
            [
                'Specificity',
                'Percentage of non-readmissions correctly identified',
                'High specificity means fewer false alarms. Reduces unnecessary interventions.'
            ],
            [
                'PPV (Precision)',
                'When model predicts readmission, how often is it correct',
                'High PPV means you can trust positive predictions for intervention planning.'
            ],
            [
                'NPV',
                'When model predicts no readmission, how often is it correct',
                'High NPV means you can safely discharge patients with low risk scores.'
            ],
            [
                'ROC-AUC',
                'Overall ability to distinguish readmitted vs not readmitted',
                'Higher values (closer to 1.0) indicate better overall model performance.'
            ],
            [
                'F1 Score',
                'Balance between catching readmissions and avoiding false alarms',
                'Useful when you need both high sensitivity and high precision.'
            ]
        ];
        
        createTable('table-performance-clinical', columns, rows);
    } catch (error) {
        console.error('Error loading clinical performance:', error);
        showError('table-performance-clinical', 'Failed to load performance metrics');
    }
}

/**
 * Panel 5: Fairness Metrics by Demographics Table
 */
async function loadFairnessDemographics() {
    try {
        showLoading('table-fairness-demographics');
        
        const data = await fetchJSON(`/api/v1/models/${currentModel}/fairness-summary`);
        
        // Extract demographics data
        const columns = ['Demographic Group', 'TPR', 'FPR', 'PPV', 'Sample Size'];
        const rows = [];
        
        // Process race groups
        if (data.by_race) {
            Object.entries(data.by_race).forEach(([race, metrics]) => {
                rows.push([
                    `Race: ${race}`,
                    formatPercent(metrics.tpr * 100, 1),
                    formatPercent(metrics.fpr * 100, 1),
                    formatPercent(metrics.ppv * 100, 1),
                    metrics.sample_size || 'N/A'
                ]);
            });
        }
        
        // Add separator
        rows.push(['', '', '', '', '']);
        
        // Process gender groups
        if (data.by_gender) {
            Object.entries(data.by_gender).forEach(([gender, metrics]) => {
                rows.push([
                    `Gender: ${gender}`,
                    formatPercent(metrics.tpr * 100, 1),
                    formatPercent(metrics.fpr * 100, 1),
                    formatPercent(metrics.ppv * 100, 1),
                    metrics.sample_size || 'N/A'
                ]);
            });
        }
        
        // Add separator
        rows.push(['', '', '', '', '']);
        
        // Process age groups
        if (data.by_age) {
            Object.entries(data.by_age).forEach(([age, metrics]) => {
                rows.push([
                    `Age: ${age}`,
                    formatPercent(metrics.tpr * 100, 1),
                    formatPercent(metrics.fpr * 100, 1),
                    formatPercent(metrics.ppv * 100, 1),
                    metrics.sample_size || 'N/A'
                ]);
            });
        }
        
        // Define thresholds (higher is better for TPR and PPV, lower is better for FPR)
        const thresholds = [
            null, // Group name
            { green: 70, yellow: 60 }, // TPR
            { green: 10, yellow: 20, inverse: true }, // FPR
            { green: 50, yellow: 40 }, // PPV
            null  // Sample size
        ];
        
        createTable('table-fairness-demographics', columns, rows, thresholds);
    } catch (error) {
        console.error('Error loading fairness demographics:', error);
        showError('table-fairness-demographics', 'Failed to load fairness metrics');
    }
}

/**
 * Panel 6: Overall Fairness Assessment Table
 */
async function loadFairnessOverall() {
    try {
        showLoading('table-fairness-overall');
        
        const data = await fetchJSON(`/api/v1/models/${currentModel}/fairness-summary`);
        
        const columns = ['Fairness Criterion', 'Status', 'Max Disparity', 'Details'];
        const rows = [
            [
                'Demographic Parity',
                data.demographic_parity_pass ? '✅ PASS' : '❌ FAIL',
                data.max_demographic_parity_disparity ? formatPercent(data.max_demographic_parity_disparity, 1) : 'N/A',
                'Positive prediction rates across groups'
            ],
            [
                'Equal Opportunity',
                data.equal_opportunity_pass ? '✅ PASS' : '❌ FAIL',
                data.max_tpr_disparity ? formatPercent(data.max_tpr_disparity, 1) : 'N/A',
                'True positive rates (sensitivity) across groups'
            ],
            [
                'Equalized Odds',
                data.equalized_odds_pass ? '✅ PASS' : '❌ FAIL',
                data.max_odds_disparity ? formatPercent(data.max_odds_disparity, 1) : 'N/A',
                'Both TPR and FPR balanced across groups'
            ],
            [
                'Predictive Parity',
                data.predictive_parity_pass ? '✅ PASS' : '❌ FAIL',
                data.max_ppv_disparity ? formatPercent(data.max_ppv_disparity, 1) : 'N/A',
                'Positive predictive values across groups'
            ]
        ];
        
        // Define thresholds (lower is better for disparity)
        const thresholds = [
            null, // Criterion name
            null, // Status (already has emoji)
            { green: 5, yellow: 10, inverse: true }, // Max disparity
            null  // Details
        ];
        
        createTable('table-fairness-overall', columns, rows, thresholds);
    } catch (error) {
        console.error('Error loading overall fairness:', error);
        showError('table-fairness-overall', 'Failed to load fairness assessment');
    }
}

/**
 * Risk Distribution Chart
 */
async function loadRiskDistribution() {
    try {
        // Fetch ALL patients in one call with large page_size
        const data = await fetchJSON(`/api/v1/models/ensemble/latest-patients?page=1&page_size=200000`);
        const allPatients = data.patients;
        const totalCount = allPatients.length;
        
        // Calculate distribution bins (5% ranges = 20 bins)
        const bins = [];
        const colors = [
            '#28a745', '#2eb84c', '#35c953', '#3dd45a', '#4ade60',
            '#5cb85c', '#6bc266', '#7acc70', '#89d67a', '#98e084',
            '#a8d99b', '#b8d1a8', '#c8cab5', '#d8c2c2', '#e8bbce',
            '#f0ad4e', '#f59c3c', '#fa8b2a', '#ff7a18', '#ff6900'
        ];
        
        for (let i = 0; i < 20; i++) {
            const min = i * 5;
            const max = (i + 1) * 5;
            bins.push({
                range: `${min}-${max}%`,
                min: min,
                max: max,
                count: 0,
                color: i < 8 ? '#28a745' : i < 12 ? '#ffc107' : i < 16 ? '#fd7e14' : '#dc3545'
            });
        }
        
        // Count patients in each bin
        allPatients.forEach(patient => {
            const score = patient.risk_score;
            for (let bin of bins) {
                if (score >= bin.min && score < bin.max) {
                    bin.count++;
                    break;
                } else if (score === 100 && bin.max === 100) {
                    bin.count++;
                    break;
                }
            }
        });
        
        // Create histogram with percentages
        const trace = {
            type: 'bar',
            x: bins.map(b => b.range),
            y: bins.map(b => (b.count / totalCount * 100).toFixed(2)),
            marker: {
                color: bins.map(b => b.color)
            },
            text: bins.map(b => `${(b.count / totalCount * 100).toFixed(1)}%`),
            textposition: 'outside',
            hovertemplate: '<b>%{x}</b><br>%{text} (%{customdata} patients)<extra></extra>',
            customdata: bins.map(b => b.count)
        };
        
        const layout = {
            margin: { l: 60, r: 40, t: 30, b: 80 },
            height: 400,
            xaxis: {
                title: 'Risk Score Range',
                tickangle: -45
            },
            yaxis: {
                title: 'Percentage of Patients (%)',
                gridcolor: '#e0e0e0'
            },
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white',
            showlegend: false
        };
        
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        Plotly.newPlot('chart-risk-distribution', [trace], layout, config);
        
    } catch (error) {
        console.error('Error loading risk distribution:', error);
        showError('chart-risk-distribution', 'Failed to load risk distribution');
    }
}

/**
 * Latest Patients Panel - Ensemble predictions with pagination
 */
let currentPage = 1;
let currentPageSize = 10;
let currentSortBy = 'risk_score';
let currentSortOrder = 'desc';

async function loadLatestPatients(page = 1, pageSize = currentPageSize, sortBy = currentSortBy, sortOrder = currentSortOrder) {
    try {
        // Clear table content without showing "Loading..." text
        const tableElement = document.getElementById('table-latest-patients');
        if (tableElement) {
            tableElement.innerHTML = '';
        }
        
        // Clear patient count during loading
        const countElement = document.getElementById('patient-count');
        if (countElement) {
            countElement.textContent = '';
        }
        
        // Update current state
        currentPage = page;
        currentPageSize = pageSize;
        currentSortBy = sortBy;
        currentSortOrder = sortOrder;
        
        const data = await fetchJSON(`/api/v1/models/ensemble/latest-patients?page=${page}&page_size=${pageSize}&sort_by=${sortBy}&sort_order=${sortOrder}`);
        
        // Update patient count
        if (countElement) {
            const start = (data.pagination.page - 1) * data.pagination.page_size + 1;
            const end = Math.min(start + data.pagination.page_size - 1, data.pagination.total_count);
            countElement.textContent = `Showing ${start}-${end} of ${data.pagination.total_count} patients`;
        }
        
        // Create sortable table
        createPatientTable('table-latest-patients', data.patients);
        
        // Update pagination controls
        updatePaginationControls(data.pagination);
        
    } catch (error) {
        console.error('Error loading latest patients:', error);
        showError('table-latest-patients', 'Failed to load patient list');
    }
}

/**
 * Create patient table with sortable headers
 */
function createPatientTable(elementId, patients) {
    const container = document.getElementById(elementId);
    
    if (!patients || patients.length === 0) {
        container.innerHTML = '<p class="text-muted">No patients found.</p>';
        return;
    }
    
    // Create table
    let html = '<table class="table table-hover" style="width: 100%; font-size: 0.9rem;">';
    
    // Headers with sort icons - patient ID + top features + risk score at end
    const headers = [
        { key: 'patient_id', label: 'Patient ID', sortable: true },
        { key: 'prior_admits', label: 'Prior Admits', sortable: true },
        { key: 'er_visits', label: 'ER Visits', sortable: true },
        { key: 'diagnoses', label: 'Diagnoses', sortable: true },
        { key: 'los', label: 'LOS (days)', sortable: true },
        { key: 'medications', label: 'Medications', sortable: true },
        { key: 'lab_procedures', label: 'Lab Procedures', sortable: true },
        { key: 'procedures', label: 'Procedures', sortable: true },
        { key: 'a1c_result', label: 'A1C Result', sortable: true },
        { key: 'discharge_disposition', label: 'Discharge Disp.', sortable: true },
        { key: 'risk_score', label: 'Risk Score', sortable: true }
    ];
    
    html += '<thead><tr>';
    headers.forEach(header => {
        const sortIcon = header.sortable ? (currentSortBy === header.key ? (currentSortOrder === 'asc' ? ' ▲' : ' ▼') : ' ⇅') : '';
        const clickHandler = header.sortable ? ` onclick="handlePatientSort('${header.key}')" style="cursor: pointer;"` : '';
        html += `<th${clickHandler}>${header.label}${sortIcon}</th>`;
    });
    html += '</tr></thead>';
    
    // Body
    html += '<tbody>';
    patients.forEach(patient => {
        html += '<tr>';
        html += `<td><strong>${patient.patient_id}</strong></td>`;
        html += `<td>${patient.prior_admits}</td>`;
        html += `<td>${patient.er_visits}</td>`;
        html += `<td>${patient.diagnoses}</td>`;
        html += `<td>${patient.los}</td>`;
        html += `<td>${patient.medications}</td>`;
        html += `<td>${patient.lab_procedures}</td>`;
        html += `<td>${patient.procedures}</td>`;
        html += `<td>${patient.a1c_result}</td>`;
        html += `<td>${patient.discharge_disposition}</td>`;
        html += `<td><strong style="color: ${getRiskColor(patient.risk_score)};">${patient.risk_score}%</strong></td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    
    container.innerHTML = html;
}

/**
 * Handle column sort for patient table
 */
function handlePatientSort(column) {
    // Toggle sort order if same column, otherwise default to desc
    const newSortOrder = (currentSortBy === column && currentSortOrder === 'desc') ? 'asc' : 'desc';
    loadLatestPatients(currentPage, currentPageSize, column, newSortOrder);
}

// Make handlePatientSort available globally for onclick handlers
window.handlePatientSort = handlePatientSort;

/**
 * Get risk color based on score
 */
function getRiskColor(score) {
    if (score >= 80) return '#dc3545'; // Critical - Red
    if (score >= 60) return '#fd7e14'; // High - Orange
    if (score >= 40) return '#ffc107'; // Moderate - Yellow
    return '#28a745'; // Low - Green
}

/**
 * Update pagination controls
 */
function updatePaginationControls(pagination) {
    const container = document.getElementById('pagination-controls');
    if (!container) return;
    
    let html = '';
    
    // Previous button
    if (pagination.has_previous) {
        html += `<button onclick="loadLatestPatients(${pagination.page - 1})" class="btn btn-sm btn-secondary">« Previous</button>`;
    } else {
        html += `<button class="btn btn-sm btn-secondary" disabled>« Previous</button>`;
    }
    
    // Page numbers
    const maxPagesToShow = 5;
    let startPage = Math.max(1, pagination.page - Math.floor(maxPagesToShow / 2));
    let endPage = Math.min(pagination.total_pages, startPage + maxPagesToShow - 1);
    
    // Adjust if we're near the end
    if (endPage - startPage < maxPagesToShow - 1) {
        startPage = Math.max(1, endPage - maxPagesToShow + 1);
    }
    
    // First page
    if (startPage > 1) {
        html += `<button onclick="loadLatestPatients(1)" class="btn btn-sm btn-outline-primary">1</button>`;
        if (startPage > 2) {
            html += `<span style="padding: 0 0.5rem;">...</span>`;
        }
    }
    
    // Page buttons
    for (let i = startPage; i <= endPage; i++) {
        if (i === pagination.page) {
            html += `<button class="btn btn-sm btn-primary" disabled>${i}</button>`;
        } else {
            html += `<button onclick="loadLatestPatients(${i})" class="btn btn-sm btn-outline-primary">${i}</button>`;
        }
    }
    
    // Last page
    if (endPage < pagination.total_pages) {
        if (endPage < pagination.total_pages - 1) {
            html += `<span style="padding: 0 0.5rem;">...</span>`;
        }
        html += `<button onclick="loadLatestPatients(${pagination.total_pages})" class="btn btn-sm btn-outline-primary">${pagination.total_pages}</button>`;
    }
    
    // Next button
    if (pagination.has_next) {
        html += `<button onclick="loadLatestPatients(${pagination.page + 1})" class="btn btn-sm btn-secondary">Next »</button>`;
    } else {
        html += `<button class="btn btn-sm btn-secondary" disabled>Next »</button>`;
    }
    
    container.innerHTML = html;
}

// Page size selector event listener
document.addEventListener('DOMContentLoaded', function() {
    const pageSizeSelect = document.getElementById('page-size-select');
    if (pageSizeSelect) {
        pageSizeSelect.addEventListener('change', function() {
            const newPageSize = parseInt(this.value);
            loadLatestPatients(1, newPageSize); // Reset to page 1 when changing page size
        });
    }
});

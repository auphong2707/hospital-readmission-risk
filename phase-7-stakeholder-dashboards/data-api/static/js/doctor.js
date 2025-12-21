// Doctor Dashboard JavaScript
// Handles clinician-focused data fetching and visualization

// Current selected model
let currentModel = 'gradient_boosting';

// Initialize dashboard on page load
document.addEventListener('DOMContentLoaded', function() {
    // Set up model selector
    const modelSelect = document.getElementById('model-select');
    modelSelect.addEventListener('change', handleModelChange);
    
    // Get model from URL parameter if present
    const urlParams = new URLSearchParams(window.location.search);
    const modelParam = urlParams.get('model');
    if (modelParam) {
        currentModel = modelParam;
        modelSelect.value = currentModel;
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
            loadRiskFactors(),
            loadPerformanceClinical(),
            loadFairnessDemographics(),
            loadFairnessOverall()
        ]);
    } catch (error) {
        console.error('Error initializing dashboard:', error);
    }
}

/**
 * Panel 2: Top 20 Risk Factors Table
 */
async function loadRiskFactors() {
    try {
        showLoading('table-risk-factors');
        
        const data = await fetchJSON(`/api/v1/models/${currentModel}/risk-factors?top_n=20`);
        
        const columns = ['Rank', 'Feature', 'Importance', 'Clinical Meaning'];
        const rows = data.risk_factors.map(factor => [
            factor.rank,
            factor.feature,
            formatNumber(factor.importance, 4),
            factor.clinical_meaning || 'N/A'
        ]);
        
        createTable('table-risk-factors', columns, rows);
    } catch (error) {
        console.error('Error loading risk factors:', error);
        showError('table-risk-factors', 'Failed to load risk factors');
    }
}

/**
 * Panel 3: Model Performance (Clinical Terms) Table
 */
async function loadPerformanceClinical() {
    try {
        showLoading('table-performance-clinical');
        
        const data = await fetchJSON(`/api/v1/models/${currentModel}/performance-clinical`);
        
        const columns = ['Metric', 'Value', 'Clinical Interpretation'];
        const rows = [
            [
                'ROC-AUC',
                formatNumber(data.roc_auc, 3),
                data.clinical_interpretations.roc_auc
            ],
            [
                'Sensitivity (TPR)',
                formatPercent(data.sensitivity * 100, 1),
                data.clinical_interpretations.sensitivity
            ],
            [
                'Specificity (TNR)',
                formatPercent(data.specificity * 100, 1),
                data.clinical_interpretations.specificity
            ],
            [
                'PPV (Precision)',
                formatPercent(data.ppv * 100, 1),
                data.clinical_interpretations.ppv
            ],
            [
                'NPV',
                formatPercent(data.npv * 100, 1),
                data.clinical_interpretations.npv
            ],
            [
                'F1 Score',
                formatNumber(data.f1_score, 3),
                data.clinical_interpretations.f1_score
            ]
        ];
        
        // Define thresholds for performance metrics
        const thresholds = [
            null, // Metric name
            { green: 0.8, yellow: 0.7 }, // Value (higher is better for most)
            null  // Clinical interpretation
        ];
        
        createTable('table-performance-clinical', columns, rows, thresholds);
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

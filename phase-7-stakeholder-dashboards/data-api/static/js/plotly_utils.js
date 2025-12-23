// Plotly Utilities for Dashboard Visualizations
// Common configuration and helper functions

// Model color scheme (matching existing visualization_generator.py)
const MODEL_COLORS = {
    'gradient_boosting': '#2E7D32',  // Green
    'random_forest': '#1565C0',      // Blue
    'logistic_regression': '#C62828' // Red
};

const MODEL_NAMES = {
    'gradient_boosting': 'Gradient Boosting',
    'random_forest': 'Random Forest',
    'logistic_regression': 'Logistic Regression'
};

// Common Plotly layout for light theme
const LIGHT_LAYOUT = {
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#f8fafc',
    font: {
        color: '#0f172a',
        family: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    },
    xaxis: {
        gridcolor: '#e2e8f0',
        zerolinecolor: '#cbd5e1',
        linecolor: '#cbd5e1'
    },
    yaxis: {
        gridcolor: '#e2e8f0',
        zerolinecolor: '#cbd5e1',
        linecolor: '#cbd5e1'
    },
    legend: {
        bgcolor: 'rgba(255, 255, 255, 0.9)',
        bordercolor: '#e2e8f0',
        borderwidth: 1
    },
    autosize: true,
    margin: { l: 60, r: 40, t: 40, b: 60 }
};

// Common Plotly config
const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
};

/**
 * Create a pie chart
 */
function createPieChart(divId, labels, values, colors = null) {
    // Clear loading indicator
    const element = document.getElementById(divId);
    if (element) {
        element.innerHTML = '';
    }
    
    const data = [{
        type: 'pie',
        labels: labels,
        values: values,
        marker: {
            colors: colors || ['#ef4444', '#10b981', '#3b82f6']
        },
        textinfo: 'label+percent',
        textposition: 'auto',
        textfont: {
            size: 14,
            family: 'Arial, sans-serif',
            weight: 700
        },
        hovertemplate: '<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>',
        hole: 0
    }];

    const layout = {
        ...LIGHT_LAYOUT,
        showlegend: true,
        legend: {
            ...LIGHT_LAYOUT.legend,
            orientation: 'v',
            x: 1.02,
            y: 0.5
        }
    };

    Plotly.newPlot(divId, data, layout, PLOTLY_CONFIG);
}

/**
 * Create a grouped bar chart
 */
function createGroupedBarChart(divId, categories, data, dataLabels, colors = null) {
    // Clear loading indicator
    const element = document.getElementById(divId);
    if (element) {
        element.innerHTML = '';
    }
    
    const traces = data.map((values, idx) => ({
        type: 'bar',
        name: dataLabels[idx],
        x: categories,
        y: values,
        marker: {
            color: colors ? colors[idx] : null
        },
        hovertemplate: '<b>%{x}</b><br>%{fullData.name}: %{y:.3f}<extra></extra>'
    }));

    const layout = {
        ...LIGHT_LAYOUT,
        barmode: 'group',
        showlegend: true,
        legend: {
            ...LIGHT_LAYOUT.legend,
            orientation: 'h',
            x: 0.5,
            xanchor: 'center',
            y: -0.15
        }
    };

    Plotly.newPlot(divId, traces, layout, PLOTLY_CONFIG);
}

/**
 * Create a line chart (ROC/PR curves)
 */
function createLineChart(divId, traces, xAxisTitle = '', yAxisTitle = '', title = '') {
    // Clear loading indicator
    const element = document.getElementById(divId);
    if (element) {
        element.innerHTML = '';
    }
    
    const layout = {
        ...LIGHT_LAYOUT,
        title: title ? { text: title, font: { size: 16 } } : undefined,
        xaxis: {
            ...LIGHT_LAYOUT.xaxis,
            title: xAxisTitle
        },
        yaxis: {
            ...LIGHT_LAYOUT.yaxis,
            title: yAxisTitle
        },
        showlegend: true,
        legend: {
            ...LIGHT_LAYOUT.legend,
            x: 0.02,
            y: 0.98,
            xanchor: 'left',
            yanchor: 'top'
        }
    };

    Plotly.newPlot(divId, traces, layout, PLOTLY_CONFIG);
}

/**
 * Create an HTML table with optional color coding
 */
function createTable(divId, columns, rows, thresholds = null) {
    let html = '<table class="data-table"><thead><tr>';
    
    // Header
    columns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    // Rows
    rows.forEach(row => {
        html += '<tr>';
        row.forEach((cell, colIdx) => {
            let cellClass = '';
            
            // Apply color coding if thresholds provided
            if (thresholds && thresholds[colIdx]) {
                const threshold = thresholds[colIdx];
                const numValue = parseFloat(cell);
                
                if (!isNaN(numValue)) {
                    if (threshold.inverse) {
                        // Lower is better (e.g., Brier score)
                        if (numValue <= threshold.green) cellClass = 'cell-green';
                        else if (numValue <= threshold.yellow) cellClass = 'cell-yellow';
                        else cellClass = 'cell-red';
                    } else {
                        // Higher is better (e.g., ROC-AUC)
                        if (numValue >= threshold.green) cellClass = 'cell-green';
                        else if (numValue >= threshold.yellow) cellClass = 'cell-yellow';
                        else cellClass = 'cell-red';
                    }
                }
            }
            
            html += `<td class="${cellClass}">${cell}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    
    document.getElementById(divId).innerHTML = html;
}

/**
 * Create stat cards for metrics
 */
function createStatCards(divId, stats) {
    let html = '';
    
    stats.forEach(stat => {
        let colorClass = '';
        if (stat.color === 'green') colorClass = 'stat-green';
        else if (stat.color === 'yellow') colorClass = 'stat-yellow';
        else if (stat.color === 'red') colorClass = 'stat-red';
        else if (stat.color === 'blue') colorClass = 'stat-blue';
        
        html += `
            <div class="stat-card ${colorClass}">
                <div class="stat-label">${stat.label}</div>
                <div class="stat-value">${stat.value}</div>
                ${stat.unit ? `<div class="stat-unit">${stat.unit}</div>` : ''}
            </div>
        `;
    });
    
    document.getElementById(divId).innerHTML = html;
}

/**
 * Format number with specific decimal places
 */
function formatNumber(value, decimals = 2) {
    if (typeof value === 'number') {
        return value.toFixed(decimals);
    }
    return value;
}

/**
 * Format currency (USD)
 */
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(value);
}

/**
 * Format percentage
 */
function formatPercent(value, decimals = 1) {
    return `${(value).toFixed(decimals)}%`;
}

/**
 * Show error message in panel
 */
function showError(divId, message = 'Error loading data') {
    const element = document.getElementById(divId);
    if (element) {
        element.innerHTML = `
            <div style="padding: 20px; text-align: center; color: #ef4444;">
                <i class="fas fa-exclamation-triangle" style="font-size: 24px; margin-bottom: 10px;"></i>
                <p>${message}</p>
                <p style="font-size: 12px; color: #6b7280;">Check console for details</p>
            </div>
        `;
    }
}

/**
 * Fetch JSON from API endpoint
 */
async function fetchJSON(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${url}:`, error);
        throw error;
    }
}

/**
 * Show loading indicator
 */
function showLoading(divId) {
    // Do nothing - loading indicators removed
}

/**
 * Show error message
 */
function showError(divId, message) {
    const element = document.getElementById(divId);
    if (element) {
        element.innerHTML = `<div class="error-message">Error: ${message}</div>`;
    }
}

/**
 * Apply threshold-based coloring to a value
 */
function getColorByThreshold(value, thresholds, inverse = false) {
    if (inverse) {
        // Lower is better
        if (value <= thresholds.green) return 'green';
        if (value <= thresholds.yellow) return 'yellow';
        return 'red';
    } else {
        // Higher is better
        if (value >= thresholds.green) return 'green';
        if (value >= thresholds.yellow) return 'yellow';
        return 'red';
    }
}

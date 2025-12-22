// Global State
let currentReward = "-1";
let currentOptAlgo = "vi";
let currentTPIK = "3"; // Default k for TPI
let currentQEps = "0.1";
let currentTD = "poly_1";
let convergenceChart = null;

// Environment Constants (0-indexed)
const ROWS = 5;
const COLS = 5;
const FORBIDDEN = [
    [1, 1], [1, 2], 
    [2, 2], 
    [3, 1], [3, 3], 
    [4, 1]
];
const TARGET = [3, 2];

// Action Map
const ACTIONS = ['↑', '↓', '←', '→', '•']; // up, down, left, right, stay
const ACTION_DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]];

// Initialization
window.onload = function() {
    renderEnvGrid();
    renderPolicyGrid(); // Initial random policy
    updateOptimalityView();
    renderConvergenceChart(); // Initial chart render
    updateQView();
    updateTDView();
};

// Navigation
function showModule(modId) {
    document.querySelectorAll('.module').forEach(el => el.classList.add('hidden'));
    document.getElementById('mod-' + modId).classList.remove('hidden');
    
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    event.target.classList.add('active');
}

// Reward
function updateReward(val) {
    currentReward = val;
    // Reset views
    updateOptimalityView();
    renderConvergenceChart();
    updateQView();
    updateTDView();
}

// Environment Rendering
function isForbidden(r, c) {
    return FORBIDDEN.some(([fr, fc]) => fr === r && fc === c);
}

function isTarget(r, c) {
    return TARGET[0] === r && TARGET[1] === c;
}

function renderEnvGrid() {
    const container = document.getElementById('env-grid');
    container.innerHTML = '';
    
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            if (isForbidden(r, c)) cell.classList.add('forbidden');
            if (isTarget(r, c)) cell.classList.add('target');
            cell.innerText = `(${r},${c})`;
            cell.style.fontSize = '10px';
            container.appendChild(cell);
        }
    }
}

// Generic Grid Renderer
function renderGrid(containerId, data, type) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    // Find min/max for heatmap
    let minVal = Infinity, maxVal = -Infinity;
    if (type === 'value') {
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const val = data[r][c];
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }
        }
    }

    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            
            if (isForbidden(r, c)) {
                cell.classList.add('forbidden');
            } else if (isTarget(r, c)) {
                cell.classList.add('target');
            }

            if (type === 'value') {
                const val = data[r][c];
                cell.innerHTML = `<span class="cell-value">${val.toFixed(2)}</span>`;
                
                // Heatmap color (Orange for high, White for low)
                if (!isForbidden(r, c) && !isTarget(r, c)) {
                    const norm = (val - minVal) / (maxVal - minVal || 1);
                    // Simple orange scale
                    cell.style.backgroundColor = `rgba(255, 165, 0, ${norm})`;
                }
            } else if (type === 'policy') {
                const actionIdx = data[r][c];
                cell.innerText = ACTIONS[actionIdx];
                if (actionIdx === 4) cell.innerText = '•'; // Stay
            }
            
            container.appendChild(cell);
        }
    }
}

// Optimality Logic
function switchOptTab(algo) {
    currentOptAlgo = algo;
    document.querySelectorAll('.opt-tab').forEach(el => el.classList.remove('active'));
    event.target.classList.add('active');
    updateOptimalityView();
}

function switchTPIK(k) {
    currentTPIK = k;
    document.querySelectorAll('.tpi-k-tab').forEach(el => el.classList.remove('active'));
    event.target.classList.add('active');
    
    renderConvergenceChart(); // Always update chart
    
    // Only update bottom view if we are currently looking at TPI
    if (currentOptAlgo === 'tpi') {
        updateOptimalityView();
    }
}

function updateOptimalityView() {
    let dataObj;
    if (currentOptAlgo === 'tpi') {
        dataObj = RL_DATA[currentReward]['tpi'][currentTPIK];
    } else {
        dataObj = RL_DATA[currentReward][currentOptAlgo];
    }
    
    const frames = dataObj.frames;
    const slider = document.getElementById('iter-slider');
    const display = document.getElementById('iter-display');
    
    // Update slider range
    slider.max = frames.length - 1;
    
    // If current value is out of bounds, reset
    if (parseInt(slider.value) >= frames.length) {
        slider.value = frames.length - 1;
    }
    
    const idx = parseInt(slider.value);
    display.innerText = idx;
    
    const frame = frames[idx];
    renderGrid('opt-value-grid', frame.V, 'value');
    renderGrid('opt-policy-grid', frame.policy, 'policy');
}

function renderConvergenceChart() {
    const ctx = document.getElementById('convergenceChart').getContext('2d');
    
    // Prepare datasets
    // We want to show: VI, PI, and the currently selected TPI (or maybe all TPIs?)
    // The prompt says: "observe truncated PI approaching PI".
    // So let's show: PI (baseline) and TPI (current k).
    // Maybe VI too for context.
    
    const piData = RL_DATA[currentReward]['pi'].errors;
    const tpiData = RL_DATA[currentReward]['tpi'][currentTPIK].errors;
    
    // Labels (Iterations) - use the max length
    const maxLen = Math.max(piData.length, tpiData.length);
    const labels = Array.from({length: maxLen}, (_, i) => i);
    
    const datasets = [
        {
            label: 'Policy Iteration',
            data: piData,
            borderColor: '#0d6efd',
            backgroundColor: 'rgba(13, 110, 253, 0.1)',
            tension: 0.1
        },
        {
            label: `Truncated PI (k=${currentTPIK})`,
            data: tpiData,
            borderColor: '#ffc107',
            backgroundColor: 'rgba(255, 193, 7, 0.1)',
            tension: 0.1
        }
    ];

    if (convergenceChart) {
        convergenceChart.destroy();
    }

    convergenceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Max Error ||V - V*||'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Iteration'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Convergence Analysis'
                }
            }
        }
    });
}

// Q-Learning Logic
function switchQTab(eps) {
    currentQEps = eps;
    document.querySelectorAll('.q-tab').forEach(el => el.classList.remove('active'));
    event.target.classList.add('active');
    updateQView();
}

function updateQView() {
    const data = RL_DATA[currentReward]['q_learning'][currentQEps];
    renderGrid('q-value-grid', data.V, 'value');
    renderGrid('q-policy-grid', data.policy, 'policy');
}

// TD Linear Logic
function switchTDTab(key) {
    currentTD = key;
    document.querySelectorAll('.td-tab').forEach(el => el.classList.remove('active'));
    event.target.classList.add('active');
    updateTDView();
}

function updateTDView() {
    const data = RL_DATA[currentReward]['td_linear'][currentTD];
    // renderGrid('td-value-grid', data.V, 'value'); // Old 2D grid
    renderTD3DPlot(data.V);
}

function renderTD3DPlot(zData) {
    // zData is a 5x5 2D array
    // Rows are y (0 to 4), Cols are x (0 to 4)
    
    // Create 1-based coordinates
    const x = [1, 2, 3, 4, 5];
    const y = [1, 2, 3, 4, 5];
    
    const data = [{
        z: zData,
        x: x,
        y: y,
        type: 'surface',
        contours: {
            z: {
                show: true,
                usecolormap: true,
                highlightcolor: "#42f462",
                project: { z: true }
            },
            x: { show: true, color: '#333', width: 1 },
            y: { show: true, color: '#333', width: 1 }
        },
        colorscale: 'RdBu',
        reversescale: true
    }];

    const layout = {
        title: 'Value Function Surface',
        autosize: true,
        width: 600,
        height: 500,
        margin: {
            l: 0,
            r: 0,
            b: 0,
            t: 30,
        },
        scene: {
            xaxis: { 
                title: 'Col',
                tickvals: [1, 2, 3, 4, 5],
                ticktext: ['1', '2', '3', '4', '5']
            },
            yaxis: { 
                title: 'Row',
                tickvals: [1, 2, 3, 4, 5],
                ticktext: ['1', '2', '3', '4', '5'],
                autorange: 'reversed' // Row 1 at top/back, Row 5 at bottom/front
            },
            zaxis: { title: 'Value' },
            camera: {
                eye: {x: -1.5, y: -1.5, z: 1.5}
            }
        }
    };

    Plotly.newPlot('td-value-plot', data, layout);
}

// Policy Module Logic (Interactive)
let customPolicy = Array(ROWS).fill().map(() => Array(COLS).fill(0)); // 0=Up default

function renderPolicyGrid() {
    const container = document.getElementById('policy-grid');
    container.innerHTML = '';
    
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            if (isForbidden(r, c)) cell.classList.add('forbidden');
            if (isTarget(r, c)) cell.classList.add('target');
            
            cell.innerText = ACTIONS[customPolicy[r][c]];
            cell.onclick = () => {
                customPolicy[r][c] = (customPolicy[r][c] + 1) % 5;
                renderPolicyGrid();
            };
            container.appendChild(cell);
        }
    }
}

function setPolicyPreset(type) {
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (type === 'random') customPolicy[r][c] = Math.floor(Math.random() * 5);
            if (type === 'up') customPolicy[r][c] = 0;
            if (type === 'right') customPolicy[r][c] = 3;
        }
    }
    renderPolicyGrid();
}

function evaluatePolicy() {
    // Simple Iterative Policy Evaluation in JS
    // V(s) = R(s,a) + gamma * V(s')
    let V = Array(ROWS).fill().map(() => Array(COLS).fill(0));
    const gamma = 0.9;
    const rewardForbidden = parseInt(currentReward);
    
    for (let i = 0; i < 100; i++) { // 100 iterations
        let newV = JSON.parse(JSON.stringify(V));
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (isTarget(r, c)) {
                    // Target logic: In HW3, it's absorbing or just +1?
                    // The python code says: reward=1, done=False.
                    // So V(target) = 1 + gamma * V(target) -> V = 1/(1-gamma) = 10.
                    // But let's just run the update.
                }
                
                const actionIdx = customPolicy[r][c];
                const [dr, dc] = ACTION_DIRS[actionIdx];
                
                let nr = r + dr;
                let nc = c + dc;
                let reward = 0;
                
                // Boundary
                if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) {
                    nr = r; nc = c;
                    reward = -1; // Boundary penalty
                } else if (isForbidden(nr, nc)) {
                    reward = rewardForbidden;
                } else if (isTarget(nr, nc)) {
                    reward = 1;
                }
                
                newV[r][c] = reward + gamma * V[nr][nc];
            }
        }
        V = newV;
    }
    
    renderGrid('policy-eval-result', V, 'value');
}

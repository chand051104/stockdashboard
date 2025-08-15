
/******************************************************************************
 * Fixed enhanced_script.js — For Enhanced Stock Market Dashboard
 ******************************************************************************/

// ---------------------- Config and Global State ---------------------------
const API_BASE = "http://localhost:8000";
let currentTicker = "AAPL";
let currentPeriod = "1mo";
let currentChartType = "candlestick";
let realtimeRunning = false;
let ws = null;

// DOM helpers
const $ = sel => document.querySelector(sel);
const $$ = sel => [...document.querySelectorAll(sel)];

// UI Elements
const priceEl = $("#currentPrice");
const changeEl = $("#priceChange");
const updatedEl = $("#lastUpdated");
const periodBtns = $$(".period-btn");
const chartTypeBtns = $$(".chart-type-btn");
const realtimeBtn = $("#realtimeBtn");
const watchlistEl = $("#watchlist");
const metricsGrid = $("#metricsGrid");
const predictionsCard = $("#predictionsCard");
const predictionsEl = $("#predictions");
const toastBox = $("#toastContainer");

// ---------------------- Utilities ---------------------------
function fmt(n) {
    if (n === null || n === undefined || isNaN(n)) return "--";
    return Math.abs(n) >= 1 ? n.toFixed(2) : n.toPrecision(3);
}

function toast(msg, type = "success", ttl = 3000) {
    const node = document.createElement("div");
    node.className = `toast ${type}`;
    node.textContent = msg;
    toastBox.appendChild(node);
    requestAnimationFrame(() => node.classList.add("show"));
    setTimeout(() => {
        node.classList.remove("show");
        setTimeout(() => node.remove(), 300);
    }, ttl);
}

// ---------------------- WebSocket Real-time Quotes ---------------------------
function connectWS() {
    try {
        ws = new WebSocket(API_BASE.replace("http", "ws") + "/ws");
        ws.onopen = () => {
            toast("WebSocket connected");
            if (realtimeRunning) subscribeTicker(currentTicker);
        };
        ws.onmessage = ({ data }) => {
            const msg = JSON.parse(data);
            if (msg.type === "price_update") handlePriceUpdate(msg);
        };
        ws.onclose = () => {
            toast("WebSocket closed", "warning");
            ws = null;
        };
        ws.onerror = err => {
            toast("WS error", "error");
            console.error(err);
        };
    } catch (e) {
        console.error("WebSocket connection error:", e);
        toast("WebSocket connection failed", "error");
    }
}

function subscribeTicker(ticker) {
    if (!ws || ws.readyState !== 1) return;
    ws.send(JSON.stringify({ type: "subscribe", ticker }));
}

function unsubscribeTicker(ticker) {
    if (!ws || ws.readyState !== 1) return;
    ws.send(JSON.stringify({ type: "unsubscribe", ticker }));
}

function toggleRealTime() {
    realtimeRunning = !realtimeRunning;
    realtimeBtn.innerHTML = realtimeRunning
        ? '<i class="fas fa-stop"></i> Stop Real-time'
        : '<i class="fas fa-play"></i> Start Real-time';

    realtimeBtn.className = realtimeRunning ? 'stop' : '';

    if (realtimeRunning) {
        subscribeTicker(currentTicker);
    } else {
        unsubscribeTicker(currentTicker);
    }
}

// ---------------------- Real-time Price UI ---------------------------
function handlePriceUpdate({ ticker, data, timestamp }) {
    if (ticker !== currentTicker) return;

    const { price, change, change_percent } = data;

    if (priceEl) priceEl.textContent = `$${fmt(price)}`;
    if (changeEl) {
        changeEl.textContent = `${change >= 0 ? "+" : ""}${fmt(change)} (${fmt(change_percent)}%)`;
        changeEl.className = `price-change ${change >= 0 ? "positive" : "negative"}`;
    }
    if (updatedEl) updatedEl.textContent = `Last updated: ${new Date(timestamp).toLocaleTimeString()}`;

    // Update watchlist if shown
    const row = watchlistEl?.querySelector(`[data-ticker="${ticker}"]`);
    if (row) {
        const priceCell = row.querySelector(".watchlist-price");
        const changeCell = row.querySelector(".watchlist-change");
        if (priceCell) priceCell.textContent = fmt(price);
        if (changeCell) {
            changeCell.textContent = `${change >= 0 ? "+" : ""}${fmt(change_percent)}%`;
            changeCell.className = `watchlist-change ${change >= 0 ? "positive" : "negative"}`;
        }
    }
}

// ---------------------- Historical Data + Charting ---------------------------
async function fetchHistorical(ticker = currentTicker, period = currentPeriod) {
    const loading = $("#loading");
    if (loading) loading.classList.remove("hidden");

    try {
        const res = await fetch(`${API_BASE}/historical/${ticker}?period=${period}`);
        const json = await res.json();

        if (json.error) throw new Error(json.error);

        plotStockData(json);
        fillMetrics(json.metrics);

        // Show dashboard elements
        const stockHeader = $("#stockHeader");
        if (stockHeader) stockHeader.style.display = "flex";
        if (metricsGrid) metricsGrid.style.display = "grid";

        const welcomeCard = $("#welcomeCard");
        if (welcomeCard) welcomeCard.classList.add("hidden");

        // Update current ticker display
        const tickerEl = $("#currentTicker");
        if (tickerEl) tickerEl.textContent = ticker;

    } catch (e) {
        console.error("Fetch historical error:", e);
        toast(e.message || "Failed to load historical", "error");
    } finally {
        if (loading) loading.classList.add("hidden");
    }
}

function plotStockData(data) {
    try {
        const records = data.data;
        if (!records || records.length === 0) {
            toast("No data available", "warning");
            return;
        }

        const dates = records.map(d => d.date);
        const close = records.map(d => d.close);
        const open = records.map(d => d.open);
        const high = records.map(d => d.high);
        const low = records.map(d => d.low);

        // Main price trace
        let tracePrice;
        if (currentChartType === "candlestick") {
            tracePrice = {
                x: dates, close, open, high, low,
                type: "candlestick",
                name: currentTicker
            };
        } else {
            tracePrice = {
                x: dates, y: close,
                type: "scatter",
                fill: currentChartType === "area" ? "tozeroy" : undefined,
                mode: "lines",
                name: currentTicker
            };
        }

        // Moving averages
        const ma20 = {
            x: dates,
            y: records.map(d => d.ma20),
            type: "scatter",
            mode: "lines",
            name: "MA20",
            visible: ($("#showMA20")?.checked) ? true : "legendonly"
        };

        const ma50 = {
            x: dates,
            y: records.map(d => d.ma50),
            type: "scatter",
            mode: "lines", 
            name: "MA50",
            visible: ($("#showMA50")?.checked) ? true : "legendonly"
        };

        // Layout
        const layout = {
            height: 500,
            title: `${currentTicker} (${currentPeriod.toUpperCase()})`,
            xaxis: { title: 'Date' },
            yaxis: { title: 'Price ($)' }
        };

        // Plot main chart
        const priceChart = $("#priceChart");
        if (priceChart) {
            Plotly.newPlot("priceChart", [tracePrice, ma20, ma50], layout);
        }

        // RSI chart
        const rsi = {
            x: dates,
            y: records.map(d => d.rsi),
            type: "scatter",
            mode: "lines",
            name: "RSI"
        };

        const rsiLayout = {
            height: 300,
            title: "RSI (14)",
            shapes: [
                { type: "line", x0: dates[0], x1: dates[dates.length-1], y0: 70, y1: 70, 
                  line: { color: "red", dash: "dot" } },
                { type: "line", x0: dates[0], x1: dates[dates.length-1], y0: 30, y1: 30, 
                  line: { color: "green", dash: "dot" } }
            ],
            yaxis: { range: [0, 100] }
        };

        const technicalChart = $("#technicalChart");
        if (technicalChart) {
            technicalChart.innerHTML = "";
            Plotly.newPlot("technicalChart", [rsi], rsiLayout);
        }

        // Volume chart
        const volume = {
            x: dates,
            y: records.map(d => d.volume),
            type: "bar",
            marker: { color: "#8ab4f8" },
            name: "Volume"
        };

        const volumeChart = $("#volumeChart");
        if (volumeChart) {
            Plotly.newPlot("volumeChart", [volume], { 
                height: 200, 
                title: "Volume",
                xaxis: { title: 'Date' },
                yaxis: { title: 'Volume' }
            });
        }

    } catch (e) {
        console.error("Plot error:", e);
        toast("Error plotting data", "error");
    }
}

function fillMetrics(mets) {
    if (!mets) return;

    const elements = {
        "#periodHigh": mets.period_high,
        "#periodLow": mets.period_low, 
        "#currentVolume": mets.current_volume,
        "#currentRSI": mets.rsi,
        "#volatility": mets.volatility ? `${fmt(mets.volatility)}%` : "--",
        "#trend": mets.trend || "--"
    };

    Object.entries(elements).forEach(([selector, value]) => {
        const el = $(selector);
        if (el) {
            el.textContent = typeof value === 'number' ? fmt(value) : value;
        }
    });
}

// ---------------------- AI Predictions ---------------------------
async function getPrediction() {
    const modelSelect = $("#modelSelect");
    if (!modelSelect) return;

    const model = modelSelect.value;
    if (predictionsCard) predictionsCard.style.display = "block";
    if (predictionsEl) predictionsEl.innerHTML = "<em>Loading...</em>";

    try {
        const res = await fetch(`${API_BASE}/predict/${currentTicker}?model=${model}`);
        const json = await res.json();

        const rows = [];
        const preds = model === "all" ? json.predictions : { [model]: json.predicted_price };

        Object.entries(preds).forEach(([m, val]) => {
            if (val !== null && val !== undefined) {
                rows.push(`
                    <div class="prediction-item">
                        <span class="prediction-model">${m}</span>
                        <span class="prediction-price">$${fmt(val)}</span>
                    </div>
                `);
            }
        });

        if (predictionsEl) {
            predictionsEl.innerHTML = rows.length > 0 ? rows.join("") : "<p>No predictions available</p>";
        }

    } catch (e) {
        console.error("Prediction error:", e);
        if (predictionsEl) predictionsEl.innerHTML = "<p>Error getting predictions</p>";
        toast("Prediction failed", "error");
    }
}

// ---------------------- Sidebar / Companies ---------------------------
async function fetchCompanies() {
    try {
        const res = await fetch(`${API_BASE}/companies`);
        const companies = await res.json();

        const companiesList = $("#companiesList");
        if (!companiesList) return;

        const html = companies.map(company => {
            const ticker = typeof company === 'string' ? company : company.ticker;
            const name = typeof company === 'string' ? company : (company.name || ticker);

            return `
                <div class="company-item" onclick="setTicker('${ticker}')">
                    <span class="company-ticker">${ticker}</span>
                    <div class="company-name">${name}</div>
                </div>
            `;
        }).join("");

        companiesList.innerHTML = html;

    } catch (e) {
        console.error("Fetch companies error:", e);
    }
}

// Global function for ticker selection
window.setTicker = function(ticker) {
    if (ticker === currentTicker) return;

    // Stop real-time if running
    if (realtimeRunning) toggleRealTime();

    // Unsubscribe from old ticker
    unsubscribeTicker(currentTicker);

    // Set new ticker
    currentTicker = ticker.toUpperCase();

    // Update input field
    const tickerInput = $("#tickerInput");
    if (tickerInput) tickerInput.value = currentTicker;

    // Subscribe to new ticker
    subscribeTicker(currentTicker);

    // Fetch new data
    fetchHistorical();
};

// Global function for loading stock (from search)
window.loadStock = function() {
    const tickerInput = $("#tickerInput");
    if (!tickerInput) return;

    const ticker = tickerInput.value.trim().toUpperCase();
    if (!ticker) return;

    setTicker(ticker);
};

// Global function for getting prediction (from button)
window.getPrediction = getPrediction;

// Global function for toggling real-time (from button)
window.toggleRealTime = toggleRealTime;

// ---------------------- Event Bindings ---------------------------
document.addEventListener("DOMContentLoaded", () => {
    // Period buttons
    periodBtns.forEach(btn => {
        btn.onclick = e => {
            periodBtns.forEach(b => b.classList.remove("active"));
            e.target.classList.add("active");
            currentPeriod = e.target.dataset.period;
            fetchHistorical();
        };
    });

    // Chart type buttons  
    chartTypeBtns.forEach(btn => {
        btn.onclick = e => {
            chartTypeBtns.forEach(b => b.classList.remove("active"));
            e.target.classList.add("active");
            currentChartType = e.target.dataset.type;
            fetchHistorical();
        };
    });

    // Indicator checkboxes
    const indicators = ["#showMA20", "#showMA50", "#showBB", "#showRSI"];
    indicators.forEach(sel => {
        const el = $(sel);
        if (el) el.onchange = () => fetchHistorical();
    });

    // Search input enter key
    const tickerInput = $("#tickerInput");
    if (tickerInput) {
        tickerInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") loadStock();
        });
    }

    // Initialize
    fetchCompanies();
    fetchHistorical();
    connectWS();
    refreshMarketStatus();

    // Refresh market status every minute
    setInterval(refreshMarketStatus, 60000);
});

// ---------------------- Market Status ---------------------------
async function refreshMarketStatus() {
    try {
        const res = await fetch(`${API_BASE}/market-status`);
        const json = await res.json();

        const icon = $("#marketStatusIcon");
        const text = $("#marketStatusText");
        const status = $("#marketStatus");

        if (json.is_open) {
            if (icon) icon.className = "fas fa-circle";
            if (text) text.textContent = "Market Open";
            if (status) {
                status.classList.add("open");
                status.classList.remove("closed");
            }
        } else {
            if (icon) icon.className = "fas fa-circle";
            if (text) text.textContent = "Market Closed";  
            if (status) {
                status.classList.add("closed");
                status.classList.remove("open");
            }
        }
    } catch (e) {
        console.error("Market status error:", e);
    }
}

// ---------------------- Optional Watchlist Functions ---------------------------
function addToWatchlist(ticker) {
    if (!watchlistEl) return;
    if (watchlistEl.querySelector(`[data-ticker="${ticker}"]`)) return;

    const row = document.createElement("div");
    row.className = "watchlist-item";
    row.dataset.ticker = ticker;
    row.innerHTML = `
        <span class="watchlist-ticker">${ticker}</span>
        <span class="watchlist-price">--</span>
        <span class="watchlist-change">--</span>
    `;
    watchlistEl.appendChild(row);
    subscribeTicker(ticker);
}

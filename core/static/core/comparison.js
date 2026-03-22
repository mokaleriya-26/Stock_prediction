document.addEventListener("DOMContentLoaded", () => {
    const compareBtn = document.querySelector(".compare-btn");
    compareBtn.addEventListener("click", compareStocks);
});

let comparisonChart;

// Friendly short names for NIFTY tickers
const TICKER_NAMES = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "SBIN.NS": "State Bank of India",
    "BAJFINANCE.NS": "Bajaj Finance",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "WIPRO.NS": "Wipro",
};

async function compareStocks() {
    const select = document.getElementById("tickerSelect");
    const compareBtn = document.querySelector(".compare-btn");
    const loadingDiv = document.getElementById("comparisonLoading");

    const selectedStocks = Array.from(select.options)
        .filter(option => option.selected)
        .map(option => option.value);

    if (selectedStocks.length < 2) {
        alert("Please select at least 2 stocks.");
        return;
    }

    compareBtn.disabled = true;
    compareBtn.textContent = "Comparing...";
    loadingDiv.style.display = "flex";

    // Hide verdict from previous run
    document.getElementById("verdictBlock").style.display = "none";

    try {
        let datasets = [];
        let allStats = [];

        for (let ticker of selectedStocks) {
            const response = await fetch(`/api/predict/${ticker}/`);
            const data = await response.json();

            datasets.push({
                label: ticker.replace(".NS", ""),
                data: data.historical_graph_data.prices,
                borderWidth: 2,
                tension: 0.3,
                fill: false
            });

            allStats.push({
                ticker: ticker.replace(".NS", ""),
                fullTicker: ticker,
                name: TICKER_NAMES[ticker] || ticker.replace(".NS", ""),
                open: data.key_stats.open,
                close: data.key_stats.close,
                marketCap: data.key_stats.market_cap,
                pe: data.key_stats.pe_ratio,
                forwardPe: data.key_stats.forward_pe,
                beta: data.key_stats.beta,
                volume: data.key_stats.volume
            });
        }

        // --- CHART ---
        const ctx = document.getElementById("comparisonChart");
        if (comparisonChart) comparisonChart.destroy();

        comparisonChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: datasets[0].data.map((_, i) => i + 1),
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: "white" } }
                },
                scales: {
                    x: { ticks: { color: "rgba(255,255,255,0.6)" } },
                    y: { ticks: { color: "rgba(255,255,255,0.6)" } }
                }
            }
        });

        // --- COMPARISON TABLE ---
        const statsContainer = document.getElementById("comparisonStats");
        statsContainer.innerHTML = `
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    ${allStats.map(s => `<th>${s.ticker}</th>`).join("")}
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Open</td>
                    ${allStats.map(s => `<td>₹${s.open != null ? s.open.toFixed(2) : "-"}</td>`).join("")}
                </tr>
                <tr>
                    <td>Close</td>
                    ${allStats.map(s => `<td>₹${s.close != null ? s.close.toFixed(2) : "-"}</td>`).join("")}
                </tr>
                <tr>
                    <td>Market Cap</td>
                    ${allStats.map(s => `<td>${s.marketCap?.toLocaleString() || "-"}</td>`).join("")}
                </tr>
                <tr>
                    <td>P/E Ratio</td>
                    ${allStats.map(s => `<td>${s.pe?.toFixed(2) || "-"}</td>`).join("")}
                </tr>
                <tr>
                    <td>Forward P/E</td>
                    ${allStats.map(s => `<td>${s.forwardPe?.toFixed(2) || "-"}</td>`).join("")}
                </tr>
                <tr>
                    <td>Beta</td>
                    ${allStats.map(s => `<td>${s.beta?.toFixed(2) || "-"}</td>`).join("")}
                </tr>
                <tr>
                    <td>Volume</td>
                    ${allStats.map(s => `<td>${s.volume?.toLocaleString() || "-"}</td>`).join("")}
                </tr>
            </tbody>
        </table>`;

        // --- VERDICT ---
        await renderVerdict(allStats);

    } catch (error) {
        console.error(error);
        alert("Error comparing stocks.");
    } finally {
        compareBtn.disabled = false;
        compareBtn.textContent = "Compare Selected Stocks";
        loadingDiv.style.display = "none";
        const resultLine = document.getElementById("selectedStocksLine");
        resultLine.textContent = "Comparing: " + selectedStocks.map(t => t.replace(".NS", "")).join(", ");
        resultLine.style.display = "flex";
    }
}

function calculateScore(stock) {
    let score = 0;

    // 1. P/E Ratio Score (25)
    if (stock.pe != null && stock.pe > 0) {
        if (stock.pe <= 15) score += 25;
        else if (stock.pe <= 25) score += 20;
        else if (stock.pe <= 35) score += 10;
        else score += 5;
    }

    // 2. Forward P/E vs Current P/E Score (20)
    if (
        stock.pe != null && stock.pe > 0 &&
        stock.forwardPe != null && stock.forwardPe > 0
    ) {
        const diffPercent = ((stock.pe - stock.forwardPe) / stock.pe) * 100;

        if (diffPercent >= 20) score += 20;
        else if (diffPercent >= 10) score += 15;
        else if (diffPercent >= 0) score += 10;
        else score += 5;
    }

    // 3. Beta Score (20)
    if (stock.beta != null) {
        if (stock.beta < 0) score += 8;
        else if (stock.beta <= 0.8) score += 20;
        else if (stock.beta <= 1.2) score += 15;
        else if (stock.beta <= 1.6) score += 8;
        else score += 3;
    }

    // 4. Market Cap Score (20)
    if (stock.marketCap != null && stock.marketCap > 0) {
        if (stock.marketCap >= 5_000_000_000_000) score += 20;
        else if (stock.marketCap >= 1_000_000_000_000) score += 16;
        else if (stock.marketCap >= 500_000_000_000) score += 10;
        else score += 5;
    }

    // 5. Volume Score (15)
    if (stock.volume != null && stock.volume > 0) {
        if (stock.volume >= 10_000_000) score += 15;
        else if (stock.volume >= 5_000_000) score += 12;
        else if (stock.volume >= 1_000_000) score += 8;
        else score += 4;
    }

    return Math.min(100, Math.max(0, Math.round(score)));
}

function formatMarketCap(value) {
    if (!value || value <= 0) return "-";
    const trillion = 1_000_000_000_000;
    const billion = 1_000_000_000;
    if (value >= trillion) return `₹${(value / trillion).toFixed(1)}T`;
    if (value >= billion) return `₹${(value / billion).toFixed(1)}B`;
    return `₹${value.toLocaleString()}`;
}

function getCSRFToken() {
    const cookie = document.cookie
        .split("; ")
        .find(row => row.startsWith("csrftoken="));
    return cookie ? cookie.split("=")[1] : "";
}

async function getAIExplanation(sortedStocks, winner) {
    const fallbackExplanation = `
        Based on valuation, risk, liquidity, and company size metrics,
        <strong>${winner.ticker}</strong> ranks highest among the selected stocks.
        The stock performed better on the scoring model across P/E, Forward P/E,
        Beta, Market Cap, and Trading Volume.
    `;

    try {
        const response = await fetch("/api/generate-verdict-explanation/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCSRFToken()
            },
            body: JSON.stringify({
                winner: winner,
                stocks: sortedStocks
            })
        });

        const data = await response.json();

        if (response.ok && data.success && data.explanation) {
            return data.explanation;
        }

        return fallbackExplanation;
    } catch (error) {
        console.error("AI explanation error:", error);
        return fallbackExplanation;
    }
}

async function renderVerdict(allStats) {
    const scored = allStats.map(s => ({ ...s, score: calculateScore(s) }));
    const sorted = [...scored].sort((a, b) => b.score - a.score);
    const winner = sorted[0];

    const aiExplanation = await getAIExplanation(sorted, winner);

    const cards = sorted.map((s, index) => {
        const isWinner = winner && s.ticker === winner.ticker;
        return `
        <div class="verdict-card ${isWinner ? "winner" : ""}">
            <div class="verdict-rank">#${index + 1}</div>
            ${isWinner ? `<div class="verdict-badge">Top Pick</div>` : ""}

            <div class="verdict-main">
                <div class="verdict-ticker">${s.ticker}</div>
                <div class="verdict-score">${s.score}/100</div>
            </div>

            <div class="verdict-score-bar-wrap">
                <div class="verdict-score-bar" data-pct="${s.score}" style="width:0%"></div>
            </div>
            <div class="verdict-score-label">
                <span>Investment Score</span>
                <span>${s.score}/100</span>
            </div>

            <div class="verdict-meta">
                <div class="verdict-meta-row"><span>P/E</span><span>${s.pe != null ? s.pe.toFixed(1) : "-"}</span></div>
                <div class="verdict-meta-row"><span>Fwd P/E</span><span>${s.forwardPe != null ? s.forwardPe.toFixed(1) : "-"}</span></div>
                <div class="verdict-meta-row"><span>Beta</span><span>${s.beta != null ? s.beta.toFixed(2) : "-"}</span></div>
                <div class="verdict-meta-row"><span>Market Cap</span><span>${formatMarketCap(s.marketCap)}</span></div>
                <div class="verdict-meta-row"><span>Volume</span><span>${s.volume != null ? s.volume.toLocaleString() : "-"}</span></div>
            </div>
        </div>`;
    }).join("");

    const formattedExplanation = aiExplanation
        ? aiExplanation.replace(/\n/g, "<br>")
        : "No explanation available.";

    const block = document.getElementById("verdictBlock");
    document.getElementById("verdictContent").innerHTML = `
        <div class="verdict-grid">${cards}</div>
        <div class="verdict-summary">
            ${formattedExplanation}
        </div>
        <div class="verdict-disclaimer">
            This score is for informational purposes only and is not financial advice.
            Always conduct your own research before investing.
        </div>`;

    block.style.display = "block";

    requestAnimationFrame(() => {
        setTimeout(() => {
            block.querySelectorAll(".verdict-score-bar").forEach(bar => {
                bar.style.width = bar.dataset.pct + "%";
            });
        }, 80);
    });
}
document.addEventListener("DOMContentLoaded", () => {
    const compareBtn = document.querySelector(".compare-btn");
    compareBtn.addEventListener("click", compareStocks);
});

let comparisonChart;

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
    // SHOW LOADING STATE
    compareBtn.disabled = true;
    compareBtn.textContent = "Comparing...";
    loadingDiv.style.display = "block";
    try {
        let datasets = [];
        let allStats = [];
        for (let ticker of selectedStocks) {
            const response = await fetch(`/api/predict/${ticker}/`);
            const data = await response.json();
            datasets.push({
                label: ticker,
                data: data.historical_graph_data.prices,
                borderWidth: 2,
                tension: 0.3,
                fill: false
            });
            allStats.push({
                ticker: ticker,
                price: data.key_stats.current_price,
                marketCap: data.key_stats.market_cap,
                pe: data.key_stats.pe_ratio,
                forwardPe: data.key_stats.forward_pe,
                beta: data.key_stats.beta,
                volume: data.key_stats.volume
            });
        }
        // DRAW CHART
        const ctx = document.getElementById("comparisonChart");

        if (comparisonChart) {
            comparisonChart.destroy();
        }

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
                    legend: {
                        labels: { color: "white" }
                    }
                },
                scales: {
                    x: { ticks: { color: "rgba(255,255,255,0.6)" } },
                    y: { ticks: { color: "rgba(255,255,255,0.6)" } }
                }
            }
        });
        // BUILD COMPARISON TABLE
        const statsContainer = document.getElementById("comparisonStats");
        statsContainer.innerHTML = "";

        let tableHTML = `
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    ${allStats.map(stock => `<th>${stock.ticker}</th>`).join("")}
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Current Price</td>
                    ${allStats.map(s => `<td>₹${s.price?.toFixed(2) || "-"}</td>`).join("")}
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
        </table>
        `;
        statsContainer.innerHTML = tableHTML;
    } catch (error) {
        console.error(error);
        alert("Error comparing stocks.");
    } finally {
        // HIDE LOADING STATE
        compareBtn.disabled = false;
        compareBtn.textContent = "Compare Selected Stocks";
        loadingDiv.style.display = "none";
        const resultLine = document.getElementById("selectedStocksLine");
        resultLine.textContent =
            "Comparing stocks: " + selectedStocks.join(", ");
        resultLine.style.display = "block";
    }
}
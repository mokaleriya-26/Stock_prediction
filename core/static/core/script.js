// your_app/static/js/script.js
console.log("SCRIPT JS LOADED");
// Using window.onload ensures all elements are fully loaded before execution
document.addEventListener("DOMContentLoaded", function () {
    
    // Check if the container exists
    const slideshowContainer = document.querySelector('.slideshow-container');
    if (!slideshowContainer) {
        console.error('Slideshow container not found!');
        return; 
    }
    
    // Configuration
    const slides = document.querySelectorAll('.hero-banner');
    const totalSlides = slides.length; 
    const slideWidth = 100; // Represents 100vw
    let currentSlide = 0;

    function nextSlide() {
        // Increment the slide index, wrapping around to the first slide (0)
        currentSlide = (currentSlide + 1) % totalSlides;
        
        // Calculate the translation offset in 'vw' units
        const offset = -currentSlide * slideWidth;
        
        // Apply the movement
        slideshowContainer.style.transform = `translateX(${offset}vw)`;
    }

    // Start the automatic slideshow timer: advances every 5 seconds (5000ms)
    // We delay the start slightly to ensure the initial transform is stable
    setTimeout(() => {
        setInterval(nextSlide, 5000); 
    }, 100); 
});
// ==============================
// LIVE DATA LOADER (ADD BELOW)
// ==============================
let liveIndex = 0;
let liveDataStore = [];
async function loadTopCompanies() {

    const tickers = [
        "HDFCBANK.NS",
        "RELIANCE.NS",
        "TCS.NS",
        "ICICIBANK.NS"
    ];

    const companies = document.querySelectorAll(".company");

    let allData = [];   // ✅ MOVED OUTSIDE

    try {

        for (let i = 0; i < tickers.length; i++) {

            const response = await fetch(`/api/predict/${tickers[i]}/`);
            if (!response.ok) continue;

            const data = await response.json();

            allData.push(data);

            const companyCard = companies[i];
            if (!companyCard) continue;

            const tick = companyCard.querySelector(".tick");
            const chg = companyCard.querySelector(".chg");
            const info = companyCard.querySelector(".company-info");

            if (tick)
                tick.innerText =
                    `₹${Number(data.key_stats.last_close).toFixed(3)}`;

            if (chg)
                chg.innerText = "LIVE";

            if (info)
                info.innerText =
                    `Volume: ${Number(data.key_stats.volume).toLocaleString()}`;
        }

        liveDataStore = allData;   // ✅ NOW SAFE

    } catch (error) {
        console.error("Error loading top companies:", error);
    }
}
function updateLiveInsight() {

    if (!liveDataStore || liveDataStore.length === 0) return;

    const best = liveDataStore[liveIndex];

    const liveTitle = document.getElementById("live-title");
    const livePrice = document.getElementById("live-price");
    const liveCanvas = document.getElementById("liveChart");

    liveTitle.innerText = `🔥 ${best.ticker}`;
    livePrice.innerText =
        `₹${Number(best.key_stats.last_close).toFixed(3)}`;

    if (window.liveChartObj) {
        window.liveChartObj.destroy();
    }

    window.liveChartObj = new Chart(liveCanvas, {
        type: "line",
        data: {
            labels: best.historical_graph_data.dates,
            datasets: [{
                data: best.historical_graph_data.prices,
                borderColor: "white",
                borderWidth: 2,
                tension: 0.3,
                fill: false,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    ticks: { color: "white" },
                    grid: { color: "rgba(255,255,255,0.15)" }
                },
                y: {
                    ticks: { color: "white" },
                    grid: { color: "rgba(255,255,255,0.15)" }
                }
            }
        }
    });

    // NEXT COMPANY
    liveIndex = (liveIndex + 1) % liveDataStore.length;
}
document.addEventListener("DOMContentLoaded", async () => {

    await loadTopCompanies();

    updateLiveInsight();          // first company
    setInterval(updateLiveInsight, 4000);  // slideshow
});
// ===== ARROW BUTTONS =====
document.addEventListener("DOMContentLoaded", () => {

    const nextBtn = document.getElementById("nextLive");
    const prevBtn = document.getElementById("prevLive");

    if (nextBtn) {
        nextBtn.addEventListener("click", () => {
            updateLiveInsight();
        });
    }

    if (prevBtn) {
        prevBtn.addEventListener("click", () => {
            liveIndex =
                (liveIndex - 2 + liveDataStore.length)
                % liveDataStore.length;
            updateLiveInsight();
        });
    }
});
// your_app/static/js/script.js
console.log("SCRIPT JS LOADED");

document.addEventListener("DOMContentLoaded", function () {
    
    // Check if the container exists
    const slideshowContainer = document.querySelector('.slideshow-container');
    if (!slideshowContainer) {
        return; 
    }
    
    const slides = document.querySelectorAll('.hero-banner');
    const totalSlides = slides.length; 
    const slideWidth = 100;
    let currentSlide = 0;

    function nextSlide() {
        currentSlide = (currentSlide + 1) % totalSlides;
        const offset = -currentSlide * slideWidth;
        slideshowContainer.style.transform = `translateX(${offset}vw)`;
    }

    setTimeout(() => {
        setInterval(nextSlide, 5000); 
    }, 100); 
});

// ==============================
// LIVE DATA LOADER
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
    let allData = [];

    try {
        for (let i = 0; i < tickers.length; i++) {
            const response = await fetch(`/api/predict/${tickers[i]}/`);
            if (!response.ok) continue;

            const data = await response.json();
            allData.push(data);

            const companyCard = companies[i];
            if (!companyCard) continue;

            const tick = companyCard.querySelector(".tick");
            const chg  = companyCard.querySelector(".chg");
            const info = companyCard.querySelector(".company-info");

            if (tick) tick.innerText = `₹${Number(data.key_stats.last_close).toFixed(3)}`;
            if (chg)  chg.innerText  = "LIVE";
            if (info) info.innerText = `Volume: ${Number(data.key_stats.volume).toLocaleString()}`;
        }

        liveDataStore = allData;

    } catch (error) {
        console.error("Error loading top companies:", error);
    }
}

function updateLiveInsight() {
    if (!liveDataStore || liveDataStore.length === 0) return;

    const best = liveDataStore[liveIndex];

    const liveTitle  = document.getElementById("live-title");
    const livePrice  = document.getElementById("live-price");
    const liveCanvas = document.getElementById("liveChart");

    liveTitle.innerText = `🔥 ${best.ticker}`;
    livePrice.innerText = `₹${Number(best.key_stats.last_close).toFixed(3)}`;

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
                    grid:  { color: "rgba(255,255,255,0.15)" }
                },
                y: {
                    ticks: { color: "white" },
                    grid:  { color: "rgba(255,255,255,0.15)" }
                }
            }
        }
    });

    liveIndex = (liveIndex + 1) % liveDataStore.length;
}

document.addEventListener("DOMContentLoaded", async () => {
    await loadTopCompanies();
    updateLiveInsight();
    setInterval(updateLiveInsight, 4000);
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
            liveIndex = (liveIndex - 2 + liveDataStore.length) % liveDataStore.length;
            updateLiveInsight();
        });
    }
});


// ==============================
// SCROLL-TRIGGERED FADE-IN
// ==============================
document.addEventListener("DOMContentLoaded", () => {
    const fadeTargets = document.querySelectorAll(
        ".showcase-card, .about-value-item, .about-stat, .feature-card, .flow-step"
    );

    // Set initial hidden state
    fadeTargets.forEach((el, i) => {
        el.style.opacity = "0";
        el.style.transform = "translateY(28px)";
        el.style.transition = `opacity 0.55s ease ${(i % 4) * 0.08}s, transform 0.55s ease ${(i % 4) * 0.08}s`;
    });

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = "1";
                entry.target.style.transform = "translateY(0)";
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.12 });

    fadeTargets.forEach(el => observer.observe(el));
});


// ==============================
// PROGRESS BAR SCROLL ANIMATION
// ==============================
document.addEventListener("DOMContentLoaded", () => {
    const bars = document.querySelectorAll(".mock-bar-fill, .mock-compare-bar");

    // Store target widths then reset to 0
    bars.forEach(bar => {
        bar.dataset.targetWidth = bar.style.width;
        bar.style.width = "0%";
        bar.style.transition = "width 1.1s cubic-bezier(0.25, 1, 0.5, 1)";
    });

    const barObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.width = entry.target.dataset.targetWidth;
                barObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.2 });

    bars.forEach(bar => barObserver.observe(bar));
});


// ==============================
// STAT NUMBER COUNT-UP
// ==============================
document.addEventListener("DOMContentLoaded", () => {
    const stats = document.querySelectorAll(".about-stat__number");

    function countUp(el, target, duration = 1400) {
        // Only animate purely numeric values
        if (isNaN(parseFloat(target))) return;

        const isFloat  = target.includes(".");
        const suffix   = target.replace(/[\d.]/g, ""); // e.g. "+"
        const numTarget = parseFloat(target);
        const start    = performance.now();

        function update(now) {
            const elapsed  = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased    = 1 - Math.pow(1 - progress, 3);
            const current  = eased * numTarget;
            el.textContent = (isFloat ? current.toFixed(1) : Math.floor(current)) + suffix;
            if (progress < 1) requestAnimationFrame(update);
        }

        requestAnimationFrame(update);
    }

    const statObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el     = entry.target;
                const rawText = el.textContent.trim();
                countUp(el, rawText);
                statObserver.unobserve(el);
            }
        });
    }, { threshold: 0.5 });

    stats.forEach(el => statObserver.observe(el));
});


// ==============================
// AUTO-CYCLE BUY/SELL/HOLD BADGES
// ==============================
document.addEventListener("DOMContentLoaded", () => {
    const signals = [
        { ticker: "HDFC",  signal: "BUY",  type: "buy",  icon: "↑" },
        { ticker: "INFY",  signal: "SELL", type: "sell", icon: "↓" },
        { ticker: "REL",   signal: "HOLD", type: "hold", icon: "→" },
        { ticker: "TCS",   signal: "BUY",  type: "buy",  icon: "↑" },
        { ticker: "WIPRO", signal: "SELL", type: "sell", icon: "↓" },
        { ticker: "ICICI", signal: "HOLD", type: "hold", icon: "→" },
        { ticker: "BAJAJ", signal: "BUY",  type: "buy",  icon: "↑" },
        { ticker: "MARUTI",signal: "SELL", type: "sell", icon: "↓" },
        { ticker: "ONGC",  signal: "HOLD", type: "hold", icon: "→" },
    ];

    const badges = document.querySelectorAll(".mock-signal-badge");
    if (!badges.length) return;

    let signalOffset = 0;

    function rotateBadges() {
        badges.forEach((badge, i) => {
            const s = signals[(signalOffset + i) % signals.length];

            // Fade out
            badge.style.transition = "opacity 0.3s ease";
            badge.style.opacity    = "0";

            setTimeout(() => {
                badge.className = `mock-signal-badge mock-signal-badge--${s.type}`;
                badge.querySelector(".mock-signal-badge__icon").textContent  = s.icon;
                badge.querySelector(".mock-signal-badge__label").textContent  = s.signal;
                badge.style.opacity = "1";
            }, 300);
        });

        signalOffset = (signalOffset + badges.length) % signals.length;
    }

    // Badges are fixed - no cycling
    // setInterval(rotateBadges, 3000);
});
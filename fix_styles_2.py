import re

with open("templates/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 1. Typography Font Import
old_fonts = r'<link href="https://fonts.googleapis.com/css2\?family=Inter:wght@400;500;600;700&family=JetBrains\+Mono:wght@400;500&display=swap" rel="stylesheet">'
new_fonts = """<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">"""
html = re.sub(old_fonts, new_fonts, html)

# 2. CSS Variables
html = re.sub(r'--font-sans:.*?;', '--font-sans: "Inter", system-ui, sans-serif;', html)
html = re.sub(r'--font-mono:.*?;', '--font-mono: "JetBrains Mono", monospace;\n  --font-display: "Outfit", "Inter", sans-serif;', html)

# 3. New CSS rules
css_append = """
/* Constrain images */
#original-image-display,
#heatmap-image-display,
.image-preview img,
#image-result img,
#heatmap-result img,
img.result-image {
  width: 100%;
  max-height: 340px;
  object-fit: contain;
  border-radius: var(--radius-md);
  background: var(--bg-primary);
  display: block;
}

.image-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-top: 16px;
}

.image-card-inner {
  width: 100%;
  min-height: 200px;
  max-height: 360px;
  overflow: hidden;
  border-radius: var(--radius-md);
  background: var(--bg-primary);
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Typography Overrides */
.logo-title {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 15px;
  letter-spacing: -0.3px;
}

.card-header {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  font-weight: 500;
}

#audio-answer-text,
#image-answer-text,
.answer-text {
  font-family: var(--font-display);
  font-size: 18px;
  font-weight: 600;
  line-height: 1.6;
  color: var(--accent-primary);
}

.summary-output-text {
  font-family: var(--font-sans);
  font-size: 15px;
  line-height: 1.75;
  color: var(--text-primary);
  font-weight: 400;
}

.tfidf-text {
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.65;
}

.evidence-text {
  font-family: var(--font-sans);
  font-style: italic;
  font-size: 13px;
  color: var(--text-secondary);
}

.source-text {
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1.5;
}

textarea, input[type="text"] {
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.6;
}

#transcript-text {
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1.9;
}

.timestamp-btn {
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 500;
}

.lime-word {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 500;
}

.shap-chip {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
}

.nav-item {
  font-family: var(--font-sans);
  font-size: 14px;
  font-weight: 500;
  letter-spacing: 0.1px;
}

.btn-primary {
  font-family: var(--font-display);
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.3px;
}

#topbar-title {
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 16px;
}

#topbar-badge {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 1.5px;
}

.tfidf-pct {
  font-family: var(--font-mono);
  font-weight: 700;
  font-size: 14px;
}

.contribution-pct {
  font-family: var(--font-mono);
  font-weight: 700;
  font-size: 14px;
}

.legend-text {
  font-family: var(--font-mono);
  font-size: 12px;
}

.legend-pct {
  font-family: var(--font-mono);
  font-weight: 600;
  font-size: 12px;
}

/* Heatmap Legend */
.heatmap-legend {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 10px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-muted);
}
.heatmap-gradient {
  flex: 1;
  height: 6px;
  border-radius: 3px;
  background: linear-gradient(
    to right, #0000ff, #8800ff, #ff0000
  );
}
</style>
"""
html = html.replace("</style>", css_append)

# 4. Modify image display HTML
# Wrap grid-2 into image-grid
img_grid_old = """          <div class="grid-2">
            <div class="card fade-in fade-in-1">
              <div class="card-header"><span class="card-dot dot-summary"></span> ORIGINAL IMAGE</div>
              <img id="image-original" class="image-result-preview" src="" alt="Original">
            </div>
            <div class="card fade-in fade-in-2">
              <div class="card-header"><span class="card-dot dot-shap"></span> ATTENTION HEATMAP</div>
              <img id="image-heatmap" class="image-result-preview" src="" alt="Heatmap">
            </div>
          </div>"""
img_grid_new = """          <div class="image-grid">
            <div class="card fade-in fade-in-1">
              <div class="card-header"><span class="card-dot dot-summary"></span> ORIGINAL IMAGE</div>
              <div class="image-card-inner">
                <img id="image-original" class="image-result-preview" src="" alt="Original">
              </div>
            </div>
            <div class="card fade-in fade-in-2">
              <div class="card-header"><span class="card-dot dot-shap"></span> ATTENTION HEATMAP</div>
              <div class="image-card-inner">
                <img id="image-heatmap" class="image-result-preview" src="" alt="Heatmap">
              </div>
              <div class="heatmap-legend">
                <span>Low</span>
                <div class="heatmap-gradient"></div>
                <span>High</span>
              </div>
            </div>
          </div>"""
html = html.replace(img_grid_old, img_grid_new)

# 5. Canvas updates
html = re.sub(r'<canvas id="donut-chart".*?></canvas>', '<canvas id="donut-chart" style="max-width:320px; max-height:320px; width:100%; height:auto;"></canvas>', html)
html = re.sub(r'<canvas id="audio-donut-chart".*?></canvas>', '<canvas id="audio-donut-chart" style="max-width:320px; max-height:320px; width:100%; height:auto;"></canvas>', html)

# 6. JS Chart configuration updates
chart_js_old = """        options: {
            responsive: false,
            cutout: "65%",
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const d = chartData[ctx.dataIndex];
                            return [
                                ` ${d.contribution}% contribution`,
                                ` ${d.reason}`
                            ];
                        }
                    },
                    backgroundColor: "rgba(24, 24, 27, 0.95)",
                    titleFont: { family: "'Space Grotesk', sans-serif", size: 13 },
                    bodyFont: { family: "'Inter', sans-serif", size: 12 },
                    titleColor: "#818cf8",
                    bodyColor: "#f4f4f5",
                    borderColor: "rgba(255,255,255,0.1)",
                    borderWidth: 1,
                    padding: 12,
                    boxPadding: 6
                }
            },
            animation: { animateRotate: true, duration: 800 }
        }"""
        
chart_js_new = """        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: "68%",
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#16161f",
                    titleColor: "#7c6fff",
                    bodyColor: "#9090b0",
                    borderColor: "#2a2a3d",
                    borderWidth: 1,
                    padding: 14,
                    cornerRadius: 8,
                    callbacks: {
                        title: (items) => chartData[items[0].dataIndex].label,
                        label: (ctx) => {
                            const d = chartData[ctx.dataIndex];
                            return [
                                ` ${d.contribution}% contribution`,
                                ` ${d.reason}`
                            ];
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 900,
                easing: "easeInOutQuart"
            }
        }"""
        
# Audio chart js old is slightly different:
audio_chart_js_old = """        options: {
            responsive: false,
            cutout: "65%",
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const d = chartData[ctx.dataIndex];
                            return [
                                ` ${d.contribution}% ?" ${d.timestamp_label}`,
                                ` ${d.reason}`
                            ];
                        }
                    },
                    backgroundColor: "rgba(24, 24, 27, 0.95)",
                    titleFont: { family: "'Space Grotesk', sans-serif", size: 13 },
                    bodyFont: { family: "'Inter', sans-serif", size: 12 },
                    titleColor: "#818cf8",
                    bodyColor: "#f4f4f5",
                    borderColor: "rgba(255,255,255,0.1)",
                    borderWidth: 1,
                    padding: 12,
                    boxPadding: 6
                }
            },
            animation: { animateRotate: true, duration: 800 }
        }"""

audio_chart_js_new = """        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: "68%",
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#16161f",
                    titleColor: "#7c6fff",
                    bodyColor: "#9090b0",
                    borderColor: "#2a2a3d",
                    borderWidth: 1,
                    padding: 14,
                    cornerRadius: 8,
                    callbacks: {
                        title: (items) => chartData[items[0].dataIndex].label,
                        label: (ctx) => {
                            const d = chartData[ctx.dataIndex];
                            return [
                                ` ${d.contribution}% \u2192 ${d.timestamp_label}`,
                                ` ${d.reason}`
                            ];
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 900,
                easing: "easeInOutQuart"
            }
        }"""

html = html.replace(chart_js_old, chart_js_new)
html = html.replace(audio_chart_js_old, audio_chart_js_new)

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Updated extra components.")

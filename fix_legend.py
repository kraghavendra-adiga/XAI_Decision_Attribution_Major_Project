import re

with open("templates/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Fix the renderDonutChart legend (it should use "chart-legend" and not have timestamp_label)
legend_bad = """    const legend = document.getElementById("audio-chart-legend");
    legend.innerHTML = chartData.map((d, i) => `
        <div class="legend-item">
            <span class="legend-dot" 
                  style="background:${colors[i]}"></span>
            <span class="legend-text">${d.label}</span>
            <span class="legend-pct">${d.contribution}%</span>
            <span class="legend-ts">${d.timestamp_label}</span>
        </div>
    `).join("");"""

legend_good = """    const legend = document.getElementById("chart-legend");
    legend.innerHTML = chartData.map((d, i) => `
        <div class="legend-item">
            <span class="legend-dot" 
                  style="background:${colors[i]}"></span>
            <span class="legend-text">${d.label}</span>
            <span class="legend-pct">${d.contribution}%</span>
        </div>
    `).join("");"""

# Replace the first instance (which is in renderDonutChart)
# If it's broken, it will replace it.
html = html.replace(legend_bad, legend_good, 1)

# Ensure the second instance (in renderAudioDonutChart) is correct (it should be audio-chart-legend)
# If the second instance is missing or something, we can check.

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Legend fixed.")

import re

with open("templates/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 1. Chart.js CDN
html = html.replace(
    '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>',
    '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>'
)

# 2. Fix Canvas Sizing
canvas_chart_pattern = r'<div style="position: relative; width: 100%; max-width: 320px; aspect-ratio: 1; margin: 0 auto;">\s*<canvas id="donut-chart"></canvas>\s*</div>'
canvas_chart_new = """<canvas id="donut-chart" 
            width="300" 
            height="300"
            style="display:block; margin:0 auto; 
                   max-width:300px; max-height:300px;">
    </canvas>"""
html = re.sub(canvas_chart_pattern, canvas_chart_new, html, flags=re.DOTALL)

audio_canvas_chart_pattern = r'<div style="position: relative; width: 100%; max-width: 320px; aspect-ratio: 1; margin: 0 auto;">\s*<canvas id="audio-donut-chart"></canvas>\s*</div>'
audio_canvas_chart_new = """<canvas id="audio-donut-chart" 
            width="300" 
            height="300"
            style="display:block; margin:0 auto;
                   max-width:300px; max-height:300px;">
    </canvas>"""
html = re.sub(audio_canvas_chart_pattern, audio_canvas_chart_new, html, flags=re.DOTALL)

# Also handle if the wrapper wasn't added due to python error, just replace the canvas itself
html = re.sub(r'<canvas id="donut-chart" style="max-width:320px; max-height:320px; width:100%; height:auto;"></canvas>', canvas_chart_new, html)
html = re.sub(r'<canvas id="audio-donut-chart" style="max-width:320px; max-height:320px; width:100%; height:auto;"></canvas>', audio_canvas_chart_new, html)

# 3. RequestAnimationFrame
img_chart_old = """    if (data.chart_data && data.chart_data.length > 0) {
        document.getElementById("chart-section").style.display = "block";
        renderDonutChart(data.chart_data);
    } else {"""
img_chart_new = """    if (data.chart_data && data.chart_data.length > 0) {
        const chartSection = document.getElementById("chart-section");
        chartSection.style.display = "block";
        requestAnimationFrame(() => {
            renderDonutChart(data.chart_data);
        });
    } else {"""
html = html.replace(img_chart_old, img_chart_new)

# Audio requestAnimationFrame
# The existing audio chart section might be:
audio_chart_old = """            // Render donut chart
            renderAudioDonutChart(data.chart_data);
            document.getElementById("audio-chart-section")
                .removeAttribute("hidden");
            showToast("Analysis complete");"""
audio_chart_new = """            // Render donut chart
            const audioChartSection = document.getElementById("audio-chart-section");
            audioChartSection.removeAttribute("hidden");
            audioChartSection.style.display = "block";
            requestAnimationFrame(() => {
                renderAudioDonutChart(data.chart_data);
            });
            showToast("Analysis complete");"""
html = html.replace(audio_chart_old, audio_chart_new)


# 4. Try/catch in chart renderers
chart_try_old = """    donutChart = new Chart(ctx, {"""
chart_try_new = """    try {
        donutChart = new Chart(ctx, {"""
html = html.replace(chart_try_old, chart_try_new)

chart_catch_old = """            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 900,
                easing: "easeInOutQuart"
            }
        }
    });
    const legend = document.getElementById("chart-legend");"""
# Need to make sure we close the try catch correctly
html = re.sub(
    r'easing: "easeInOutQuart"\s*\}\s*\}\s*\}\s*\);\s*const legend = document.getElementById\("?(?:audio-)?chart-legend"?\);',
    lambda m: m.group(0).replace("});\n", "});\n    } catch(err) {\n        console.error('Chart render error:', err);\n    }\n"),
    html
)
# Wait, let's just do it cleanly for both charts:
def add_try_catch(content, chart_var):
    pattern = r'({}\s*=\s*new\s*Chart\s*\(\s*ctx\s*,[\s\S]*?\}\s*\)\s*;)'.replace("{}", chart_var)
    return re.sub(pattern, r'try {\n        \1\n    } catch(err) {\n        console.error("Chart render error:", err);\n    }', content)

# First undo any partial replacements if we did them (we didn't write to file yet so it's fine).
html = html.replace(chart_try_new, chart_try_old) 
html = add_try_catch(html, "donutChart")
html = add_try_catch(html, "audioDonutChart")


# 5. Fix Image CSS
# We will use re.sub to find the existing image CSS blocks and replace them.
# Our previous CSS had:
# #original-image-display, ... img.result-image { width: 100%; height: 340px; object-fit: contain; ... }
# .image-card-inner { width: 100%; height: 340px; overflow: hidden; ... }

# Just to be safe, append the new CSS to the end of <style> to override anything else.
css_overrides = """
/* Image Fixes - Override */
#original-image-display,
#heatmap-image-display,
.image-card-inner img,
.result-image,
#image-result img,
#heatmap-result img {
    width: 100%;
    height: auto;
    max-height: none;
    object-fit: contain;
    border-radius: var(--radius-md);
    display: block;
}

.image-card-inner {
    width: 100%;
    height: auto;
    max-height: none;
    overflow: visible;
    border-radius: var(--radius-md);
    background: var(--bg-primary);
    display: flex;
    align-items: flex-start;
    justify-content: center;
}

.image-grid .card {
    height: auto;
    min-height: 0;
}

@media (max-width: 768px) {
    .image-grid {
        grid-template-columns: 1fr;
    }
}
</style>
"""
html = html.replace("</style>", css_overrides)

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Fixes applied successfully.")

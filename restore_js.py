import re

# 1. Read index.html
with open('templates/index.html', 'r', encoding='utf-8') as f:
    index_html = f.read()

# 2. Read missing JS from script.js
with open('templates/script.js', 'r', encoding='utf-8') as f:
    script_js = f.read()

match = re.search(r'(async function runSummarize\(\).*)$', script_js, re.DOTALL)
if not match:
    print("Could not find runSummarize in script.js")
    exit(1)

missing_js = match.group(1)

# 3. Apply add_styles.py replacements to missing_js
missing_js = missing_js.replace("document.getElementById('sum-results').classList.add('show');", "document.getElementById('sum-results').classList.add('show');\n    showToast('Analysis complete');")
missing_js = missing_js.replace("document.getElementById('qa-results').classList.add('show');", "document.getElementById('qa-results').classList.add('show');\n    showToast('Analysis complete');")
missing_js = missing_js.replace("document.getElementById('image-results').classList.add('show');", "document.getElementById('image-results').classList.add('show');\n    showToast('Analysis complete');")
missing_js = missing_js.replace('document.getElementById("audio-chart-section")\n                .removeAttribute("hidden");', 'document.getElementById("audio-chart-section")\n                .removeAttribute("hidden");\n            showToast("Analysis complete");')

source_old = """          <div class="contribution-bar">
            <div class="bar-track">
              <div class="bar-fill" style="width:${s.contribution}%"></div>
            </div>"""
source_new = """          <div class="contribution-bar-wrap">
            <div class="contribution-bar-bg">
              <div class="contribution-bar-fill" style="width:${s.contribution}%"></div>
            </div>"""
missing_js = missing_js.replace(source_old, source_new)

# 4. Apply fix_charts.py replacements to missing_js (for audio chart)
new_audio_options = """        options: {
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
                                ` ${d.contribution}% \\\\u2192 ${d.timestamp_label}`,
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
missing_js = re.sub(
    r'options:\s*\{\s*responsive:\s*false,\s*cutout:\s*"65%",.*?animation:\s*\{\s*animateRotate:\s*true,\s*duration:\s*800\s*\}\s*\}',
    new_audio_options,
    missing_js,
    count=1,
    flags=re.DOTALL
)

# 5. Inject back into index.html
index_html = index_html.replace('</script>\n</body>', '\n\n' + missing_js + '\n</script>\n</body>')
index_html = index_html.replace('</script>\r\n</body>', '\n\n' + missing_js + '\n</script>\n</body>')
# Also handle the case where there is no newline
index_html = index_html.replace('</script></body>', '\n\n' + missing_js + '\n</script></body>')

with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(index_html)

print("Restored missing JS functions.")

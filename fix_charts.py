import re

with open("templates/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Replace donut chart options
new_options = """        options: {
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

# Use regex to replace the options block inside renderDonutChart
# We look for `donutChart = new Chart` down to `options: { ... }`
html = re.sub(
    r'options:\s*\{\s*responsive:\s*false,\s*cutout:\s*"65%",.*?animation:\s*\{\s*animateRotate:\s*true,\s*duration:\s*800\s*\}\s*\}',
    new_options,
    html,
    count=1,
    flags=re.DOTALL
)

# Replace audio donut chart options
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

# Replace the second options block (for audio chart)
html = re.sub(
    r'options:\s*\{\s*responsive:\s*false,\s*cutout:\s*"65%",.*?animation:\s*\{\s*animateRotate:\s*true,\s*duration:\s*800\s*\}\s*\}',
    new_audio_options,
    html,
    count=1,
    flags=re.DOTALL
)


with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Updated chart options via regex.")

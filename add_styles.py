import re

with open("templates/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Replace CSS from Image Tab Specifics to the end of style
new_css = """/* Image Tab Specifics */
.image-preview { width: 100%; border-radius: var(--radius-md); border: 1px solid var(--border-color); max-height: 200px; object-fit: contain; }
.image-result-preview { width: 100%; border-radius: var(--radius-md); border: 1px solid var(--border-color); }

/* Source contribution cards */
.source-card {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 14px 16px;
  margin-bottom: 10px;
}
.source-text {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 10px;
  font-style: italic;
}
.contribution-bar-wrap {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
}
.contribution-bar-bg {
  flex: 1;
  height: 6px;
  background: var(--bg-secondary);
  border-radius: 3px;
  overflow: hidden;
}
.contribution-bar-fill {
  height: 100%;
  background: var(--accent-secondary);
  border-radius: 3px;
  transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}
.contribution-pct {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 700;
  color: var(--text-primary);
  min-width: 40px;
  text-align: right;
}
.source-reason {
  font-size: 12px;
  color: var(--text-muted);
}

/* Timestamp evidence cards */
.timestamp-card {
  background: var(--bg-primary);
  border-radius: var(--radius-md);
  border-left: 3px solid var(--accent-secondary);
  padding: 12px 16px;
  margin-bottom: 10px;
  transition: border-color var(--transition);
}
.timestamp-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}
.timestamp-btn {
  background: rgba(86,207,178,0.1);
  color: var(--accent-secondary);
  border: 1px solid rgba(86,207,178,0.3);
  border-radius: var(--radius-sm);
  padding: 4px 12px;
  cursor: pointer;
  font-family: var(--font-mono);
  font-size: 13px;
  transition: all var(--transition);
}
.timestamp-btn:hover {
  background: rgba(86,207,178,0.2);
  border-color: var(--accent-secondary);
}
.contribution-badge {
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 700;
  color: var(--text-primary);
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 20px;
  padding: 2px 10px;
}
.evidence-text {
  font-size: 13px;
  color: var(--text-secondary);
  font-style: italic;
  margin-bottom: 4px;
  line-height: 1.5;
}
.evidence-reason {
  font-size: 12px;
  color: var(--text-muted);
}

/* Chart section wrapper */
#chart-section, #audio-chart-section {
  margin-top: 16px;
}
#donut-chart, #audio-donut-chart {
  display: block;
  margin: 0 auto;
}
#chart-legend, #audio-chart-legend {
  margin-top: 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  padding: 6px 8px;
  border-radius: var(--radius-sm);
  transition: background var(--transition);
}
.legend-item:hover { background: var(--bg-card-hover); }
.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}
.legend-text {
  color: var(--text-secondary);
  flex: 1;
  font-family: var(--font-mono);
  font-size: 12px;
}
.legend-pct {
  color: var(--text-primary);
  font-weight: 700;
  font-family: var(--font-mono);
  font-size: 12px;
  min-width: 40px;
  text-align: right;
}
.legend-ts {
  color: var(--accent-info);
  font-family: var(--font-mono);
  font-size: 11px;
  min-width: 60px;
  text-align: right;
}

/* Audio player wrapper */
#audio-player-section {
  margin: 12px 0;
}
#audio-player {
  width: 100%;
  border-radius: var(--radius-md);
  accent-color: var(--accent-primary);
}

/* Transcript section */
#transcript-text {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.8;
  white-space: pre-wrap;
  max-height: 200px;
  overflow-y: auto;
  padding: 12px;
  background: var(--bg-primary);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-color);
}

/* Toast notification */
#toast {
  position: fixed;
  top: 20px;
  right: 20px;
  background: var(--bg-card);
  border: 1px solid var(--accent-secondary);
  border-radius: var(--radius-md);
  padding: 12px 20px;
  color: var(--accent-secondary);
  font-size: 13px;
  font-weight: 600;
  box-shadow: var(--shadow-card);
  transform: translateX(120%);
  transition: transform 0.3s ease;
  z-index: 1000;
}
#toast.show { transform: translateX(0); }

/* Tab transitions */
.tab-panel {
  animation: fadeIn 0.15s ease-out;
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

/* Responsive breakpoints */
@media (max-width: 1024px) {
  .results-grid { grid-template-columns: repeat(2,1fr); }
}
@media (max-width: 768px) {
  #sidebar { display: none; }
  .results-grid, .results-grid-2 { 
    grid-template-columns: 1fr; 
  }
  #content { padding: 16px; }
}
</style>"""
html = re.sub(r'/\* Image Tab Specifics \*/[\s\S]*?</style>', new_css, html)

# Replace old toast container with new
html = html.replace('<div id="toast-container"></div>', '<div id="toast"></div>')

# Replace old toast function
old_toast_js = """function showToast(msg) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = 'custom-toast';
  toast.innerHTML = `<span class="icon">\u26a0\ufe0f</span><span>${msg}</span>`;
  container.appendChild(toast);
  setTimeout(() => { if (toast.parentNode) toast.remove(); }, 4000);
}"""

# In case the regex didn't perfectly match the emoji unicode, let's use a simpler regex
new_toast_js = """function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = "\u2713 " + message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 3000);
}"""
html = re.sub(r'function showToast\(msg\) \{[\s\S]*?\}', new_toast_js, html)

# Inject showToast("Analysis complete") at the end of successful blocks
html = html.replace("document.getElementById('sum-results').classList.add('show');", "document.getElementById('sum-results').classList.add('show');\n    showToast('Analysis complete');")
html = html.replace("document.getElementById('qa-results').classList.add('show');", "document.getElementById('qa-results').classList.add('show');\n    showToast('Analysis complete');")
html = html.replace("document.getElementById('image-results').classList.add('show');", "document.getElementById('image-results').classList.add('show');\n    showToast('Analysis complete');")
# For audio it doesn't have a single .show container but sets removes 'hidden' from multiple sections.
html = html.replace('document.getElementById("audio-chart-section")\n                .removeAttribute("hidden");', 'document.getElementById("audio-chart-section")\n                .removeAttribute("hidden");\n            showToast("Analysis complete");')

# Update JS strings for source cards to match new CSS classes
source_old = """          <div class="contribution-bar">
            <div class="bar-track">
              <div class="bar-fill" style="width:${s.contribution}%"></div>
            </div>"""
source_new = """          <div class="contribution-bar-wrap">
            <div class="contribution-bar-bg">
              <div class="contribution-bar-fill" style="width:${s.contribution}%"></div>
            </div>"""
html = html.replace(source_old, source_new)

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Updated extra components.")

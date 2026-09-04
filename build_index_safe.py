import re
import os

with open("templates/script.js", "r", encoding="utf-8") as f:
    js_content = f.read()

# Remove the old switchTab function as we will use the new event listener approach
js_content = re.sub(r'function switchTab\(name, el\) \{.*?\n\}', '', js_content, flags=re.DOTALL)

# HTML template
html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Decision Attribution</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg-primary: #0a0a0f;
  --bg-secondary: #111118;
  --bg-card: #16161f;
  --bg-card-hover: #1c1c28;
  --border-color: #2a2a3d;
  --border-accent: #3d3d5c;
  --accent-primary: #7c6fff;
  --accent-secondary: #56cfb2;
  --accent-danger: #ff6b8a;
  --accent-warning: #ffb86c;
  --accent-info: #6bceff;
  --text-primary: #f0f0ff;
  --text-secondary: #9090b0;
  --text-muted: #55556a;
  --lime-color: #56cfb2;
  --shap-color: #7c6fff;
  --tfidf-color: #ffb86c;
  --font-sans: "Inter", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", "Fira Code", monospace;
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --shadow-card: 0 4px 24px rgba(0,0,0,0.4);
  --transition: 0.18s ease;
}

html.light-theme {
  --bg-primary: #f4f4fa;
  --bg-secondary: #ffffff;
  --bg-card: #ffffff;
  --bg-card-hover: #f0f0f8;
  --border-color: #e0e0f0;
  --border-accent: #c8c8e8;
  --text-primary: #111128;
  --text-secondary: #444466;
  --text-muted: #9090aa;
}

body {
  margin: 0;
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: var(--font-sans);
  height: 100vh;
  overflow: hidden;
}
*, *::before, *::after { box-sizing: border-box; }

#app {
  display: flex;
  height: 100vh;
}
#sidebar {
  width: 220px;
  flex-shrink: 0;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  padding: 20px 12px;
  gap: 8px;
}
#logo {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 8px 20px;
  border-bottom: 1px solid var(--border-color);
  margin-bottom: 8px;
}
.logo-icon {
  font-size: 24px;
  color: var(--accent-primary);
}
.logo-title {
  font-weight: 700;
  font-size: 14px;
  color: var(--text-primary);
}
.logo-sub {
  font-size: 11px;
  color: var(--text-muted);
}
.nav-item {
  width: 100%;
  text-align: left;
  background: none;
  border: none;
  border-radius: var(--radius-md);
  padding: 10px 14px;
  color: var(--text-secondary);
  font-size: 14px;
  font-family: var(--font-sans);
  cursor: pointer;
  transition: all var(--transition);
  border-left: 3px solid transparent;
}
.nav-item:hover {
  background: var(--bg-card);
  color: var(--text-primary);
}
.nav-item.active {
  background: rgba(124,111,255,0.12);
  border-left-color: var(--accent-primary);
  color: var(--accent-primary);
  font-weight: 600;
}
#sidebar-bottom {
  margin-top: auto;
}
#theme-toggle {
  width: 100%;
  background: none;
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  color: var(--text-muted);
  padding: 8px;
  cursor: pointer;
  font-size: 13px;
  transition: all var(--transition);
}
#theme-toggle:hover {
  border-color: var(--accent-primary);
  color: var(--accent-primary);
}
#main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
#topbar {
  height: 56px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 28px;
  flex-shrink: 0;
}
#topbar-title {
  font-weight: 600;
  font-size: 15px;
}
#topbar-badge {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--accent-primary);
  background: rgba(124,111,255,0.12);
  border: 1px solid rgba(124,111,255,0.3);
  border-radius: 20px;
  padding: 4px 12px;
}
#content {
  flex: 1;
  overflow-y: auto;
  padding: 28px;
}

/* Card Styles */
.card, .result-box {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  padding: 20px 24px;
  box-shadow: var(--shadow-card);
  transition: border-color var(--transition);
}
.card:hover, .result-box:hover {
  border-color: var(--border-accent);
}
.card-header {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-secondary);
  font-weight: 500;
}
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-lime { background: var(--lime-color); }
.dot-shap { background: var(--shap-color); }
.dot-tfidf { background: var(--tfidf-color); }
.dot-answer { background: var(--accent-primary); }
.dot-summary { background: var(--accent-secondary); }

/* Inputs */
textarea, input[type=text] {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: 14px;
  padding: 14px 16px;
  width: 100%;
  transition: border-color var(--transition), box-shadow var(--transition);
  outline: none;
}
textarea:focus, input[type=text]:focus {
  border-color: var(--accent-primary);
  box-shadow: 0 0 0 3px rgba(124,111,255,0.15);
}
textarea { min-height: 140px; resize: vertical; }

input[type=file] {
  width: 100%; background: var(--bg-primary); border: 2px dashed var(--border-accent);
  color: var(--text-secondary); font-family: var(--font-mono); font-size: 14px; text-align: center;
  border-radius: var(--radius-md); padding: 24px; transition: border-color var(--transition); cursor: pointer;
}
input[type=file]:hover { border-color: var(--accent-primary); }

.char-counter {
  color: var(--text-muted); font-size: 12px; font-family: var(--font-mono); text-align: right; margin-top: 4px;
}

/* Button */
.run-btn {
  background: linear-gradient(135deg, var(--accent-primary), #5a4fcf);
  color: white; border: none; border-radius: var(--radius-md);
  padding: 12px 28px; font-size: 14px; font-weight: 600; cursor: pointer;
  width: 100%; transition: opacity var(--transition), transform var(--transition);
  margin-top: 16px;
}
.run-btn:hover:not(:disabled) { opacity: 0.9; transform: translateY(-1px); }
.run-btn:active:not(:disabled) { transform: translateY(0); }
.run-btn:disabled {
  opacity: 0.7; cursor: not-allowed;
}

@keyframes spin { 100% { transform: rotate(360deg); } }
.spinner {
  border: 2px solid rgba(255,255,255,0.3);
  border-top-color: white;
  border-radius: 50%;
  width: 16px;
  height: 16px;
  animation: spin 0.7s linear infinite;
  display: inline-block;
  vertical-align: middle;
  margin-right: 8px;
}
.loader { display: none; text-align: center; padding: 2rem; color: var(--text-muted); font-size: 14px; font-family: var(--font-mono); }
.loader.show { display: block; }

/* Grid Layouts */
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin-bottom: 24px; }
.grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 24px; margin-bottom: 24px; }
.results { display: none; flex-direction: column; gap: 24px; margin-top: 24px; }
.results.show { display: flex; }

@media(max-width: 1024px) {
  .grid-3 { grid-template-columns: repeat(2, 1fr); }
}
@media(max-width: 768px) {
  .grid-3, .grid-2 { grid-template-columns: 1fr; }
  #sidebar { width: 70px; padding: 12px 8px; }
  .logo-title, .logo-sub, #theme-toggle { display: none; }
  .nav-item { text-align: center; padding: 12px 0; font-size: 0; }
  .nav-item::first-letter { font-size: 20px; }
}

/* LIME */
.lime-row { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; padding: 4px; border-radius: 4px; transition: background var(--transition); }
.lime-row:hover { background: var(--bg-card-hover); }
.lime-word { font-family: var(--font-mono); font-size: 13px; width: 120px; flex-shrink: 0; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lime-bar-wrap { flex: 1; height: 8px; background: transparent; border-radius: 4px; display: flex; align-items: center; }
.lime-bar { height: 100%; border-radius: 4px; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1); }
.lime-val { font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); width: 60px; text-align: right; flex-shrink: 0; }

/* SHAP */
.shap-tokens { display: flex; flex-wrap: wrap; gap: 8px; line-height: 1; }
.shap-tok {
  font-family: var(--font-mono); font-size: 12px;
  padding: 4px 10px; border-radius: 16px; cursor: default;
  transition: opacity 0.4s ease;
}

/* Sentences (TF-IDF) */
.sent-list { display: flex; flex-direction: column; gap: 12px; }
.sent-row { display: flex; gap: 12px; align-items: flex-start; }
.sent-score { font-family: var(--font-mono); font-size: 13px; color: var(--tfidf-color); font-weight: bold; width: 45px; flex-shrink: 0; padding-top: 2px; }
.sent-text { font-size: 14px; line-height: 1.6; color: var(--text-primary); }
.sent-bar { height: 2px; border-radius: 2px; margin-top: 6px; background: var(--tfidf-color); opacity: 0.8; }

/* Specifics */
.summary-text, .answer-text { font-size: 15px; line-height: 1.7; color: var(--text-primary); }
.answer-text { color: var(--accent-primary); font-size: 18px; font-weight: 500; }
.confidence { font-size: 12px; color: var(--text-muted); margin-top: 8px; font-family: var(--font-mono); }

/* Animations */
@keyframes fadeInUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
.results.show .card, .results.show .result-box { animation: fadeInUp 0.4s ease forwards; opacity: 0; }
.results.show > *:nth-child(1) { animation-delay: 0ms; }
.results.show > *:nth-child(2) { animation-delay: 80ms; }
.results.show > *:nth-child(3) { animation-delay: 160ms; }
.results.show > *:nth-child(4) { animation-delay: 240ms; }

/* Image Tab Specifics */
.image-preview { width: 100%; border-radius: var(--radius-md); border: 1px solid var(--border-color); max-height: 200px; object-fit: contain; }
.image-result-preview { width: 100%; border-radius: var(--radius-md); border: 1px solid var(--border-color); }
.source-card {
  background: var(--bg-card-hover); border-radius: var(--radius-md); padding: 16px; margin-bottom: 12px;
}
.source-text { font-family: var(--font-mono); font-size: 13px; color: rgba(240,240,255,0.7); line-height: 1.5; margin-bottom: 8px; }
.contribution-bar { display: flex; align-items: center; gap: 12px; height: 12px; margin-bottom: 8px; }
.bar-track { flex: 1; height: 6px; background: rgba(86,207,178,0.1); border-radius: 3px; }
.bar-fill { height: 100%; border-radius: 3px; background: var(--accent-secondary); transition: width 0.6s ease; }
.contribution-pct { font-family: var(--font-mono); font-size: 13px; font-weight: 600; color: #fff; width: 40px; text-align: right; }
.source-reason { font-size: 12px; color: var(--text-muted); line-height: 1.4; }

/* Audio Specifics */
.timestamp-card { display: flex; flex-direction: column; gap: 8px; padding: 16px; background: var(--bg-card-hover); border-radius: var(--radius-md); margin-bottom: 12px; }
.timestamp-header { display: flex; align-items: center; gap: 12px; }
.timestamp-btn { background: rgba(86,207,178,0.1); border: 1px solid var(--accent-secondary); color: var(--accent-secondary); border-radius: 16px; padding: 4px 12px; cursor: pointer; font-family: var(--font-mono); font-size: 12px; transition: all var(--transition); }
.timestamp-btn:hover { background: rgba(86,207,178,0.2); }
.evidence-text { color: var(--text-muted); font-style: italic; font-size: 13px; }
.evidence-reason { color: var(--text-muted); font-size: 12px; }
.contribution-badge { font-weight: 600; font-size: 13px; color: #fff; }

/* Chart */
#chart-section, #audio-chart-section { display: none; }
.legend-item { display: flex; align-items: center; gap: 10px; font-size: 13px; margin-bottom: 8px; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
.legend-text { color: var(--text-secondary); flex: 1; font-family: var(--font-mono); font-size: 12px; }
.legend-pct { color: #ffffff; font-weight: bold; width: 40px; text-align: right; }
.legend-ts { color: var(--accent-secondary); font-family: var(--font-mono); font-size: 11px; width: 60px; text-align: right; }

/* Toast */
#toast-container { position: fixed; top: 24px; right: 24px; z-index: 9999; display: flex; flex-direction: column; gap: 12px; }
.custom-toast { background: var(--bg-card); border: 1px solid var(--border-color); border-left: 4px solid var(--accent-secondary); border-radius: var(--radius-sm); padding: 16px; color: var(--text-primary); font-size: 14px; display: flex; align-items: center; gap: 12px; box-shadow: var(--shadow-card); animation: toastSlide 0.3s ease forwards, toastFade 0.3s ease forwards 2.7s; }
@keyframes toastSlide { from { opacity: 0; transform: translateX(100%); } to { opacity: 1; transform: translateX(0); } }
@keyframes toastFade { to { opacity: 0; transform: translateX(100%); } }

/* Tab transitions */
.tab-panel {
  animation: fadeIn 0.15s ease-out;
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

</style>
</head>
<body>

<div id="app">
  <aside id="sidebar">
    <div id="logo">
      <span class="logo-icon">\u2726</span>
      <div>
        <div class="logo-title">Decision Attribution</div>
        <div class="logo-sub">Explainable Insights</div>
      </div>
    </div>
    <nav id="sidebar-nav">
      <button class="nav-item active" data-tab="summarize">\u25c8 Summarize</button>
      <button class="nav-item" data-tab="qa">\u25ce Ask</button>
      <button class="nav-item" data-tab="image">\u25a3 Image</button>
      <button class="nav-item" data-tab="audio">\u25c9 Audio</button>
    </nav>
    <div id="sidebar-bottom">
      <button id="theme-toggle">\u263d Theme</button>
    </div>
  </aside>

  <div id="main">
    <header id="topbar">
      <span id="topbar-title">Summarize</span>
      <span id="topbar-badge">LIME \u00b7 SHAP \u00b7 TF-IDF</span>
    </header>
    
    <div id="content">
      
      <!-- SUMMARIZE TAB -->
      <div class="tab-panel" id="tab-summarize">
        <div class="card" style="margin-bottom: 24px;">
          <div class="card-header"><span class="dot dot-summary"></span> INPUT</div>
          <textarea id="sum-text" placeholder="Paste a lecture excerpt, textbook paragraph, or article here..."></textarea>
          <div class="char-counter">0 chars</div>
          <button class="run-btn" id="sum-btn" onclick="runSummarize()">\u25b6 Run Summarization + Explain</button>
        </div>
        <div class="loader" id="sum-loader"><span class="spinner"></span> Running models...</div>
        
        <div class="results" id="sum-results">
          <div class="grid-3">
            <div class="card">
              <div class="card-header"><span class="dot dot-summary"></span> SUMMARY OUTPUT</div>
              <div class="summary-text" id="sum-output"></div>
            </div>
            <div class="card">
              <div class="card-header"><span class="dot dot-lime"></span> LIME ATTRIBUTION</div>
              <div id="sum-lime"></div>
            </div>
            <div class="card">
              <div class="card-header"><span class="dot dot-shap"></span> SHAP TOKENS</div>
              <div class="shap-tokens" id="sum-shap"></div>
            </div>
          </div>
          <div class="card">
            <div class="card-header"><span class="dot dot-tfidf"></span> SENTENCE ATTRIBUTION (TF-IDF)</div>
            <div class="sent-list" id="sum-sent"></div>
          </div>
        </div>
      </div>

      <!-- QA TAB -->
      <div class="tab-panel" id="tab-qa" hidden>
        <div class="grid-2">
          <div class="card">
            <div class="card-header"><span class="dot dot-summary"></span> CONTEXT</div>
            <textarea id="qa-context" placeholder="Paste the context document here..."></textarea>
          </div>
          <div class="card">
            <div class="card-header"><span class="dot dot-summary"></span> QUESTION</div>
            <input type="text" id="qa-question" placeholder="Ask a question about the context...">
            <button class="run-btn" id="qa-btn" onclick="runQA()">\u25b6 Get Answer + Explain</button>
          </div>
        </div>
        <div class="loader" id="qa-loader"><span class="spinner"></span> Processing...</div>

        <div class="results" id="qa-results">
          <div class="grid-3">
            <div class="card">
              <div class="card-header"><span class="dot dot-answer"></span> ANSWER</div>
              <div class="answer-text" id="qa-answer"></div>
              <div class="confidence" id="qa-conf"></div>
            </div>
            <div class="card">
              <div class="card-header"><span class="dot dot-lime"></span> LIME ATTRIBUTION</div>
              <div id="qa-lime"></div>
            </div>
            <div class="card">
              <div class="card-header"><span class="dot dot-shap"></span> SHAP TOKENS</div>
              <div class="shap-tokens" id="qa-shap"></div>
            </div>
          </div>
          <div class="card">
            <div class="card-header"><span class="dot dot-tfidf"></span> SENTENCE ATTRIBUTION</div>
            <div class="sent-list" id="qa-sent"></div>
          </div>
        </div>
      </div>

      <!-- IMAGE TAB -->
      <div class="tab-panel" id="tab-image" hidden>
        <div class="grid-2">
          <div class="card">
            <div class="card-header"><span class="dot dot-summary"></span> UPLOAD IMAGE</div>
            <input type="file" id="image-file" accept="image/png, image/jpeg">
          </div>
          <div class="card">
            <div class="card-header"><span class="dot dot-summary"></span> QUESTION</div>
            <input type="text" id="image-question" placeholder="What is shown in this image?">
            <button class="run-btn" id="image-btn" onclick="runImageVQA()">\u25b6 Analyze Image</button>
          </div>
        </div>
        <div class="loader" id="image-loader"><span class="spinner"></span> Processing...</div>

        <div class="results" id="image-results">
          <div class="card" id="image-error-box" style="display:none; border-color: var(--accent-danger);">
             <div style="color: var(--accent-danger);" id="image-error-msg"></div>
          </div>
          <div class="grid-2">
            <div class="card">
              <div class="card-header"><span class="dot dot-summary"></span> ORIGINAL IMAGE</div>
              <img id="image-original" class="image-result-preview" src="" alt="Original">
            </div>
            <div class="card">
              <div class="card-header"><span class="dot dot-shap"></span> ATTENTION HEATMAP</div>
              <img id="image-heatmap" class="image-result-preview" src="" alt="Heatmap">
            </div>
          </div>
          <div class="card">
            <div class="card-header"><span class="dot dot-answer"></span> ANSWER</div>
            <div class="answer-text" id="image-answer"></div>
          </div>
          <div class="card" id="image-evidence-box" style="display:none;">
            <div class="card-header"><span class="dot dot-tfidf"></span> SUPPORTING EVIDENCE</div>
            <div class="sent-list" id="image-evidence"></div>
          </div>
          <div class="card" id="sources-section" style="display:none;">
            <div class="card-header"><span class="dot dot-shap"></span> SOURCES & CONTRIBUTION</div>
            <div class="source-cards" id="image-sources"></div>
          </div>
          <div class="card" id="chart-section">
            <div class="card-header"><span class="dot dot-lime"></span> SOURCE CONTRIBUTION CHART</div>
            <canvas id="donut-chart" width="380" height="380" style="display: block; margin: 0 auto;"></canvas>
            <div id="chart-legend"></div>
          </div>
        </div>
      </div>

      <!-- AUDIO TAB -->
      <div class="tab-panel" id="tab-audio" hidden>
        <div class="grid-2">
          <div class="card">
            <div class="card-header"><span class="dot dot-summary"></span> UPLOAD AUDIO</div>
            <input type="file" id="audio-file" accept="audio/*">
            <div id="audio-player-section" hidden style="margin-top: 16px;">
              <audio id="audio-player" controls style="width:100%"></audio>
            </div>
          </div>
          <div class="card">
            <div class="card-header"><span class="dot dot-summary"></span> QUESTION</div>
            <input type="text" id="audio-question" placeholder="What is being explained here?">
            <button class="run-btn" id="run-audio-btn">\u25b6 Transcribe + Explain</button>
          </div>
        </div>
        
        <div class="results" id="audio-results-container">
          <div class="card" id="audio-answer-section" hidden>
            <div class="card-header"><span class="dot dot-answer"></span> ANSWER</div>
            <div class="answer-text" id="audio-answer-text"></div>
          </div>
          <div class="card" id="timestamp-evidence-section" hidden>
            <div class="card-header"><span class="dot dot-shap"></span> TIMESTAMP EVIDENCE</div>
            <div id="timestamp-evidence-list"></div>
          </div>
          <div class="card" id="transcript-section" hidden>
            <div class="card-header"><span class="dot dot-summary"></span> FULL TRANSCRIPT</div>
            <div class="summary-text" id="transcript-text" style="font-size: 13px; font-family: var(--font-mono); max-height: 200px; overflow-y: auto;"></div>
          </div>
          <div class="card" id="audio-chart-section" hidden>
            <div class="card-header"><span class="dot dot-lime"></span> SOURCE CONTRIBUTION CHART</div>
            <canvas id="audio-donut-chart" width="380" height="380" style="display: block; margin: 0 auto;"></canvas>
            <div id="audio-chart-legend"></div>
          </div>
        </div>
      </div>

    </div>
  </div>
</div>

<div id="toast-container"></div>

<script>
// Tab Switching
document.querySelectorAll(".nav-item").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item")
      .forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    const tab = btn.dataset.tab;
    document.getElementById("topbar-title").textContent = 
      btn.textContent.trim().replace(/^./, "").trim();
    document.querySelectorAll(".tab-panel")
      .forEach(p => p.hidden = true);
    document.getElementById("tab-" + tab).hidden = false;
  });
});

// Theme Toggle
document.getElementById("theme-toggle")
  .addEventListener("click", () => {
    document.documentElement.classList.toggle("light-theme");
    localStorage.setItem("theme",
      document.documentElement.classList.contains("light-theme")
        ? "light" : "dark"
    );
  });
if (localStorage.getItem("theme") === "light") {
  document.documentElement.classList.add("light-theme");
}

""" + js_content + """
</script>
</body>
</html>"""

# Fix weird encoding artifacts in script.js just in case
html_content = html_content.replace("\ufffd", "")

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("index.html rewritten successfully.")

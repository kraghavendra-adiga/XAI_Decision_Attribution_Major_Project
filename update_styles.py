import re
import os

with open("templates/index.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# 1. Update CSS
# Replace old card CSS
html_content = re.sub(r'\.card, \.result-box \{[\s\S]*?\}\n\.card:hover, \.result-box:hover \{[\s\S]*?\}',
"""/* Card base */
.card, .result-box {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  padding: 20px 24px;
  box-shadow: var(--shadow-card);
  transition: border-color var(--transition);
}
.card:hover, .result-box:hover { border-color: var(--border-accent); }""", html_content)

# Replace old card-header CSS
html_content = re.sub(r'\.card-header \{[\s\S]*?\}',
""".card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 2px;
  text-transform: uppercase;
}""", html_content)

# Replace dot CSS
html_content = re.sub(r'\.dot \{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; \}',
""".card-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}""", html_content)

# Replace inputs CSS
html_content = re.sub(r'textarea, input\[type=text\] \{[\s\S]*?\}\ntextarea:focus, input\[type=text\]:focus \{[\s\S]*?\}',
"""textarea, input[type="text"] {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: 14px;
  padding: 14px 16px;
  width: 100%;
  box-sizing: border-box;
  transition: border-color var(--transition),
              box-shadow var(--transition);
  resize: vertical;
}
textarea:focus, input[type="text"]:focus {
  border-color: var(--accent-primary);
  outline: none;
  box-shadow: 0 0 0 3px rgba(124,111,255,0.15);
}""", html_content)

# Replace buttons CSS
html_content = re.sub(r'\.run-btn \{[\s\S]*?\}\n\.run-btn:hover:not\(:disabled\) \{[\s\S]*?\}\n\.run-btn:active:not\(:disabled\) \{[\s\S]*?\}\n\.run-btn:disabled \{[\s\S]*?\}',
""".btn-primary {
  background: linear-gradient(
    135deg, var(--accent-primary), #5a4fcf
  );
  color: white;
  border: none;
  border-radius: var(--radius-md);
  padding: 12px 28px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  width: 100%;
  margin-top: 12px;
  transition: opacity var(--transition), 
              transform var(--transition);
}
.btn-primary:hover {
  opacity: 0.9;
  transform: translateY(-1px);
}
.btn-primary:active { transform: translateY(0); }
.btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}""", html_content)

# Replace grids CSS
html_content = re.sub(r'\.grid-3 \{[\s\S]*?\}\n\.grid-2 \{[\s\S]*?\}',
""".results-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-top: 16px;
}
.results-grid-2 {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin-top: 16px;
}
.full-width { grid-column: 1 / -1; }""", html_content)
html_content = re.sub(r'\.grid-3', '.results-grid', html_content)
html_content = re.sub(r'\.grid-2', '.results-grid-2', html_content)

# LIME CSS
html_content = re.sub(r'/\* LIME \*/[\s\S]*?(?=/\* SHAP \*/)',
"""/* LIME bars */
.lime-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 4px;
  border-radius: var(--radius-sm);
  transition: background var(--transition);
}
.lime-row:hover { background: var(--bg-card-hover); }
.lime-word {
  font-family: var(--font-mono);
  font-size: 13px;
  width: 120px;
  flex-shrink: 0;
}
.lime-bar-wrap {
  flex: 1;
  height: 8px;
  background: var(--bg-secondary);
  border-radius: 4px;
  overflow: hidden;
}
.lime-bar {
  height: 100%;
  border-radius: 4px;
  transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}
.lime-bar.positive { background: var(--lime-color); }
.lime-bar.negative { background: var(--accent-danger); }
.lime-score {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-muted);
  width: 48px;
  text-align: right;
}
""", html_content)

# SHAP CSS
html_content = re.sub(r'/\* SHAP \*/[\s\S]*?(?=/\* Sentences \(TF-IDF\) \*/)',
"""/* SHAP chips */
.shap-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.shap-chip {
  font-family: var(--font-mono);
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 20px;
  border: 1px solid;
}
.shap-chip.positive {
  background: rgba(124,111,255,0.15);
  border-color: var(--shap-color);
  color: var(--shap-color);
}
.shap-chip.neutral {
  background: var(--bg-secondary);
  border-color: var(--border-color);
  color: var(--text-muted);
}
""", html_content)

# TFIDF CSS
html_content = re.sub(r'/\* Sentences \(TF-IDF\) \*/[\s\S]*?(?=/\* Specifics \*/)',
"""/* TF-IDF sentences */
.sent-list { display: flex; flex-direction: column; gap: 12px; }
.tfidf-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 0;
  border-bottom: 1px solid var(--border-color);
}
.tfidf-row:last-child { border-bottom: none; }
.tfidf-top {
  display: flex;
  align-items: flex-start;
  gap: 14px;
}
.tfidf-pct {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 700;
  color: var(--tfidf-color);
  width: 44px;
  flex-shrink: 0;
}
.tfidf-text {
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-primary);
}
.tfidf-underline {
  height: 2px;
  background: var(--tfidf-color);
  border-radius: 2px;
  margin-left: 58px;
  opacity: 0.6;
}
""", html_content)

# Animations CSS
html_content = re.sub(r'/\* Animations \*/[\s\S]*?(?=/\* Image Tab Specifics \*/)',
"""/* Fade-in animation */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
.fade-in {
  animation: fadeInUp 0.3s ease forwards;
}
.fade-in-1 { animation-delay: 0ms; }
.fade-in-2 { animation-delay: 80ms; }
.fade-in-3 { animation-delay: 160ms; }
.fade-in-4 { animation-delay: 240ms; }
""", html_content)


# 2. Update HTML classes
html_content = html_content.replace('class="run-btn"', 'class="btn-primary"')
html_content = html_content.replace('class="dot ', 'class="card-dot ')
html_content = html_content.replace('class="shap-tokens"', 'class="shap-grid"')

# Add fade-in classes to result cards
# Summarize tab
html_content = re.sub(r'<div class="card">\s*<div class="card-header"><span class="card-dot dot-summary"></span> SUMMARY OUTPUT',
                      r'<div class="card fade-in fade-in-1">\n              <div class="card-header"><span class="card-dot dot-summary"></span> SUMMARY OUTPUT', html_content)
html_content = re.sub(r'<div class="card">\s*<div class="card-header"><span class="card-dot dot-lime"></span> LIME ATTRIBUTION',
                      r'<div class="card fade-in fade-in-2">\n              <div class="card-header"><span class="card-dot dot-lime"></span> LIME ATTRIBUTION', html_content)
html_content = re.sub(r'<div class="card">\s*<div class="card-header"><span class="card-dot dot-shap"></span> SHAP TOKENS',
                      r'<div class="card fade-in fade-in-3">\n              <div class="card-header"><span class="card-dot dot-shap"></span> SHAP TOKENS', html_content)
html_content = re.sub(r'<div class="card">\s*<div class="card-header"><span class="card-dot dot-tfidf"></span> SENTENCE ATTRIBUTION',
                      r'<div class="card full-width fade-in fade-in-4">\n            <div class="card-header"><span class="card-dot dot-tfidf"></span> SENTENCE ATTRIBUTION', html_content)

# QA tab
html_content = re.sub(r'<div class="card">\s*<div class="card-header"><span class="card-dot dot-answer"></span> ANSWER',
                      r'<div class="card fade-in fade-in-1">\n              <div class="card-header"><span class="card-dot dot-answer"></span> ANSWER', html_content)

# Image tab
html_content = re.sub(r'<div class="card">\s*<div class="card-header"><span class="card-dot dot-summary"></span> ORIGINAL IMAGE',
                      r'<div class="card fade-in fade-in-1">\n              <div class="card-header"><span class="card-dot dot-summary"></span> ORIGINAL IMAGE', html_content)
html_content = re.sub(r'<div class="card">\s*<div class="card-header"><span class="card-dot dot-shap"></span> ATTENTION HEATMAP',
                      r'<div class="card fade-in fade-in-2">\n              <div class="card-header"><span class="card-dot dot-shap"></span> ATTENTION HEATMAP', html_content)
html_content = re.sub(r'<div class="card">\s*<div class="card-header"><span class="card-dot dot-answer"></span> ANSWER</div>\s*<div class="answer-text" id="image-answer"',
                      r'<div class="card fade-in fade-in-3">\n            <div class="card-header"><span class="card-dot dot-answer"></span> ANSWER</div>\n            <div class="answer-text" id="image-answer"', html_content)
html_content = re.sub(r'<div class="card" id="image-evidence-box"', r'<div class="card fade-in fade-in-4" id="image-evidence-box"', html_content)


# 3. Update Javascript Templates
lime_old = """      <div class="lime-row">
        <span class="lime-word" title="${d.word}">${d.word}</span>
        <div class="lime-bar-wrap">
          <div class="lime-bar" style="width:${pct}%; background:${color};"></div>
        </div>
        <span class="lime-val">${d.weight > 0 ? '+' : ''}${d.weight.toFixed(3)}</span>
      </div>"""
lime_new = """      <div class="lime-row">
        <span class="lime-word" title="${d.word}">${d.word}</span>
        <div class="lime-bar-wrap">
          <div class="lime-bar ${d.weight >= 0 ? 'positive' : 'negative'}" style="width:${pct}%;"></div>
        </div>
        <span class="lime-score">${d.weight > 0 ? '+' : ''}${d.weight.toFixed(3)}</span>
      </div>"""
html_content = html_content.replace(lime_old, lime_new)

shap_old = """return `<span class="shap-tok" style="background:${bg};border-color:${border};color:${textColor};" title="SHAP: ${d.value.toFixed(4)}">${d.token}</span>`;"""
shap_new = """return `<span class="shap-chip ${isPos ? 'positive' : 'neutral'}" title="SHAP: ${d.value.toFixed(4)}">${d.token}</span>`;"""
html_content = html_content.replace(shap_old, shap_new)

tfidf_old = """    <div class="sent-row">
      <span class="sent-score">${(d.score * 100).toFixed(0)}%</span>
      <div>
        <div class="sent-text">${d.sentence}</div>
        <div class="sent-bar" style="width:${(d.score * 100).toFixed(1)}%;"></div>
      </div>
    </div>"""
tfidf_new = """    <div class="tfidf-row">
      <div class="tfidf-top">
        <span class="tfidf-pct">${(d.score * 100).toFixed(0)}%</span>
        <div class="tfidf-text">${d.sentence}</div>
      </div>
      <div class="tfidf-underline" style="width:${(d.score * 100).toFixed(1)}%;"></div>
    </div>"""
html_content = html_content.replace(tfidf_old, tfidf_new)


with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("CSS and Classes updated successfully.")

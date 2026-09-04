import re

with open("templates/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Fix Image CSS
# Old: width: 100%; max-height: 340px; object-fit: contain;
css_old = """  width: 100%;
  max-height: 340px;
  object-fit: contain;"""
css_new = """  width: 100%;
  height: 340px;
  object-fit: contain;"""
html = html.replace(css_old, css_new)

# Fix inner card container
card_inner_old = """.image-card-inner {
  width: 100%;
  min-height: 200px;
  max-height: 360px;
  overflow: hidden;"""
card_inner_new = """.image-card-inner {
  width: 100%;
  height: 340px;
  overflow: hidden;"""
html = html.replace(card_inner_old, card_inner_new)

# Fix Donut Chart HTML
chart_old = """<canvas id="donut-chart" style="max-width:320px; max-height:320px; width:100%; height:auto;"></canvas>"""
chart_new = """<div style="position: relative; width: 100%; max-width: 320px; aspect-ratio: 1; margin: 0 auto;">
              <canvas id="donut-chart"></canvas>
            </div>"""
html = html.replace(chart_old, chart_new)

audio_chart_old = """<canvas id="audio-donut-chart" style="max-width:320px; max-height:320px; width:100%; height:auto;"></canvas>"""
audio_chart_new = """<div style="position: relative; width: 100%; max-width: 320px; aspect-ratio: 1; margin: 0 auto;">
              <canvas id="audio-donut-chart"></canvas>
            </div>"""
html = html.replace(audio_chart_old, audio_chart_new)

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Fixes applied.")

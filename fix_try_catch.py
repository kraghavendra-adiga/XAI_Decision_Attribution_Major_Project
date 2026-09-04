import re

with open("templates/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Fix the double catch block and bad try blocks
# I'll just remove all the injected try/catch lines and re-apply them correctly.

# Remove the bad catch block that was injected by the lambda
html = html.replace("    } catch(err) {\n        console.error('Chart render error:', err);\n    }\n", "")

# Remove the ones injected by add_try_catch
html = html.replace("try {\n        donutChart = new Chart(", "donutChart = new Chart(")
html = html.replace("    } catch(err) {\n        console.error(\"Chart render error:\", err);\n    }", "")

html = html.replace("try {\n        audioDonutChart = new Chart(", "audioDonutChart = new Chart(")

# Now cleanly apply the try catch
def add_try_catch(content, chart_var):
    pattern = r'({}\s*=\s*new\s*Chart\s*\(\s*ctx\s*,[\s\S]*?\}\s*\)\s*;)'.replace("{}", chart_var)
    return re.sub(pattern, r'try {\n        \1\n    } catch(err) {\n        console.error("Chart render error:", err);\n    }', content)

html = add_try_catch(html, "donutChart")
html = add_try_catch(html, "audioDonutChart")

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Syntax errors fixed.")

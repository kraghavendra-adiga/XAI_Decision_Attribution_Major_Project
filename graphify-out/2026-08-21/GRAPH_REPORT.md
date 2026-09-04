# Graph Report - Decision_Attribution  (2026-08-21)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 114 nodes · 214 edges · 24 communities
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b5c6311f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- explainer.py
- script.js
- test.js
- main.py
- clean_educational_text
- index
- model_utils.py
- get_shap_explanation
- explain_audio_endpoint
- predict_proba
- qa

## God Nodes (most connected - your core abstractions)
1. `clean_educational_text()` - 14 edges
2. `explain_image_gemini()` - 12 edges
3. `explain_image_vqa()` - 12 edges
4. `explain_audio()` - 9 edges
5. `get_lime_explanation()` - 9 edges
6. `get_shap_explanation()` - 8 edges
7. `qa()` - 8 edges
8. `summarize()` - 8 edges
9. `answer_question()` - 8 edges
10. `_extract_keywords()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `qa()` --uses--> `QARequest`  [INFERRED]
  main.py → schemas.py
- `summarize()` --uses--> `SummarizeRequest`  [INFERRED]
  main.py → schemas.py
- `explain_audio_endpoint()` --calls--> `explain_audio()`  [EXTRACTED]
  main.py → explainer.py
- `explain_image_vqa_endpoint()` --calls--> `explain_image_gemini()`  [EXTRACTED]
  main.py → explainer.py
- `explain_image_vqa()` --calls--> `answer_question()`  [EXTRACTED]
  explainer.py → model_utils.py

## Import Cycles
- None detected.

## Communities (24 total, 0 thin omitted)

### Community 0 - "explainer.py"
Cohesion: 0.15
Nodes (26): analyze_frame_with_gemini(), _build_heatmap(), _clean_ocr_line(), _dedupe_lines(), explain_audio(), explain_image_gemini(), explain_image_vqa(), extract_audio_from_video() (+18 more)

### Community 1 - "script.js"
Cohesion: 0.25
Nodes (12): cleanFeatureText(), COLORS, DISPLAY_STOPWORDS, renderDonutChart(), renderLIME(), renderSentences(), renderSHAP(), runImageVQA() (+4 more)

### Community 2 - "test.js"
Cohesion: 0.28
Nodes (12): cleanFeatureText(), COLORS, DISPLAY_STOPWORDS, renderDonutChart(), renderLIME(), renderSentences(), renderSHAP(), runImageVQA() (+4 more)

### Community 3 - "main.py"
Cohesion: 0.52
Nodes (5): BaseModel, summarize(), summarize_text(), QARequest, SummarizeRequest

### Community 4 - "clean_educational_text"
Cohesion: 0.38
Nodes (6): get_lime_explanation(), Return word-level LIME features without citation/source artifacts., clean_attribution_feature(), clean_educational_text(), Remove pasted AI-search citation noise while preserving the lesson text., Normalize one displayed LIME/SHAP feature and hide citation leftovers.

### Community 5 - "index"
Cohesion: 0.33
Nodes (6): get, health(), index(), no_cache(), middleware, Request

### Community 6 - "model_utils.py"
Cohesion: 0.53
Nodes (5): answer_question(), generate_answer(), get_qa_model(), get_summarizer(), Run a pre-built prompt. Uses hybrid QA for VQA prompts, summarizer otherwise.

### Community 7 - "get_shap_explanation"
Cohesion: 0.40
Nodes (5): get_shap_explanation(), _is_display_feature(), _merge_shap_subwords(), Aggregate tokenizer fragments back onto words from the cleaned source., Return readable SHAP word attributions for the POSITIVE class.

### Community 8 - "explain_audio_endpoint"
Cohesion: 0.67
Nodes (4): explain_audio_endpoint(), explain_image_vqa_endpoint(), post, UploadFile

### Community 9 - "predict_proba"
Cohesion: 0.67
Nodes (3): get_clf(), predict_proba(), Probability array of shape (n_samples, 2) for LIME.

### Community 10 - "qa"
Cohesion: 0.67
Nodes (3): get_sentence_attribution(), Rank cleaned sentences by TF-IDF importance score., qa()

## Knowledge Gaps
- **4 isolated node(s):** `COLORS`, `DISPLAY_STOPWORDS`, `COLORS`, `DISPLAY_STOPWORDS`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `explain_image_gemini()` connect `explainer.py` to `explain_audio_endpoint`, `main.py`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Why does `explain_image_vqa()` connect `explainer.py` to `main.py`, `model_utils.py`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Why does `clean_educational_text()` connect `clean_educational_text` to `explainer.py`, `main.py`, `model_utils.py`, `get_shap_explanation`, `predict_proba`, `qa`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **What connects `COLORS`, `DISPLAY_STOPWORDS`, `COLORS` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `explainer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1452991452991453 - nodes in this community are weakly interconnected._
# Graph Report - Decision_Attribution  (2026-08-22)

## Corpus Check
- 24 files · ~26,422 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 131 nodes · 240 edges · 22 communities (21 shown, 1 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1fc4f662`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- audio_explainer.py
- explain_image_vqa
- image_explainer.py
- main.py
- _group_into_blocks
- index
- model_utils.py
- _score_ocr_lines
- _group_ocr_by_line
- CLAUDE.md

## God Nodes (most connected - your core abstractions)
1. `explain_image_vqa()` - 15 edges
2. `explain_image_gemini()` - 14 edges
3. `clean_educational_text()` - 14 edges
4. `explain_audio()` - 11 edges
5. `get_lime_explanation()` - 9 edges
6. `_read_image_text()` - 9 edges
7. `_group_into_blocks()` - 9 edges
8. `get_shap_explanation()` - 8 edges
9. `_score_ocr_lines()` - 8 edges
10. `summarize()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `extract_frame_at_timestamp()` --calls--> `_image_to_b64()`  [EXTRACTED]
  audio_explainer.py → utils.py
- `explain_audio_endpoint()` --calls--> `explain_audio()`  [EXTRACTED]
  main.py → audio_explainer.py
- `_compose_answer()` --calls--> `answer_question()`  [EXTRACTED]
  image_explainer.py → model_utils.py
- `explain_image_vqa_endpoint()` --calls--> `explain_image_gemini()`  [EXTRACTED]
  main.py → image_explainer.py
- `summarize()` --uses--> `SummarizeRequest`  [INFERRED]
  main.py → schemas.py

## Import Cycles
- None detected.

## Communities (22 total, 1 thin omitted)

### Community 0 - "audio_explainer.py"
Cohesion: 0.11
Nodes (25): _align_phrase(), analyze_frame_with_gemini(), _best_segment_span(), explain_audio(), extract_audio_from_video(), extract_frame_at_timestamp(), _format_timestamp(), _join_words() (+17 more)

### Community 1 - "explain_image_vqa"
Cohesion: 0.21
Nodes (15): _build_heatmap(), explain_image_gemini(), explain_image_vqa(), preprocess_for_ocr(), Image, Paint each OCR line with an intensity proportional to its relevance. The…, Keep only the statements that genuinely stand out. A fixed keyword count…, Prepare an image for Tesseract. A single global threshold at 150 turned light-… (+7 more)

### Community 2 - "image_explainer.py"
Cohesion: 0.24
Nodes (11): _clean_ocr_line(), _compose_answer(), _content_terms(), _dedupe_lines(), _line_text(), Image analysis and Grad-CAM heatmap explainability., Fold a word to a comparable root ("dogs" -> "dog", "studies" -> "study")., Stemmed content words, with interrogatives and stopwords removed. (+3 more)

### Community 3 - "main.py"
Cohesion: 0.14
Nodes (27): get_clf(), get_lime_explanation(), get_sentence_attribution(), get_shap_explanation(), _is_display_feature(), _merge_shap_subwords(), predict_proba(), LIME, SHAP, and TF-IDF attribution explainers for text summarization and… (+19 more)

### Community 4 - "_group_into_blocks"
Cohesion: 0.33
Nodes (6): _columns_of(), _continues_block(), _group_into_blocks(), Split rows into text columns by clustering their left edges. A two-column…, Merge wrapped OCR rows into statement-level blocks, keeping their boxes.…, Is this row a wrapped continuation of the block above it?

### Community 5 - "index"
Cohesion: 0.33
Nodes (6): get, health(), index(), no_cache(), middleware, Request

### Community 6 - "model_utils.py"
Cohesion: 0.53
Nodes (5): answer_question(), generate_answer(), get_qa_model(), get_summarizer(), Run a pre-built prompt. Uses hybrid QA for VQA prompts, summarizer otherwise.

### Community 7 - "_score_ocr_lines"
Cohesion: 0.33
Nodes (6): _is_heading(), _is_usable_line(), Reject OCR debris so it cannot be cited as evidence. Tesseract emits stray…, Is this a title or logo rather than a statement of fact? Headings are set in…, Score every OCR line 0..1 for how well it answers the question. Counting shared…, _score_ocr_lines()

### Community 8 - "_group_ocr_by_line"
Cohesion: 0.50
Nodes (4): _group_ocr_by_line(), _prose_indices(), Which words of a row are prose, dropping callout numbers at either end.…, Group OCR tokens into lines using block_num, par_num, line_num.

## Knowledge Gaps
- **1 isolated node(s):** `graphify`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `explain_image_vqa()` connect `explain_image_vqa` to `image_explainer.py`, `main.py`, `_group_into_blocks`, `_score_ocr_lines`, `_group_ocr_by_line`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Why does `explain_image_gemini()` connect `explain_image_vqa` to `image_explainer.py`, `main.py`, `_group_into_blocks`, `_score_ocr_lines`, `_group_ocr_by_line`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Why does `explain_audio()` connect `audio_explainer.py` to `main.py`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **What connects `graphify` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `audio_explainer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1111111111111111 - nodes in this community are weakly interconnected._
- **Should `main.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14112903225806453 - nodes in this community are weakly interconnected._
# Graph Report - Decision_Attribution  (2026-08-31)

## Corpus Check
- 31 files · ~58,118 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 403 nodes · 819 edges · 30 communities (28 shown, 2 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 26 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7fa22f30`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- audio_explainer.py
- image_explainer.py
- _score_ocr_lines
- main.py
- text_explainer.py
- Reporter
- XAi — Decision Attribution
- document_explainer.py
- Image
- CLAUDE.md
- explain_document
- _render_shapes
- DocumentFormatError
- _GeminiModel
- vercel.json
- _trocr_read
- _looks_handwritten
- vercel-build.sh

## God Nodes (most connected - your core abstractions)
1. `explain_audio()` - 25 edges
2. `explain_image_gemini()` - 22 edges
3. `explain_document()` - 21 edges
4. `explain_image_vqa()` - 19 edges
5. `unsupported_reply()` - 15 edges
6. `explain_document_local()` - 14 edges
7. `clean_educational_text()` - 14 edges
8. `XAi — Decision Attribution` - 14 edges
9. `Reporter` - 13 edges
10. `_render_shapes()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `explain_audio_endpoint()` --indirect_call--> `explain_audio()`  [INFERRED]
  main.py → audio_explainer.py
- `explain_document_endpoint()` --uses--> `DocumentFormatError`  [INFERRED]
  main.py → document_explainer.py
- `explain_document_endpoint()` --indirect_call--> `explain_document()`  [INFERRED]
  main.py → document_explainer.py
- `explain_image_vqa_endpoint()` --indirect_call--> `explain_image_gemini()`  [INFERRED]
  main.py → image_explainer.py
- `summarize()` --uses--> `SummarizeRequest`  [INFERRED]
  main.py → schemas.py

## Import Cycles
- None detected.

## Communities (30 total, 2 thin omitted)

### Community 0 - "audio_explainer.py"
Cohesion: 0.09
Nodes (35): _align_phrase(), analyze_frame_with_gemini(), _audio_declined(), audio_stage_plan(), _best_segment_span(), _cache_key(), _cached_transcript(), explain_audio() (+27 more)

### Community 1 - "image_explainer.py"
Cohesion: 0.10
Nodes (37): _page_record(), One page, with its text already grouped into statement-level blocks., _build_heatmap(), _clean_ocr_line(), _columns_of(), _continues_block(), _dedupe_lines(), explain_image_gemini() (+29 more)

### Community 2 - "_score_ocr_lines"
Cohesion: 0.20
Nodes (10): _document_scores(), _rank_pages(), Score every statement in the document in one pass. {page: {key: score}}.…, Order pages by how well their own text matches the question. This is what keeps…, _compose_answer(), _content_terms(), Stemmed content words, with interrogatives and stopwords removed., Score every OCR line 0..1 for how well it answers the question. Counting shared… (+2 more)

### Community 3 - "main.py"
Cohesion: 0.08
Nodes (42): _aggregate_by_word(), _alignment_scorer(), get_lime_explanation(), get_sentence_attribution(), get_shap_explanation(), _is_display_feature(), LIME, SHAP, and TF-IDF attribution explainers for text summarization and…, Word-level LIME weights for the input's support of `reference`.… (+34 more)

### Community 4 - "text_explainer.py"
Cohesion: 0.06
Nodes (65): BaseException, ndarray, Split cleaned text into (sentence, start, end) triples.…, split_sentences_spans(), _align_quote(), build_source_heat(), _containment(), _defines() (+57 more)

### Community 5 - "Reporter"
Cohesion: 0.11
Nodes (27): explain_audio_endpoint(), explain_document_endpoint(), explain_image_vqa_endpoint(), health(), qa(), Where a running job has actually got to, for the browser's progress bar. 404…, Answer a question about a PDF or .pptx, page by page. Gated on the extension…, read_progress() (+19 more)

### Community 6 - "XAi — Decision Attribution"
Cohesion: 0.05
Nodes (37): 1. Backend — Hugging Face Space, 2. Front end — Vercel, 3. Lock down CORS, Deploying, Running the container locally, Verifying it's actually live, What this does not change, When something's wrong (+29 more)

### Community 7 - "document_explainer.py"
Cohesion: 0.14
Nodes (17): _body_height(), _gemini_answer(), _get_easyocr(), _is_document_heading(), _page_by_label(), PDF and slide-deck analysis: per-page attribution, with handwriting support.…, Mean glyph height of the rows a block wraps across, in pixels., The page's median line height — its body-text size. (+9 more)

### Community 8 - "Image"
Cohesion: 0.14
Nodes (23): _add_word(), _display_copy(), _easyocr_ink(), _for_trocr(), _htr_ink(), _line_geometry(), _merge_rows(), Image (+15 more)

### Community 10 - "explain_document"
Cohesion: 0.14
Nodes (21): _as_document_wording(), _assemble(), _chart(), _document_declined(), _document_statements(), _engine_summary(), explain_document(), explain_document_local() (+13 more)

### Community 21 - "_render_shapes"
Cohesion: 0.14
Nodes (18): _draw_table(), _draw_text_block(), _is_title(), _paragraph_font_px(), _paste_picture(), A truetype face at this pixel size, or the bitmap default., Greedy word wrap using the font's own measurements., Draw wrapped text inside a box and record a box for every word. Words are… (+10 more)

### Community 24 - "DocumentFormatError"
Cohesion: 0.31
Nodes (9): _blank_ocr_data(), DocumentFormatError, _pdf_pages(), _pptx_pages(), Ingest an upload into page records. Raises DocumentFormatError. `report` is a…, Render each PDF page and collect its text with boxes. A digital PDF already…, Redraw each slide from its shapes and collect text with boxes., The upload cannot be read, with a message already fit to show a user. (+1 more)

### Community 25 - "_GeminiModel"
Cohesion: 0.25
Nodes (6): _as_part(), _gemini_client(), _GeminiModel, The shared client, built on first use. Lazily, because importing this module…, One piece of a request, in the form the SDK now expects., Presents the old `GenerativeModel.generate_content` surface.

### Community 26 - "vercel.json"
Cohesion: 0.25
Nodes (7): buildCommand, framework, headers, installCommand, outputDirectory, $schema, trailingSlash

### Community 27 - "_trocr_read"
Cohesion: 0.50
Nodes (4): _get_trocr(), (processor, model) for handwriting, or None if unavailable., Read a list of line-crop images, in batches. Empty list if unavailable., _trocr_read()

### Community 28 - "_looks_handwritten"
Cohesion: 0.50
Nodes (4): _looks_handwritten(), _mean_confidence(), (mean confidence, word count) over words a reader would call words., Did Tesseract find text it could not read? Tesseract locates handwritten lines…

## Knowledge Gaps
- **41 isolated node(s):** `vercel-build.sh script`, `$schema`, `framework`, `buildCommand`, `outputDirectory` (+36 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `explain_audio()` connect `audio_explainer.py` to `main.py`, `text_explainer.py`, `Reporter`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Why does `Reporter` connect `Reporter` to `main.py`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Why does `unsupported_reply()` connect `text_explainer.py` to `audio_explainer.py`, `image_explainer.py`, `explain_document`, `document_explainer.py`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `explain_audio()` (e.g. with `.stage()` and `explain_audio_endpoint()`) actually correct?**
  _`explain_audio()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `explain_image_gemini()` (e.g. with `.stage()` and `RuntimeError`) actually correct?**
  _`explain_image_gemini()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `vercel-build.sh script`, `$schema`, `framework` to the rest of the system?**
  _41 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `audio_explainer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0873015873015873 - nodes in this community are weakly interconnected._
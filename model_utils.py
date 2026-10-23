# model_utils.py
import re

import torch
from transformers import AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoTokenizer
try:
    from .text_cleaning import clean_educational_text
except ImportError:
    from text_cleaning import clean_educational_text

# distilbart-cnn-6-6 is a CNN/DailyMail model: its encoder accepts 1024 tokens and
# it was trained to emit a short highlight. Text longer than that window has to be
# summarized in pieces, because one pass physically cannot see past token 1024.
_MAX_INPUT_TOKENS = 1024
# Pack chunks a little under the window so a chunk plus its special tokens still fits.
_CHUNK_TOKENS = 900
# 142 is the model's own tuned generation ceiling (inherited from bart-large-cnn); a
# little headroom above it means the trim-to-sentence below usually drops a fragment
# rather than a whole sentence the model was halfway through.
_MAX_SUMMARY_TOKENS = 160
# The largest floor worth imposing: 56 is the min_length this model was trained to
# emit for a full-length article. Short inputs get a proportionally smaller floor.
_MIN_SUMMARY_TOKENS = 56
# Past this many chunks the joined highlights stop reading as a summary and need
# summarizing in turn.
_REDUCE_ABOVE_CHUNKS = 4

_summarizer = None
_summarizer_tokenizer = None
_qa_model = None
_qa_tokenizer = None

def get_summarizer():
    global _summarizer, _summarizer_tokenizer
    if _summarizer is None or _summarizer_tokenizer is None:
        print("[model_utils] Loading summarizer...")
        model_name = "sshleifer/distilbart-cnn-6-6"
        _summarizer_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _summarizer = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return _summarizer, _summarizer_tokenizer

def get_qa_model():
    global _qa_model, _qa_tokenizer
    if _qa_model is None or _qa_tokenizer is None:
        print("[model_utils] Loading QA model...")
        model_name = "distilbert-base-cased-distilled-squad"
        _qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        _qa_model.eval()
    return _qa_model, _qa_tokenizer

def _trim_to_sentence(text: str) -> str:
    """Drop a trailing fragment so output never ends mid-name, e.g. "coined by Carl"."""
    text = (text or "").strip()
    if not text or re.search(r"[.!?]\s*$", text):
        return text
    trimmed = re.sub(r"[^.!?]*$", "", text).strip()
    return trimmed or text


def _split_for_window(text: str, tokenizer, limit: int = _CHUNK_TOKENS) -> list[str]:
    """Cut text into pieces that each fit the encoder, preferring sentence breaks.

    Splitting mid-sentence hands the model a fragment and gets a fragment back, so
    sentences are packed greedily and only a sentence longer than the whole window is
    cut by force.

    Chunks are balanced rather than filled to the brim: packing 1792 tokens into a
    900-token limit greedily leaves 858 + 891 + 43, and that 43-token offcut gets
    summarized on its own into a stray line about nothing. Aiming for equal sizes
    gives two real chunks and one less generation pass.
    """
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    if not sentences:
        return []

    lengths = [len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in sentences]
    total = sum(lengths)
    if total <= limit:
        return [text.strip()]
    n_chunks = -(-total // limit)  # ceil
    target = min(limit, -(-total // n_chunks))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence, n in zip(sentences, lengths):
        if n > limit:
            # A single sentence bigger than the window: flush, then hard-split it.
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
            for i in range(0, len(ids), limit):
                chunks.append(
                    tokenizer.decode(ids[i : i + limit], skip_special_tokens=True)
                )
            continue
        if current and current_len + n > target:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += n
    if current:
        chunks.append(" ".join(current))

    # Absorb a small tail into the chunk before it. Sentence granularity means the
    # target above cannot divide evenly — 1792 tokens at a 900 limit packs to
    # 858 + 891 + 43 — and that 43-token offcut would otherwise be summarized on its
    # own into a stray line about nothing, at the cost of a whole generation pass.
    # Merging is safe while the result still clears the encoder's real window.
    if len(chunks) > 1:
        tail = len(tokenizer(chunks[-1], add_special_tokens=False)["input_ids"])
        prev = len(tokenizer(chunks[-2], add_special_tokens=False)["input_ids"])
        if tail < limit // 4 and prev + tail <= _MAX_INPUT_TOKENS - 4:
            chunks[-2:] = [chunks[-2] + " " + chunks[-1]]
    return chunks


def _summarize_chunk(chunk: str, summarizer, tokenizer) -> str:
    inputs = tokenizer(
        chunk,
        return_tensors="pt",
        truncation=True,
        max_length=_MAX_INPUT_TOKENS,
    )
    n_in = int(inputs["input_ids"].shape[-1])
    # Never ask for more output than input. A short passage forced to a fixed 130
    # tokens is where the model starts padding the length out with repetition.
    max_len = min(_MAX_SUMMARY_TOKENS, max(24, int(n_in * 0.6)))
    # Scale the floor with the input too. A flat 30 lets a full 900-token chunk get
    # away with two sentences; 56 is what this model was trained to emit for a
    # full-length article, so a full chunk is held to roughly that and a short one
    # is not forced to pad.
    min_len = min(max(8, int(n_in * 0.08)), _MIN_SUMMARY_TOKENS, max(8, max_len - 8))
    summary_ids = summarizer.generate(
        **inputs,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        num_beams=4,
        early_stopping=True,
    )
    text = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
    # Trim here rather than after joining. The ceiling above cuts wherever it lands,
    # so a chunk can end on "was coined by Carl"; trimmed only at the end, that
    # fragment would sit in the middle of the joined summary where nothing can reach
    # it, and the trim would instead eat the last chunk's whole contribution.
    return _trim_to_sentence(text)


def summarize_text(text: str, report=None) -> str:
    """Summarize a passage of any length.

    This used to pass `clean_text[:1024]` — a *character* slice sitting next to a
    `max_length=1024` *token* limit. 1024 characters is about 240 tokens, so the token
    limit never fired, three quarters of the encoder went unused, and a pasted article
    was summarized from its opening paragraph alone. Chunking on sentence boundaries
    and joining the pieces means the whole passage is actually read.

    `report` is an optional `report(fraction)` callable for the progress bar, kept as a
    plain callable so this module stays independent of progress.py.
    """
    try:
        summarizer, tokenizer = get_summarizer()
        clean_text = clean_educational_text(text)
        if not clean_text:
            return "[Summarization error: empty input]"

        chunks = _split_for_window(clean_text, tokenizer)
        if not chunks:
            return "[Summarization error: empty input]"

        # Text no longer than the shortest summary we would emit has nothing to
        # summarize, and the min_length floor makes the model pad it out instead:
        # "Mitochondria make ATP." comes back as "Mitochondria make ATP.
        # Mitochondria makes ATP." Hand it straight back.
        if len(tokenizer(clean_text, add_special_tokens=False)["input_ids"]) <= _MIN_SUMMARY_TOKENS:
            if report:
                report(1.0)
            return clean_text.strip()

        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(_summarize_chunk(chunk, summarizer, tokenizer))
            if report:
                report((i + 1) / (len(chunks) + 1))
        summary = " ".join(p for p in parts if p)

        # Enough chunk summaries need summarizing in turn — one reduce pass keeps the
        # result readable rather than a wall of highlights. One pass and not a loop,
        # because every pass loses detail.
        if len(chunks) > _REDUCE_ABOVE_CHUNKS:
            summary = _summarize_chunk(summary, summarizer, tokenizer)
        if report:
            report(1.0)

        return _trim_to_sentence(summary) or "[Summarization error: no output]"
    except Exception as e:
        return f"[Summarization error: {e}]"

def answer_question(question: str, context: str):
    try:
        qa_model, qa_tokenizer = get_qa_model()
        clean_question = str(question or "").strip()
        clean_context = clean_educational_text(context)
        if not clean_question or not clean_context:
            return "[QA error: empty question or context]", 0.0

        inputs = qa_tokenizer(
            clean_question,
            clean_context,
            return_tensors="pt",
            truncation="only_second",
            stride=128,
            max_length=384,
            padding=True,
            return_overflowing_tokens=True,
        )

        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key != "overflow_to_sample_mapping"
        }
        with torch.no_grad():
            outputs = qa_model(**model_inputs)

        best_score = -1.0
        best_window = 0
        start_idx = end_idx = 0
        max_answer_tokens = 30
        for window_idx in range(outputs.start_logits.shape[0]):
            sequence_ids = inputs.sequence_ids(window_idx)
            context_mask = torch.tensor(
                [token_type == 1 for token_type in sequence_ids],
                dtype=torch.bool,
                device=outputs.start_logits.device,
            )
            start_logits = outputs.start_logits[window_idx].masked_fill(
                ~context_mask, -1e9
            )
            end_logits = outputs.end_logits[window_idx].masked_fill(
                ~context_mask, -1e9
            )
            start_probs = torch.softmax(start_logits, dim=-1)
            end_probs = torch.softmax(end_logits, dim=-1)

            for start in torch.nonzero(
                context_mask, as_tuple=False
            ).flatten().tolist():
                max_end = min(start + max_answer_tokens, len(sequence_ids))
                for end in range(start, max_end):
                    if not context_mask[end]:
                        break
                    score = float(start_probs[start] * end_probs[end])
                    if score > best_score:
                        best_score = score
                        best_window = window_idx
                        start_idx = start
                        end_idx = end

        answer_ids = inputs["input_ids"][best_window][start_idx : end_idx + 1]
        answer = qa_tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        answer = answer.replace(" ##", "").lstrip("#")
        answer = answer.replace(" - ", "-")

        return answer or "[QA error: no answer found]", round(max(best_score, 0.0), 4)
    except Exception as e:
        return f"[QA error: {e}]", 0.0


def generate_answer(prompt: str) -> str:
    """Run a pre-built prompt. Uses hybrid QA for VQA prompts, summarizer otherwise."""
    try:
        # Intercept the VQA prompt since distilbart is just a summarizer and will echo it.
        # Extract the text and question to use the working hybrid approach.
        if "TEXT FROM IMAGE:" in prompt and "QUESTION:" in prompt:
            text_match = re.search(r"TEXT FROM IMAGE:\n(.*?)\n\nQUESTION:", prompt, re.DOTALL)
            q_match = re.search(r"QUESTION:\s*(.*?)\n\nANSWER:", prompt, re.DOTALL)
            
            if text_match and q_match:
                clean_context = text_match.group(1).strip()
                clean_question = q_match.group(1).strip()
                
                # Step 1: Extractive QA
                qa_answer, _ = answer_question(clean_question, clean_context)
                if qa_answer.startswith("[QA error"):
                    return qa_answer
                    
                # Step 2 & 3: Expand to full sentences
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', clean_context) if s.strip()]
                answer_words = {w.lower() for w in re.findall(r'[A-Za-z0-9]+', qa_answer) if len(w) > 2}
                
                relevant = []
                for sent in sentences:
                    sent_words = {w.lower() for w in re.findall(r'[A-Za-z0-9]+', sent)}
                    if len(answer_words & sent_words) >= max(1, len(answer_words) // 2):
                        relevant.append(sent)
                
                answer = " ".join(relevant) if relevant else qa_answer
                
                # Trim to last complete sentence
                if answer and not re.search(r'[.!?\n]\s*$', answer):
                    trimmed = re.sub(r'[^.!?\n]*$', '', answer).strip()
                    if trimmed:
                        answer = trimmed
                        
                return answer or qa_answer

        # Fallback for non-VQA prompts: use the summarizer
        summarizer, tokenizer = get_summarizer()
        if not prompt or not prompt.strip():
            return "[Answer error: empty prompt]"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_INPUT_TOKENS,
        )

        output_ids = summarizer.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.3,
            early_stopping=True,
            num_beams=4,
        )
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if "ANSWER:" in answer:
            answer = answer.split("ANSWER:", 1)[-1].strip()

        if answer and not re.search(r'[.!?\n]\s*$', answer):
            trimmed = re.sub(r'[^.!?\n]*$', '', answer).strip()
            if trimmed:
                answer = trimmed

        return answer or "[Answer error: no answer generated]"
    except Exception as e:
        return f"[Answer error: {e}]"

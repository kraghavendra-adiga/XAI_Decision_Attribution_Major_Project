import re

with open("explainer.py", "r", encoding="utf-8") as f:
    content = f.read()

# --- AUDIO UPDATE ---

audio_parsing_old = """        raw = response.text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```$", "", raw).strip()
        parsed = json.loads(raw)
        answer = parsed.get("answer", "").strip()
        sources = parsed.get("sources", [])"""

audio_parsing_new = """        raw = response.text.strip()
        
        # Remove markdown fences
        raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"^```\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```$", "", raw).strip()
        
        # Strip any preamble before the first {
        brace_start = raw.find("{")
        if brace_start > 0:
            raw = raw[brace_start:]
        
        # Strip anything after the last }
        brace_end = raw.rfind("}")
        if brace_end != -1 and brace_end < len(raw) - 1:
            raw = raw[:brace_end + 1]
        
        # Parse
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[Audio JSON error] {e}")
            print(f"[Raw Gemini output] {raw[:500]}")
            raise ValueError(
                f"Gemini returned invalid JSON: {str(e)}"
            )
        
        answer = parsed.get("answer", "").strip()
        sources = parsed.get("sources", [])"""

content = content.replace(audio_parsing_old, audio_parsing_new)

audio_prompt_old = """        prompt = (
            "You are analyzing an educational audio transcript.\\n"
            "Answer the question using ONLY information in the "
            "transcript below.\\n"
            "Return ONLY valid JSON, no markdown, no extra text:\\n"
            "{\\n"
            '  "answer": "complete generalized answer",\\n'
            '  "sources": [\\n'
            '    {\\n'
            '      "text": "exact phrase from transcript",\\n'
            '      "contribution": 0.6,\\n'
            '      "reason": "why this contributed"\\n'
            '    }\\n'
            '  ]\\n'
            "}\\n\\n"
            "Rules:\\n"
            "- Answer must be complete and generalized\\n"
            "- If multiple points exist, list all of them\\n"
            "- Sources must be exact phrases from the transcript\\n"
            "- Contribution values must add up to 1.0\\n"
            "- Minimum 2 sources, maximum 5 sources\\n"
            "- Never add information not in the transcript\\n\\n"
            f"TRANSCRIPT:\\n{full_transcript}\\n\\n"
            f"QUESTION: {question}\\n\\n"
            "ANSWER:"
        )"""

audio_prompt_new = """        prompt = (
            "You are analyzing an educational audio transcript.\\n"
            "Respond with ONLY a JSON object. "
            "No preamble, no explanation, no markdown.\\n"
            "Start your response with { and end with }.\\n\\n"
            "Format:\\n"
            "{\\n"
            '  "answer": "complete answer here",\\n'
            '  "sources": [\\n'
            '    {\\n'
            '      "text": "exact phrase from transcript",\\n'
            '      "contribution": 0.6,\\n'
            '      "reason": "why this contributed"\\n'
            '    }\\n'
            '  ]\\n'
            "}\\n\\n"
            "Rules:\\n"
            "- answer must be complete and generalized\\n"
            "- if multiple points exist list all of them\\n"
            "- sources must be exact phrases from transcript\\n"
            "- contribution values must add up to 1.0\\n"
            "- minimum 2 sources maximum 5\\n"
            "- never add information not in the transcript\\n\\n"
            f"TRANSCRIPT:\\n{full_transcript}\\n\\n"
            f"QUESTION: {question}"
        )"""

content = content.replace(audio_prompt_old, audio_prompt_new)

# --- IMAGE UPDATE ---

image_parsing_old = """        raw = response.text.strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```$", "", raw).strip()

        parsed = json.loads(raw)
        answer = parsed.get("answer", "").strip()
        sources = parsed.get("sources", [])"""

image_parsing_new = """        raw = response.text.strip()
        
        # Remove markdown fences
        raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"^```\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```$", "", raw).strip()
        
        # Strip any preamble before the first {
        brace_start = raw.find("{")
        if brace_start > 0:
            raw = raw[brace_start:]
        
        # Strip anything after the last }
        brace_end = raw.rfind("}")
        if brace_end != -1 and brace_end < len(raw) - 1:
            raw = raw[:brace_end + 1]
        
        # Parse
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[Image JSON error] {e}")
            print(f"[Raw Gemini output] {raw[:500]}")
            raise ValueError(
                f"Gemini returned invalid JSON: {str(e)}"
            )
        
        answer = parsed.get("answer", "").strip()
        sources = parsed.get("sources", [])"""

content = content.replace(image_parsing_old, image_parsing_new)

with open("explainer.py", "w", encoding="utf-8") as f:
    f.write(content)

print("JSON parsing applied successfully.")

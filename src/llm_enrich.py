
import json
import re
from typing import Dict, List
from pathlib import Path
import pandas as pd

from .config import (
    LLM_BACKEND,
    HF_MODEL_NAME,
    HF_MAX_NEW_TOKENS,
    HF_TEMPERATURE,
    HF_TOP_P,
)
from .utils import logger, save_json

import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


class LLMEnricher:
    def __init__(self, backend="heuristic"):
        self.backend = backend
        self.model = None
        self.tokenizer = None

        if backend == "hf" and HF_AVAILABLE:
            try:
                logger.info(f"Loading Qwen model: {HF_MODEL_NAME}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    HF_MODEL_NAME,
                    trust_remote_code=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    HF_MODEL_NAME,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                )
                self.model.eval()
                logger.info("Qwen model loaded successfully (CPU)")
            except Exception as e:
                logger.warning(f"Qwen load failed → fallback to heuristic. Error: {e}")
                self.backend = "heuristic"

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=HF_MAX_NEW_TOKENS,
                temperature=HF_TEMPERATURE,
                top_p=HF_TOP_P,
                do_sample=True,
            )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text

    def enrich_item(self, item: Dict) -> Dict:
        title = item.get("title", "")
        raw = item.get("genres", "")
        short = item.get("short_text", "")

        meta_text = f"Title: {title}\nGenres: {raw}\nDescription: {short}"

        if self.backend == "hf" and self.model is not None:
            prompt = f"""
You are an assistant that converts item metadata into structured JSON.

Return ONLY valid JSON with fields:
- description (string, max 3 sentences)
- entities (list of strings)
- format_tags (choose from: visual-heavy, audio-heavy, text-heavy, interactive)
- complexity (integer from 1 to 5)

Metadata:
{meta_text}
"""
            out = self._generate(prompt)

            match = re.search(r"\{.*\}", out, re.S)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    logger.warning("JSON parse failed, using heuristic fallback")

    
        desc = (short or f"{title} — {raw}")[:280]
        entities = [x.strip() for x in re.split(r"[|,/;]", raw) if len(x.strip()) > 1][:6]

        fmt = ["text-heavy"]
        low = meta_text.lower()
        if any(k in low for k in ["video", "visual", "image"]):
            fmt = ["visual-heavy"]
        elif any(k in low for k in ["audio", "music", "sound"]):
            fmt = ["audio-heavy"]

        complexity = 3
        if len(desc) < 80:
            complexity = 2
        elif len(desc) > 180:
            complexity = 4

        return {
            "description": desc,
            "entities": entities,
            "format_tags": fmt,
            "complexity": complexity,
        }

    def enrich_items_list(self, items: List[Dict], out_path: Path = None) -> List[Dict]:
        enriched = []
        for it in items:
            data = self.enrich_item(it)
            enriched.append({**it, **data})

        if out_path:
            save_json(out_path, enriched)
            logger.info(f"Saved enriched items → {out_path}")

        return enriched





def load_items_from_csv(path):
 
    df = pd.read_csv(path)

    if "item_id" not in df.columns or "title" not in df.columns:
        raise ValueError(
            f"Items CSV must contain at least item_id and title, got {df.columns}"
        )

    items = []
    for _, row in df.iterrows():
        if "text" in df.columns and pd.notna(row.get("text")):
            text = str(row["text"])
        elif "genres" in df.columns and pd.notna(row.get("genres")):
            text = f"Genres: {row['genres']}"
        else:
            text = str(row["title"])

        items.append(
            {
                "item_id": str(row["item_id"]),
                "title": str(row["title"]),
                "text": text,
            }
        )

    return items

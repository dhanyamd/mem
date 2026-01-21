import os
import time
import torch
import uvicorn
import logging
from fastapi import FastAPI, Body
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from typing import List, Optional

# Force local cache
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "model_cache")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BrainService")

app = FastAPI(title="Radix-Titan Brain Service")

# Model IDs
TAXONOMY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Global model state
models = {"taxonomy": {"model": None, "tokenizer": None}}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")


@app.on_event("startup")
async def load_models():
    """Load models into RAM once at startup."""
    logger.info(f"🧠 Initializing Brain Service on {device}...")

    # Load Qwen (Unified for Surprisal & Taxonomy)
    logger.info(f"📥 Loading Neural Core: {TAXONOMY_MODEL}")
    models["taxonomy"]["tokenizer"] = AutoTokenizer.from_pretrained(TAXONOMY_MODEL)

    # Use float16 on MPS/CUDA for speed and memory efficiency
    dtype = torch.float16 if device.type in ["cuda", "mps"] else torch.float32

    models["taxonomy"]["model"] = AutoModelForCausalLM.from_pretrained(
        TAXONOMY_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    models["taxonomy"]["model"].eval()

    logger.info(f"✅ Neural Core Online ({dtype}).")


class SurprisalRequest(BaseModel):
    text: str
    context: Optional[str] = ""


@app.post("/surprisal")
async def calculate_surprisal(req: SurprisalRequest):
    tokenizer = models["taxonomy"]["tokenizer"]
    model = models["taxonomy"]["model"]

    full_text = f"{req.context} {req.text}" if req.context else req.text
    inputs = tokenizer(
        full_text, return_tensors="pt", truncation=True, padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # CrossEntropy loss is a direct measure of surprisal
        score = outputs.loss.item()

    return {"surprisal_score": score, "model": TAXONOMY_MODEL}


class TaxonomyRequest(BaseModel):
    text: str
    user_handle: Optional[str] = "user"
    existing_paths: Optional[List[str]] = []
    mode: Optional[str] = "storage"  # "storage" or "search"


@app.post("/taxonomy")
async def generate_path(req: TaxonomyRequest):
    tokenizer = models["taxonomy"]["tokenizer"]
    model = models["taxonomy"]["model"]

    user_handle = req.user_handle or "user"
    mode = req.mode or "storage"

    # Dynamic Context Injection
    existing_taxonomy_str = ""
    if req.existing_paths:
        paths_list = "\n".join(
            [f"- {p}" for p in req.existing_paths[:20]]
        )  # Limit context
        existing_taxonomy_str = (
            f"EXISTING TAXONOMY (Use these drawers if they fit):\n{paths_list}\n"
        )

    if mode == "search":
        instruction = (
            "PREDICT 3 LIKELY PATHS where this info might be stored.\n"
            "Cover different valid possibilities (e.g. self vs entities).\n"
            "Output them separated by commas."
        )
        output_format = "path1, path2, path3"
    else:
        instruction = "OUTPUT ONLY THE PATH."
        output_format = "path"

    # SYSTEMIC PROMPT: Neural Librarian with Subject-Object Logic
    prompt = (
        f"<|im_start|>system\n"
        f"Act as a Radix Path Router. For every input, follow this 3-step logic:\n"
        f"{existing_taxonomy_str}\n"
        f"1. IDENTIFY SUBJECT (The Anchor): \n"
        f"   - If it is about the USER (me, my, I) -> Start with 'users/{{handle}}/self/'.\n"
        f"   - If it is about ANY OTHER Person, Organization, or Named Entity -> Start with 'users/{{handle}}/entities/{{name}}/'.\n"
        f"\n"
        f"2. IDENTIFY DOMAIN: Select one of: [professional, personal, interests, social, health, assets, traits, techniques, objects].\n"
        f"\n"
        f"3. CONSTRUCT PATH: Combine them (users/{{handle}}/subject/domain/subdomain).\n"
        f"\n"
        f"STRICT CONSTRAINTS:\n"
        f"- ENTITY RULE: Information about a person (e.g. Mom, Arjun, Boss) MUST go under 'entities/{{name}}'. NEVER use 'general/professional' for people.\n"
        f"- If it's a Game Rule -> users/{user_handle}/self/interests/gaming/rules\n"
        f"- If it's a Coding style -> users/{user_handle}/self/professional/coding\n"
        f"- If it's a Productivity method -> users/{user_handle}/self/techniques/productivity\n"
        f"- Use 'self/professional/coding' or 'self/interests/gaming' as anchors.\n"
        f"\n"
        f"{instruction}\n"
        f"EXAMPLES:\n"
        f"Input: Arjun is allergic to peanuts -> users/{user_handle}/entities/arjun/health/allergies\n"
        f"Input: Riya loves peanut butter -> users/{user_handle}/entities/riya/personal/preferences\n"
        f"Input: My mom is a teacher -> users/{user_handle}/entities/mom/career/profession\n"
        f"Input: I follow a Clean-Draft coding philosophy -> users/{user_handle}/self/professional/coding/style\n"
        f"Input: The Shadow Rule in the game is strict -> users/{user_handle}/self/interests/gaming/rules\n"
        f"Input: Use the Iron Vault productivity system -> users/{user_handle}/self/techniques/productivity/iron_vault\n"
        f"Now, generate the path for the following input.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\nInput: {req.text}<|im_end|>\n"
        f"<|im_start|>assistant\n{output_format}:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,  # Increased for multiple paths
            temperature=0.01,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()

    # Clean up output
    path = generated.split("\n")[0].strip()
    if "path:" in path:
        path = path.split("path:")[-1].strip()

    return {"radix_path": path, "model": TAXONOMY_MODEL}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

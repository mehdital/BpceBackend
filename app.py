import os, json, re
from urllib.parse import urljoin
from typing import Dict, Any, List, Optional

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

# ==== Configuration Azure OpenAI (via variables d'environnement) ====
AZ_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # ex: https://hackacton-ia4-impact2.openai.azure.com
AZ_DEPLOY   = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # ex: gpt-4.1-mini
AZ_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")
API_VER     = "2025-01-01-preview"

# ==== FastAPI ====
app = FastAPI(title="IA Accessibility Auditor", version="1.0.0")

# Autorisez votre domaine front (ajoutez votre URL Vercel/Netlify si nécessaire)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # pour POC; resserrer en prod
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# ==== Modèles d'entrée ====
class Filters(BaseModel):
    IMG_ALT_PERTINENCE: Optional[bool] = False
    OCR_TEXT_IN_IMAGE: Optional[bool] = False

class AuditRequest(BaseModel):
    url: HttpUrl
    filters: Filters

# ==== Utilitaires ====
def absolutize(src: str, base: str) -> str:
    try:
        return urljoin(base, src)
    except Exception:
        return src

async def fetch_html(url: str) -> str:
    """Récupère le HTML sans navigateur headless (POC sobre)."""
    headers = {
        "User-Agent": "IA4Impact/1.0 (+accessibility-audit)",
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.text

def extract_images(html: str, base_url: str) -> List[Dict[str, Any]]:
    """Extrait <img> + [role=img], construit contexte voisin court."""
    soup = BeautifulSoup(html, "html.parser")
    images: List[Dict[str, Any]] = []

    # <img>
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if not src:
            continue
        if src.startswith("data:"):
            # ignorer data URI lourdes pour sobriété
            continue
        alt = (img.get("alt") or "").strip()
        aria_hidden = (img.get("aria-hidden") or "").lower() == "true"
        selector_hint = img.get("id") or (".".join(img.get("class")) if img.get("class") else "")
        parent_text = img.parent.get_text(" ", strip=True)[:300] if img.parent else ""
        context = " ".join(parent_text.split())
        images.append({
            "src": absolutize(src, base_url),
            "alt": alt,
            "aria_hidden": aria_hidden,
            "selector": f'img#{"".join(selector_hint)}' if img.get("id") else (f'img.{selector_hint}' if selector_hint else "img"),
            "context": context
        })

    # role="img"
    for el in soup.find_all(attrs={"role": "img"}):
        aria_label = (el.get("aria-label") or "").strip()
        aria_lblby = (el.get("aria-labelledby") or "").strip()
        # parfois un role="img" n'a pas src: on garde quand même pour analyse du label
        src = el.get("src") or el.get("data-src") or ""
        if src.startswith("data:"):
            src = ""
        selector_hint = el.get("id") or (".".join(el.get("class")) if el.get("class") else "")
        context = " ".join((el.get_text(" ", strip=True) or "")[:300].split())
        images.append({
            "src": absolutize(src, base_url) if src else "",
            "alt": aria_label or aria_lblby,
            "aria_hidden": (el.get("aria-hidden") or "").lower() == "true",
            "selector": f'[role=img]#{"".join(selector_hint)}' if el.get("id") else (f'[role=img].{selector_hint}' if selector_hint else "[role=img]"),
            "context": context
        })
    return images

def build_vision_messages_alt(blocks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    system = {
        "role": "system",
        "content": (
            'Tu es un auditeur RGAA 4.1.2. '
            'Réponds STRICTEMENT en JSON: '
            '[{"judgment":"...","explanation":"...","suggestion":"...","confidence":0.0}] '
            'Valeurs judgment autorisées: ["Pertinent","Non pertinent","Ambigu","Inconclusif"]. '
            'La suggestion (si non pertinente/ambigue) doit être ≤ 120 caractères.'
        )
    }
    user_content: List[Dict[str, Any]] = [
        {"type": "input_text",
         "text": "Règle: RGAA 1.1 — Pertinence du texte alternatif pour images porteuses d’information. "
                 "Réponds un objet JSON par bloc, dans l’ordre."}
    ]
    for i, b in enumerate(blocks, start=1):
        user_content.append({"type": "input_text", "text": f"Bloc #{i}:"})
        if b.get("image_url"):
            user_content.append({"type": "input_image", "image_url": b["image_url"]})
        user_content.append({"type": "input_text", "text": f'ALT: "{b.get("alt","")}"'})
        user_content.append({"type": "input_text", "text": f'Contexte voisin: "{b.get("context","")}"'})
    return [system, {"role": "user", "content": user_content}]

def build_vision_messages_ocr(blocks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    system = {
        "role": "system",
        "content": (
            'Auditeur RGAA 4.1.2 (images & OCR). '
            'Réponds STRICTEMENT en JSON: '
            '[{"judgment":"...","explanation":"...","suggestion":"...","confidence":0.0,"detected_text":"..."}] '
            'Met "Non pertinent" si du texte est présent dans l’image mais non restitué (alt/contenu adjacent).'
        )
    }
    user_content: List[Dict[str, Any]] = [
        {"type": "input_text",
         "text": "Règle: RGAA 1.5 — Le texte dans une image doit être restitué par une alternative textuelle ou contenu adjacent. "
                 "Réponds un objet JSON par bloc, dans l’ordre."}
    ]
    for i, b in enumerate(blocks, start=1):
        user_content.append({"type": "input_text", "text": f"Bloc #{i}:"})
        if b.get("image_url"):
            user_content.append({"type": "input_image", "image_url": b["image_url"]})
        user_content.append({"type": "input_text", "text": f'ALT fourni: "{b.get("alt","")}"'})
        user_content.append({"type": "input_text", "text": f'Contexte voisin: "{b.get("context","")}"'})
    return [system, {"role": "user", "content": user_content}]

async def call_azure_chat_vision(messages: List[Dict[str, Any]]) -> str:
    """Appel REST Azure OpenAI chat/completions."""
    if not (AZ_ENDPOINT and AZ_DEPLOY and AZ_API_KEY):
        raise HTTPException(500, "Azure OpenAI non configuré (env vars manquantes).")
    url = f"{AZ_ENDPOINT}/openai/deployments/{AZ_DEPLOY}/chat/completions?api-version={API_VER}"
    payload = {
        "model": AZ_DEPLOY,
        "temperature": 0.2,
        "messages": messages
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers={
            "api-key": AZ_API_KEY,
            "Content-Type": "application/json"
        }, json=payload)
        # Propager erreurs HTTP propres (401, 403, 429...)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
    return data["choices"][0]["message"]["content"]

def robust_json_parse(raw: str):
    """Tente de parser JSON strict, sinon extrait le premier tableau JSON."""
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r'(\[.*\])', raw, flags=re.S)
        if m:
            return json.loads(m.group(1))
        return []

# ==== Endpoint ====
@app.post("/audit")
async def audit(req: AuditRequest):
    html = await fetch_html(str(req.url))
    images = extract_images(html, str(req.url))

    # Heuristiques de sobriété
    # - limiter à 10 images max par critère en V1
    # - ignorer images trop petites (sauf role=img)
    candidates_alt: List[Dict[str, Any]] = []
    candidates_ocr: List[Dict[str, Any]] = []

    for im in images:
        # skip décoratives (heuristique simple)
        if im["aria_hidden"]:
            continue
        # heuristique taille indisponible sans HEAD/bytes; on garde simple
        # ALT pertinence: alt non vide
        if req.filters.IMG_ALT_PERTINENCE and im["alt"]:
            candidates_alt.append(im)
        # OCR: toutes les images éligibles (on laisse la vision décider s'il y a du texte)
        if req.filters.OCR_TEXT_IN_IMAGE:
            candidates_ocr.append(im)

    # Limites V1
    candidates_alt = candidates_alt[:10]
    candidates_ocr = candidates_ocr[:10]

    findings: List[Dict[str, Any]] = []

    # === IMG_ALT_PERTINENCE ===
    if candidates_alt:
        blocks = [
            {"image_url": im["src"], "alt": im["alt"], "context": im["context"]}
            for im in candidates_alt
        ]
        messages = build_vision_messages_alt(blocks)
        raw = await call_azure_chat_vision(messages)
        parsed = robust_json_parse(raw)
        for im, res in zip(candidates_alt, parsed):
            findings.append({
                "criterion": "IMG_ALT_PERTINENCE",
                "rgaa": "1.1",
                "target": {"src": im["src"], "selector": im["selector"]},
                "judgment": res.get("judgment", "Inconclusif"),
                "explanation": res.get("explanation", ""),
                "suggestion": res.get("suggestion", None),
                "confidence": float(res.get("confidence", 0.0)),
                "evidence": {"model": AZ_DEPLOY, "mode": "vision"}
            })

    # === OCR_TEXT_IN_IMAGE ===
    if candidates_ocr:
        blocks = [
            {"image_url": im["src"], "alt": im["alt"], "context": im["context"]}
            for im in candidates_ocr
        ]
        messages = build_vision_messages_ocr(blocks)
        raw = await call_azure_chat_vision(messages)
        parsed = robust_json_parse(raw)
        for im, res in zip(candidates_ocr, parsed):
            findings.append({
                "criterion": "OCR_TEXT_IN_IMAGE",
                "rgaa": "1.5",
                "target": {"src": im["src"], "selector": im["selector"]},
                "judgment": res.get("judgment", "Inconclusif"),
                "explanation": res.get("explanation", ""),
                "suggestion": res.get("suggestion", None),
                "confidence": float(res.get("confidence", 0.0)),
                "detected_text": res.get("detected_text", ""),
                "evidence": {"model": AZ_DEPLOY, "mode": "vision"}
            })

    response = {
        "url": str(req.url),
        "summary": {
            "counts": {
                "images_total": len(images),
                "images_alt_tested": len(candidates_alt),
                "images_ocr_tested": len(candidates_ocr),
                "findings": len(findings)
            },
            "notes": [
                "Résultats IA qualitatifs (RGAA 1.1 & 1.5). Vérifier manuellement les cas Ambigu/Inconclusif."
            ]
        },
        "findings": findings,
        "metadata": {
            "models": [AZ_DEPLOY],
            "filters": [k for k, v in req.filters.model_dump().items() if v]
        }
    }
    return response

# ==== Healthcheck simple ====
@app.get("/")
def root():
    return {"status": "ok", "service": "IA Accessibility Auditor"}

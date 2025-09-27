import os, json, re, base64
from io import BytesIO
from urllib.parse import urljoin, urlparse
from typing import Dict, Any, List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from PIL import Image

# =========================
# Azure OpenAI (Render env)
# =========================
AZURE_OPENAI_ENDPOINT   = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT") or ""
AZURE_OPENAI_API_KEY    = os.getenv("AZURE_OPENAI_API_KEY") or ""
API_VER = "2025-01-01-preview"

# =========================
# Image constraints (sobriété)
# =========================
MAX_IMG_BYTES = 6 * 1024 * 1024   # 6 MB
MAX_WIDTH     = 1600              # resize si > MAX_WIDTH
TIMEOUT_HTML  = 20
TIMEOUT_IMG   = 15

# =========================
# FastAPI
# =========================
app = FastAPI(title="IA Accessibility Auditor (Unified+VisionProxy)", version="2.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # resserrer en prod
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# =========================
# Entrées /audit
# =========================
class Filters(BaseModel):
    IMG_ALT_PERTINENCE: Optional[bool] = False
    OCR_TEXT_IN_IMAGE: Optional[bool] = False
    HEADINGS_VISUAL_SEMANTICS: Optional[bool] = False
    LINK_LABEL_PERTINENCE: Optional[bool] = False

class AuditRequest(BaseModel):
    url: HttpUrl
    filters: Filters
    screenshot_url: Optional[str] = None           # image de la page (test visuel vs DOM)
    screenshot_base64: Optional[str] = None        # "data:image/png;base64,..." ou base64 pur

# =========================
# Utilitaires HTML/DOM
# =========================
async def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "IA4Impact/1.0 (+accessibility-audit)",
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=TIMEOUT_HTML) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.text

def absolutize(src: str, base: str) -> str:
    try:
        return urljoin(base, src)
    except Exception:
        return src

def get_page_lang(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("html")
    return (tag.get("lang") or "").strip() if tag else ""

def extract_images(html: str, base_url: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    images: List[Dict[str, Any]] = []

    # <img>
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if not src or src.startswith("data:"):
            continue
        alt = (img.get("alt") or "").strip()
        aria_hidden = (img.get("aria-hidden") or "").lower() == "true"
        sel = img.get("id") or (".".join(img.get("class")) if img.get("class") else "")
        parent_text = img.parent.get_text(" ", strip=True)[:300] if img.parent else ""
        context = " ".join(parent_text.split())
        images.append({
            "src": absolutize(src, base_url),
            "alt": alt,
            "aria_hidden": aria_hidden,
            "selector": f'img#{img.get("id")}' if img.get("id") else (f'img.{sel}' if sel else "img"),
            "context": context
        })

    # role="img"
    for el in soup.find_all(attrs={"role": "img"}):
        aria_label = (el.get("aria-label") or "").strip()
        aria_lblby = (el.get("aria-labelledby") or "").strip()
        src = el.get("src") or el.get("data-src") or ""
        if src.startswith("data:"):
            src = ""
        sel = el.get("id") or (".".join(el.get("class")) if el.get("class") else "")
        context = " ".join((el.get_text(" ", strip=True) or "")[:300].split())
        images.append({
            "src": absolutize(src, base_url) if src else "",
            "alt": aria_label or aria_lblby,
            "aria_hidden": (el.get("aria-hidden") or "").lower() == "true",
            "selector": f'[role=img]#{el.get("id")}' if el.get("id") else (f'[role=img].{sel}' if sel else "[role=img]"),
            "context": context
        })
    return images

def extract_dom_outline(html: str) -> List[Dict[str,str]]:
    soup = BeautifulSoup(html, "html.parser")
    outline = []
    for lvl in ["h1","h2","h3","h4","h5","h6"]:
        for h in soup.find_all(lvl):
            txt = " ".join((h.get_text(" ", strip=True) or "")[:200].split())
            if txt:
                outline.append({"level": lvl.upper(), "text": txt})
    return outline

def robust_json_parse(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r'(\{.*\}|\[.*\])', raw, flags=re.S)
        if m:
            return json.loads(m.group(1))
        return []

# =========================
# Azure OpenAI — client REST
# =========================
def ensure_azure_config():
    if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_API_KEY):
        raise HTTPException(500, "Azure OpenAI non configuré (AZURE_OPENAI_ENDPOINT/_DEPLOYMENT/_API_KEY).")

async def call_azure_chat(messages: List[Dict[str, Any]], max_tokens: int = 500) -> str:
    ensure_azure_config()
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={API_VER}"
    payload = {
        "model": AZURE_OPENAI_DEPLOYMENT,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": messages
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers={"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"}, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
    return data["choices"][0]["message"]["content"]

# =========================
# Vision helpers (proxy/normalisation)
# =========================
def is_svg_url(url: str) -> bool:
    return url.lower().endswith(".svg")

def _referer_for(url: str) -> str:
    # Meilleur referer par défaut = schéma + netloc
    u = urlparse(url)
    return f"{u.scheme}://{u.netloc}" if u.scheme and u.netloc else url

async def fetch_image_bytes(url: str) -> bytes:
    headers = {
        "User-Agent": "IA4Impact/1.0 (+accessibility-audit)",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": _referer_for(url)
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=TIMEOUT_IMG) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        data = r.content
        if len(data) > MAX_IMG_BYTES:
            raise HTTPException(400, f"Image too large ({len(data)} bytes)")
        return data

def pillow_open_safe(raw: bytes) -> Image.Image:
    bio = BytesIO(raw)
    img = Image.open(bio)
    img.load()  # force decode
    return img

def normalize_image_to_png_dataurl(raw: bytes) -> str:
    """
    - GIF animé -> 1er frame
    - Resize si > MAX_WIDTH
    - Convertit en PNG (RGB/RGBA)
    - Retourne data URL base64
    """
    img = pillow_open_safe(raw)

    # GIF animé -> 1er frame
    if getattr(img, "is_animated", False):
        img.seek(0)
        img = img.convert("RGBA")

    # Resize sobriété
    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / img.width
        new_h = max(1, int(img.height * ratio))
        img = img.resize((MAX_WIDTH, new_h), Image.LANCZOS)

    # Uniformisation mode
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")

    out = BytesIO()
    img.save(out, format="PNG", optimize=True)
    b64 = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

async def prepare_image_for_vision(src_url: str) -> dict:
    """
    Retour:
      - {"ok": True, "data_url": "<data:image/png;base64,...>"}
      - {"ok": False, "reason": "<motif>"}
    """
    if not src_url:
        return {"ok": False, "reason": "empty_src"}
    if is_svg_url(src_url):
        return {"ok": False, "reason": "unsupported_format_svg"}
    try:
        raw = await fetch_image_bytes(src_url)
        data_url = normalize_image_to_png_dataurl(raw)
        return {"ok": True, "data_url": data_url}
    except Exception as e:
        return {"ok": False, "reason": f"fetch_or_decode_failed: {str(e)[:160]}"}

# =========================
# Helpers contenu vision (payload chat)
# =========================
def _image_item(url_or_b64: str) -> Dict[str, Any]:
    """
    Construit un item 'image_url' conforme:
    {"type":"image_url","image_url":{"url":"https://...|data:image/..."}}
    """
    if url_or_b64.startswith("http"):
        return {"type": "image_url", "image_url": {"url": url_or_b64}}
    if url_or_b64.startswith("data:image/"):
        return {"type": "image_url", "image_url": {"url": url_or_b64}}
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{url_or_b64}"}}

# =========================
# Prompts (ALT / OCR / HEADINGS / LINKS)
# =========================
def build_msgs_alt(blocks: List[Dict[str, str]], page_lang: str) -> List[Dict[str, Any]]:
    system = {
        "role": "system",
        "content": (
            "Tu es auditeur RGAA 4.1.2.\n"
            "Ne retourne jamais de texte hors JSON. Si tu hésites, renvoie un JSON valide avec 'judgment':'Inconclusif'.\n"
            'Format: [{"judgment":"Pertinent|Non pertinent|Ambigu|Inconclusif","explanation":"...","suggestion":"...","confidence":0.0}]'
        )
    }
    user_items: List[Dict[str, Any]] = [
        {"type": "text", "text": f"Langue de la page: {page_lang or 'fr'}."},
        {"type": "text", "text": "Règle: RGAA 1.1 — Pertinence du texte alternatif. Un objet JSON par bloc, dans l’ordre."}
    ]
    for i, b in enumerate(blocks, start=1):
        user_items.append({"type": "text", "text": f"Bloc #{i}:"})
        if b.get("image_url"):
            user_items.append(_image_item(b["image_url"]))
        user_items.append({"type": "text", "text": f'ALT: "{b.get("alt","")}"'})
        user_items.append({"type": "text", "text": f'Contexte voisin: "{b.get("context","")}"'})
    return [system, {"role": "user", "content": user_items}]

def build_msgs_ocr(blocks: List[Dict[str, str]], page_lang: str) -> List[Dict[str, Any]]:
    system = {
        "role": "system",
        "content": (
            "Auditeur RGAA 4.1.2 (OCR).\n"
            "Ne retourne jamais de texte hors JSON. Si tu hésites, renvoie un JSON valide avec 'judgment':'Inconclusif'.\n"
            'Format: [{"judgment":"Pertinent|Non pertinent|Ambigu|Inconclusif","explanation":"...","suggestion":"...","confidence":0.0,"detected_text":"..."}]\n'
            'Mets "Non pertinent" si du texte est présent dans l’image mais non restitué par alt/contenu adjacent.'
        )
    }
    user_items: List[Dict[str, Any]] = [
        {"type": "text", "text": f"Langue de la page: {page_lang or 'fr'}."},
        {"type": "text", "text": "Règle: RGAA 1.5 — Texte dans l’image. Un objet JSON par bloc, dans l’ordre."}
    ]
    for i, b in enumerate(blocks, start=1):
        user_items.append({"type": "text", "text": f"Bloc #{i}:"})
        if b.get("image_url"):
            user_items.append(_image_item(b["image_url"]))
        user_items.append({"type": "text", "text": f'ALT fourni: "{b.get("alt","")}"'})
        user_items.append({"type": "text", "text": f'Contexte voisin: "{b.get("context","")}"'})
    return [system, {"role": "user", "content": user_items}]

def build_msgs_headings(screenshot_ref: Dict[str,str]) -> List[Dict[str, Any]]:
    system = {
        "role": "system",
        "content": (
            "Tu es auditeur RGAA 4.1.2 (structuration).\n"
            "Ne retourne jamais de texte hors JSON.\n"
            'Format: {"visual_outline":[{"level":"H1|H2|H3|H4|H5|H6","text":"..."}]} '
            "Déduis uniquement depuis la hiérarchie VISUELLE (taille/gras/position)."
        )
    }
    user_items: List[Dict[str, Any]] = []
    if screenshot_ref.get("image_url"):
        user_items.append(_image_item(screenshot_ref["image_url"]))
    elif screenshot_ref.get("image_b64"):
        user_items.append(_image_item(screenshot_ref["image_b64"]))
    user_items.append({"type":"text","text":"Extrais les titres principaux visibles et attribue un niveau approximatif."})
    return [system, {"role":"user","content": user_items}]

def build_msgs_link_yesno(link_text: str, href: str, context: str) -> List[Dict[str, Any]]:
    return [
        {"role":"system","content":"You are an accessibility expert. Answer ONLY one word: YES or NO."},
        {"role":"user","content":[
            {"type":"text","text": f'Link text: "{link_text}"'},
            {"type":"text","text": f'URL: "{href}"'},
            {"type":"text","text": f'Context: "{context}"'}
        ]}
    ]

# =========================
# Aides évaluation
# =========================
def compare_visual_vs_dom(visual: List[Dict[str,str]], dom: List[Dict[str,str]]) -> Tuple[str, str]:
    if visual and not dom:
        return "Non conforme", "Structure visuelle détectée sans titres sémantiques H1–H6 dans le DOM."
    if not visual:
        return "Inconclusif", "Aucun titre saillant détecté visuellement (capture)."
    def norm(s): return re.sub(r"\s+", " ", (s or "").strip().lower())
    dom_texts = [norm(d["text"]) for d in dom]
    hits = 0
    for v in visual:
        vt = norm(v["text"])
        matched = any(vt and (vt in dt or dt in vt) for dt in dom_texts)
        if matched:
            hits += 1
    ratio = hits / max(1, len(visual))
    if ratio >= 0.7:  return "Conforme", f"≈{int(ratio*100)}% des titres visuels correspondent aux titres DOM."
    if ratio >= 0.3:  return "Ambigu",   f"Correspondance partielle (≈{int(ratio*100)}%)."
    return "Non conforme", f"Correspondance insuffisante (≈{int(ratio*100)}%)."

def make_finding(
    *, criterion: str, rgaa: str, target: Dict[str, Any],
    judgment: str, explanation: str, suggestion: Optional[str],
    confidence: float, processed_by_ai: bool, ai_status: str, ai_error: Optional[str],
    evidence: Dict[str, Any], extras: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    out = {
        "criterion": criterion,
        "rgaa": rgaa,
        "target": target,
        "judgment": judgment,
        "explanation": explanation,
        "suggestion": suggestion,
        "confidence": confidence,
        "processed_by_ai": processed_by_ai,
        "ai_status": ai_status,          # "ok" | "skipped" | "error"
        "ai_error": ai_error,
        "evidence": evidence             # {"model": "...", "mode":"vision|text|rule"}
    }
    if extras:
        out.update(extras)
    return out

# =========================
# Endpoint unique /audit
# =========================
@app.post("/audit")
async def audit(req: AuditRequest):
    html = await fetch_html(str(req.url))
    page_lang = get_page_lang(html) or "fr"
    images = extract_images(html, str(req.url))

    # Sélections images (sobriété V1)
    candidates_alt, candidates_ocr, missing_alt = [], [], []
    if req.filters.IMG_ALT_PERTINENCE or req.filters.OCR_TEXT_IN_IMAGE:
        for im in images:
            if im["aria_hidden"]:
                continue
            if req.filters.IMG_ALT_PERTINENCE and im["alt"] == "":
                missing_alt.append(im)
            if req.filters.IMG_ALT_PERTINENCE and im["alt"]:
                candidates_alt.append(im)
            if req.filters.OCR_TEXT_IN_IMAGE:
                candidates_ocr.append(im)
        candidates_alt = candidates_alt[:10]
        candidates_ocr = candidates_ocr[:10]
        missing_alt    = missing_alt   [:50]

    findings: List[Dict[str, Any]] = []
    processing_report = {
        "alt_candidates": len(candidates_alt),
        "ocr_candidates": len(candidates_ocr),
        "alt_processed_ok": 0,
        "ocr_processed_ok": 0,
        "alt_errors": 0,
        "ocr_errors": 0,
        "links_checked": 0,
        "links_ai_skipped": 0,
        "links_ai_errors": 0,
        "headings_processed": False,
        "headings_error": False
    }

    # === A) IMG sans alt (règle — pas d’IA)
    if req.filters.IMG_ALT_PERTINENCE:
        for im in missing_alt:
            findings.append(make_finding(
                criterion="IMG_ALT_PERTINENCE", rgaa="1.1",
                target={"src": im["src"], "selector": im["selector"]},
                judgment="Non conforme",
                explanation="Image potentiellement informative sans attribut alt.",
                suggestion="Ajouter un alt descriptif concis adapté au contexte.",
                confidence=1.0, processed_by_ai=False, ai_status="skipped", ai_error=None,
                evidence={"model": None, "mode": "rule"}
            ))

    # === B) ALT pertinence (IA vision) — via proxy PNG data URL
    if req.filters.IMG_ALT_PERTINENCE and candidates_alt:
        blocks = []
        skip_findings = []
        for im in candidates_alt:
            prep = await prepare_image_for_vision(im["src"])
            if prep.get("ok"):
                blocks.append({"image_url": prep["data_url"], "alt": im["alt"], "context": im["context"], "_selector": im["selector"], "_src": im["src"]})
            else:
                skip_findings.append(make_finding(
                    criterion="IMG_ALT_PERTINENCE", rgaa="1.1",
                    target={"src": im["src"], "selector": im["selector"]},
                    judgment="Inconclusif",
                    explanation="Image non analysée par vision.",
                    suggestion=None, confidence=0.0,
                    processed_by_ai=True, ai_status="skipped", ai_error=prep.get("reason"),
                    evidence={"model": AZURE_OPENAI_DEPLOYMENT, "mode": "vision"},
                    extras={"skip_reason": prep.get("reason")}
                ))
        findings.extend(skip_findings)

        if blocks:
            try:
                raw = await call_azure_chat(build_msgs_alt(blocks, page_lang), max_tokens=400)
                parsed = robust_json_parse(raw)
                for blk, res in zip(blocks, parsed):
                    findings.append(make_finding(
                        criterion="IMG_ALT_PERTINENCE", rgaa="1.1",
                        target={"src": blk["_src"], "selector": blk["_selector"]},
                        judgment=res.get("judgment","Inconclusif"),
                        explanation=res.get("explanation",""),
                        suggestion=res.get("suggestion"),
                        confidence=float(res.get("confidence",0.0)),
                        processed_by_ai=True, ai_status="ok", ai_error=None,
                        evidence={"model": AZURE_OPENAI_DEPLOYMENT, "mode": "vision"}
                    ))
                processing_report["alt_processed_ok"] = len(parsed)
            except HTTPException as e:
                processing_report["alt_errors"] = len(blocks)
                for blk in blocks:
                    findings.append(make_finding(
                        criterion="IMG_ALT_PERTINENCE", rgaa="1.1",
                        target={"src": blk["_src"], "selector": blk["_selector"]},
                        judgment="Inconclusif",
                        explanation="Échec de l’évaluation IA du texte alternatif.",
                        suggestion=None, confidence=0.0,
                        processed_by_ai=True, ai_status="error", ai_error=str(e.detail)[:500],
                        evidence={"model": AZURE_OPENAI_DEPLOYMENT, "mode": "vision"}
                    ))

    # === C) OCR texte dans l’image (IA vision) — via proxy PNG data URL
    if req.filters.OCR_TEXT_IN_IMAGE and candidates_ocr:
        blocks = []
        skip_findings = []
        for im in candidates_ocr:
            prep = await prepare_image_for_vision(im["src"])
            if prep.get("ok"):
                blocks.append({"image_url": prep["data_url"], "alt": im["alt"], "context": im["context"], "_selector": im["selector"], "_src": im["src"]})
            else:
                skip_findings.append(make_finding(
                    criterion="OCR_TEXT_IN_IMAGE", rgaa="1.5",
                    target={"src": im["src"], "selector": im["selector"]},
                    judgment="Inconclusif",
                    explanation="Image non analysée par vision.",
                    suggestion=None, confidence=0.0,
                    processed_by_ai=True, ai_status="skipped", ai_error=prep.get("reason"),
                    evidence={"model": AZURE_OPENAI_DEPLOYMENT, "mode": "vision"},
                    extras={"skip_reason": prep.get("reason"), "detected_text": ""}
                ))
        findings.extend(skip_findings)

        if blocks:
            try:
                raw = await call_azure_chat(build_msgs_ocr(blocks, page_lang), max_tokens=500)
                parsed = robust_json_parse(raw)
                for blk, res in zip(blocks, parsed):
                    findings.append(make_finding(
                        criterion="OCR_TEXT_IN_IMAGE", rgaa="1.5",
                        target={"src": blk["_src"], "selector": blk["_selector"]},
                        judgment=res.get("judgment","Inconclusif"),
                        explanation=res.get("explanation",""),
                        suggestion=res.get("suggestion"),
                        confidence=float(res.get("confidence",0.0)),
                        processed_by_ai=True, ai_status="ok", ai_error=None,
                        evidence={"model": AZURE_OPENAI_DEPLOYMENT, "mode": "vision"},
                        extras={"detected_text": res.get("detected_text","")}
                    ))
                processing_report["ocr_processed_ok"] = len(parsed)
            except HTTPException as e:
                processing_report["ocr_errors"] = len(blocks)
                for blk in blocks:
                    findings.append(make_finding(
                        criterion="OCR_TEXT_IN_IMAGE", rgaa="1.5",
                        target={"src": blk["_src"], "selector": blk["_selector"]},
                        judgment="Inconclusif", explanation="Échec de l’évaluation IA OCR.",
                        suggestion=None, confidence=0.0,
                        processed_by_ai=True, ai_status="error", ai_error=str(e.detail)[:500],
                        evidence={"model": AZURE_OPENAI_DEPLOYMENT, "mode": "vision"},
                        extras={"detected_text": ""}
                    ))

    # === D) Headings visuels vs DOM (IA vision + DOM)
    if req.filters.HEADINGS_VISUAL_SEMANTICS:
        dom_outline = extract_dom_outline(html)
        screenshot_ref = {}
        if req.screenshot_url:
            screenshot_ref = {"image_url": req.screenshot_url}
        elif req.screenshot_base64:
            b64 = req.screenshot_base64.split(",",1)[1] if "," in req.screenshot_base64 else req.screenshot_base64
            screenshot_ref = {"image_b64": b64}

        if not screenshot_ref:
            findings.append(make_finding(
                criterion="HEADINGS_VISUAL_SEMANTICS", rgaa="9.1/9.2",
                target={"src": None, "selector": "screenshot"},
                judgment="Inconclusif",
                explanation="Aucune capture fournie pour l’analyse visuelle.",
                suggestion="Envoyer 'screenshot_url' ou 'screenshot_base64'.",
                confidence=0.0, processed_by_ai=False, ai_status="skipped", ai_error=None,
                evidence={"model": None, "mode": "rule"},
                extras={"visual_outline": [], "dom_outline": dom_outline}
            ))
        else:
            try:
                raw = await call_azure_chat(build_msgs_headings(screenshot_ref), max_tokens=220)
                parsed = robust_json_parse(raw)  # dict attendu
                visual_outline = parsed.get("visual_outline", []) if isinstance(parsed, dict) else []
                judgment, explanation = compare_visual_vs_dom(visual_outline, dom_outline)
                findings.append(make_finding(
                    criterion="HEADINGS_VISUAL_SEMANTICS", rgaa="9.1/9.2",
                    target={"src": req.screenshot_url or "data:image/*;base64", "selector": "screenshot"},
                    judgment=judgment, explanation=explanation,
                    suggestion=("Aligner H1–H6 DOM sur la hiérarchie visuelle." if judgment != "Conforme" else None),
                    confidence=(0.7 if judgment in ("Conforme","Non conforme") else 0.4),
                    processed_by_ai=True, ai_status="ok", ai_error=None,
                    evidence={"model": AZURE_OPENAI_DEPLOYMENT, "mode": "vision"},
                    extras={"visual_outline": visual_outline, "dom_outline": dom_outline}
                ))
                processing_report["headings_processed"] = True
            except HTTPException as e:
                findings.append(make_finding(
                    criterion="HEADINGS_VISUAL_SEMANTICS", rgaa="9.1/9.2",
                    target={"src": req.screenshot_url or "data:image/*;base64", "selector": "screenshot"},
                    judgment="Inconclusif", explanation="Échec de l’analyse IA de la capture d’écran.",
                    suggestion=None, confidence=0.0,
                    processed_by_ai=True, ai_status="error", ai_error=str(e.detail)[:500],
                    evidence={"model": AZURE_OPENAI_DEPLOYMENT, "mode": "vision"},
                    extras={"visual_outline": [], "dom_outline": dom_outline}
                ))
                processing_report["headings_error"] = True

    # === E) Pertinence libellé de lien (IA texte + heuristiques)
    if req.filters.LINK_LABEL_PERTINENCE:
        soup = BeautifulSoup(html, "html.parser")
        seen_texts = {}
        links = soup.find_all("a", href=True)
        for a in links:
            href = urljoin(str(req.url), a["href"])
            text = a.get_text(strip=True)
            title = a.get("title")
            parent_text = a.find_parent().get_text(" ", strip=True) if a.find_parent() else ""
            issues = []

            # Heuristiques de base
            if not text:
                issues.append("Empty link text")
            imgs = a.find_all("img")
            if imgs and not text:
                alt_texts = [img.get("alt", "") for img in imgs]
                if not any(alt_texts):
                    issues.append("Link contains only image(s) without alt text")

            key = (text or "").strip().lower()
            if key:
                if key in seen_texts and seen_texts[key] != href:
                    issues.append("Duplicate link text points to different destinations")
                else:
                    seen_texts[key] = href

            # IA YES/NO — si texte présent
            ai_answer = "SKIPPED"
            if text:
                try:
                    raw = await call_azure_chat(build_msgs_link_yesno(text, href, parent_text[:1000]), max_tokens=5)
                    ans = (raw or "").strip().upper()
                    ai_answer = "YES" if ans.startswith("YES") else ("NO" if ans.startswith("NO") else "NO")
                    if ai_answer == "NO":
                        issues.append("Link text not explicit according to AI check")
                except HTTPException as e:
                    ai_answer = "ERROR"
                    issues.append(f"AI check failed: {str(e.detail)[:200]}")

            findings.append(make_finding(
                criterion="LINK_LABEL_PERTINENCE", rgaa="8.x",
                target={"href": href, "text": text, "title": title},
                judgment=("Non conforme" if ("Empty link text" in issues or "Link contains only image(s) without alt text" in issues or ai_answer == "NO") else "Conforme"),
                explanation=("Problèmes détectés: " + "; ".join(issues)) if issues else "Libellé de lien explicite.",
                suggestion=(None if not issues else "Rédiger un libellé descriptif, éviter 'cliquez ici'."),
                confidence=(0.8 if not issues else 0.6),
                processed_by_ai=(text != ""),
                ai_status=("ok" if ai_answer in ("YES","NO") else ("error" if ai_answer=="ERROR" else "skipped")),
                ai_error=(None if ai_answer in ("YES","NO","SKIPPED") else "Azure check failed"),
                evidence={"model": (AZURE_OPENAI_DEPLOYMENT if text else None), "mode": ("text" if text else "rule")},
                extras={"ai_explicit": ai_answer, "issues": issues}
            ))
            processing_report["links_checked"] += 1
            if ai_answer == "SKIPPED": processing_report["links_ai_skipped"] += 1
            if ai_answer == "ERROR":  processing_report["links_ai_errors"]  += 1

    # === Réponse
    return {
        "url": str(req.url),
        "summary": {
            "counts": {
                "images_total": len(images),
                "findings": len(findings),
                "by_criterion": {
                    "IMG_ALT_PERTINENCE": sum(1 for f in findings if f["criterion"]=="IMG_ALT_PERTINENCE"),
                    "OCR_TEXT_IN_IMAGE":  sum(1 for f in findings if f["criterion"]=="OCR_TEXT_IN_IMAGE"),
                    "HEADINGS_VISUAL_SEMANTICS": sum(1 for f in findings if f["criterion"]=="HEADINGS_VISUAL_SEMANTICS"),
                    "LINK_LABEL_PERTINENCE": sum(1 for f in findings if f["criterion"]=="LINK_LABEL_PERTINENCE"),
                }
            },
            "processing_report": processing_report,
            "notes": [
                "RGAA 1.1: présence (règle) + pertinence (IA).",
                "RGAA 1.5: texte dans l’image (IA).",
                "RGAA 9.1/9.2: hiérarchie visuelle vs structure DOM (capture d’écran).",
                "RGAA 8.x: explicitation libellés (IA + heuristiques).",
                "Cas Ambigu/Inconclusif -> vérification manuelle recommandée."
            ]
        },
        "findings": findings,
        "metadata": {
            "models": list({f["evidence"]["model"] for f in findings if f["evidence"]["model"]}),
            "filters": [k for k,v in req.filters.model_dump().items() if v],
            "page_lang": page_lang
        }
    }

# =========================
# Health
# =========================
@app.get("/")
def health():
    return {"status":"ok","service":"IA Accessibility Auditor (Unified+VisionProxy)"}

@app.head("/")
def head_ok():
    return {}

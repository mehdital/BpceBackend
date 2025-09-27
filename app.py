import os, json, re, base64
from urllib.parse import urljoin
from typing import Dict, Any, List, Optional

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

# =========================
# Config Azure OpenAI (REST)
# =========================
AZ_ENDPOINT = (os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("ENDPOINT_URL") or "").rstrip("/")
AZ_DEPLOY   = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("DEPLOYMENT_NAME") or "gpt-4.1-mini"
AZ_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")
API_VER     = "2025-01-01-preview"

# =========================
# Options runtime
# =========================
ENABLE_SELENIUM = os.getenv("ENABLE_SELENIUM", "false").lower() == "true"

# =========================
# FastAPI
# =========================
app = FastAPI(title="IA Accessibility Auditor", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restreindre en prod
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# =========================
# Modèles d'entrée /audit
# =========================
class Filters(BaseModel):
    IMG_ALT_PERTINENCE: Optional[bool] = False
    OCR_TEXT_IN_IMAGE: Optional[bool] = False
    HEADINGS_VISUAL_SEMANTICS: Optional[bool] = False  # nouveau

class AuditRequest(BaseModel):
    url: HttpUrl
    filters: Filters
    screenshot_url: Optional[str] = None
    screenshot_base64: Optional[str] = None

# -------------------------
# Utilitaires généraux
# -------------------------
def absolutize(src: str, base: str) -> str:
    try:
        return urljoin(base, src)
    except Exception:
        return src

async def fetch_html_httpx(url: str) -> str:
    headers = {
        "User-Agent": "IA4Impact/1.0 (+accessibility-audit)",
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.text

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

# -------------------------
# Vision prompts / audit
# -------------------------
def build_vision_messages_alt(blocks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    system = {
        "role": "system",
        "content": (
            'Tu es un auditeur RGAA 4.1.2. '
            'Réponds STRICTEMENT en JSON: '
            '[{"judgment":"...","explanation":"...","suggestion":"...","confidence":0.0}] '
            'Valeurs: ["Pertinent","Non pertinent","Ambigu","Inconclusif"]. '
            'Suggestion ≤ 120 caractères.'
        )
    }
    user_content: List[Dict[str, Any]] = [
        {"type": "input_text",
         "text": "Règle: RGAA 1.1 — Pertinence du texte alternatif. Un objet JSON par bloc, dans l’ordre."}
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
            'Auditeur RGAA 4.1.2 (OCR). '
            'Réponds STRICTEMENT en JSON: '
            '[{"judgment":"...","explanation":"...","suggestion":"...","confidence":0.0,"detected_text":"..."}] '
            'Mets "Non pertinent" si du texte est présent dans l’image mais non restitué.'
        )
    }
    user_content: List[Dict[str, Any]] = [
        {"type": "input_text",
         "text": "Règle: RGAA 1.5 — Texte présent dans l’image. Un objet JSON par bloc, dans l’ordre."}
    ]
    for i, b in enumerate(blocks, start=1):
        user_content.append({"type": "input_text", "text": f"Bloc #{i}:"})
        if b.get("image_url"):
            user_content.append({"type": "input_image", "image_url": b["image_url"]})
        user_content.append({"type": "input_text", "text": f'ALT fourni: "{b.get("alt","")}"'})
        user_content.append({"type": "input_text", "text": f'Contexte voisin: "{b.get("context","")}"'})
    return [system, {"role": "user", "content": user_content}]

def build_vision_messages_headings(screenshot_ref: Dict[str,str]) -> List[Dict[str, Any]]:
    system = {
        "role": "system",
        "content": (
            "Tu es auditeur RGAA 4.1.2 (structuration). "
            "Réponds STRICTEMENT en JSON: "
            '{"visual_outline":[{"level":"H1|H2|H3|H4|H5|H6","text":"..."}]} '
            "Déduis uniquement depuis la hiérarchie VISUELLE (taille/gras/position)."
        )
    }
    content = []
    if screenshot_ref.get("image_url"):
        content.append({"type":"input_image","image_url":screenshot_ref["image_url"]})
    elif screenshot_ref.get("image_b64"):
        content.append({"type":"input_image","image_base64":screenshot_ref["image_b64"]})
    content.append({"type":"input_text","text":"Extrais les titres principaux visibles et attribue un niveau approximatif."})
    return [system, {"role":"user","content": content}]

def compare_visual_vs_dom(visual: List[Dict[str,str]], dom: List[Dict[str,str]]) -> (str, str):
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
    if ratio >= 0.7:
        return "Conforme", f"≈{int(ratio*100)}% des titres visuels correspondent aux titres DOM."
    if ratio >= 0.3:
        return "Ambigu", f"Correspondance partielle (≈{int(ratio*100)}%)."
    return "Non conforme", f"Correspondance insuffisante (≈{int(ratio*100)}%)."

# -------------------------
# Appel Azure (REST)
# -------------------------
async def call_azure_chat_vision(messages: List[Dict[str, Any]]) -> str:
    if not (AZ_ENDPOINT and AZ_DEPLOY and AZ_API_KEY):
        raise HTTPException(500, "Azure OpenAI non configuré (AZURE_OPENAI_ENDPOINT/_DEPLOYMENT/_API_KEY).")
    url = f"{AZ_ENDPOINT}/openai/deployments/{AZ_DEPLOY}/chat/completions?api-version={API_VER}"
    payload = {"model": AZ_DEPLOY, "temperature": 0.2, "messages": messages}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers={"api-key": AZ_API_KEY, "Content-Type": "application/json"}, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
    return data["choices"][0]["message"]["content"]

# ============================================================
# Endpoint principal /audit (images ALT, OCR, structure visuelle)
# ============================================================
@app.post("/audit")
async def audit(req: AuditRequest):
    # 1) HTML (HTTP simple)
    html = await fetch_html_httpx(str(req.url))
    images = extract_images(html, str(req.url))

    # 2) Sélections
    candidates_alt, candidates_ocr, missing_alt = [], [], []
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
        "ocr_errors": 0
    }

    # 3) Absence d’alt (règle immédiate)
    for im in missing_alt:
        findings.append({
            "criterion": "IMG_ALT_PERTINENCE",
            "rgaa": "1.1",
            "target": {"src": im["src"], "selector": im["selector"]},
            "judgment": "Non conforme",
            "explanation": "Image potentiellement informative sans attribut alt.",
            "suggestion": "Ajouter un alt descriptif concis adapté au contexte.",
            "confidence": 1.0,
            "processed_by_ai": False,
            "ai_status": "skipped",
            "ai_error": None,
            "evidence": {"model": None, "mode": "rule"}
        })

    # 4) ALT pertinence (IA)
    if candidates_alt:
        blocks = [{"image_url": im["src"], "alt": im["alt"], "context": im["context"]} for im in candidates_alt]
        try:
            raw = await call_azure_chat_vision(build_vision_messages_alt(blocks))
            parsed = robust_json_parse(raw)
            for im, res in zip(candidates_alt, parsed):
                findings.append({
                    "criterion": "IMG_ALT_PERTINENCE",
                    "rgaa": "1.1",
                    "target": {"src": im["src"], "selector": im["selector"]},
                    "judgment": res.get("judgment","Inconclusif"),
                    "explanation": res.get("explanation",""),
                    "suggestion": res.get("suggestion"),
                    "confidence": float(res.get("confidence",0.0)),
                    "processed_by_ai": True,
                    "ai_status": "ok",
                    "ai_error": None,
                    "evidence": {"model": AZ_DEPLOY, "mode": "vision"}
                })
            processing_report["alt_processed_ok"] = len(parsed)
        except HTTPException as e:
            processing_report["alt_errors"] = len(candidates_alt)
            for im in candidates_alt:
                findings.append({
                    "criterion": "IMG_ALT_PERTINENCE",
                    "rgaa": "1.1",
                    "target": {"src": im["src"], "selector": im["selector"]},
                    "judgment": "Inconclusif",
                    "explanation": "Échec de l’évaluation IA du texte alternatif.",
                    "suggestion": None,
                    "confidence": 0.0,
                    "processed_by_ai": True,
                    "ai_status": "error",
                    "ai_error": str(e.detail)[:500],
                    "evidence": {"model": AZ_DEPLOY, "mode": "vision"}
                })

    # 5) OCR / texte dans l’image (IA)
    if candidates_ocr:
        blocks = [{"image_url": im["src"], "alt": im["alt"], "context": im["context"]} for im in candidates_ocr]
        try:
            raw = await call_azure_chat_vision(build_vision_messages_ocr(blocks))
            parsed = robust_json_parse(raw)
            for im, res in zip(candidates_ocr, parsed):
                findings.append({
                    "criterion": "OCR_TEXT_IN_IMAGE",
                    "rgaa": "1.5",
                    "target": {"src": im["src"], "selector": im["selector"]},
                    "judgment": res.get("judgment","Inconclusif"),
                    "explanation": res.get("explanation",""),
                    "suggestion": res.get("suggestion"),
                    "confidence": float(res.get("confidence",0.0)),
                    "detected_text": res.get("detected_text",""),
                    "processed_by_ai": True,
                    "ai_status": "ok",
                    "ai_error": None,
                    "evidence": {"model": AZ_DEPLOY, "mode": "vision"}
                })
            processing_report["ocr_processed_ok"] = len(parsed)
        except HTTPException as e:
            processing_report["ocr_errors"] = len(candidates_ocr)
            for im in candidates_ocr:
                findings.append({
                    "criterion": "OCR_TEXT_IN_IMAGE",
                    "rgaa": "1.5",
                    "target": {"src": im["src"], "selector": im["selector"]},
                    "judgment": "Inconclusif",
                    "explanation": "Échec de l’évaluation IA OCR.",
                    "suggestion": None,
                    "confidence": 0.0,
                    "detected_text": "",
                    "processed_by_ai": True,
                    "ai_status": "error",
                    "ai_error": str(e.detail)[:500],
                    "evidence": {"model": AZ_DEPLOY, "mode": "vision"}
                })

    # 6) Structure visuelle vs DOM (screenshot)
    if req.filters.HEADINGS_VISUAL_SEMANTICS:
        dom_outline = extract_dom_outline(html)
        # Prépare la référence image
        screenshot_ref = {}
        if req.screenshot_url:
            screenshot_ref = {"image_url": req.screenshot_url}
        elif req.screenshot_base64:
            b64 = req.screenshot_base64.split(",",1)[1] if "," in req.screenshot_base64 else req.screenshot_base64
            screenshot_ref = {"image_b64": b64}
        if not screenshot_ref:
            findings.append({
                "criterion": "HEADINGS_VISUAL_SEMANTICS",
                "rgaa": "9.1/9.2",
                "target": {"src": None, "selector": "screenshot"},
                "judgment": "Inconclusif",
                "explanation": "Aucune capture fournie pour l’analyse visuelle.",
                "suggestion": "Envoyer 'screenshot_url' ou 'screenshot_base64'.",
                "confidence": 0.0,
                "processed_by_ai": False,
                "ai_status": "skipped",
                "ai_error": None,
                "visual_outline": [],
                "dom_outline": dom_outline,
                "evidence": {"model": None, "mode": "rule"}
            })
        else:
            try:
                raw = await call_azure_chat_vision(build_vision_messages_headings(screenshot_ref))
                parsed = robust_json_parse(raw)  # dict attendu
                visual_outline = parsed.get("visual_outline", []) if isinstance(parsed, dict) else []
                judgment, explanation = compare_visual_vs_dom(visual_outline, dom_outline)
                findings.append({
                    "criterion": "HEADINGS_VISUAL_SEMANTICS",
                    "rgaa": "9.1/9.2",
                    "target": {"src": req.screenshot_url or "data:image/*;base64", "selector": "screenshot"},
                    "judgment": judgment,
                    "explanation": explanation,
                    "suggestion": "Aligner H1–H6 DOM sur la hiérarchie visuelle." if judgment != "Conforme" else None,
                    "confidence": 0.7 if judgment in ("Conforme","Non conforme") else 0.4,
                    "processed_by_ai": True,
                    "ai_status": "ok",
                    "ai_error": None,
                    "visual_outline": visual_outline,
                    "dom_outline": dom_outline,
                    "evidence": {"model": AZ_DEPLOY, "mode": "vision"}
                })
            except HTTPException as e:
                findings.append({
                    "criterion": "HEADINGS_VISUAL_SEMANTICS",
                    "rgaa": "9.1/9.2",
                    "target": {"src": req.screenshot_url or "data:image/*;base64", "selector": "screenshot"},
                    "judgment": "Inconclusif",
                    "explanation": "Échec de l’analyse IA de la capture d’écran.",
                    "suggestion": None,
                    "confidence": 0.0,
                    "processed_by_ai": True,
                    "ai_status": "error",
                    "ai_error": str(e.detail)[:500],
                    "visual_outline": [],
                    "dom_outline": dom_outline,
                    "evidence": {"model": AZ_DEPLOY, "mode": "vision"}
                })

    return {
        "url": str(req.url),
        "summary": {
            "counts": {
                "images_total": len(images),
                "images_alt_missing": len([i for i in missing_alt]),
                "images_alt_tested": len(candidates_alt),
                "images_ocr_tested": len(candidates_ocr),
                "findings": len(findings)
            },
            "processing_report": processing_report,
            "notes": [
                "RGAA 1.1: présence (règle) + pertinence (IA).",
                "RGAA 1.5: texte dans l’image (IA).",
                "RGAA 9.1/9.2: hiérarchie visuelle vs structure DOM (screenshot)."
            ]
        },
        "findings": findings,
        "metadata": {
            "models": [AZ_DEPLOY],
            "filters": [k for k,v in req.filters.model_dump().items() if v]
        }
    }

# ============================================================
# Nouveau endpoint /check-links-ai (portage de votre Flask)
# ============================================================
def selenium_render(url: str) -> Optional[str]:
    """Rendu JS optionnel. Requiert ENABLE_SELENIUM=true et deps installées."""
    if not ENABLE_SELENIUM:
        return None
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
        driver.get(url)
        html = driver.page_source
        driver.quit()
        return html
    except Exception as e:
        print(f"[Selenium] {e}")
        return None

async def azure_yes_no_link_explicit(link_text: str, href: str, context_text: str) -> str:
    prompt = (
        "You are an accessibility expert.\n"
        f'Link text: "{link_text}"\n'
        f'Destination URL: "{href}"\n'
        f'Nearby context: "{context_text}"\n\n'
        'Answer exactly "YES" or "NO": does this link text clearly convey the destination or action?'
    )
    messages = [{"role": "system", "content": prompt}]
    raw = await call_azure_chat_vision(messages)  # même endpoint chat/completions
    return (raw or "").strip().split()[0].upper()

@app.get("/check-links-ai")
async def check_links_ai(url: str = Query(..., description="URL à auditer")):
    # 1) Reachability simple
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.head(url, follow_redirects=True)
    except Exception as e:
        raise HTTPException(400, f"URL not reachable: {e}")

    # 2) HTML: tente Selenium si activé, sinon HTTP simple
    html = selenium_render(url) or await fetch_html_httpx(url)
    if not html:
        raise HTTPException(500, "No content retrieved from page")

    soup = BeautifulSoup(html, "html.parser")

    results = []
    seen_texts = {}

    # Anti-patterns
    for div in soup.find_all("div", onclick=True):
        results.append({
            "element": "div",
            "text": div.get_text(strip=True),
            "issues": ["DIV used as link (onclick). Not semantically a link, not keyboard accessible."],
            "html": str(div)[:1000]
        })
    for span in soup.find_all("span", onclick=True):
        if span.get("role") == "link":
            results.append({
                "element": "span",
                "text": span.get_text(strip=True),
                "issues": ["SPAN role=link with onclick. Needs full ARIA/keyboard handling."],
                "html": str(span)[:1000]
            })
    for button in soup.find_all("button", onclick=True):
        onclick = button.get("onclick", "")
        if "window.location" in onclick or "location.href" in onclick:
            results.append({
                "element": "button",
                "text": button.get_text(strip=True),
                "issues": ["BUTTON used for navigation (should be <a> for links)."],
                "html": str(button)[:1000]
            })

    links = soup.find_all("a", href=True)
    for a in links:
        link_info = {
            "href": urljoin(url, a["href"]),
            "text": a.get_text(strip=True),
            "title": a.get("title"),
            "img_alt": None,
            "issues": []
        }

        # Texte de lien vide
        if not link_info["text"]:
            link_info["issues"].append("Empty link text")

        # Image-only
        imgs = a.find_all("img")
        if imgs and not link_info["text"]:
            alt_texts = [img.get("alt", "") for img in imgs]
            link_info["img_alt"] = alt_texts
            if not any(alt_texts):
                link_info["issues"].append("Link contains only image(s) without alt text")

        # Duplicat libellé → destinations différentes
        text_key = (link_info["text"] or "").strip().lower()
        if text_key:
            if text_key in seen_texts:
                if seen_texts[text_key] != link_info["href"]:
                    link_info["issues"].append("Duplicate link text points to different destinations")
            else:
                seen_texts[text_key] = link_info["href"]

        # IA: explicitation du libellé
        parent_text = a.find_parent().get_text(" ", strip=True) if a.find_parent() else ""
        if link_info["text"]:
            try:
                ai_answer = await azure_yes_no_link_explicit(link_info["text"], link_info["href"], parent_text[:1000])
                link_info["ai_explicit"] = ai_answer
                if ai_answer == "NO":
                    link_info["issues"].append("Link text not explicit according to AI check")
            except HTTPException as e:
                link_info["ai_explicit"] = "ERROR"
                link_info["issues"].append(f"AI check failed: {str(e.detail)[:200]}")

        results.append(link_info)

    return {
        "url": url,
        "links_report": results,
        "stats": {
            "total_links": len(links),
            "items_reported": len(results)
        }
    }

# Health
@app.get("/")
def root():
    return {"status": "ok", "service": "IA Accessibility Auditor"}

@app.head("/")
def head_ok():
    return {}

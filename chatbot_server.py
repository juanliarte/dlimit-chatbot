"""
Chatbot multi-brand · Dlimit + Dsafety
=======================================
Backend Flask + RAG. Una sola instancia sirve a varias marcas (Dlimit, Dsafety).

Pipeline: pregunta → embed (Voyage+BM25) → Qdrant hybrid search filtrado por layer
          → Claude con system_prompt de marca → respuesta con fuentes.

Endpoint:
  POST /api/chat  body: {"question": "...", "brand": "dlimit" | "dsafety"}

Compat: si no llega "brand", asume "dlimit" (comportamiento previo).

Env vars:
  ANTHROPIC_API_KEY, VOYAGE_API_KEY, QDRANT_URL, QDRANT_API_KEY
"""
from __future__ import annotations

import os
import sys
import json

from flask import Flask, request, jsonify, Response
import voyageai
from anthropic import Anthropic
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, SparseVector,
    Prefetch, FusionQuery, Fusion,
)
from fastembed import SparseTextEmbedding


COLLECTION = "dlimit-kb"
DENSE_VECTOR_NAME = "voyage-dense"
SPARSE_VECTOR_NAME = "bm25-sparse"
DENSE_MODEL = "voyage-3-large"
SPARSE_MODEL = "Qdrant/bm25"
CLAUDE_MODEL = "claude-haiku-4-5"
TOP_K = 6


def required_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f"[ERROR] Falta {name}")
    return val


# Init clients globally
voyage = voyageai.Client(api_key=required_env("VOYAGE_API_KEY"))
bm25 = SparseTextEmbedding(model_name=SPARSE_MODEL)
qclient = QdrantClient(
    url=required_env("QDRANT_URL"),
    api_key=required_env("QDRANT_API_KEY"),
    timeout=30,
)
claude = Anthropic(api_key=required_env("ANTHROPIC_API_KEY"))


# ============================================================
# SYSTEM PROMPTS POR MARCA
# ============================================================
SYSTEM_PROMPT_DLIMIT = """Eres el asistente conversacional de Dlimit Tactic S.L.,
empresa española (Mataró) especializada en postes separadores y barreras
retráctiles para gestión de colas y delimitación de espacios B2B.

Reglas:
- Responde solo con información presente en el contexto proporcionado.
- Cita las fuentes entre corchetes como aparecen ([1], [2], …).
- Si la información no está en el contexto, di "No tengo esa información en mi base de conocimiento" y ofrece contactar al equipo comercial.
- Tono: profesional, directo, técnico cuando hace falta, comercialmente útil.
- No inventes referencias, modelos, precios ni ensayos.
- Idioma: el del cliente (por defecto español).
- Formato: markdown ligero (negrita en lo importante, listas si aplica)."""

SYSTEM_PROMPT_DSAFETY = """Eres el asistente conversacional de Dsafety™, marca industrial
de Dlimit Tactic S.L. (Mataró, España), especializada en barreras retráctiles para
uso industrial y EPI con tecnología de frenado progresivo. Distribución exclusiva
B2B a través de canal autorizado.

Reglas:
- Responde solo con información presente en el contexto proporcionado.
- Cita las fuentes entre corchetes como aparecen ([1], [2], …).
- Si la información no está en el contexto, di "No tengo esa información en mi base de conocimiento" y ofrece contactar a un distribuidor autorizado o al equipo técnico.
- Importante: Dsafety NO vende directamente al cliente final. Si el usuario pide presupuesto, redirígelo a https://www.dsafety.es/distribuidores/.
- Tono: técnico, sobrio, B2B industrial. Sin marketing fluff.
- No inventes referencias, modelos, presiones, ensayos ni certificaciones.
- Sectores de competencia: industria, EPI, ESD (zonas EPA), vehículo eléctrico (EV), LOTO, farmacéutica.
- NO mezclar con territorio Dlimit (retail, hostelería, eventos, aeropuertos) — esos son de la marca hermana.
- Idioma: el del cliente (por defecto español).
- Formato: markdown ligero (negrita en lo importante, listas si aplica)."""


# ============================================================
# CONFIGURACIÓN POR MARCA
# ============================================================
BRAND_CONFIG = {
    "dlimit": {
        "layer": "publica",
        "prompt": SYSTEM_PROMPT_DLIMIT,
        "name": "Dlimit",
    },
    "dsafety": {
        "layer": "dsafety",
        "prompt": SYSTEM_PROMPT_DSAFETY,
        "name": "Dsafety",
    },
}
DEFAULT_BRAND = "dlimit"


# ============================================================
# RAG — hybrid search
# ============================================================
def hybrid_search(question: str, layer: str, top_k: int = TOP_K):
    dense = voyage.embed([question], model=DENSE_MODEL, input_type="query",
                         truncation=True).embeddings[0]
    sparse_emb = list(bm25.query_embed([question]))[0]
    sparse = SparseVector(indices=sparse_emb.indices.tolist(),
                          values=sparse_emb.values.tolist())
    flt = Filter(must=[FieldCondition(key="layer", match=MatchValue(value=layer))])
    res = qclient.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=dense, using=DENSE_VECTOR_NAME, limit=top_k * 4, filter=flt),
            Prefetch(query=sparse, using=SPARSE_VECTOR_NAME, limit=top_k * 4, filter=flt),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return res.points


# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)


# CORS para los dominios de producción
ALLOWED_ORIGINS = {
    "https://www.dsafety.es",
    "https://dsafety.es",
    "https://www.dlimit.net",
    "https://dlimit.net",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    # añadir aquí orígenes de desarrollo o staging si hace falta
}


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Max-Age"] = "3600"
    return response


@app.route("/api/chat", methods=["OPTIONS"])
def chat_options():
    return ("", 204)


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "brands": list(BRAND_CONFIG.keys())})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    brand = (data.get("brand") or DEFAULT_BRAND).lower()

    if not question:
        return jsonify({"error": "empty"}), 400

    if brand not in BRAND_CONFIG:
        return jsonify({"error": f"unknown brand '{brand}'", "valid": list(BRAND_CONFIG.keys())}), 400

    config = BRAND_CONFIG[brand]
    layer = config["layer"]
    system_prompt = config["prompt"]

    points = hybrid_search(question, layer=layer)
    sources = []
    context_parts = []
    for i, p in enumerate(points, 1):
        pl = p.payload
        src = {
            "n": i,
            "title": pl.get("title") or pl.get("source", "?"),
            "source": pl.get("source", "?"),
            "page": pl.get("page", 1),
            "doc_type": pl.get("doc_type", "?"),
            "url": pl.get("source_path", ""),
            "score": round(float(p.score), 3),
            "snippet": pl.get("text", "")[:300],
        }
        sources.append(src)
        loc = f"{pl.get('source','?')}, p.{pl.get('page','?')}"
        context_parts.append(f"[{i}] ({loc}) {pl.get('text','')}")

    if not points:
        fallback = "No tengo información sobre eso en mi base de conocimiento."
        if brand == "dsafety":
            fallback += " ¿Puedes contactar a un distribuidor autorizado en https://www.dsafety.es/distribuidores/ o al equipo técnico?"
        else:
            fallback += " ¿Puedes contactar a nuestro equipo comercial?"
        return jsonify({"answer": fallback, "sources": [], "brand": brand})

    user_msg = (
        f"Pregunta del cliente:\n{question}\n\n"
        f"Contexto KB {config['name']}:\n" + "\n\n".join(context_parts) +
        f"\n\nResponde basándote SOLO en el contexto. Cita con [n]."
    )
    resp = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=900,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    return jsonify({
        "answer": resp.content[0].text,
        "sources": sources,
        "tokens": {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens},
        "model": CLAUDE_MODEL,
        "brand": brand,
    })


# ============================================================
# DEMO HTML — sigue sirviendo el frontend Dlimit
# (el frontend Dsafety se inyecta en su propio sitio web)
# ============================================================
HTML_PAGE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Chatbot multi-brand · Demo</title>
<style>
  :root {
    --primary:#1F3A5F; --accent:#2E75B6; --bg:#F5F7FA; --card:#fff;
    --muted:#6b6b6b; --border:#e2e6ea; --user:#E8F0FB; --bot:#fff;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:#222;height:100vh;display:flex;flex-direction:column}
  header{background:var(--primary);color:#fff;padding:14px 22px;display:flex;align-items:center;gap:14px}
  header h1{margin:0;font-size:18px;font-weight:700;letter-spacing:.5px}
  header .badge{background:var(--accent);padding:3px 10px;border-radius:12px;font-size:12px}
  header .meta{margin-left:auto;font-size:13px;opacity:.85}
  header select{margin-left:14px;padding:5px 10px;border-radius:6px;border:1px solid rgba(255,255,255,.3);background:rgba(255,255,255,.1);color:#fff;font-size:13px}
  main{flex:1;overflow:hidden;display:grid;grid-template-columns:1fr 320px;max-width:1400px;width:100%;margin:0 auto;padding:18px;gap:18px}
  #chat{background:var(--card);border:1px solid var(--border);border-radius:14px;display:flex;flex-direction:column;overflow:hidden}
  #messages{flex:1;overflow-y:auto;padding:22px}
  .msg{margin:14px 0;padding:14px 16px;border-radius:14px;line-height:1.55;max-width:85%;white-space:pre-wrap}
  .msg.user{background:var(--user);margin-left:auto;border-bottom-right-radius:4px}
  .msg.bot{background:var(--bot);border:1px solid var(--border);border-bottom-left-radius:4px}
  .msg.bot strong{color:var(--primary)}
  .msg.thinking{font-style:italic;color:var(--muted);background:transparent;border:1px dashed var(--border)}
  .form{display:flex;gap:10px;padding:14px;border-top:1px solid var(--border);background:#fafbfc}
  textarea{flex:1;border:1px solid var(--border);border-radius:10px;padding:10px 12px;font-size:14px;font-family:inherit;resize:none;height:48px}
  button{background:var(--primary);color:#fff;border:0;border-radius:10px;padding:0 20px;font-size:14px;font-weight:600;cursor:pointer}
  button:disabled{opacity:.5;cursor:not-allowed}
  aside{background:var(--card);border:1px solid var(--border);border-radius:14px;overflow-y:auto;padding:18px}
  aside h3{margin:0 0 12px 0;font-size:14px;color:var(--primary);text-transform:uppercase;letter-spacing:1px}
  .src{border:1px solid var(--border);border-radius:10px;padding:10px;margin-bottom:10px;font-size:12px;line-height:1.5}
  .src .head{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
  .src .num{background:var(--accent);color:#fff;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;margin-right:6px}
  .src .score{color:var(--muted);font-size:11px}
  .src .title{font-weight:600;color:var(--primary);margin-bottom:4px}
  .src .meta{color:var(--muted);font-size:11px;margin-bottom:6px}
  .src .snippet{color:#444}
  .stats{font-size:11px;color:var(--muted);padding:8px 12px;background:#f0f4f8;border-radius:8px;margin-top:14px}
  .empty{color:var(--muted);font-size:13px;text-align:center;padding:30px 14px}
</style>
</head>
<body>
<header>
  <h1>Chatbot multi-brand · Demo</h1>
  <select id="brand">
    <option value="dlimit">Dlimit</option>
    <option value="dsafety">Dsafety</option>
  </select>
  <span class="badge">RAG · Claude + Voyage + Qdrant</span>
  <span class="meta" id="brandMeta">Capa: publica</span>
</header>
<main>
  <div id="chat">
    <div id="messages">
      <div class="empty">Escribe una pregunta. Cambia la marca arriba para consultar la KB de Dlimit o Dsafety.</div>
    </div>
    <form id="form" class="form" onsubmit="return ask(event)">
      <textarea id="q" placeholder="Pregunta…" required></textarea>
      <button type="submit" id="send">Enviar</button>
    </form>
  </div>
  <aside id="sidebar">
    <h3>Fuentes</h3>
    <div id="sources"><div class="empty">Aparecen tras enviar.</div></div>
  </aside>
</main>
<script>
const messages = document.getElementById('messages');
const sourcesDiv = document.getElementById('sources');
const q = document.getElementById('q');
const send = document.getElementById('send');
const brandSel = document.getElementById('brand');
const brandMeta = document.getElementById('brandMeta');

brandSel.addEventListener('change', () => {
  brandMeta.textContent = 'Capa: ' + (brandSel.value === 'dsafety' ? 'dsafety' : 'publica');
});

function add(role, html, cls=''){
  const d = document.createElement('div');
  d.className = 'msg '+role+(cls?' '+cls:'');
  d.innerHTML = html;
  messages.appendChild(d);
  messages.scrollTop = messages.scrollHeight;
  return d;
}

function fmt(text){
  return text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/\[(\d+)\]/g,'<sup style="color:var(--accent);font-weight:700">[$1]</sup>');
}

function renderSources(srcs, tokens){
  if(!srcs.length){ sourcesDiv.innerHTML='<div class="empty">Sin fuentes</div>'; return; }
  let html = '';
  for(const s of srcs){
    const link = s.url ? `<a href="${s.url}" target="_blank" style="color:var(--accent);text-decoration:none">↗</a>` : '';
    html += `<div class="src">
      <div class="head"><div><span class="num">${s.n}</span><strong>${s.doc_type}</strong> · p.${s.page}</div>
      <span class="score">score ${s.score}</span></div>
      <div class="title">${s.title}</div>
      <div class="meta">${s.source} ${link}</div>
      <div class="snippet">${s.snippet}…</div>
    </div>`;
  }
  if(tokens) html += `<div class="stats">Tokens: in=${tokens.in} · out=${tokens.out}</div>`;
  sourcesDiv.innerHTML = html;
}

async function ask(e){
  if(e) e.preventDefault();
  const text = q.value.trim();
  if(!text) return false;
  add('user', fmt(text));
  q.value='';
  send.disabled=true;
  const thinking = add('bot', 'Pensando…', 'thinking');
  try{
    const r = await fetch('/api/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:text, brand:brandSel.value})
    });
    const data = await r.json();
    thinking.remove();
    add('bot', fmt(data.answer));
    renderSources(data.sources||[], data.tokens);
  }catch(err){
    thinking.remove();
    add('bot', '<em>Error: '+err.message+'</em>');
  }
  send.disabled=false;
  q.focus();
  return false;
}

q.addEventListener('keydown', e=>{
  if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); ask(); }
});
q.focus();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("[INFO] Iniciando chatbot multi-brand (Dlimit + Dsafety)")
    print("[INFO] Brands disponibles:", list(BRAND_CONFIG.keys()))
    print("[INFO] Abre http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)

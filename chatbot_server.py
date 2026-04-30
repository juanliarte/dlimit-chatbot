"""
Chatbot Dlimit · servidor demo local
====================================
Backend Flask + frontend HTML embebido.
Pipeline: pregunta → embed (Voyage+BM25) → Qdrant hybrid search → Claude → respuesta.

Uso:
  export ANTHROPIC_API_KEY=...
  export VOYAGE_API_KEY=...
  export QDRANT_URL=https://...:6333
  export QDRANT_API_KEY=...
  python chatbot_server.py
  → abre http://localhost:5000 en tu navegador
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

SYSTEM_PROMPT = """Eres el asistente conversacional de Dlimit Tactic S.L.,
empresa española (Mataró) especializada en postes separadores y barreras
retráctiles para gestión de colas y delimitación de espacios B2B.

REGLA DE IDIOMA (prioridad máxima):
- Detecta el idioma del último mensaje del cliente y responde SIEMPRE en ese
  mismo idioma. Soporta como mínimo: español, inglés, francés, alemán,
  italiano, portugués y catalán.
- Si el mensaje es muy corto o ambiguo (un saludo como "hello", "ok", "hi"...),
  responde en el idioma de ese saludo. Ej: "hello" → respuesta en inglés.
- Solo si el idioma es realmente imposible de determinar, usa español.

Reglas de contenido:
- Responde solo con información presente en el contexto proporcionado.
- NUNCA incluyas referencias numéricas tipo [1], [2], (fuente 3), etc. en la
  respuesta. Integra la información de forma natural sin marcadores.
- Si la información no está en el contexto, dilo amablemente y ofrece contactar
  al equipo comercial en info@dlimit.es.
- Tono: profesional, directo, técnico cuando hace falta, comercialmente útil.
- No inventes referencias, modelos, precios ni ensayos.
- Formato: markdown ligero (negrita en lo importante, listas si aplica).
- Respuestas concisas: ve al grano, sin preámbulos innecesarios."""


def hybrid_search(question: str, top_k: int = TOP_K, layer: str = "publica"):
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


def fallback_no_context(question: str) -> str:
    """Genera una respuesta breve en el idioma del cliente cuando no hay contexto."""
    resp = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=180,
        system=(
            "Responde SIEMPRE en el mismo idioma que el mensaje del cliente "
            "(español, inglés, francés, alemán, italiano, portugués o catalán). "
            "Si es un saludo como 'hello', responde en ese idioma. "
            "Sé breve, cordial y profesional. NUNCA uses marcadores [1], [2] ni similares."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Mensaje del cliente: \"{question}\"\n\n"
                "Si es un saludo o pregunta general, salúdale brevemente y "
                "preséntate como el asistente de Dlimit Tactic, especialista "
                "en postes separadores y barreras retráctiles, e invítale a "
                "preguntar sobre productos, materiales o sectores. "
                "Si pregunta algo concreto que no puedes responder, dile amablemente "
                "que no tienes esa información y que puede contactar a info@dlimit.es. "
                "Todo en el idioma del cliente."
            ),
        }],
    )
    return resp.content[0].text


app = Flask(__name__)


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "empty"}), 400

    points = hybrid_search(question)
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
        context_parts.append(f"({loc}) {pl.get('text','')}")

    if not points:
        return jsonify({"answer": fallback_no_context(question), "sources": []})

    user_msg = (
        f"Customer message (respond in the SAME language as this message):\n"
        f"{question}\n\n"
        f"Knowledge base context (Dlimit):\n" + "\n\n".join(context_parts) +
        f"\n\nUse ONLY the context above. Do NOT add reference markers like [1], [2]."
    )
    resp = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=900,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return jsonify({
        "answer": resp.content[0].text,
        "sources": sources,
        "tokens": {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens},
        "model": CLAUDE_MODEL,
    })


HTML_PAGE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Dlimit</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --accent:#2A9D2A;
    --ink:#111;
    --muted:#8a8a8a;
    --rule:#ececec;
    --bg:#fff;
    --user-bg:#f4f4f4;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{
    margin:0;
    font-family:'Inter',-apple-system,BlinkMacSystemFont,'Helvetica Neue',Arial,sans-serif;
    font-weight:400;
    font-size:15px;
    line-height:1.6;
    color:var(--ink);
    background:var(--bg);
    -webkit-font-smoothing:antialiased;
  }
  .wrap{
    max-width:680px;
    margin:0 auto;
    height:100vh;
    display:flex;
    flex-direction:column;
    padding:0 24px;
  }
  #messages{
    flex:1;
    overflow-y:auto;
    padding:48px 0 24px;
  }
  .empty{
    color:var(--muted);
    font-size:15px;
    line-height:1.6;
    padding:80px 0 0;
    max-width:420px;
  }
  .empty strong{color:var(--ink);font-weight:500}
  .msg{
    margin:0 0 28px;
    line-height:1.65;
    white-space:pre-wrap;
    font-size:15px;
  }
  .msg.user{
    background:var(--user-bg);
    padding:12px 16px;
    border-radius:14px;
    margin-left:auto;
    max-width:80%;
    width:fit-content;
  }
  .msg.bot{
    color:var(--ink);
    padding:0;
  }
  .msg.bot strong{font-weight:600}
  .msg.bot h3,.msg.bot h4{
    font-weight:600;
    margin:18px 0 6px;
    font-size:15px;
  }
  .msg.bot ul,.msg.bot ol{padding-left:20px;margin:8px 0}
  .msg.bot li{margin:4px 0}
  .msg.thinking{
    color:var(--muted);
    font-size:14px;
  }
  .dots{display:inline-block}
  .dots span{
    display:inline-block;
    width:4px;height:4px;
    border-radius:50%;
    background:var(--muted);
    margin:0 2px;
    animation:b 1.4s infinite;
  }
  .dots span:nth-child(2){animation-delay:.2s}
  .dots span:nth-child(3){animation-delay:.4s}
  @keyframes b{0%,80%,100%{opacity:.2}40%{opacity:1}}

  form{
    border-top:1px solid var(--rule);
    padding:18px 0 24px;
    display:flex;
    gap:12px;
    align-items:flex-end;
    background:var(--bg);
  }
  textarea{
    flex:1;
    border:0;
    outline:0;
    resize:none;
    font:inherit;
    font-size:15px;
    color:var(--ink);
    padding:10px 0;
    min-height:24px;
    max-height:200px;
    background:transparent;
    line-height:1.5;
  }
  textarea::placeholder{color:var(--muted)}
  button{
    border:0;
    background:var(--accent);
    color:#fff;
    width:36px;height:36px;
    border-radius:50%;
    cursor:pointer;
    display:flex;align-items:center;justify-content:center;
    transition:opacity .15s;
    flex-shrink:0;
  }
  button:hover{opacity:.85}
  button:disabled{opacity:.3;cursor:not-allowed}
  button svg{width:16px;height:16px}
</style>
</head>
<body>
<div class="wrap">
  <div id="messages">
    <div class="empty">
      Asistente <strong>Dlimit</strong>. Pregunta lo que quieras sobre productos, materiales, instalación o sectores.
    </div>
  </div>
  <form id="form" onsubmit="return ask(event)">
    <textarea id="q" placeholder="Escribe tu pregunta…" required rows="1"></textarea>
    <button type="submit" id="send" aria-label="Enviar">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>
    </button>
  </form>
</div>
<script>
const messages = document.getElementById('messages');
const q = document.getElementById('q');
const send = document.getElementById('send');
const empty = document.querySelector('.empty');

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
    .replace(/^### (.+)$/gm,'<h4>$1</h4>')
    .replace(/^## (.+)$/gm,'<h3>$1</h3>')
    .replace(/^# (.+)$/gm,'<h3>$1</h3>')
    .replace(/^\s*[-*]\s+(.+)$/gm,'<li>$1</li>')
    .replace(/(<li>.*?<\/li>)(?:\n(?=<li>))/gs,'$1')
    .replace(/((?:<li>.*?<\/li>\s*)+)/gs,'<ul>$1</ul>');
}

function autoresize(){
  q.style.height='auto';
  q.style.height = Math.min(q.scrollHeight, 200)+'px';
}

async function ask(e){
  if(e) e.preventDefault();
  const text = q.value.trim();
  if(!text) return false;
  if(empty && empty.parentNode) empty.remove();
  add('user', text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'));
  q.value=''; autoresize();
  send.disabled=true;
  const thinking = add('bot', '<span class="dots"><span></span><span></span><span></span></span>', 'thinking');
  try{
    const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:text})});
    const data = await r.json();
    thinking.remove();
    add('bot', fmt(data.answer));
  }catch(err){
    thinking.remove();
    add('bot', '<em>Error: '+err.message+'</em>');
  }
  send.disabled=false;
  q.focus();
  return false;
}

q.addEventListener('input', autoresize);
q.addEventListener('keydown', e=>{
  if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); ask(); }
});

q.focus();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("[INFO] Iniciando servidor Dlimit Chatbot Demo...")
    print("[INFO] Abre http://localhost:5000 en tu navegador")
    app.run(host="127.0.0.1", port=5000, debug=False)

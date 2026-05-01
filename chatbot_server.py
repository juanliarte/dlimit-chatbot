"""
Chatbot Dlimit · servidor demo local
====================================
Backend Flask + frontend HTML embebido.
Pipeline: pregunta -> embed (Voyage+BM25) -> Qdrant hybrid search -> Claude -> respuesta.

Uso:
  export ANTHROPIC_API_KEY=...
  export VOYAGE_API_KEY=...
  export QDRANT_URL=https://...:6333
  export QDRANT_API_KEY=...
  python chatbot_server.py
  -> abre http://localhost:5000 en tu navegador
"""
from __future__ import annotations

import os
import sys
import json
import threading
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone

from flask import Flask, request, jsonify, Response
import voyageai
from anthropic import Anthropic
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, SparseVector,
    Prefetch, FusionQuery, Fusion,
)
from fastembed import SparseTextEmbedding

# Módulo de respuesta automática con catálogo + tarifa
from email_responder import send_info_email


COLLECTION = "dlimit-kb"
DENSE_VECTOR_NAME = "voyage-dense"
SPARSE_VECTOR_NAME = "bm25-sparse"
DENSE_MODEL = "voyage-3-large"
SPARSE_MODEL = "Qdrant/bm25"
CLAUDE_MODEL = "claude-haiku-4-5"
TOP_K = 6

# Brevo lead capture (opcional - si no hay key, se desactiva sin romper)
BREVO_API_KEY = os.environ.get("BREVO_API_KEY", "").strip()
LEAD_NOTIFICATION_EMAIL = os.environ.get("LEAD_NOTIFICATION_EMAIL", "info@dlimit.es").strip()
LEAD_SENDER_EMAIL = os.environ.get("LEAD_SENDER_EMAIL", "noreply@dlimit.es").strip()
LEAD_SENDER_NAME = "Dlimit Chatbot"


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

SYSTEM_PROMPT = """Eres asesor comercial-tecnico de Dlimit Tactic S.L. (Mataro, Espana),
fabricante europeo de postes separadores con cinta extensible y
barreras retractiles para gestion de colas y delimitacion de espacios B2B.

REGLA DE IDIOMA (PRIORIDAD MAXIMA):
- Detecta el idioma del ultimo mensaje del cliente y responde SIEMPRE en ese
  mismo idioma. Soporta: espanol, ingles, frances, aleman, italiano,
  portugues, catalan.
- Saludo corto ("hello", "ok", "ciao") -> responde en ese idioma.
- Solo si es imposible determinar, usa espanol.

ROL:
No eres un asistente generico ni un catalogo. Eres un asesor que:
1. Entiende necesidad real del cliente
2. Recomienda solucion concreta de Dlimit
3. Avanza la conversacion hacia compra, presupuesto o lead cualificado
4. Acompana con material descargable cuando ayuda al cierre

REGLAS DE COMPORTAMIENTO:
- Respuestas cortas: 1-3 lineas maximo. Listas solo si son imprescindibles.
- Termina cada respuesta con una pregunta o un CTA que avance la venta.
- NUNCA des precio directo. Pide contexto antes (uso propio o reventa, cantidad, sector).
- NUNCA des toda la info de golpe. Ve por capas segun lo que el cliente pregunta.
- NUNCA inventes referencias, modelos, precios, plazos ni ensayos.
- NUNCA incluyas marcadores tipo [1], [2], (fuente 3), etc.
- Responde solo con informacion del contexto recuperado de la KB.
  Si no esta en contexto, dilo y escala (no inventes).

ESTILO:
Tono: profesional, directo, tecnico cuando hace falta, comercialmente util.
Tratamiento: "tu" para web publica (cercano, B2B moderno).
Prohibido: "estaremos encantados", "soluciones innovadoras", "amplio abanico",
"no dudes en contactarnos", "estamos a tu disposicion".
Preferido: "Depende del uso", "Lo habitual aqui es", "Te explico rapido",
"Para ese sector usamos...", "Con X reduces Y".

FAMILIAS DE PRODUCTO DLIMIT:
- **Dbasic**: gama economica, uso ligero (oficinas, eventos puntuales)
- **Dstandard**: gama estandar, uso medio (retail, hosteleria)
- **Dclassic**: gama clasica con base elegante (hoteles, recepciones premium)
- **Dline**: gama profesional, base reforzada
- **Dsafety**: enfoque seguridad y exterior (industria, obras)
- **Dterminal**: pensado para transporte (aeropuertos, estaciones)
- **Dlimit**: gama insignia, premium, maxima durabilidad
- **Daccessory**: bastidores, carteles, cordones, ganchos, accesorios

Cintas estandar 4m (mayor que la media de mercado: 2-3m).
Opciones intensivas: 6m y 9m. Personalizacion CMYK sublimacion
(logos, colores corporativos, mensajes).

DIFERENCIACION VS COMPETENCIA:
- Fabricacion propia en Mataro (Espana), no reseller
- 8 familias para cubrir todos los presupuestos y entornos
- Cinta 4m estandar = menos postes para misma distancia = menos coste total
- Personalizacion completa: cinta CMYK + colores RAL del poste + bases
- Plazos cortos en Espana y Europa
- Catalogo extensible: bastidores A3/A4, ganchos, cordones, paredes, tapas

RECURSOS DESCARGABLES (usalos siempre que ayuden al cierre):
La web dlimit.net tiene catalogos PDF y fichas tecnicas que el cliente
puede descargar gratis, sin formulario. Acompana SIEMPRE que detectes
interes concreto en una familia o sector.

Catalogos por familia (URLs directas):
- Dbasic: https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dbasic-ES.pdf
- Dstandard: https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dstandard-ES.pdf
- Dclassic: https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dclassic-ES.pdf
- Dline: https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dline-ES.pdf
- Dsafety: https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dsafety-ES.pdf
- Dterminal: https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dterminal-ES.pdf
- Dlimit (gama insignia): https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dlimit-ES.pdf

Pagina general de descargas: https://www.dlimit.net/descargas.html

Cuando y como acompanar con PDF:
- Cliente menciona una familia -> "Te paso el PDF de [familia]: [URL]"
- Cliente pregunta por un sector -> "Te dejo el PDF de [familia recomendada]
  para que veas medidas y acabados: [URL]"
- Cliente compara opciones -> Pasar 2 PDFs para comparar.
- Cliente pide ficha tecnica -> Enlazar PDF directo de la familia.

Formato del enlace: SIEMPRE URL completa, en su propia linea. Sin acortar.

LOGICA DE VENTA:
1. Detectar intencion: informacion, presupuesto, comparacion, distribucion
2. Entender contexto: uso (propio/reventa), sector, cantidad, entorno
3. Recomendar familia y configuracion
4. Acompanar con PDF si ayuda al cierre
5. Avanzar a accion: presupuesto, datos de contacto, llamada con comercial

GESTION DE INTENCIONES:

Si pregunta por PRECIO:
-> "Depende del uso. Es para uso propio o reventa? Que cantidad estimas?"
-> Tras respuesta: "Te preparo presupuesto, me dejas tu email?"

Si es TECNICO (materiales, mecanismos, ensayos):
-> Responder con datos del contexto.
-> Ofrecer PDF: "Te paso la ficha completa: [URL]"
-> Preguntar entorno (interior/exterior, intensidad, costa, etc.).

Si esta PERDIDO o pregunta general:
-> Ofrecer 3 opciones: comprar | info tecnica | distribucion.
-> "Que te interesa mas?"

Si quiere COMPRAR:
-> Pedir cantidad, sector, ubicacion.
-> Acompanar con PDF de la familia recomendada.
-> Cerrar con presupuesto o derivar a comercial.

Si pide DISTRIBUCION/REVENTA:
-> Pedir pais, sector, volumen anual estimado.
-> Escalar a info@dlimit.es

Si pregunta por SECTOR (aeropuertos, hospitales, hoteles, retail, eventos,
ferias, museos, hosteleria, estaciones):
-> Recomendar familia + configuracion especifica.
-> Pasar PDF de la familia recomendada.

Si pregunta cosas OFF-TOPIC:
-> "Soy el asesor de Dlimit, te ayudo con sistemas de gestion de colas.
   Que necesitas?"

OBJECIONES:

"Caro" / "Otros son mas baratos":
-> "Con cinta de 4m necesitas menos postes que con 2-3m. Coste total mas bajo.
   Quieres que te lo calculemos?"

"Ya tengo proveedor":
-> "Entendido. Si en algun momento quieres comparar acabados o personalizacion,
   estamos aqui. Que proveedor usas?"

"No estoy seguro":
-> "Normal. Para que entorno lo usarias? Con eso te oriento mejor."

"Quiero mirarlo antes de decidir":
-> "Perfecto. Te paso el catalogo de la gama que mejor te encaja: [URL]
   De que sector hablamos?"

CAPTURA DE LEAD:
Cuando el cliente muestra intencion clara de compra/presupuesto, pide
de uno en uno (no como formulario):
1. Email de contacto (siempre)
2. Empresa (siempre)
3. Cantidad estimada
4. Sector / uso final
5. Pais o region (para plazos y envio)

Recordatorio RGPD si pide email: "Solo lo usamos para enviarte el presupuesto."


ENVIO AUTOMATICO DE INFORMACION POR EMAIL:

Cuando detectes interes concreto del cliente (cantidad mencionada, sector
especifico, familia recomendada, intencion de compra clara), OFRECE
ESPONTANEAMENTE el envio de informacion por email - no esperes a que el
cliente lo pida. Frase tipo:

  "Quieres que te envie el catalogo completo y la tarifa de precios por
   email? Te lo mando ahora mismo y Ester, nuestra responsable comercial,
   te llama en 24 h para condiciones especiales."

Si el cliente acepta, pidele su email (uno solo, sin formulario):
  "Perfecto, a que email te lo mando?"

Cuando obtengas un email VALIDO en la conversacion y haya interes real,
al FINAL de tu respuesta y EN UNA LINEA SEPARADA, incluye este marcador
EXACTO:

  [SEND_INFO_EMAIL: email@cliente.com]

El sistema detecta automaticamente ese marcador y dispara el envio del
email con catalogo + tarifa adjuntos. Ester ve la conversacion completa
y llama al cliente en 24 h.

REGLAS DEL MARCADOR (criticas):
- Solo emitelo si el cliente ha dado un email valido y ha aceptado el envio.
- Una vez emitido en una conversacion, NO lo vuelvas a emitir.
- NUNCA inventes un email. Si el cliente no lo dio, pideselo, no inventes.
- NUNCA emitas el marcador si el cliente NO ha aceptado el envio.
- En la respuesta visible al cliente, confirma con frase breve:
  "Perfecto, te lo envio ahora mismo a [email]. Ester te llama en 24 h."
- El marcador se quita automaticamente del mensaje visible al cliente,
  solo el sistema lo lee.

ESCALADO A HUMANO:
Escala en estos casos:
- Volumen >50 postes
- Personalizacion avanzada (logos exclusivos, colores fuera de RAL estandar)
- Solicitud de distribucion / reventa
- Condiciones de pago especiales
- Proyecto publico o licitacion

Mensaje de escalado:
"Esto lo revisa mejor nuestro equipo comercial. Escribenos a info@dlimit.es
o llama al +34 932 526 915 (L-V 9:00-18:00 CET). Te paso ya o quieres
que te llamen ellos?"

FORMATO:
- Markdown ligero: negrita en terminos clave (familia, numero, sector).
- Listas solo cuando enumeras opciones reales (3+ items).
- Sin emojis salvo respuesta a saludo en idioma extranjero.
- URLs siempre en linea propia, completas, sin acortar.

REGLA FINAL:
SIEMPRE: entender -> guiar -> acompanar (con PDF si aplica) -> cerrar.
NUNCA: responder sin preguntar, explicar sin avanzar, informar sin intencion de venta."""


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
    """Genera respuesta breve en el idioma del cliente cuando no hay contexto."""
    resp = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        system=(
            "Eres asesor comercial de Dlimit Tactic, fabricante de postes separadores "
            "con cinta extensible. Responde SIEMPRE en el mismo idioma que el cliente "
            "(espanol, ingles, frances, aleman, italiano, portugues, catalan). "
            "Saluda brevemente, presentate, y ofrece 3 opciones para avanzar: "
            "(1) informacion de producto, (2) presupuesto, (3) distribucion. "
            "Termina con una pregunta concreta. Tono profesional, directo, sin "
            "frases comerciales vacias. Maximo 3 lineas. NUNCA uses [1], [2]."
        ),
        messages=[{
            "role": "user",
            "content": (
                f'Mensaje del cliente: "{question}"\n\n'
                "Si es saludo, salude breve y presenta las 3 opciones. "
                "Si es pregunta concreta sin contexto disponible, di amablemente "
                "que no tienes esa informacion y ofrece contactar al equipo "
                "comercial en info@dlimit.es. Todo en el idioma del cliente."
            ),
        }],
    )
    return resp.content[0].text



# ============================================================
# CAPTURA DE LEADS - INTEGRACION BREVO
# ============================================================

def detect_lead(question, answer):
    """Clasifica si la conversacion es un lead. Llama a Claude Haiku."""
    try:
        prompt = (
            f"Analiza esta interaccion del chatbot comercial de Dlimit (postes separadores B2B):\n\n"
            f"CLIENTE: {question}\n"
            f"BOT: {answer}\n\n"
            f"Devuelve SOLO un JSON valido (sin markdown) con estos campos:\n"
            f'{{"is_lead": true|false, "email": "..."|null, "sector": "..."|null, '
            f'"quantity": "..."|null, "country": "..."|null, "urgency": "low"|"medium"|"high", '
            f'"summary": "...", "language": "es"|"en"|"fr"|"de"|"it"|"pt"|"ca"}}\n\n'
            f"Reglas:\n"
            f"- is_lead=true SOLO si el cliente muestra intencion clara de compra/presupuesto/distribucion\n"
            f"  (mencionar cantidad, sector, ubicacion, dejar email, pedir presupuesto explicito)\n"
            f"- Saludos, preguntas generales o tecnicas sin contexto comercial -> is_lead=false\n"
            f"- email: extraer email del cliente si lo escribio (regex valido), si no null\n"
            f"- urgency=high si menciona plazos cortos, evento proximo, urgencia explicita\n"
            f"- summary: 1-2 lineas en espanol describiendo el interes del cliente\n"
        )
        resp = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        print(f"[detect_lead] error: {e}", flush=True)
        return {"is_lead": False}


def brevo_request(method, path, body=None):
    """Llama a la API de Brevo. Devuelve (status_code, body)."""
    url = f"https://api.brevo.com/v3{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "api-key": BREVO_API_KEY,
            "accept": "application/json",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode("utf-8")
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return 0, str(e)


def upsert_brevo_contact(lead):
    """Crea o actualiza Contact en Brevo CRM."""
    if not lead.get("email"):
        return None
    body = {
        "email": lead["email"],
        "attributes": {
            "SECTOR_DLIMIT": lead.get("sector") or "",
            "QUANTITY_ESTIMATE": lead.get("quantity") or "",
            "COUNTRY_DLIMIT": lead.get("country") or "",
            "URGENCY": lead.get("urgency") or "",
            "LANGUAGE": lead.get("language") or "",
            "LAST_INTERACTION_SUMMARY": lead.get("summary") or "",
            "SOURCE": "chatbot-dlimit-net",
        },
        "listIds": [],
        "updateEnabled": True,
    }
    status, resp = brevo_request("POST", "/contacts", body)
    if status in (200, 201, 204):
        return resp.get("id") if isinstance(resp, dict) else -1
    print(f"[brevo_contact] error {status}: {resp}", flush=True)
    return None


def send_lead_email(lead, question, answer):
    """Envia email transaccional al equipo comercial."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    urgency_color = {"high": "#D32F2F", "medium": "#F57C00", "low": "#666666"}.get(
        lead.get("urgency", "low"), "#666666"
    )
    answer_html = answer.replace(chr(10), "<br/>")
    html = (
        '<html><body style="font-family: Arial, sans-serif; color: #111; max-width: 640px;">'
        '<h2 style="color: #2A9D2A; margin-bottom: 4px;">Nuevo lead chatbot Dlimit</h2>'
        f'<p style="color: #888; font-size: 12px; margin-top: 0;">{timestamp}</p>'
        '<table style="border-collapse: collapse; margin: 16px 0; width: 100%;">'
        f'<tr><td style="padding:6px 12px;background:#f5f5f5;font-weight:bold;">Email</td><td style="padding:6px 12px;"><a href="mailto:{lead.get("email","")}">{lead.get("email","(no facilitado)")}</a></td></tr>'
        f'<tr><td style="padding:6px 12px;background:#f5f5f5;font-weight:bold;">Sector</td><td style="padding:6px 12px;">{lead.get("sector") or "-"}</td></tr>'
        f'<tr><td style="padding:6px 12px;background:#f5f5f5;font-weight:bold;">Cantidad</td><td style="padding:6px 12px;">{lead.get("quantity") or "-"}</td></tr>'
        f'<tr><td style="padding:6px 12px;background:#f5f5f5;font-weight:bold;">Pais</td><td style="padding:6px 12px;">{lead.get("country") or "-"}</td></tr>'
        f'<tr><td style="padding:6px 12px;background:#f5f5f5;font-weight:bold;">Idioma</td><td style="padding:6px 12px;">{lead.get("language") or "-"}</td></tr>'
        f'<tr><td style="padding:6px 12px;background:#f5f5f5;font-weight:bold;">Urgencia</td><td style="padding:6px 12px;color:{urgency_color};font-weight:bold;">{(lead.get("urgency") or "low").upper()}</td></tr>'
        '</table>'
        '<h3 style="color:#2A9D2A;margin-bottom:4px;">Resumen</h3>'
        f'<p>{lead.get("summary","")}</p>'
        '<h3 style="color:#2A9D2A;margin-bottom:4px;">Conversacion</h3>'
        f'<div style="background:#f5f5f5;padding:12px;border-radius:8px;margin-bottom:8px;"><strong>Cliente:</strong><br/>{question}</div>'
        f'<div style="background:#fff;border:1px solid #eee;padding:12px;border-radius:8px;"><strong>Bot:</strong><br/>{answer_html}</div>'
        '<p style="color:#888;font-size:12px;margin-top:24px;">Lead capturado automaticamente por chatbot Dlimit IA. Contacto registrado tambien en Brevo CRM.</p>'
        '</body></html>'
    )
    body = {
        "sender": {"email": LEAD_SENDER_EMAIL, "name": LEAD_SENDER_NAME},
        "to": [{"email": LEAD_NOTIFICATION_EMAIL, "name": "Equipo Comercial Dlimit"}],
        "subject": f'[LEAD {(lead.get("urgency") or "low").upper()}] {lead.get("sector") or "Chatbot"} - {lead.get("email") or "Sin email"}',
        "htmlContent": html,
    }
    status, resp = brevo_request("POST", "/smtp/email", body)
    if status in (200, 201):
        return True
    print(f"[brevo_email] error {status}: {resp}", flush=True)
    return False


def process_lead_async(question, answer):
    """Detecta lead y notifica. Thread separado para no bloquear respuesta."""
    if not BREVO_API_KEY:
        return
    try:
        lead = detect_lead(question, answer)
        if not lead.get("is_lead"):
            return
        print(f"[LEAD detected] {lead}", flush=True)
        if lead.get("email"):
            cid = upsert_brevo_contact(lead)
            print(f"[brevo_contact] upserted id={cid}", flush=True)
        send_lead_email(lead, question, answer)
        print(f"[brevo_email] sent to {LEAD_NOTIFICATION_EMAIL}", flush=True)
    except Exception as e:
        print(f"[process_lead_async] error: {e}", flush=True)


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
        answer_text = fallback_no_context(question)
        if BREVO_API_KEY:
            threading.Thread(target=process_lead_async, args=(question, answer_text), daemon=True).start()
        return jsonify({"answer": answer_text, "sources": []})

    user_msg = (
        f"Customer message (respond in the SAME language as this message):\n"
        f"{question}\n\n"
        f"Knowledge base context (Dlimit):\n" + "\n\n".join(context_parts) +
        f"\n\nUse ONLY the context above. Do NOT add reference markers like [1], [2]."
        f" Follow the commercial advisor logic: understand, guide, accompany with PDF if helpful, close."
    )
    resp = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=900,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    answer_text = resp.content[0].text

    # ─── Detectar marcador de envio automatico de catalogo+tarifa ───
    _email_marker = re.search(r'\[SEND_INFO_EMAIL:\s*([^\]\s]+)\s*\]', answer_text)
    if _email_marker:
        _target_email = _email_marker.group(1).strip().rstrip('.,;:')
        if "@" in _target_email and "." in _target_email.split("@")[-1]:
            try:
                send_info_email(
                    client_email=_target_email,
                    conversation_summary=(
                        f"Pregunta del cliente:\n{question}\n\n"
                        f"Respuesta del bot:\n{answer_text[:800]}"
                    ),
                    detected_language="es",
                    source="chat",
                    blocking=False,
                )
            except Exception:
                pass  # No bloquear la respuesta si falla
        # Quitar el marcador del mensaje visible al cliente
        answer_text = re.sub(r'\[SEND_INFO_EMAIL:[^\]]*\]', '', answer_text).strip()

    # Detectar lead y notificar en background (no bloquea la respuesta al cliente)
    if BREVO_API_KEY:
        threading.Thread(
            target=process_lead_async,
            args=(question, answer_text),
            daemon=True,
        ).start()

    return jsonify({
        "answer": answer_text,
        "sources": sources,
        "tokens": {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens},
        "model": CLAUDE_MODEL,
    })



@app.route("/api/send-info-email", methods=["POST", "OPTIONS"])
def api_send_info_email():
    """
    Endpoint que dispara el envío automático del email comercial
    (catálogo + tarifa) al cliente que lo solicita.

    Acepta dos fuentes:
      - source="chat"        — el bot detectó intención y el cliente dejó email
      - source="web-button"  — clic en un botón "Pedir información" de la web
    """
    if request.method == "OPTIONS":
        return _cors_response()

    data = request.get_json(force=True) or {}
    client_email = (data.get("client_email") or "").strip()

    if not client_email or "@" not in client_email:
        return jsonify({"ok": False, "error": "invalid_email"}), 400

    if data.get("source") == "web-button":
        ctx = (
            f"El cliente clicó en un botón de la web.\n"
            f"Página: {data.get('page_url', '?')}\n"
            f"Título: {data.get('page_title', '?')}\n"
            f"Mensaje del cliente: {data.get('message') or '(no añadió mensaje)'}"
        )
    else:
        ctx = data.get("conversation_summary", "") or data.get("message", "")

    send_info_email(
        client_email=client_email,
        client_name=data.get("client_name"),
        conversation_summary=ctx,
        detected_language=data.get("language", "es"),
        detected_family=data.get("family"),
        detected_sector=data.get("sector"),
        source=data.get("source", "chat"),
        blocking=False,
    )

    response = jsonify({"ok": True, "queued": True})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def _cors_response():
    response = jsonify({"ok": True})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


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
    max-width:460px;
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
  .msg.bot a{color:var(--accent);text-decoration:underline;word-break:break-word}
  .msg.bot a:hover{opacity:.8}
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
      Asesor comercial <strong>Dlimit</strong>. Te ayudo a elegir poste, pedir presupuesto o resolver dudas tecnicas. Que necesitas?
    </div>
  </div>
  <form id="form" onsubmit="return ask(event)">
    <textarea id="q" placeholder="Escribe tu pregunta..." required rows="1"></textarea>
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
    .replace(/(https?:\/\/[^\s<>"]+)/g,'<a href="$1" target="_blank" rel="noopener">$1</a>')
    .replace(/(?:^|\s)([\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,})/g,' <a href="mailto:$1">$1</a>')
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

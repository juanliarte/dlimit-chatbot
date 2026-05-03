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
import time
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

# Módulo de tracking persistente (SQLite en disco /var/data)
from tracking import (
    init_db as tracking_init_db,
    log_conversation as tracking_log,
    update_lead_info as tracking_update_lead,
    export_csv as tracking_export_csv,
    stats as tracking_stats,
)


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

# Admin key para descarga de tracking CSV / stats
ADMIN_EXPORT_KEY = os.environ.get("ADMIN_EXPORT_KEY", "").strip()


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

SYSTEM_PROMPT = """Eres asesor comercial-tecnico de Dlimit Tactic S.L., fabricante europeo
de postes separadores con cinta retractil y barreras de delimitacion B2B. Trabajas
con un catalogo de 9 familias auditadas + accesorios.

REGLA DE IDIOMA (PRIORIDAD MAXIMA):
- Detecta el idioma del ULTIMO mensaje del cliente y responde SIEMPRE en ese idioma.
- Soporta: espanol, ingles, frances, aleman, italiano, portugues, catalan.
- Saludo corto ("hello", "ok", "ciao") -> responde en ese idioma.
- Solo si es imposible determinar, usa espanol.

ROL:
No eres un catalogo. Eres un asesor que (1) entiende la necesidad real,
(2) recomienda la familia Dlimit correcta, (3) avanza hacia presupuesto
o lead cualificado, (4) acompana con material descargable cuando ayuda al cierre.

REGLAS DE COMPORTAMIENTO:
- Respuestas cortas: 1-3 lineas maximo. Listas solo si son imprescindibles.
- Cada respuesta termina con pregunta o CTA que avance la venta.
- NUNCA des precio directo. Pide contexto antes (uso propio o reventa, cantidad, sector).
- NUNCA inventes referencias, modelos, precios, plazos, ensayos ni certificaciones.
- NUNCA incluyas marcadores tipo [1], [2], (fuente 3).
- Responde SOLO con informacion del contexto recuperado. Si no esta, dilo y escala.

ESTILO:
- Tono: profesional, directo, tecnico cuando hace falta, comercialmente util.
- Tratamiento: "tu" (cercano, B2B moderno).
- Prohibido: "estaremos encantados", "soluciones innovadoras", "amplio abanico",
  "no dudes en contactarnos", "estamos a tu disposicion".
- Preferido: "Depende del uso", "Lo habitual aqui es", "Te explico rapido",
  "Para ese sector usamos...", "Con X reduces Y".

LAS 9 FAMILIAS DEL CATALOGO (posicionamiento real auditado):

- Dbasic: gama economica reforzada. 99% polimeros sin pintura, Marcado CE,
  ISO 9001 200.000 ciclos, EUIPO patentado, frenado One-Way + ShockResisting.
  Para tienda pequena, autonomo, oficina, evento puntual con presupuesto.

- Dstandard: cinta retractil moderna por defecto. 99% polimeros sin pintura,
  4 modelos (Poste 3m/6m, Mural 3m/6m), 4 RAL, 14 colores de cinta, PANTONE
  personalizable. Bloqueo de cinta solo en modelo 6m. Equilibrio coste/calidad.
  Para retail, hosteleria, recepciones, banca, salud, eventos medios.

- Dclassic: ceremonial premium. Aluminio Ø50mm pintado al polvo, INOX satinado
  o laton dorado pulido, cordon trenzado, 9 colores cordon, ganchos cromado/dorado,
  cumplimiento DDA. NO usa anodizado.
  Para hotel 5*, gala, alfombra roja, eventos representativos.

- Dline: discrecion arquitectonica. Cordon elastico Ø6mm en carretes 25m, 6 modelos
  (Basic/Dual/Mini movil + Fix), 4 acabados oficiales (Negro RAL 9005, Gris RAL 7035,
  Blanco RAL 9003, INOX Satinado), 4 colores cordon (Negro/Rojo/Gris/Blanco).
  INOX Satinado es la version oficial para EXTERIOR cultural.
  Para museos, galerias, patrimonio.

- Dlimit: la mas versatil del catalogo (frase oficial). Acero Ø70mm pared 1,5mm
  pintado al polvo carta RAL completa. 3 modelos tubo (Normal/Dual/Mini), 6 bases
  (Limit/Eco/Mag/Max/Fix/Unfix), 4 cierres (rapido/seguridad/magnetico/antipanico),
  3 acabados (RAL/INOX satinado/aluminio anodizado plata u oro alto brillo),
  5 murales hasta 7m, carro porta postes 15 unidades, DDA. NO es premium ceremonial.
  Para empresa eventos, hotel medio-alto, centro convenciones, hospital,
  ayuntamiento, aeropuerto regional.

- Dsafety: la barrera retractil mas segura del mercado (frase oficial). Industrial.
  99% polimeros, 5 modelos (Basic/Dual/Double/Cone/Wall), cinta hasta 10m
  (DOUBLE 20m), 4 RAL, 10 colores cinta, ShockResisting, 3 sistemas instalacion
  (mecanico/magnetico/ventosa). Marcado CE + ISO 9001 + 2 patentes EUIPO.
  Para almacen, fabrica, obra, taller, mantenimiento industrial.
  ACLARAR: NO es barrera estructural, es delimitacion visible al impacto incidental.

- Dterminal: especialista 24/7 transporte. Aluminio anodizado en EXTRUSION,
  Cabezal 360° (reconfigura cola sin mover poste), Base MAG (3 imanes neodimio
  ~150kg sobre disco ferritico adhesivo). 3 postes (Normal 98cm/Dual 98cm/Mini 63cm),
  2 bases (Limit/Mag), 4 plafones autoportantes (1m y 1,8m), 2 murales, carro 15 postes,
  4 portacarteles 360°, 8 RAL + Anodizado, 14 cintas, DDA.
  Para aeropuerto principal, AVE, metro denso, terminal maritima, hub 24/7.

- Esafety: the new security standard for E-Mobility (frase oficial). Sello GS aleman,
  carcasa PRFV irrombible (GRP), muelle inoxidable, cinta 7m que delimita 49m² con
  solo 4 postes (30% menos coste). 5 modelos (E-BASIC/DUAL/DOUBLE/WALL/CONE),
  advertencias en 13 idiomas, vinilos E-VINIL High Voltage / No Trespassing,
  sistema antirrobo, 6 kits oficiales del catalogo.
  Para estacion recarga EV, hub carga, mantenimiento HV, subestacion, talleres EV.
  ACLARAR: es delimitacion visible certificada, NO aislamiento electrico activo.

- Esdsafety: crea zona EPA segun UNE-EN-61340-5.1. Cinta textil plastificada de
  nylon practicamente irrombible, mensajes EPA en 9 idiomas oficiales (EN/DE/FR/
  ES/IT/SE/PT/PL + Personalizado), sistema antirrobo especifico (tornillo central
  + 2 tetones laterales). 5 modelos, sello GS, muelle inoxidable.
  Para sala SMD, EMS, lab I+D electronico, sala blanca electronica.
  ACLARAR: es el perimetro visible de la zona EPA, NO la zona EPA completa.
  La conformidad UNE-EN-61340-5.1 completa requiere tambien suelo, calzado,
  pulseras y procedimientos del cliente.

- Daccessory: complemento (display A4/A3, portacarteles, terminales murales,
  conos carretera, kits magneticos/ventosa). NUNCA opcion principal: completa
  una familia ya elegida. NO es un poste.

LONGITUDES OFICIALES DE CINTA POR FAMILIA:
- Dbasic: 3m  - Dstandard: 3m / 6m  - Dlimit: mural 5m / 7m
- Dline: cordon Ø6mm carrete 25m  - Dclassic: cordon trenzado segun proyecto
- Dsafety: 10m (DOUBLE 20m)  - Dterminal: 3m / 3,7m / 5m
- Esafety: 7m (DOUBLE 14m)   - Esdsafety: 7m (DOUBLE 14m)
NUNCA digas "cinta 4m estandar" ni "9m". Usa la longitud real por familia.
Personalizacion CMYK sublimacion con tinta al agua. PANTONE en Dstandard/Dlimit/Dterminal.

DIFERENCIACION VS COMPETENCIA:
- Fabricacion europea propia, no reseller.
- 9 familias para todos los presupuestos, sectores y entornos (de Dbasic economico
  a Esdsafety zona EPA segun norma).
- Dos tecnologias patentadas (One-Way frenado + ShockResisting impacto), 2 patentes
  EUIPO (006608147-0001 y 006607248-0001), Marcado CE, ISO 9001 hasta 200.000 ciclos.
- Sello GS aleman en Esafety y Esdsafety (estandar europeo mas exigente).
- Cumplimiento DDA accesibilidad en Dclassic, Dlimit y Dterminal.
- Personalizacion completa: cinta CMYK + colores RAL + bases modulares.

CASCADA DE DECISION (en orden, primer match gana):
1. Riesgo electrico activo / EV / recarga / HV -> Esafety.
2. EPA / ESD / SMD / sala blanca electronica -> Esdsafety.
3. Aeropuerto principal / AVE / metro denso / 24/7 transporte -> Dterminal.
4. Carretillas / almacen / fabrica / obra / industrial -> Dsafety.
5. Museo / galeria / patrimonio / exterior cultural -> Dline.
6. Hotel 5* / gala / ceremonial premium -> Dclassic.
7. Eventos profesionales / hosteleria intensiva / cinta retractil con bases
   modulares y carro -> Dlimit.
8. Retail / hosteleria estandar / banca / salud / publico medio -> Dstandard.
9. Presupuesto minimo absoluto -> Dbasic.
10. Solo necesita accesorio para familia ya elegida -> Daccessory.

RECURSOS DESCARGABLES (acompana cuando ayude al cierre, URL completa, sin acortar):
General: https://www.dlimit.net/descargas.html
- Dbasic:    https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dbasic-ES.pdf
- Dstandard: https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dstandard-ES.pdf
- Dclassic:  https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dclassic-ES.pdf
- Dline:     https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dline-ES.pdf
- Dlimit:    https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dlimit-ES.pdf
- Dsafety:   https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dsafety-ES.pdf
- Dterminal: https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dterminal-ES.pdf
- Esafety:   https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Esafety-ES.pdf
- Esdsafety: https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-ESDsafety-ES.pdf

LOGICA DE VENTA (4 pasos):
1. Detectar intencion (informacion, presupuesto, comparacion, distribucion).
2. Entender contexto (uso propio/reventa, sector, cantidad, entorno).
3. Recomendar familia y configuracion (segun CASCADA DE DECISION).
4. Acompanar con PDF + avanzar a accion (presupuesto, email, comercial).

GESTION DE INTENCIONES:
- PRECIO -> "Depende del uso. Es para uso propio o reventa? Que cantidad estimas?"
  Tras respuesta: "Te preparo presupuesto, me dejas tu email?"
- TECNICO -> Datos del contexto. Ofrecer ficha: "Te paso la ficha completa: [URL]"
  Preguntar entorno (interior/exterior, intensidad, costa).
- PERDIDO -> 3 opciones: comprar | info tecnica | distribucion. "Que te interesa mas?"
- COMPRAR -> Pedir cantidad + sector + ubicacion. PDF + presupuesto o derivar a comercial.
- DISTRIBUCION/REVENTA -> Pais + sector + volumen anual. Escalar a info@dlimit.es.
- SECTOR (aeropuertos, hospitales, hoteles, retail, eventos, ferias, museos,
  hosteleria, estaciones, recarga EV, sala SMD) -> Recomendar familia + PDF.
- OFF-TOPIC -> "Soy el asesor de Dlimit, te ayudo con sistemas de gestion de
  colas y delimitacion. Que necesitas?"

OBJECIONES TIPICAS:
- "Caro / otros mas baratos" -> "Con cinta hasta 10m (Dsafety) o 7m (Esafety)
  necesitas menos postes que con 2-3m. Coste total mas bajo. Quieres que te lo calculemos?"
- "Ya tengo proveedor" -> "Entendido. Si quieres comparar acabados o
  personalizacion, aqui estamos. Que proveedor usas?"
- "No estoy seguro" -> "Normal. Para que entorno lo usarias? Con eso te oriento mejor."
- "Quiero mirar antes" -> "Perfecto. Te paso el catalogo de la gama que mejor
  te encaja: [URL]. De que sector hablamos?"
- "Es de mala calidad" (Dbasic) -> "Es funcional, fiable, con tecnologia patentada
  One-Way y Marcado CE. Cumple su funcion al menor coste."
- "Es muy caro" (Dclassic/Dlimit/Dterminal) -> "Forma parte del coste de imagen
  del proyecto o del KPI de operatividad. Igual que iluminacion o catering, comunica categoria."
- "Aguanta uso intensivo?" (Dclassic) -> "Pensado para eventos y hospitality,
  no 24/7 de transporte. Para 24/7 te recomiendo Dterminal."
- "Parece poco firme" (Dline) -> "Su funcion no es contener, es senalizar el
  limite con respeto al espacio."
- "Esafety aisla la electricidad?" -> "No. Es delimitacion visible certificada
  con sello GS, NO aislamiento activo. La proteccion electrica la otorga la
  instalacion del cliente."
- "Esdsafety garantiza la zona EPA completa?" -> "No. Es el perimetro visible
  certificado. La conformidad UNE-EN-61340-5.1 completa requiere tambien suelo,
  calzado, pulseras y procedimientos del cliente."

CAPTURA DE LEAD (de uno en uno, no formulario):
1. Email (siempre)  2. Empresa  3. Cantidad estimada  4. Sector / uso final  5. Pais.
RGPD si pide email: "Solo lo usamos para enviarte el presupuesto."

ENVIO AUTOMATICO POR EMAIL (mecanismo critico, NO TOCAR):
Cuando detectes interes concreto (cantidad, sector, familia, intencion clara),
OFRECE proactivamente el envio:
  "Quieres que te envie el catalogo completo y la tarifa de precios por
   email? Te lo mando ahora mismo y Ester, nuestra responsable comercial,
   te llama en 24 h para condiciones especiales."

Si acepta, pide email (uno solo): "Perfecto, a que email te lo mando?"

Cuando obtengas un email VALIDO + acepto explicito, al FINAL de tu respuesta
y EN UNA LINEA SEPARADA incluye el marcador EXACTO:
  [SEND_INFO_EMAIL: email@cliente.com]

Reglas del marcador (criticas):
- Solo si el cliente dio email valido y acepto explicitamente el envio.
- Una vez emitido en una conversacion, NO repetir.
- NUNCA inventes email. Si no lo dio, pidelo.
- En la respuesta visible al cliente, confirma con una linea breve:
  "Perfecto, te lo envio ahora mismo a [email]. Ester te llama en 24 h."
- El sistema borra el marcador del mensaje visible automaticamente.

ESCALADO A HUMANO (escala en estos casos):
- Volumen >50 postes  - Personalizacion avanzada (logos exclusivos, RAL no estandar)
- Distribucion / reventa  - Condiciones de pago especiales  - Licitacion publica.
- Configuracion compleja Esafety/Esdsafety con norma especifica (REBT, sectorial).
Mensaje: "Esto lo revisa mejor nuestro equipo comercial. Escribenos a
info@dlimit.es o llama al +34 932 526 915 (L-V 9:00-18:00 CET).
Te paso ya o quieres que te llamen ellos?"

FORMATO:
- Markdown ligero: negrita en terminos clave (familia, numero, sector).
- Listas solo cuando enumeras 3+ items reales.
- Sin emojis salvo respuesta a saludo en idioma extranjero.
- URLs completas en linea propia, sin acortar.

REGLA FINAL:
SIEMPRE: entender -> guiar -> acompanar (con PDF) -> cerrar.
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


def process_lead_async(question, answer, conv_id=None):
    """Detecta lead y notifica. Thread separado para no bloquear respuesta.
    Si conv_id viene informado, actualiza la fila de tracking con info del lead."""
    if not BREVO_API_KEY:
        return
    try:
        lead = detect_lead(question, answer)
        is_lead = bool(lead.get("is_lead"))
        # Persistir info del lead en la fila de tracking
        if conv_id is not None:
            try:
                tracking_update_lead(
                    conv_id,
                    was_lead=is_lead,
                    lead_email=lead.get("email") if is_lead else None,
                    lead_urgency=lead.get("urgency") if is_lead else None,
                    lead_sector=lead.get("sector") if is_lead else None,
                    lead_quantity=lead.get("quantity") if is_lead else None,
                )
            except Exception as _e:
                print(f"[tracking_update_lead] {_e}", flush=True)
        if not is_lead:
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

# Inicializar BBDD de tracking (crea tabla si no existe).
# Si falla (p.ej. no hay disco), el chatbot sigue funcionando.
try:
    tracking_init_db()
except Exception as _e:
    print(f"[tracking_init_db] {_e}", flush=True)


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    _t_start = time.time()
    _client_ip = (
        request.headers.get("X-Forwarded-For", request.remote_addr or "")
        .split(",")[0]
        .strip()
        or None
    )
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "empty"}), 400

    _had_email = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", question))
    _session_id = (data.get("session_id") or "").strip()[:80] or None
    _lang_hint = (data.get("lang") or "").strip()[:5] or None

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
        _elapsed_ms = int((time.time() - _t_start) * 1000)
        _conv_id = tracking_log(
            session_id=_session_id,
            ip=_client_ip,
            language=_lang_hint,
            question=question,
            answer=answer_text,
            sources_count=0,
            tokens_in=0,
            tokens_out=0,
            response_time_ms=_elapsed_ms,
            had_email=_had_email,
        )
        if BREVO_API_KEY:
            threading.Thread(target=process_lead_async, args=(question, answer_text, _conv_id), daemon=True).start()
        return jsonify({"answer": answer_text, "sources": []})

    user_msg = (
        f"Customer message (respond in the SAME language as this message):\n"
        f"{question}\n\n"
        f"Knowledge base context (Dlimit):\n" + "\n\n".join(context_parts) +
        f"\n\nUse ONLY the context above. Do NOT add reference markers like [1], [2]."
        f" Follow the commercial advisor logic: understand, guide, accompany with PDF if helpful, close."
    )
    # Construir array de mensajes con historial conversacional
    history = data.get("history") or []
    valid_history = []
    if isinstance(history, list):
        for msg in history[-14:]:  # max 14 turnos previos
            if isinstance(msg, dict):
                role = msg.get("role")
                content_msg = msg.get("content")
                if role in ("user", "assistant") and isinstance(content_msg, str) and content_msg.strip():
                    valid_history.append({"role": role, "content": content_msg.strip()[:4000]})
    messages_for_claude = valid_history + [{"role": "user", "content": user_msg}]

    resp = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=900,
        system=SYSTEM_PROMPT,
        messages=messages_for_claude,
    )
    answer_text = resp.content[0].text

    # ─── Detectar marcador de envio automatico de catalogo+tarifa ───
    _email_marker = re.search(r'\[SEND_INFO_EMAIL:\s*([^\]\s]+)\s*\]', answer_text)
    if _email_marker:
        _target_email = _email_marker.group(1).strip().rstrip('.,;:')
        if "@" in _target_email and "." in _target_email.split("@")[-1]:
            try:
                # Construir resumen con todo el historial conversacional
                _summary_parts = []
                for _m in valid_history[-8:]:
                    _prefix = "Cliente" if _m["role"] == "user" else "Bot"
                    _summary_parts.append(f"{_prefix}: {_m['content'][:300]}")
                _summary_parts.append(f"Cliente: {question}")
                _summary_parts.append(f"Bot: {answer_text[:400]}")
                send_info_email(
                    client_email=_target_email,
                    conversation_summary="\n".join(_summary_parts),
                    detected_language="es",
                    source="chat",
                    blocking=False,
                )
            except Exception:
                pass  # No bloquear la respuesta si falla
        # Quitar el marcador del mensaje visible al cliente
        answer_text = re.sub(r'\[SEND_INFO_EMAIL:[^\]]*\]', '', answer_text).strip()

    # ─── Tracking persistente ───
    _elapsed_ms = int((time.time() - _t_start) * 1000)
    _conv_id = tracking_log(
        session_id=_session_id,
        ip=_client_ip,
        language=_lang_hint,
        question=question,
        answer=answer_text,
        sources_count=len(sources),
        tokens_in=getattr(resp.usage, "input_tokens", 0) or 0,
        tokens_out=getattr(resp.usage, "output_tokens", 0) or 0,
        response_time_ms=_elapsed_ms,
        had_email=_had_email,
    )

    # Detectar lead y notificar en background (no bloquea la respuesta al cliente)
    if BREVO_API_KEY:
        threading.Thread(
            target=process_lead_async,
            args=(question, answer_text, _conv_id),
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
        # use_ai por defecto: True para chat, False para web-button
        # (dentro de send_info_email se decide automaticamente segun source)
        client_message=data.get("message"),
        page_url=data.get("page_url"),
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


# Admin endpoints (tracking)
@app.route("/admin/export", methods=["GET"])
def admin_export():
    """Descarga CSV con todas las conversaciones registradas.
    Requiere ?key=ADMIN_EXPORT_KEY. 401 si no coincide."""
    key = (request.args.get("key") or "").strip()
    if not ADMIN_EXPORT_KEY or key != ADMIN_EXPORT_KEY:
        return jsonify({"error": "unauthorized"}), 401
    csv_str = tracking_export_csv()
    fname = "dlimit_conversations_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".csv"
    return Response(
        csv_str,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.route("/admin/stats", methods=["GET"])
def admin_stats_endpoint():
    """Metricas basicas del tracking. Requiere ?key=ADMIN_EXPORT_KEY."""
    key = (request.args.get("key") or "").strip()
    if not ADMIN_EXPORT_KEY or key != ADMIN_EXPORT_KEY:
        return jsonify({"error": "unauthorized"}), 401
    return jsonify(tracking_stats())


@app.route("/admin/ingest_chunks", methods=["POST"])
def admin_ingest_chunks():
    """Ingest text chunks into Qdrant. Protected by ADMIN_EXPORT_KEY.
    Body JSON: {"chunks": [{"text", "family", "title", "source",
                            "source_path", "doc_type", "layer", "page", "id"}]}
    Each chunk gets dense (Voyage) + sparse (BM25) vectors.
    Idempotent: same family+doc_type produces same id (upsert overwrites)."""
    key = (request.args.get("key") or "").strip()
    if not ADMIN_EXPORT_KEY or key != ADMIN_EXPORT_KEY:
        return jsonify({"error": "unauthorized"}), 401

    from qdrant_client.models import PointStruct
    import uuid

    data = request.get_json(force=True) or {}
    chunks = data.get("chunks", [])
    if not chunks or not isinstance(chunks, list):
        return jsonify({"error": "no chunks"}), 400

    points = []
    errors = []
    for i, chunk in enumerate(chunks):
        try:
            text = (chunk.get("text") or "").strip()
            if not text:
                errors.append({"i": i, "err": "empty text"})
                continue
            family = (chunk.get("family") or "").strip()
            doc_type = chunk.get("doc_type", "playbook")
            layer = chunk.get("layer", "publica")

            dense = voyage.embed(
                [text], model=DENSE_MODEL, input_type="document",
                truncation=True
            ).embeddings[0]
            sparse_emb = list(bm25.embed([text]))[0]
            sparse_vec = SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist(),
            )
            payload = {
                "text": text,
                "title": chunk.get("title") or f"Playbook {family}",
                "source": chunk.get("source") or f"playbook_{family}.md",
                "source_path": chunk.get("source_path", ""),
                "doc_type": doc_type,
                "family": family,
                "layer": layer,
                "page": chunk.get("page", 1),
            }
            point_id = chunk.get("id") or str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_type}-{family}")
            )
            points.append(PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR_NAME: dense,
                    SPARSE_VECTOR_NAME: sparse_vec,
                },
                payload=payload,
            ))
        except Exception as e:
            errors.append({"i": i, "err": str(e)})

    if not points:
        return jsonify({"ok": False, "ingested": 0, "errors": errors}), 400

    try:
        qclient.upsert(collection_name=COLLECTION, points=points, wait=True)
    except Exception as e:
        return jsonify({"ok": False, "ingested": 0,
                        "upsert_error": str(e), "errors": errors}), 500

    return jsonify({"ok": True, "ingested": len(points), "errors": errors})


@app.route("/admin/purge_filtered", methods=["POST"])
def admin_purge_filtered():
    """Borra puntos de Qdrant que coincidan con un filtro de payload.
    Protegido por ADMIN_EXPORT_KEY.
    Body JSON: {"filters": {"doc_type": "playbook"}, "dry_run": false}
    Para OR sobre varios valores: {"filters": {"doc_type": ["playbook", "old_doc"]}}
    Devuelve los puntos borrados (count). Operacion irreversible."""
    key = (request.args.get("key") or "").strip()
    if not ADMIN_EXPORT_KEY or key != ADMIN_EXPORT_KEY:
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(force=True) or {}
    filters = data.get("filters", {})
    dry_run = bool(data.get("dry_run", False))
    if not filters or not isinstance(filters, dict):
        return jsonify({"error": "no filters"}), 400

    must = []
    has_list = False
    for field, value in filters.items():
        if isinstance(value, list):
            has_list = True
            for v in value:
                must.append(FieldCondition(key=field, match=MatchValue(value=v)))
        else:
            must.append(FieldCondition(key=field, match=MatchValue(value=value)))

    qfilter = Filter(should=must) if has_list else Filter(must=must)

    try:
        count_resp = qclient.count(
            collection_name=COLLECTION,
            count_filter=qfilter,
            exact=True,
        )
        n = count_resp.count
        if dry_run:
            return jsonify({"ok": True, "dry_run": True, "matched": n, "filters": filters})

        qclient.delete(
            collection_name=COLLECTION,
            points_selector=qfilter,
            wait=True,
        )
        return jsonify({"ok": True, "deleted": n, "filters": filters})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "filters": filters}), 500


@app.route("/healthz", methods=["GET"])
def healthz():
    """Endpoint publico de salud para monitorizacion (no expone datos)."""
    return jsonify({"status": "ok", "ts": datetime.now(timezone.utc).isoformat()})


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
// ============================================================
// i18n del frontend (v4 - 3 mayo 2026)
// El widget pasa ?lang=es|en al cargar el iframe. Aplicamos los
// textos UI (saludo, placeholder, aria-label) ANTES de cualquier
// otra cosa, para que el visitante anglofono no vea espanol.
// ============================================================
const __DLIMIT_PARAMS = new URLSearchParams(location.search);
const __DLIMIT_LANG = (__DLIMIT_PARAMS.get('lang') || 'es').toLowerCase().startsWith('en') ? 'en' : 'es';
const __DLIMIT_I18N = {
  es: {
    greet: 'Asesor comercial <strong>Dlimit</strong>. Te ayudo a elegir poste, pedir presupuesto o resolver dudas tecnicas. Que necesitas?',
    placeholder: 'Escribe tu pregunta...',
    sendAria: 'Enviar',
    error: 'Error: '
  },
  en: {
    greet: 'Hi, I am the <strong>Dlimit</strong> sales assistant. I can help you choose a stanchion, request a quote or solve technical questions. What do you need?',
    placeholder: 'Type your question...',
    sendAria: 'Send',
    error: 'Error: '
  }
};
const __T = __DLIMIT_I18N[__DLIMIT_LANG];
document.documentElement.lang = __DLIMIT_LANG;

const messages = document.getElementById('messages');
const q = document.getElementById('q');
const send = document.getElementById('send');
const empty = document.querySelector('.empty');

// Aplicar i18n a los elementos hardcoded en el HTML
if (empty) empty.innerHTML = __T.greet;
if (q) q.placeholder = __T.placeholder;
if (send) send.setAttribute('aria-label', __T.sendAria);

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

// Historial de conversacion (se mantiene en memoria mientras la pestana este abierta)
let conversationHistory = [];
const MAX_HISTORY = 14;

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
    const historyToSend = conversationHistory.slice(-MAX_HISTORY);
    const r = await fetch('/api/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question: text, history: historyToSend})
    });
    const data = await r.json();
    thinking.remove();
    const answer = data.answer || '';
    add('bot', fmt(answer));
    conversationHistory.push({role: 'user', content: text});
    conversationHistory.push({role: 'assistant', content: answer});
    if (conversationHistory.length > MAX_HISTORY * 2) {
      conversationHistory = conversationHistory.slice(-MAX_HISTORY * 2);
    }
  }catch(err){
    thinking.remove();
    add('bot', '<em>'+__T.error+err.message+'</em>');
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


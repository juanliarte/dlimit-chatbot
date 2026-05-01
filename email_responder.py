"""
email_responder.py
==================
Módulo auto-contenido para enviar emails comerciales personalizados a clientes
de Dlimit Tactic. Se dispara desde dos fuentes:

  1. Conversación del chatbot — el bot ofrece "¿quieres recibir el catálogo
     y la tarifa por email?" y dispara este flujo si el cliente acepta.
  2. Botones "Pedir información" / "Solicitar catálogo" en la web — el cliente
     deja su email en un formulario corto y se dispara igualmente.

El email lleva:
  - Cuerpo personalizado escrito por Claude en el idioma del cliente
  - Catálogo general (PDF público de dlimit.net)
  - Tarifa PVP (PDF público de dlimit.net)
  - Firma de Ester Prieto (Responsable Comercial)
  - Reply-To: info@dlimit.es (cuando responda el cliente, va a Ester)
  - Bcc: juanliarte@gmail.com (Juan ve todo lo que sale)

Variables de entorno requeridas (configurar en Render):
  BREVO_API_KEY         — ya configurada
  ANTHROPIC_API_KEY     — ya configurada
  FROM_EMAIL_ADDRESS    — info@dlimit.es
  FROM_EMAIL_NAME       — "Dlimit Tactic · Comercial"
  REPLY_TO_EMAIL        — info@dlimit.es
  BCC_EMAIL             — juanliarte@gmail.com
  CATALOG_PDF_URL       — https://www.dlimit.net/catalogos/catalogo-general-dlimit.pdf
  TARIFA_PDF_URL        — https://www.dlimit.net/catalogos/tarifa-pvp-dlimit-2026.pdf

Uso desde chatbot_server.py:
  from email_responder import send_info_email
  send_info_email(
      client_email="cliente@hotelmallorca.es",
      conversation_summary="Cliente busca 25 postes Dclassic para hotel...",
      detected_language="es",
      detected_family="dclassic",
      detected_sector="hostelería",
      source="chat",
  )
"""

from __future__ import annotations

import os
import json
import logging
import threading
from typing import Optional

import requests
from anthropic import Anthropic

log = logging.getLogger(__name__)

# ───────────────────────── Configuración ─────────────────────────

BREVO_API_KEY      = os.environ.get("BREVO_API_KEY", "")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")

FROM_EMAIL         = os.environ.get("FROM_EMAIL_ADDRESS", "info@dlimit.es")
FROM_NAME          = os.environ.get("FROM_EMAIL_NAME", "Dlimit Tactic · Comercial")
REPLY_TO_EMAIL     = os.environ.get("REPLY_TO_EMAIL", "info@dlimit.es")
BCC_EMAIL          = os.environ.get("BCC_EMAIL", "juanliarte@gmail.com")

CATALOG_PDF_URL    = os.environ.get(
    "CATALOG_PDF_URL",
    "https://www.dlimit.net/catalogos/catalogo-general-dlimit.pdf",
)
TARIFA_PDF_URL     = os.environ.get(
    "TARIFA_PDF_URL",
    "https://www.dlimit.net/catalogos/tarifa-pvp-dlimit-2026.pdf",
)

CLAUDE_BODY_MODEL  = os.environ.get("CLAUDE_BODY_MODEL", "claude-haiku-4-5")
BREVO_API_URL      = "https://api.brevo.com/v3/smtp/email"

# Cliente Anthropic (lazy)
_anthropic_client: Optional[Anthropic] = None

def _get_claude() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("Falta ANTHROPIC_API_KEY")
        _anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


# ───────────────── Firma de Ester (todos los idiomas) ─────────────────
# Cuando entre Max para export, añadir aquí firmas alternativas por idioma.
ESTER_SIGNATURE_HTML = """
<table cellpadding="0" cellspacing="0" border="0" style="font-family:Arial,sans-serif;font-size:13px;color:#222;border-collapse:collapse;margin-top:18px">
  <tr>
    <td style="padding-right:16px;border-right:2px solid #2A9D2A;vertical-align:top">
      <strong style="font-size:15px;color:#1a1a1a">Ester Prieto</strong><br>
      <span style="color:#666">Responsable Comercial</span><br>
      <strong style="color:#1a1a1a">DLIMIT TACTIC S.L.</strong>
    </td>
    <td style="padding-left:16px;vertical-align:top;color:#444">
      Fijo: <a href="tel:+34932526915" style="color:#2A9D2A;text-decoration:none">932 526 915</a><br>
      Móvil: <a href="tel:+34670806282" style="color:#2A9D2A;text-decoration:none">670 806 282</a><br>
      <a href="mailto:eprieto@dlimit.es" style="color:#2A9D2A;text-decoration:none">eprieto@dlimit.es</a><br>
      <a href="https://www.dlimit.es" style="color:#2A9D2A;text-decoration:none">www.dlimit.es</a>
    </td>
  </tr>
</table>
<p style="font-size:10px;color:#999;margin-top:14px;line-height:1.4;font-family:Arial,sans-serif">
Este mensaje y sus archivos adjuntos son confidenciales y pueden contener información privilegiada.
Si usted no es el destinatario, por favor notifíquelo y elimínelo.
</p>
"""


# ───────────────── Generación del cuerpo con Claude ─────────────────

_BODY_PROMPT = """Eres redactor de emails comerciales para Dlimit Tactic S.L.,
empresa española (Mataró) especializada en postes separadores y barreras
retráctiles para B2B industrial. Tu tarea: redactar el cuerpo HTML de un
email de respuesta automática a un cliente que ha pedido información.

REGLAS DE FORMATO
- HTML simple compatible con Gmail/Outlook (solo <p>, <strong>, <ul>, <li>, <a>).
- 4-7 frases, máximo 6 párrafos cortos. Tono cálido, profesional, directo.
- NO incluyas firma — la firma se añade después automáticamente.
- NO uses emojis excepto un saludo inicial breve si encaja con el idioma.
- NO menciones precios concretos, NO menciones porcentajes de descuento.
- Idioma: el especificado en LANGUAGE — si es "ca" responde en catalán, si es "en" en inglés perfecto, etc.

CONTENIDO OBLIGATORIO
1. Saludo personalizado y agradecimiento por el interés
2. Si hay familia recomendada o sector mencionado, hacer referencia natural
3. Mencionar los DOS adjuntos: el catálogo general y la tarifa de precios PVP
4. Indicar que para volúmenes especiales o configuraciones particulares hay
   condiciones a medida que se ajustan caso por caso
5. Anunciar que Ester Prieto, responsable comercial, contactará en las
   próximas 24 horas laborables al teléfono que el cliente facilite, o
   invitar a llamar directamente al 932 526 915 / 670 806 282
6. Cierre cálido pero breve

CONTEXTO DEL CLIENTE
- Idioma: {language}
- Origen del contacto: {source}  (chat = conversación con el bot; web-button = clic en botón web)
- Familia recomendada: {family}
- Sector mencionado: {sector}
- Resumen de la conversación o página visitada:
{context_summary}

Devuelve ÚNICAMENTE un objeto JSON válido con dos claves:
{{
  "subject": "asunto del email, breve, en el mismo idioma",
  "html_body": "cuerpo HTML del email (sin <html> ni <body>, solo el contenido)"
}}
"""


def _generate_body(
    *,
    language: str,
    source: str,
    family: Optional[str],
    sector: Optional[str],
    context_summary: str,
) -> dict:
    """Llama a Claude Haiku para generar subject + html_body en JSON."""
    prompt = _BODY_PROMPT.format(
        language=language or "es",
        source=source or "chat",
        family=family or "(no detectada)",
        sector=sector or "(no detectado)",
        context_summary=context_summary[:2000] or "(sin contexto adicional)",
    )
    claude = _get_claude()
    resp = claude.messages.create(
        model=CLAUDE_BODY_MODEL,
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()

    # Limpieza típica si Claude envuelve en ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip("` \n")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Claude devolvió JSON no parseable, usando fallback. Raw=%r", raw[:300])
        parsed = {
            "subject": "Información Dlimit Tactic — catálogo y tarifa",
            "html_body": (
                "<p>Hola,</p>"
                "<p>Gracias por tu interés en los productos de Dlimit Tactic. "
                "Te adjunto el catálogo general y la tarifa PVP.</p>"
                "<p>Para volúmenes especiales o configuraciones particulares, "
                "ajustamos condiciones caso por caso.</p>"
                "<p>Ester Prieto, nuestra responsable comercial, contactará "
                "contigo en las próximas 24 horas. Si lo prefieres, puedes "
                "llamar al 932 526 915.</p>"
                "<p>Un saludo,</p>"
            ),
        }
    return parsed


# ───────────────── Envío vía Brevo ─────────────────

def _send_via_brevo(
    *,
    to_email: str,
    to_name: Optional[str],
    subject: str,
    html_content: str,
) -> dict:
    """POST a Brevo /v3/smtp/email con adjuntos por URL."""
    if not BREVO_API_KEY:
        raise RuntimeError("Falta BREVO_API_KEY")

    payload = {
        "sender": {"name": FROM_NAME, "email": FROM_EMAIL},
        "replyTo": {"email": REPLY_TO_EMAIL, "name": "Ester Prieto · Dlimit"},
        "to": [{"email": to_email, "name": to_name or to_email}],
        "bcc": [{"email": BCC_EMAIL, "name": "Juan Liarte"}],
        "subject": subject,
        "htmlContent": html_content,
        "attachment": [
            {"url": CATALOG_PDF_URL, "name": "Catalogo-Dlimit.pdf"},
            {"url": TARIFA_PDF_URL,  "name": "Tarifa-PVP-Dlimit.pdf"},
        ],
        "tags": ["info-auto", "dlimit-chatbot"],
    }
    headers = {
        "accept": "application/json",
        "api-key": BREVO_API_KEY,
        "content-type": "application/json",
    }
    r = requests.post(BREVO_API_URL, json=payload, headers=headers, timeout=30)
    if not r.ok:
        log.error("Brevo error %s: %s", r.status_code, r.text)
        r.raise_for_status()
    return r.json()


# ───────────────── API pública del módulo ─────────────────

def send_info_email(
    *,
    client_email: str,
    client_name: Optional[str] = None,
    conversation_summary: str = "",
    detected_language: str = "es",
    detected_family: Optional[str] = None,
    detected_sector: Optional[str] = None,
    source: str = "chat",
    blocking: bool = False,
) -> dict:
    """
    Punto de entrada principal del módulo.

    Args:
        client_email: dirección del cliente al que enviar el email.
        client_name: nombre del cliente si lo conocemos (opcional).
        conversation_summary: resumen de la conversación o contexto de la
            página desde donde se disparó el email.
        detected_language: 'es', 'en', 'fr', 'de', 'it', 'pt', 'ca'
        detected_family: 'dbasic', 'dstandard', 'dclassic', 'dline',
            'dsafety', 'dterminal', 'dlimit', 'daccessory', o None.
        detected_sector: 'hostelería', 'hospitales', 'transporte', etc.
        source: 'chat' o 'web-button'
        blocking: si True, bloquea hasta que el email esté enviado.
            Si False (default), corre en thread asíncrono y devuelve {ok: True}
            inmediatamente para no bloquear la respuesta del bot al cliente.

    Returns:
        dict con {ok: True, message_id: ...} si síncrono, o {ok: True, queued: True}
        si asíncrono.
    """
    if not client_email or "@" not in client_email:
        return {"ok": False, "error": "invalid_email"}

    def _run():
        try:
            body = _generate_body(
                language=detected_language,
                source=source,
                family=detected_family,
                sector=detected_sector,
                context_summary=conversation_summary,
            )
            html_full = body["html_body"] + ESTER_SIGNATURE_HTML
            res = _send_via_brevo(
                to_email=client_email,
                to_name=client_name,
                subject=body["subject"],
                html_content=html_full,
            )
            log.info("Email enviado a %s (msg_id=%s)", client_email,
                     res.get("messageId", "?"))
            return res
        except Exception as e:
            log.exception("Error enviando info email a %s: %s", client_email, e)
            return None

    if blocking:
        result = _run()
        return {"ok": bool(result), "result": result}

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "queued": True}


# ───────────────── Health check / smoke test ─────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    test_email = sys.argv[1] if len(sys.argv) > 1 else "juanliarte@gmail.com"
    print(f"Lanzando smoke test → {test_email}")

    result = send_info_email(
        client_email=test_email,
        client_name="Juan Liarte (TEST)",
        conversation_summary=(
            "El cliente pregunta por postes separadores para un hotel "
            "boutique de 25 habitaciones en Mallorca. Está interesado en "
            "una solución elegante con base cromada. Ha mencionado un "
            "presupuesto orientativo de unos 25 unidades para terraza y "
            "recepción. Idioma: español. Familia recomendada en la "
            "conversación: Dclassic."
        ),
        detected_language="es",
        detected_family="dclassic",
        detected_sector="hostelería",
        source="chat",
        blocking=True,  # en el smoke test bloqueamos para ver el resultado
    )
    print("Resultado:", result)

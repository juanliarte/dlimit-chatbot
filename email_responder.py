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
    "https://www.dlimit.net/catalogos/general/DLimit_Catalogo_2024-25.pdf",
)
TARIFA_PDF_URL     = os.environ.get(
    "TARIFA_PDF_URL",
    "https://www.dlimit.net/catalogos/tarifas/DLimit_Tarifa_June_2025_NAC.pdf",
)

# Catálogos por familia (públicos en dlimit.net). Cuando el cliente llega
# desde una página de familia (data-family="dclassic"), enviamos el catálogo
# de esa familia + la tarifa. Si no hay familia detectada, mandamos el
# catálogo general + tarifa.
_FAMILY_CATALOG_URLS = {
    "dbasic":     "https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dbasic-ES.pdf",
    "dstandard":  "https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dstandard-ES.pdf",
    "dclassic":   "https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dclassic-ES.pdf",
    "dline":      "https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dline-ES.pdf",
    "dsafety":    "https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dsafety-ES.pdf",
    "dterminal":  "https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dterminal-ES.pdf",
    "dlimit":     "https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-Dlimit-ES.pdf",
    "esdsafety":  "https://www.dlimit.net/catalogos/catalogos%20familia/Catalogo-ESDsafety-ES.pdf",
}

def _resolve_catalog_for_family(family):
    """Devuelve la URL del catálogo de familia, o el general si la familia no está mapeada."""
    if not family:
        return CATALOG_PDF_URL
    fam_clean = family.strip().lower()
    return _FAMILY_CATALOG_URLS.get(fam_clean, CATALOG_PDF_URL)

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
    catalog_url: Optional[str] = None,
    catalog_name: Optional[str] = None,
) -> dict:
    """POST a Brevo /v3/smtp/email con adjuntos por URL."""
    if not BREVO_API_KEY:
        raise RuntimeError("Falta BREVO_API_KEY")

    catalog_url = catalog_url or CATALOG_PDF_URL
    catalog_name = catalog_name or "Catalogo-Dlimit.pdf"

    payload = {
        "sender": {"name": FROM_NAME, "email": FROM_EMAIL},
        "replyTo": {"email": REPLY_TO_EMAIL, "name": "Ester Prieto · Dlimit"},
        "to": [{"email": to_email, "name": to_name or to_email}],
        "bcc": [{"email": BCC_EMAIL, "name": "Juan Liarte"}],
        "subject": subject,
        "htmlContent": html_content,
        "attachment": [
            {"url": catalog_url, "name": catalog_name},
            {"url": TARIFA_PDF_URL, "name": "Tarifa-PVP-Dlimit.pdf"},
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



# ───────────────── Templates fijos (sin IA) para botones web ─────────────────
# Estos templates se usan cuando source="web-button". Más rápidos y baratos
# que llamar a Claude — solo dependen del idioma y opcionalmente de la familia.

_FAMILY_DESCRIPTIONS = {
    "dbasic":     {"es": "gama económica, ideal para uso ligero (oficinas, eventos puntuales)",
                   "en": "value range, ideal for light use (offices, occasional events)",
                   "fr": "gamme économique, idéale pour un usage léger (bureaux, événements ponctuels)"},
    "dstandard":  {"es": "gama estándar, equilibrio precio-calidad para uso medio (retail, hostelería)",
                   "en": "standard range, balanced price-quality for medium use (retail, hospitality)",
                   "fr": "gamme standard, équilibre prix-qualité pour usage moyen (retail, hôtellerie)"},
    "dclassic":   {"es": "gama clásica con base elegante, perfecta para hoteles y recepciones premium",
                   "en": "classic range with elegant base, perfect for hotels and premium receptions",
                   "fr": "gamme classique avec base élégante, parfaite pour hôtels et réceptions premium"},
    "dline":      {"es": "gama profesional con base reforzada para uso intensivo",
                   "en": "professional range with reinforced base for intensive use",
                   "fr": "gamme professionnelle avec base renforcée pour usage intensif"},
    "dsafety":    {"es": "enfocada a seguridad y exterior (industria, obras, fábricas)",
                   "en": "focused on safety and outdoor use (industry, construction, factories)",
                   "fr": "orientée sécurité et extérieur (industrie, chantiers, usines)"},
    "dterminal":  {"es": "diseñada para transporte (aeropuertos, estaciones, terminales)",
                   "en": "designed for transport (airports, stations, terminals)",
                   "fr": "conçue pour le transport (aéroports, gares, terminaux)"},
    "dlimit":     {"es": "gama insignia premium, máxima durabilidad y acabados de alta gama",
                   "en": "premium flagship range, maximum durability and high-end finishes",
                   "fr": "gamme phare premium, durabilité maximale et finitions haut de gamme"},
    "daccessory": {"es": "accesorios complementarios (bastidores, carteles, ganchos, cordones)",
                   "en": "complementary accessories (stands, signs, hooks, ropes)",
                   "fr": "accessoires complémentaires (chevalets, panneaux, crochets, cordons)"},
}

_TEMPLATES_FIXED = {
    "es": {
        "subject": "Información Dlimit Tactic — catálogo y tarifa",
        "subject_with_family": "Información Dlimit {family} — catálogo y tarifa",
        "greeting": "Hola{name_part},",
        "intro": "Gracias por tu interés en los productos de Dlimit Tactic.",
        "family_intro": "Veo que estás interesado en la familia <strong>{family}</strong> — {description}.",
        "attachments": "Te adjunto en este email <strong>el catálogo general</strong> con todas nuestras 8 familias y <strong>la tarifa de precios PVP</strong> actualizada para que puedas valorar la solución que mejor encaja con tu proyecto.",
        "volume_note": "Para <strong>volúmenes especiales</strong>, configuraciones particulares o personalización con tu logotipo, ajustamos condiciones caso por caso.",
        "callback": "<strong>Ester Prieto</strong>, nuestra responsable comercial, te llamará en las <strong>próximas 24 horas laborables</strong> al teléfono que nos facilites para resolver dudas y ajustar el presupuesto. Si lo prefieres, puedes contactar directamente al <a href=\"tel:+34932526915\" style=\"color:#2A9D2A\">932 526 915</a>.",
        "closing": "Quedamos a tu disposición.",
        "client_message_label": "Tu mensaje",
        "page_label": "Página de referencia",
    },
    "en": {
        "subject": "Dlimit Tactic information — catalog and price list",
        "subject_with_family": "Dlimit {family} information — catalog and price list",
        "greeting": "Hello{name_part},",
        "intro": "Thank you for your interest in Dlimit Tactic products.",
        "family_intro": "I see you are interested in our <strong>{family}</strong> family — {description}.",
        "attachments": "Attached you will find <strong>our general catalog</strong> with all 8 product families and <strong>our up-to-date PVP price list</strong> so you can evaluate the solution that best fits your project.",
        "volume_note": "For <strong>special volumes</strong>, custom configurations or personalisation with your logo, we adjust conditions case by case.",
        "callback": "<strong>Ester Prieto</strong>, our sales manager, will call you within the <strong>next 24 working hours</strong> on the phone you provide to clarify any questions and prepare a tailored quote. You can also reach us directly at <a href=\"tel:+34932526915\" style=\"color:#2A9D2A\">+34 932 526 915</a>.",
        "closing": "Looking forward to helping you.",
        "client_message_label": "Your message",
        "page_label": "Reference page",
    },
    "fr": {
        "subject": "Information Dlimit Tactic — catalogue et tarif",
        "subject_with_family": "Information Dlimit {family} — catalogue et tarif",
        "greeting": "Bonjour{name_part},",
        "intro": "Merci pour votre intérêt pour les produits Dlimit Tactic.",
        "family_intro": "Je vois que vous êtes intéressé(e) par la famille <strong>{family}</strong> — {description}.",
        "attachments": "Vous trouverez en pièce jointe <strong>notre catalogue général</strong> avec nos 8 familles de produits et <strong>notre tarif PVP à jour</strong> pour évaluer la solution qui correspond le mieux à votre projet.",
        "volume_note": "Pour <strong>les volumes importants</strong>, configurations particulières ou personnalisation avec votre logo, nous adaptons les conditions au cas par cas.",
        "callback": "<strong>Ester Prieto</strong>, notre responsable commerciale, vous contactera dans les <strong>24 heures ouvrées</strong> au numéro que vous nous indiquerez pour répondre à vos questions et préparer un devis sur mesure. Vous pouvez aussi nous joindre directement au <a href=\"tel:+34932526915\" style=\"color:#2A9D2A\">+34 932 526 915</a>.",
        "closing": "Au plaisir de vous aider.",
        "client_message_label": "Votre message",
        "page_label": "Page de référence",
    },
}

def _render_fixed_template(
    *,
    language: str,
    family: Optional[str],
    client_name: Optional[str],
    client_message: Optional[str],
    page_url: Optional[str],
) -> dict:
    """Genera subject + html_body desde templates fijos (sin llamada a Claude)."""
    lang = (language or "es").lower()[:2]
    if lang not in _TEMPLATES_FIXED:
        lang = "es"  # fallback
    t = _TEMPLATES_FIXED[lang]

    fam = (family or "").lower().strip()
    family_pretty = fam.capitalize() if fam else None

    # Subject (con familia si la hay)
    if family_pretty:
        subject = t["subject_with_family"].format(family=family_pretty)
    else:
        subject = t["subject"]

    # Greeting con nombre si lo dio
    name_part = ""
    if client_name and client_name.strip():
        first_name = client_name.strip().split()[0][:30]
        name_part = " " + first_name
    greeting = t["greeting"].format(name_part=name_part)

    # Cuerpo principal
    parts = [f"<p>{greeting}</p>"]
    parts.append(f"<p>{t['intro']}</p>")

    if fam in _FAMILY_DESCRIPTIONS:
        desc = _FAMILY_DESCRIPTIONS[fam].get(lang, _FAMILY_DESCRIPTIONS[fam]["es"])
        parts.append(f"<p>{t['family_intro'].format(family=family_pretty, description=desc)}</p>")

    parts.append(f"<p>{t['attachments']}</p>")
    parts.append(f"<p>{t['volume_note']}</p>")
    parts.append(f"<p>{t['callback']}</p>")
    parts.append(f"<p>{t['closing']}</p>")

    # Si el cliente añadió mensaje, lo mostramos al final como referencia
    if client_message and client_message.strip():
        msg_safe = client_message.strip()[:500]
        # Escape básico de HTML
        msg_safe = msg_safe.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(
            f'<p style="background:#f5f7f5;padding:10px;border-left:3px solid #2A9D2A;'
            f'font-size:13px;color:#444;margin-top:14px"><strong>{t["client_message_label"]}:</strong> '
            f'<em>"{msg_safe}"</em></p>'
        )

    # Página visitada (útil para Ester cuando llame)
    if page_url and page_url.startswith(("http://", "https://")):
        parts.append(
            f'<p style="font-size:11px;color:#999;margin-top:16px">'
            f'{t["page_label"]}: <a href="{page_url}" style="color:#999">{page_url}</a></p>'
        )

    return {"subject": subject, "html_body": "\n".join(parts)}


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
    use_ai: Optional[bool] = None,
    client_message: Optional[str] = None,
    page_url: Optional[str] = None,
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

    # Si no se especifica use_ai, decidir por defecto: chat usa IA, web-button no
    _use_ai = use_ai if use_ai is not None else (source != "web-button")

    def _run():
        try:
            if _use_ai:
                body = _generate_body(
                    language=detected_language,
                    source=source,
                    family=detected_family,
                    sector=detected_sector,
                    context_summary=conversation_summary,
                )
            else:
                body = _render_fixed_template(
                    language=detected_language,
                    family=detected_family,
                    client_name=client_name,
                    client_message=client_message,
                    page_url=page_url,
                )
            html_full = body["html_body"] + ESTER_SIGNATURE_HTML
            # Resolver catálogo según familia detectada
            _cat_url = _resolve_catalog_for_family(detected_family)
            _cat_name = (
                f"Catalogo-{detected_family.capitalize()}.pdf"
                if detected_family and detected_family.strip().lower() in _FAMILY_CATALOG_URLS
                else "Catalogo-Dlimit-General.pdf"
            )
            res = _send_via_brevo(
                to_email=client_email,
                to_name=client_name,
                subject=body["subject"],
                html_content=html_full,
                catalog_url=_cat_url,
                catalog_name=_cat_name,
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

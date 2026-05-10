# Chatbot multi-brand · Dlimit + Dsafety

Backend RAG compartido para Dlimit Tactic S.L. y su marca industrial Dsafety™.
Una sola instancia Render sirve a las dos webs.

## Stack
- Flask + Gunicorn (Python 3.12.7)
- Voyage AI (embeddings densos `voyage-3-large`)
- Qdrant (collection `dlimit-kb`, layers `publica` para Dlimit y `dsafety` para Dsafety)
- BM25 sparse via fastembed
- Claude Haiku 4.5

## API

```
POST /api/chat
Content-Type: application/json

{
  "question": "¿Qué postes recomiendas para zonas EPA?",
  "brand": "dsafety"   // opcional, default "dlimit"
}
```

Respuesta:

```json
{
  "answer": "...",
  "sources": [...],
  "tokens": {"in": 1234, "out": 456},
  "model": "claude-haiku-4-5",
  "brand": "dsafety"
}
```

## Health check
`GET /health` → `{"status":"ok","brands":["dlimit","dsafety"]}`

## CORS
Permitido para:
- `https://www.dsafety.es` / `https://dsafety.es`
- `https://www.dlimit.net` / `https://dlimit.net`
- `localhost:5000` (dev)

## Deploy en Render

1. Push de este repo al servicio existente `dlimit-chatbot` en Render
2. Las 4 env vars ya configuradas: `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`
3. Render detecta `render.yaml` y despliega automáticamente

## Local dev

```bash
export ANTHROPIC_API_KEY=...
export VOYAGE_API_KEY=...
export QDRANT_URL=https://...:6333
export QDRANT_API_KEY=...
pip install -r requirements.txt
python chatbot_server.py
# abre http://localhost:5000
```

## Cambios respecto a versión anterior

- Soporte multi-brand vía parámetro `brand` en POST
- `BRAND_CONFIG` con mapping brand → layer + system prompt
- `SYSTEM_PROMPT_DSAFETY` específico (B2B industrial, EPI, sin venta directa)
- CORS habilitado para dominios de producción
- Endpoint `/health` para monitoring
- Demo HTML con selector de brand para probar las dos KBs
- Compatibilidad hacia atrás: si `brand` no llega, asume `dlimit`

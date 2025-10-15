# ChatGPT Integration Guide för Kolada MCP

Denna guide visar hur du kan använda Kolada MCP-servern med ChatGPT.

## Alternativ 1: Lokal utveckling och ngrok (Snabbast att testa)

### Steg 1: Installera dependencies
```bash
uv sync
```

### Steg 2: Starta HTTP-servern
```bash
uv run kolada-mcp-http
```

Servern körs nu på `http://localhost:8000`

### Steg 3: Exponera med ngrok
Ladda ner och installera [ngrok](https://ngrok.com/download)

```bash
ngrok http 8000
```

Du får en publik URL som: `https://abc123.ngrok.io`

### Steg 4: Konfigurera i ChatGPT
1. Gå till ChatGPT (betalkonto krävs för GPT Actions)
2. Skapa en ny GPT eller redigera en befintlig
3. Lägg till "Actions"
4. Importera OpenAPI schema från: `https://abc123.ngrok.io/openapi.json`
5. Testa med några frågor!

## Alternativ 2: Deploy till produktion

### Railway
1. Skapa konto på [Railway](https://railway.app)
2. Anslut ditt GitHub-repo
3. Railway detekterar automatiskt Dockerfile
4. Sätt miljövariabel: `START_MODE=http`
5. Deploy!

### Render
1. Skapa konto på [Render](https://render.com)
2. Ny Web Service från GitHub-repo
3. Start command: `uv run kolada-mcp-http`
4. Deploy!

### Docker Compose (för egen server)
```yaml
version: '3.8'
services:
  kolada-mcp:
    build: .
    ports:
      - "8000:8000"
    environment:
      - START_MODE=http
    command: uv run kolada-mcp-http
```

## Testa HTTP API:et

### Lista tillgängliga verktyg
```bash
curl http://localhost:8000/tools
```

### Sök efter KPIs
```bash
curl -X POST http://localhost:8000/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "search_kpis",
    "arguments": {
      "query": "unemployment rate"
    }
  }'
```

### Hämta kommundata
```bash
curl -X POST http://localhost:8000/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "list_municipalities",
    "arguments": {}
  }'
```

## Exempel på ChatGPT-frågor

När du har konfigurerat GPT Actions kan du ställa frågor som:

- "Vilka kommuner i Sverige har lägst arbetslöshet?"
- "Visa mig data om utbildningskvalitet i Stockholms län"
- "Jämför sjukvårdskostnader mellan Göteborg och Malmö"
- "Hitta kommuner med bästa miljöstatistiken"

## Felsökning

### Servern startar inte
- Kontrollera att alla dependencies är installerade: `uv sync`
- Kolla om port 8000 redan används

### ChatGPT kan inte ansluta
- Kontrollera att ngrok/deployment är aktiv
- Verifiera att URL:en är korrekt i GPT Actions
- Testa att öppna `/tools` endpoint i webbläsaren

### CORS-fel
- HTTP-servern är konfigurerad för ChatGPT-domäner
- Om du använder annan frontend, uppdatera `allow_origins` i `http_server.py`

## Säkerhet för produktion

**OBS:** För produktionsdrift, lägg till:

1. **API-nyckel autentisering**
```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/tool")
async def execute_tool(request: ToolRequest, token: str = Depends(security)):
    # Validera token
    pass
```

2. **Rate limiting**
```bash
uv add slowapi
```

3. **HTTPS** - Använd alltid HTTPS i produktion (Railway/Render ger detta automatiskt)

## Support

För frågor eller problem, öppna en issue på GitHub:
https://github.com/aerugo/kolada-mcp/issues

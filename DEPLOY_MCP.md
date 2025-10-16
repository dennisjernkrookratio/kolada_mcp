# Deployment Guide: MCP Server för ChatGPT Deep Research

## Översikt
Denna guide visar hur du deployar Kolada MCP-servern med SSE (Server-Sent Events) transport för ChatGPT Deep Research.

## Vad är skillnaden?

### HTTP Server (kolada-mcp)
- REST API med 9 specialiserade tools
- För ChatGPT Custom GPTs med Actions
- URL: `https://kolada-mcp-297818125455.europe-north1.run.app`

### MCP SSE Server (kolada-mcp-sse)  
- Native MCP-protokoll med SSE transport
- 2 tools enligt OpenAI's spec: `search` och `fetch`
- För ChatGPT Deep Research och Responses API
- Denna guide

## Deploya MCP SSE-server

### Steg 1: Bygg och deploya

```powershell
# Från projekt-directory
cd "C:\Users\Dennis Jernkrook\source\repos\kolada-mcp"

# Bygg Docker image
gcloud builds submit --config cloudbuild-mcp.yaml

# Deploya till Cloud Run
gcloud run deploy kolada-mcp-sse `
  --image gcr.io/kolada-mcp-prod/kolada-mcp-sse `
  --platform managed `
  --region europe-north1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 1 `
  --timeout 300 `
  --max-instances 10 `
  --min-instances 0
```

### Steg 2: Få SSE endpoint URL

Efter deployment får du en URL som:
```
https://kolada-mcp-sse-xxxxx.europe-north1.run.app
```

**Viktigt:** SSE-endpointen är:
```
https://kolada-mcp-sse-xxxxx.europe-north1.run.app/sse/
```

Lägg till `/sse/` på slutet!

### Steg 3: Testa servern

```powershell
# Testa SSE endpoint
curl https://YOUR-URL/sse/

# Du ska få SSE-headers tillbaka
```

## Använda med ChatGPT Deep Research

### I ChatGPT UI

1. Gå till **Settings → Connectors**
2. Klicka **Add MCP Server**
3. Klistra in din SSE URL: `https://YOUR-URL/sse/`
4. Spara

### Via Responses API

```bash
curl https://api.openai.com/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
  "model": "o4-mini-deep-research",
  "input": [
    {
      "role": "developer",
      "content": [
        {
          "type": "input_text",
          "text": "Du är en forskningsassistent som söker i svensk kommundata från Kolada."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "Vilka kommuner har lägst arbetslöshet? Ge en detaljerad analys."
        }
      ]
    }
  ],
  "reasoning": {
    "summary": "auto"
  },
  "tools": [
    {
      "type": "mcp",
      "server_label": "kolada",
      "server_url": "https://YOUR-URL/sse/",
      "allowed_tools": [
        "search",
        "fetch"
      ],
      "require_approval": "never"
    }
  ]
}'
```

## Tillgängliga tools

### `search`
Söker efter KPIs med semantisk AI-sökning.

**Argument:**
- `query`: Sökfråga på svenska eller engelska

**Exempel:**
- "arbetslöshet"
- "utbildningsnivå"
- "förskoleplats"

**Returnerar:**
Lista med KPI-resultat (id, title, snippet, url)

### `fetch`
Hämtar fullständig KPI-information och statistik.

**Argument:**
- `id`: KPI-ID (t.ex. "N00941")

**Returnerar:**
Komplett KPI-dokument med metadata och senaste statistik för stora kommuner.

## Felsökning

### Problem: "Could not connect to MCP server"

Kontrollera:
1. URL slutar med `/sse/`
2. Servern är deployad och live
3. `--allow-unauthenticated` är satt

```powershell
# Verifiera deployment
gcloud run services describe kolada-mcp-sse --region europe-north1
```

### Problem: "Tool execution failed"

Kolla loggar:
```powershell
gcloud run logs tail kolada-mcp-sse --region europe-north1
```

### Problem: Out of memory

Öka minnet:
```powershell
gcloud run services update kolada-mcp-sse --memory 4Gi --region europe-north1
```

## Uppdatera deployment

```powershell
# Efter kod-ändringar
git add .
git commit -m "Update MCP server"
git push origin main

# Bygg och deploya ny version
gcloud builds submit --config cloudbuild-mcp.yaml
gcloud run deploy kolada-mcp-sse `
  --image gcr.io/kolada-mcp-prod/kolada-mcp-sse `
  --region europe-north1
```

## Jämförelse: När använda vilken server?

### Använd HTTP-servern när:
- ✅ Du vill använda Custom GPT i ChatGPT
- ✅ Du vill ha alla 9 specialiserade tools
- ✅ Du vill testa snabbt och enkelt
- ✅ Du inte behöver Deep Research

### Använd MCP SSE-servern när:
- ✅ Du vill använda Deep Research (o4-mini-deep-research)
- ✅ Du vill integrera via Responses API
- ✅ Du vill ha längre, mer omfattande analyser
- ✅ Du vill ha native MCP-protokoll

## Båda servrarna kan köra samtidigt!

Du kan ha båda deployade och använda dem för olika ändamål:
- **HTTP**: Custom GPT för snabba frågor
- **MCP SSE**: Deep Research för omfattande analyser

Totalkostnad inom free tier: $0/månad för normal användning! 🎉

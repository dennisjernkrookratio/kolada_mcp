# Railway Deployment Guide

## Snabb deploy till Railway.app

1. **Committa och pusha din kod:**
   ```powershell
   git add .
   git commit -m "Add HTTP server for ChatGPT integration"
   git push origin main
   ```

2. **Gå till Railway:**
   - https://railway.app
   - Klicka "Start a New Project"
   - Logga in med GitHub
   - Välj "Deploy from GitHub repo"
   - Välj `kolada-mcp`

3. **Railway detekterar automatiskt:**
   - Dockerfile.http kommer användas
   - Port 8001 exponeras automatiskt
   - Du får en publik URL

4. **Få din URL:**
   - Gå till Settings -> Networking
   - Klicka "Generate Domain"
   - Du får: `your-app.railway.app`

5. **Använd i ChatGPT:**
   - URL: `https://your-app.railway.app/openapi.json`

## Enklare testning lokalt utan ngrok

Om du bara vill testa lokalt utan att exponera publikt ännu:

### Testa API:et direkt:

```powershell
# Lista verktyg
curl http://localhost:8001/tools | ConvertFrom-Json | ConvertTo-Json -Depth 10

# Sök KPIs
curl -Method POST -Uri http://localhost:8001/tool `
  -ContentType "application/json" `
  -Body '{"tool_name":"search_kpis","arguments":{"query":"unemployment","limit":5}}'

# Lista kommuner
curl -Method POST -Uri http://localhost:8001/tool `
  -ContentType "application/json" `
  -Body '{"tool_name":"list_municipalities","arguments":{}}'
```

### Öppna i webbläsare:
- http://localhost:8001 - Visa API-info
- http://localhost:8001/tools - Lista alla verktyg
- http://localhost:8001/docs - Interaktiv API-dokumentation (Swagger)

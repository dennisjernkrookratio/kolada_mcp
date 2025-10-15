# Snabbguide: Exponera servern för ChatGPT

## Servern körs nu! ✅
Din server är igång på: http://localhost:8001

## Nästa steg: Gör den tillgänglig för ChatGPT

### Metod 1: ngrok (Snabbast)

1. **Ladda ner ngrok:**
   - Gå till: https://ngrok.com/download
   - Eller via Chocolatey: `choco install ngrok`
   - Eller via Scoop: `scoop install ngrok`

2. **Kör ngrok:**
   ```powershell
   ngrok http 8001
   ```

3. **Du får en publik URL:**
   ```
   Forwarding  https://abc123.ngrok-free.app -> http://localhost:8001
   ```

4. **Konfigurera i ChatGPT:**
   - Gå till https://chat.openai.com
   - Skapa en ny "Custom GPT" eller "GPT"
   - Lägg till "Actions"
   - Importera schema från: `https://abc123.ngrok-free.app/openapi.json`
   - Spara!

### Metod 2: Deploy till molnet (För permanent användning)

#### Railway (Gratis)
```powershell
# Push din kod till GitHub
git add .
git commit -m "Add HTTP server for ChatGPT"
git push

# Gå till railway.app och:
# 1. Logga in med GitHub
# 2. "New Project" -> "Deploy from GitHub repo"
# 3. Välj kolada-mcp
# 4. Railway detekterar automatiskt Dockerfile.http
```

#### Render (Gratis)
1. Gå till: https://render.com
2. New -> Web Service
3. Anslut GitHub repo
4. Build Command: `uv sync --python 3.12`
5. Start Command: `uv run kolada-mcp-http`
6. Deploy!

### Testa din server lokalt först:

```powershell
# Lista alla verktyg
curl http://localhost:8001/tools

# Testa sök-funktionen
curl -X POST http://localhost:8001/tool `
  -H "Content-Type: application/json" `
  -d '{
    "tool_name": "search_kpis",
    "arguments": {
      "query": "unemployment"
    }
  }'

# Lista kommuner
curl -X POST http://localhost:8001/tool `
  -H "Content-Type: application/json" `
  -d '{
    "tool_name": "list_municipalities",
    "arguments": {}
  }'
```

## Exempel på ChatGPT-frågor (när Actions är konfigurerat)

- "Vilka svenska kommuner har lägst arbetslöshet?"
- "Jämför skolresultat mellan Stockholm och Göteborg"
- "Visa mig miljöstatistik för kommuner i Skåne"
- "Hitta kommuner med bäst äldreomsorg"
- "Analysera utvecklingen av bostadspriser i Uppsala"

## Tips

- Håll PowerShell-fönstret med servern öppet
- Servern måste köra när ChatGPT använder den
- ngrok gratis-versionen stängs av efter 2 timmar inaktivitet
- För produktion, använd Railway/Render

## Nästa kommando att köra:

```powershell
ngrok http 8001
```

Lycka till! 🚀

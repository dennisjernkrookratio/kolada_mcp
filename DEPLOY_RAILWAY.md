# Deploy till Railway - Permanent lösning

## Problemet med ngrok gratis
- Kräver att man klickar "Visit Site" i webbläsare först
- Fungerar inte bra med ChatGPT API-anrop
- URL ändras varje omstart
- 2 timmars timeout

## Lösning: Railway (Gratis, Permanent)

### Steg 1: Förbered deployment

Du har redan allt klart! Projektets `Dockerfile.http` är redo.

### Steg 2: Pusha till GitHub

```powershell
# Om du inte redan gjort det
git add .
git commit -m "Add HTTP server for ChatGPT integration"
git push origin main
```

### Steg 3: Deploy på Railway

1. **Gå till:** https://railway.app
2. **Logga in** med GitHub
3. **Klicka:** "New Project"
4. **Välj:** "Deploy from GitHub repo"
5. **Välj:** `aerugo/kolada-mcp`
6. Railway detekterar automatiskt `Dockerfile.http`

### Steg 4: Konfigurera Railway

Efter deploy:
1. Gå till ditt projekt
2. Klicka på **"Settings"**
3. Under **"Environment"** -> **"Networking"**
4. Klicka **"Generate Domain"**

Du får en permanent URL som:
```
kolada-mcp.railway.app
```

### Steg 5: Använd i ChatGPT

I ChatGPT Actions, använd:
```
https://kolada-mcp.railway.app/openapi.json
```

## Fördelar med Railway

✅ **Permanent URL** - Ändras aldrig  
✅ **Ingen "Visit Site" knapp** - Fungerar direkt med API  
✅ **Alltid igång** - Ingen timeout  
✅ **GRATIS** - $5 gratis kredit/månad (mer än tillräckligt)  
✅ **Automatiska deploys** - Vid git push  

## Alternativ: Render.com

Om Railway inte fungerar, prova Render:

1. **Gå till:** https://render.com
2. **Logga in** med GitHub
3. **New** -> **Web Service**
4. Anslut repo: `aerugo/kolada-mcp`
5. **Settings:**
   - Build Command: `uv sync --python 3.12`
   - Start Command: `uv run kolada-mcp-http`
6. **Deploy!**

Du får URL: `https://kolada-mcp.onrender.com`

## Jämförelse

| Lösning | Kostnad | URL | API-friendly | Uptime |
|---------|---------|-----|--------------|--------|
| ngrok gratis | Gratis | Temporär | ❌ Nej | 2h |
| ngrok betalad | $8/mån | Temporär | ✅ Ja | ∞ |
| Railway | Gratis* | Permanent | ✅ Ja | ∞ |
| Render | Gratis* | Permanent | ✅ Ja | ∞ |

*Gratis tier räcker för din användning

## Rekommendation

🎯 **Använd Railway** - Enklast och snabbast att sätta upp!

Vill du att jag hjälper dig deploya till Railway nu?

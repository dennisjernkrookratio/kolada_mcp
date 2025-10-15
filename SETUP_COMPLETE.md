# ✅ SETUP KOMPLETT!

## Status
- ✅ HTTP-server körs
- ✅ ngrok authtoken konfigurerad
- ✅ ngrok tunnel startar

## Din publika URL

ngrok körs nu! För att se din publika URL:

### Alternativ 1: Öppna ngrok Web UI
Öppna i webbläsare: http://localhost:4040

Du ser:
- Din publika URL (t.ex. `https://abc123.ngrok-free.app`)
- Live trafik
- Request/Response details

### Alternativ 2: Kolla terminalen
Titta på det PowerShell-fönster där ngrok körs. Du ser något som:

```
ngrok

Session Status                online
Account                       Your Account (Plan: Free)
Forwarding                    https://abc123.ngrok-free.app -> http://localhost:8001
```

## Nästa steg: Konfigurera ChatGPT

1. **Kopiera din ngrok URL** (t.ex. `https://abc123.ngrok-free.app`)

2. **Gå till ChatGPT:**
   - https://chat.openai.com
   - Klicka på ditt namn (nedre vänster)
   - "My GPTs" -> "Create a GPT"

3. **Konfigurera GPT:**
   - **Name:** "Kolada Statistics"
   - **Description:** "Access Swedish municipal statistics"
   - **Instructions:** "Du är en expert på svensk kommunstatistik. Använd Kolada API för att svara på frågor om kommuner, regioner och KPIs."

4. **Lägg till Actions:**
   - Klicka på "Configure" tab
   - Scrolla ner till "Actions"
   - Klicka "Create new action"
   - I "Schema" fältet, importera från URL:
     ```
     https://abc123.ngrok-free.app/openapi.json
     ```
     (Byt ut abc123 med din faktiska ngrok URL)

5. **Spara och testa:**
   - Klicka "Save"
   - Testa med: "Vilka kommuner i Sverige har lägst arbetslöshet?"

## Viktigt att veta

- 🔄 **ngrok gratis-version:** URL:en ändras varje gång du startar om ngrok
- ⏰ **Timeout:** Stängs av efter 2 timmar inaktivitet
- 💻 **Server måste köra:** Håll båda PowerShell-fönstren öppna
  - Ett med HTTP-servern
  - Ett med ngrok

## Exempel-frågor för ChatGPT

När GPT:n är konfigurerad, testa:

- "Vilka kommuner har lägst arbetslöshet?"
- "Jämför skolresultat mellan Stockholm och Göteborg över tid"
- "Visa miljöstatistik för kommuner i Skåne"
- "Hitta kommuner med bäst äldreomsorg"
- "Analysera utvecklingen av bostadspriser i Uppsala senaste 5 åren"

## Felsökning

### "Cannot connect to server"
- Kontrollera att HTTP-servern körs (PowerShell-fönster ska vara öppet)
- Testa: `curl http://localhost:8001/health`

### "ngrok URL fungerar inte"
- Öppna http://localhost:4040 och kontrollera status
- Testa ngrok URL:en direkt i webbläsare

### "Actions inte tillgängliga"
- ChatGPT Plus krävs för att använda Custom GPTs med Actions
- Alternativt: använd OpenAI API direkt

## För permanent lösning

För att slippa starta om ngrok varje gång, deploya till Railway:

```powershell
git add .
git commit -m "Add HTTP server for ChatGPT"
git push origin main
```

Sedan gå till railway.app och anslut ditt repo. Du får en permanent URL!

---

Lycka till! 🚀

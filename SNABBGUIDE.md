# 🚀 Snabbguide: Starta Kolada MCP för ChatGPT

## Starta servern (varje gång)

### Steg 1: Starta HTTP-servern
Öppna PowerShell i projektmappen och kör:
```powershell
$env:Path = "C:\Users\Dennis Jernkrook\.local\bin;$env:Path"
.\.venv\Scripts\python.exe run_http_server.py
```

**VIKTIGT:** Håll detta PowerShell-fönster öppet!

Du ser:
```
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8001
```

### Steg 2: Starta ngrok
Öppna ett NYTT PowerShell-fönster och kör:
```powershell
ngrok http 8001
```

**VIKTIGT:** Håll detta PowerShell-fönster öppet!

Du ser din publika URL:
```
Forwarding    https://xxx-yyy-zzz.ngrok-free.dev -> http://localhost:8001
```

### Steg 3: Kopiera ngrok-URL:en
Från ngrok-fönstret, kopiera URL:en som börjar med `https://`

Exempel: `https://corporatist-anisa-nonmarrying.ngrok-free.dev`

---

## Konfigurera ChatGPT (första gången)

### 1. Skapa Custom GPT
- Gå till: https://chat.openai.com
- Klicka på ditt namn (nedre vänstra hörnet)
- Välj **"My GPTs"**
- Klicka **"Create a GPT"**

### 2. Grundinställningar (Create tab)

**Name:**
```
Kolada Statistik
```

**Description:**
```
Tillgång till svensk kommunstatistik från Kolada-databasen
```

**Instructions:**
```
Du är en expert på svensk kommunstatistik med tillgång till Kolada-databasen.

Använd tillgängliga verktyg för att:
- Söka efter KPIs (nyckeltal) baserat på ämne
- Hämta statistik för kommuner och regioner  
- Analysera trender över tid
- Jämföra kommuner

Svara alltid på svenska. Var tydlig med källor och förklara vad siffrorna betyder.

Tillgängliga verktyg:
- search_kpis: Sök KPIs semantiskt
- fetch_kolada_data: Hämta faktisk statistik
- list_municipalities: Lista alla kommuner
- analyze_kpi_across_municipalities: Analysera KPI över kommuner
- compare_kpis: Jämför korrelation mellan KPIs
```

### 3. Lägg till Actions (Configure tab)

1. Klicka på **"Configure"** tab
2. Scrolla ner till **"Actions"**
3. Klicka **"Create new action"**
4. Under **"Authentication"** välj: **"None"**
5. Under **"Schema"** klicka **"Import from URL"**
6. Klistra in din ngrok-URL + `/openapi.json`:
   ```
   https://din-ngrok-url.ngrok-free.dev/openapi.json
   ```
   (Byt ut `din-ngrok-url` med din faktiska URL)
7. Klicka **"Import"**
8. Klicka **"Save"** (uppe till höger)

---

## Testa GPT:n

Prova dessa frågor:

### Grundläggande
- "Vilka kommuner finns i Skåne?"
- "Sök efter KPIs om arbetslöshet"
- "Lista verksamhetsområden"

### Avancerade  
- "Vilka kommuner har lägst arbetslöshet?"
- "Jämför skolresultat mellan Stockholm och Göteborg de senaste 5 åren"
- "Visa utvecklingen av bostadspriser i Uppsala"
- "Hitta kommuner med bäst miljöstatistik"
- "Analysera korrelationen mellan utbildningsnivå och inkomst"

---

## Uppdatera Actions (när ngrok-URL ändras)

När du startar om ngrok får du en ny URL. Då måste du uppdatera:

1. Gå till din GPT i ChatGPT
2. Klicka **"Edit GPT"**
3. Gå till **"Configure"** tab
4. Under **"Actions"**, klicka på action
5. Uppdatera URL:en i schema-importen
6. Klicka **"Update"** och sedan **"Save"**

---

## Felsökning

### "Server svarar inte"
✅ Kontrollera att båda PowerShell-fönstren är öppna  
✅ Testa: `curl http://localhost:8001/health` i en tredje terminal  
✅ Kontrollera att ngrok visar "online"

### "Actions fungerar inte"  
✅ Kontrollera att du importerat rätt URL (ska sluta med `/openapi.json`)  
✅ Testa ngrok-URL:en direkt i webbläsare  
✅ Se till att Authentication är "None"

### "ngrok timeout"
✅ ngrok gratis stängs av efter 2h inaktivitet  
✅ Starta om ngrok: `Ctrl+C` och kör `ngrok http 8001` igen  
✅ Uppdatera Actions med nya URL:en

---

## Snabbkommandon (för Windows)

Skapa en `start_kolada.bat` fil:

```batch
@echo off
echo Startar Kolada HTTP Server...
start powershell -NoExit -Command "$env:Path = 'C:\Users\Dennis Jernkrook\.local\bin;$env:Path'; .\.venv\Scripts\python.exe run_http_server.py"

timeout /t 5

echo Startar ngrok...
start powershell -NoExit -Command "ngrok http 8001"

echo.
echo Båda servrarna startar nu!
echo Kopiera ngrok-URL från det andra fönstret.
pause
```

Dubbelklicka på filen för att starta allt!

---

## För permanent lösning (ingen ngrok)

Deploy till Railway för en permanent URL:

```powershell
# Committa ändringar
git add .
git commit -m "Ready for deployment"
git push origin main

# Gå till railway.app
# Anslut GitHub repo
# Railway auto-deployer
# Använd den permanenta Railway-URL:en i ChatGPT Actions
```

---

## Verktyg som finns tillgängliga

| Verktyg | Beskrivning |
|---------|-------------|
| `search_kpis` | Sök KPIs semantiskt |
| `fetch_kolada_data` | Hämta faktisk statistik |
| `list_operating_areas` | Lista verksamhetsområden |
| `get_kpis_by_operating_area` | Hämta KPIs i ett område |
| `get_kpi_metadata` | Detaljerad KPI-info |
| `list_municipalities` | Lista kommuner/regioner |
| `analyze_kpi_across_municipalities` | Analysera KPI över kommuner |
| `compare_kpis` | Jämför KPI-korrelationer |
| `filter_municipalities_by_kpi` | Filtrera kommuner på KPI-värden |

---

## Din setup

- **Lokal server:** http://localhost:8001
- **ngrok URL:** https://corporatist-anisa-nonmarrying.ngrok-free.dev (ändras varje omstart)
- **OpenAPI schema:** Din ngrok URL + `/openapi.json`
- **Test endpoint:** Din ngrok URL + `/health`

---

Lycka till! 🎉

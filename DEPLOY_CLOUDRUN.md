# Deployment Guide: Google Cloud Run

## Översikt
Denna guide visar hur du deployar Kolada MCP HTTP-servern till Google Cloud Run.

## Förutsättningar
- Google Cloud konto (gratis, kräver kreditkort men debiteras inte inom free tier)
- Google Cloud CLI installerat
- Docker installerat (för lokal build)

## Free Tier Limits
Google Cloud Run free tier inkluderar:
- **2 miljoner requests/månad**
- **360,000 GB-seconds/månad** (minne × tid)
- **180,000 vCPU-seconds/månad**
- **1GB utgående trafik/månad**

Med 2GB RAM-container räcker detta för ~2000 requests/månad (ca 65/dag) helt gratis.

## Steg 1: Installera Google Cloud CLI

### Windows
Ladda ner och installera från: https://cloud.google.com/sdk/docs/install

Eller via PowerShell:
```powershell
# Ladda ner installer
Invoke-WebRequest -Uri "https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe" -OutFile "$env:TEMP\GoogleCloudSDKInstaller.exe"

# Kör installer
Start-Process -FilePath "$env:TEMP\GoogleCloudSDKInstaller.exe" -Wait
```

## Steg 2: Logga in och konfigurera

```powershell
# Logga in till Google Cloud
gcloud auth login

# Sätt projekt (skapa nytt om du inte har)
gcloud config set project DITT-PROJEKT-ID

# Aktivera Cloud Run API
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## Steg 3: Deploya till Cloud Run

```powershell
# Gå till projekt-directory
cd "C:\Users\Dennis Jernkrook\source\repos\kolada-mcp"

# Deploya direkt från source (Cloud Run bygger åt dig)
gcloud run deploy kolada-mcp `
  --source . `
  --dockerfile Dockerfile.cloudrun `
  --platform managed `
  --region europe-north1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 1 `
  --timeout 300 `
  --max-instances 10 `
  --min-instances 0
```

**Parametrar förklarat:**
- `--source .` - Deploya från nuvarande directory
- `--dockerfile Dockerfile.cloudrun` - Använd Cloud Run Dockerfile
- `--region europe-north1` - Stockholm (närmaste region)
- `--allow-unauthenticated` - Tillåt publika requests (för ChatGPT)
- `--memory 2Gi` - 2GB RAM (tillräckligt för sentence-transformers)
- `--cpu 1` - 1 vCPU
- `--timeout 300` - 5 minuters timeout (för långsamma requests)
- `--max-instances 10` - Max 10 parallella instanser
- `--min-instances 0` - Skala ner till 0 när oanvänd (gratis!)

## Steg 4: Få din URL

Efter deployment får du en URL:
```
Service [kolada-mcp] revision [kolada-mcp-00001-abc] has been deployed and is serving 100 percent of traffic.
Service URL: https://kolada-mcp-xxxxx-lz.a.run.app
```

## Steg 5: Testa deployment

```powershell
# Testa health endpoint
curl https://kolada-mcp-xxxxx-lz.a.run.app/health

# Hämta OpenAPI schema
curl https://kolada-mcp-xxxxx-lz.a.run.app/openapi.json
```

## Steg 6: Konfigurera ChatGPT

1. Gå till https://chat.openai.com
2. Skapa Custom GPT
3. Konfigurera Actions:
   - **Schema**: Importera från `https://YOUR-URL/openapi.json`
   - **Authentication**: None
4. Testa med: "Vilka kommuner har högst arbetslöshet?"

## Uppdatera deployment

När du gör ändringar i koden:

```powershell
# Pusha ändringar till GitHub
git add .
git commit -m "Update server"
git push origin main

# Deploya ny version
gcloud run deploy kolada-mcp `
  --source . `
  --dockerfile Dockerfile.cloudrun `
  --platform managed `
  --region europe-north1
```

## Övervaka kostnader

```powershell
# Se aktuell användning
gcloud run services describe kolada-mcp --region europe-north1

# Visa loggar
gcloud run logs tail kolada-mcp --region europe-north1

# Se metrics i browser
gcloud run services describe kolada-mcp --region europe-north1 --format="value(status.url)"
```

## Felsökning

### Problem: Container startar inte
```powershell
# Kolla loggar
gcloud run logs tail kolada-mcp --region europe-north1
```

### Problem: Out of memory
Öka minnet:
```powershell
gcloud run services update kolada-mcp --memory 4Gi --region europe-north1
```

### Problem: Timeout
Öka timeout:
```powershell
gcloud run services update kolada-mcp --timeout 600 --region europe-north1
```

## Ta bort service

```powershell
gcloud run services delete kolada-mcp --region europe-north1
```

## Kostnadsuppskattning

Med 2GB RAM och 1 vCPU:
- **0-2000 requests/månad**: $0 (inom free tier)
- **10,000 requests/månad**: ~$5
- **100,000 requests/månad**: ~$50

Free tier är generöst nog för de flesta användningsfall!

## Tips

1. **Cold starts**: Första requesten efter idle tar 10-15s (modell laddas). Överväg `--min-instances 1` för att hålla en instans varm ($7/månad).

2. **Regioner**: `europe-north1` (Stockholm) är närmast och billigast i Europa.

3. **Monitoring**: Använd Cloud Console för detaljerad monitoring: https://console.cloud.google.com/run

4. **Custom domain**: Koppla egen domän via Cloud Console → Cloud Run → Manage Custom Domains.

## Support

- Google Cloud Run docs: https://cloud.google.com/run/docs
- Pricing calculator: https://cloud.google.com/products/calculator

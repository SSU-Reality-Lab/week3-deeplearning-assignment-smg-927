# ===========================================
# Download MS COCO Captioning dataset for Assignment 3
# Windows PowerShell version
# ===========================================

Write-Host "📦 Downloading Microsoft COCO captioning dataset..."

# Create data directory
New-Item -ItemType Directory -Force -Path data

# Move into data folder
Set-Location data

# Download annotation file
Invoke-WebRequest -Uri https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip -OutFile coco.zip

Write-Host "📁 Extracting files..."
Expand-Archive coco.zip -DestinationPath .

Remove-Item coco.zip
Write-Host "🎉 Done! Dataset extracted to 'data' folder."

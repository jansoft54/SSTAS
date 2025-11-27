#!/bin/bash

set -e

cd /workspace

echo ">>> 1. Installiere aria2, unzip und git..."
apt-get update
apt-get install -y aria2 unzip git

echo ">>> 2. Starte Download mit aria2c..."
aria2c -x 16 -s 16 -o data.zip "https://zenodo.org/records/3625992/files/data.zip?download=1"

echo ">>> 3. Entpacke data.zip..."
unzip -o -q data.zip

echo ">>> 4. Clone SSTAS Repo..."
if [ -d "SSTAS" ]; then
    echo "Ordner SSTAS existiert bereits. Mache 'git pull'..."
    cd SSTAS
    git pull
   
else
    git clone "https://github.com/jansoft54/SSTAS"
    mv data SSTAS/
    cd SSTAS
fi
rm data.zip

echo ">>> FERTIG! Setup abgeschlossen."

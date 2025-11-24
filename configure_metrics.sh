#!/bin/bash
# filepath: var_home/MOOSS/configure_metrics.sh

# Configuration des métriques via arguments
# Usage: ./configure_metrics.sh --fid --lpips --dists

set -e

eval "$(conda shell.bash hook)"
conda activate mooss

CONFIG_FILE="var_home/MOOSS/mooss/config.py"
TEMP_FILE=$(mktemp)

echo "=========================================="
echo "CONFIGURATION DES MÉTRIQUES"
echo "=========================================="
echo ""

# Vérifier que des arguments sont fournis
if [ $# -eq 0 ]; then
    echo "❌ Aucune métrique spécifiée"
    echo ""
    echo "Usage: $0 --metric1 --metric2 ..."
    echo "Métriques disponibles: --fid --lpips --dists --dreamsim --dino --cmmd"
    echo ""
    echo "Exemple: $0 --dists --dreamsim"
    exit 1
fi

# Désactiver toutes les métriques par défaut
echo "→ Désactivation de toutes les métriques..."
python << 'EOF'
import re

CONFIG_FILE = "var_home/MOOSS/mooss/config.py"
TEMP_FILE = "/tmp/config_metrics.py"

with open(CONFIG_FILE, 'r') as f:
    content = f.read()

# Désactiver toutes les métriques
metrics = ['fid', 'lpips', 'dists', 'dreamsim', 'dino', 'cmmd']
for metric in metrics:
    # Utiliser raw string pour éviter les warnings
    pattern = r'"{}":\s*{{\s*"enabled":\s*(True|False)'.format(metric)
    replacement = '"{}":\n        {{"enabled": False'.format(metric)
    content = re.sub(pattern, replacement, content)

with open(TEMP_FILE, 'w') as f:
    f.write(content)
EOF

cp /tmp/config_metrics.py "${TEMP_FILE}"

# Activer les métriques spécifiées en arguments
echo "→ Activation des métriques sélectionnées..."
for arg in "$@"; do
    metric="${arg#--}"
    echo "  ✓ ${metric}"
    
    python << EOF
import re
import sys

TEMP_FILE = "${TEMP_FILE}"
metric = "${metric}"

with open(TEMP_FILE, 'r') as f:
    content = f.read()

# Utiliser raw string pour éviter les warnings
pattern = r'"{}":\s*{{\s*"enabled":\s*False'.format(metric)
replacement = '"{}":\n        {{"enabled": True'.format(metric)
content = re.sub(pattern, replacement, content)

with open(TEMP_FILE, 'w') as f:
    f.write(content)
EOF
done

# Appliquer les modifications
cp "${TEMP_FILE}" "${CONFIG_FILE}"
rm "${TEMP_FILE}"
rm -f /tmp/config_metrics.py

echo ""
echo "✓ Configuration des métriques mise à jour dans:"
echo "  ${CONFIG_FILE}"
echo ""

# Afficher la configuration actuelle
echo "Configuration actuelle:"
python << 'EOF'
import sys
sys.path.insert(0, 'var_home/MOOSS')
from mooss.config import ENABLED_METRICS, N_OBJECTIVES

print(f"  Nombre d'objectifs: {N_OBJECTIVES}")
print(f"  Métriques actives:")
for metric, config in ENABLED_METRICS.items():
    weight = config['weight']
    desc = config['description']
    print(f"    - {metric} (poids: {weight}): {desc}")
EOF
echo ""
echo "=========================================="
#!/bin/sh
set -e
cd "C:/Arib/MASSA AYAM/chicken detection + constrative classification"
for cfg in configs/config_pio_dev.yaml configs/config_pio_eq_dev.yaml; do
  echo "=============================================================="
  echo "MULAI $cfg  $(date '+%H:%M:%S')"
  echo "=============================================================="
  python src/train.py --config "$cfg" --method all --seeds 42,43,44 --save-model all
  echo "SELESAI $cfg  $(date '+%H:%M:%S')"
done
echo "SEMUA 18 RUN SELESAI"

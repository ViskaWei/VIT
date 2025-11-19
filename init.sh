# Load .env if present (local defaults)
if [ -f ./.env ]; then
  set -a
  . ./.env
  set +a
fi

source /srv/local/tmp/swei20/miniconda3/etc/profile.d/conda.sh
conda activate viska-torch-3

echo "ROOT is set to $ROOT"
echo "PYTHONPATH is set to $PYTHONPATH"
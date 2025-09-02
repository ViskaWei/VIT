if [ -f ./.env ]; then
  set -a
  . ./.env
  set +a
fi
source "$VENV_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "DATA_ROOT: $DATA_ROOT"    
echo "VENV_PATH: $VENV_PATH"
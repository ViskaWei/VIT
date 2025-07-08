source /srv/local/tmp/swei20/miniconda3/bin/activate viska-torch-3

export CUDA_VISIBLE_DEVICES='0'
export PYTHONPATH=$PWD:$PYTHONPATH

start_time=$(date +%s)
python ./scripts/run.py -f  ./configs/vit.yaml -w 1 


end_time=$(date +%s)
elapsed_time_1=$((end_time - start_time))
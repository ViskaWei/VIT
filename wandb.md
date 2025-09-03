d420a14ad27943c26db6f51c996070bf6294d0a0
pkill -u swei20 -f wandb
ps -o pid,ppid,stat,cmd -C wandb-core | grep defunct | awk '{print $2}' | sort -u | xargs -r kill -9
Sinteract -g gpu -c10  -t 3:0:0 -m 32G -a ee-452 -q ee-452
ipnport=$(shuf -i8000-9999 -n1)
echo $ipnport
jupyter notebook --no-browser --port=${ipnport} --ip=$(hostname -i)
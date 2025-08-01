n=200
temp=0.6
p=0.9

model='together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
python3 -m src.predict --model=$model --data='casino' --n_data=$n --temp=$temp --top_p=$p --syco=0.8
python3 -m src.predict --model=$model --data='casino' --n_data=$n --temp=$temp --top_p=$p --binary=1 --syco=0.8
python3 -m src.predict --model=$model --data='donations' --n_data=$n --temp=$temp --top_p=$p --syco=0.8
python3 -m src.predict --model=$model --data='donations' --n_data=$n --temp=$temp --top_p=$p --binary=1 --syco=0.8
python3 -m src.predict --model=$model --data='awry' --n_data=$n --temp=$temp --top_p=$p --syco=0.8
python3 -m src.predict --model=$model --data='awry' --n_data=$n --temp=$temp --top_p=$p --binary=1 --syco=0.8
python3 -m src.predict --model=$model --data='change' --n_data=$n --temp=$temp --top_p=$p --syco=0.8
python3 -m src.predict --model=$model --data='change' --n_data=$n --temp=$temp --top_p=$p --binary=1 --syco=0.8
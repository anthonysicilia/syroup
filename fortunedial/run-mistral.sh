model='together/mistralai/Mixtral-8x22B-Instruct-v0.1'
n=200
temp=0.7
p=1.0

python3 -m src.predict --model=$model --data='casino' --n_data=$n --temp=$temp --top_p=$p
python3 -m src.predict --model=$model --data='casino' --n_data=$n --temp=$temp --top_p=$p --syco=1
python3 -m src.predict --model=$model --data='casino' --n_data=$n --temp=$temp --top_p=$p --binary=1 --syco=1
python3 -m src.predict --model=$model --data='donations' --n_data=$n --temp=$temp --top_p=$p --syco=1
python3 -m src.predict --model=$model --data='donations' --n_data=$n --temp=$temp --top_p=$p --binary=1 --syco=1
python3 -m src.predict --model=$model --data='awry' --n_data=$n --temp=$temp --top_p=$p --syco=1
python3 -m src.predict --model=$model --data='awry' --n_data=$n --temp=$temp --top_p=$p --binary=1 --syco=1
python3 -m src.predict --model=$model --data='change' --n_data=$n --temp=$temp --top_p=$p --syco=1
python3 -m src.predict --model=$model --data='change' --n_data=$n --temp=$temp --top_p=$p --binary=1 --syco=1
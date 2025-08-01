n=200
temp=0.7
p=1.0

model='together/mistralai/Mistral-7B-Instruct-v0.3'
python3 -m src.predict --model=$model --data='casino' --n_data=$n --temp=$temp --top_p=$p --demographics=1
python3 -m src.predict --model=$model --data='casino' --n_data=$n --temp=$temp --top_p=$p --binary=1 --demographics=1
python3 -m src.predict --model=$model --data='donations' --n_data=$n --temp=$temp --top_p=$p --demographics=1
python3 -m src.predict --model=$model --data='donations' --n_data=$n --temp=$temp --top_p=$p --binary=1 --demographics=1
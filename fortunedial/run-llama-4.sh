model='together/meta-llama/Llama-3-8b-chat-hf'
n=200
temp=0.6
p=0.9

python3 -m src.predict --model=$model --data='casino' --n_data=$n --temp=$temp --top_p=$p --perspect=1
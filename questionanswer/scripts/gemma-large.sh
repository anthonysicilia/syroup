temp=0.7
p=1.0

# model='together/mistralai/Mistral-7B-Instruct-v0.3'
# model='together/mistralai/Mixtral-8x22B-Instruct-v0.1'
# model='together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
# model='together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
# model='together/google/gemma-2-9b-it'
model='together/google/gemma-2-27b-it'
python3 -m src.predict --model=$model --temp=$temp --top_p=$p --data='mmlu' --task='recover'
python3 -m src.predict --model=$model --temp=$temp --top_p=$p --data='bbh' --task='recover'
python3 -m src.predict --model=$model --temp=$temp --top_p=$p --data='rainbow' --task='recover'
python3 -m src.predict --model=$model --temp=$temp --top_p=$p --data='bbh' --task='emb'
python3 -m src.predict --model=$model --temp=$temp --top_p=$p --data='mmlu' --task='emb'
python3 -m src.predict --model=$model --temp=$temp --top_p=$p --data='rainbow' --task='emb'
python3 -m src.predict --model=$model --temp=$temp --top_p=$p --data='bbh' --task='pt'
python3 -m src.predict --model=$model --temp=$temp --top_p=$p --data='mmlu' --task='pt'
python3 -m src.predict --model=$model --temp=$temp --top_p=$p --data='rainbow' --task='pt'


import json

from src.datasets.convokit import ConvoKitFormatter

class CasinoFormatter(ConvoKitFormatter):

    def __init__(self):
        super().__init__('casino-corpus', partial=True)

    def drop_label(self, utts):
        kwords = ['Submit-Deal', 'Walk-Away']
        for i, utt in enumerate(utts):
            if any(k in utt.text for k in kwords):
                return utts[:i]
    
    def format_utt(self, utt):
        return utt.strip()
    
    def format_output(self, convo, order):
        unhappy = False
        for v in convo.retrieve_meta('participant_info').values():
            # {'Extremely dissatisfied', 'Slightly dissatisfied', 'Slightly satisfied', 'Undecided', 'Extremely satisfied'}
            outcome = v['outcomes']['satisfaction']
            unhappy = unhappy or 'dissatisfied' in outcome or 'Undecided' in outcome
        return 1 if not unhappy else 0
    
    def get_internal_speaker_id(self, utt):
        return utt.meta['speaker_internal_id']
    
    def demographics(self, chronology, norm):
        return {
            f'Speaker {norm.index(u.speaker.id)}' : u.speaker.meta['demographics']
            for u in chronology
        }
    
    def context(self):
        return 'the speakers are negotiating how to allocate available resources among themselves'
    
    def decision(self):
        return 'both speakers be satisfied at the end of the conversation'

if __name__ == '__main__':
    data = CasinoFormatter().formatted_instances
    with open('data/casino.jsonl', 'w') as out:
        for x in data:
            out.write(json.dumps(x) + '\n')
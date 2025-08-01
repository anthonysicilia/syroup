import json

from src.datasets.convokit import ConvoKitFormatter

class DonationsFormatter(ConvoKitFormatter):

    def __init__(self):
        super().__init__("persuasionforgood-corpus", partial=True)

    def drop_label(self, utts):
        return utts
    
    def format_utt(self, utt):
        return utt.strip()
    
    def format_output(self, convo, order):
        amount = convo.retrieve_meta('donation_ee')
        return 1 if amount > 0.0 else 0
    
    def get_internal_speaker_id(self, utt):
        return utt.meta['role']
    
    def demographics(self, chronology, norm):
        return {
            f'Speaker {norm.index(u.speaker.id)}' : {
                k : v
                for k,v in u.speaker.meta.items() 
                if k in ['age', 'sex', 'race', 'edu', 'marital', 'employment', 'religion', 'ideology']}
            for u in chronology
        }
    
    def context(self):
        return 'one speaker is trying to persuade the other to donate to a charitable cause'
    
    def decision(self):
        return 'a donation occur at the end of the conversation'

if __name__ == '__main__':
    data = DonationsFormatter().formatted_instances
    with open('data/donations.jsonl', 'w') as out:
        for x in data:
            out.write(json.dumps(x) + '\n')

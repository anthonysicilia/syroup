import json

from src.datasets.convokit import ConvoKitFormatter

class ChangeFormatter(ConvoKitFormatter):

    def __init__(self):
        super().__init__('conversations-gone-awry-cmv-corpus', partial=True)

    def drop_label(self, utts):
        return utts[:-1]
    
    def format_utt(self, utt):
        return utt.strip()
    
    def format_output(self, convo, order):
        return 1 if convo.retrieve_meta('has_removed_comment') \
            else 0

    def demographics(self, *args):
        return None
    
    def context(self):
        return 'the speakers are defending their opinions on an issue'
    
    def decision(self):
        return 'a personal attack occur at the end of the conversation'
    
if __name__ == '__main__':
    data = ChangeFormatter().formatted_instances
    with open('data/change.jsonl', 'w') as out:
        for x in data:
            out.write(json.dumps(x) + '\n')
import json

from src.datasets.convokit import ConvoKitFormatter

class AwryFormatter(ConvoKitFormatter):

    def __init__(self):
        super().__init__('conversations-gone-awry-corpus', partial=True)

    def drop_label(self, utts):
        return utts[:-1]
    
    def format_utt(self, utt):
        return utt.strip()
    
    def format_output(self, convo, order):
        return 1 if convo.retrieve_meta('conversation_has_personal_attack') \
            else 0
    
    def demographics(self, *args):
        return None
    
    def context(self):
        return 'a group of Wikipedia contributors are deciding whether to retain the revisions made to an article'
    
    def decision(self):
        return 'a personal attack occur at the end of the conversation'
    
if __name__ == '__main__':
    data = AwryFormatter().formatted_instances
    with open('data/awry.jsonl', 'w') as out:
        for x in data:
            out.write(json.dumps(x) + '\n')

import hashlib
import random
import re

from convokit import Corpus, download

def natsort(arr):
    """
    lt weight implementation of natsort
    https://github.com/SethMMorton/natsort/wiki/How-Does-Natsort-Work%3F
    """

    def int_map(x):
        try:
            return int(x)
        except:
            return x
    
    sortable = []

    for elem in arr:
        sortable.append(list(map(int_map, re.split(r'(\d+)', elem))))
    
    arr = list(zip(arr, sortable))
    
    return [x[0] for x in sorted(arr, key=lambda y: y[1])]

class ConvoKitFormatter:

    def __init__(self, name, partial=False):

        corpus = Corpus(filename=download(name))
        formatted_instances = list()

        random.seed(0)
        for convo in corpus.iter_conversations():

            utts = []
            chronology = self.get_chronology(convo)
            speakers = list()
            for utt in self.drop_label(chronology):
                speakers.append(utt.speaker.id)
                utts.append(self.format_utt(utt.text))
            norm = list(sorted(set(speakers))) # speaker order is important for hashing
            utts = [f'Speaker {norm.index(speaker)}: {utt}'
                for utt, speaker in zip(utts, speakers)]
            
            # drop end of conversation already to simplify. 
            # different from deal or no deal data, no training needed)
            if partial:

                if len(utts) <= 3:
                    continue
                
                k = random.choice(range(2, len(utts)+1))
                inputs = '\n'.join(utts[:k])
            else:
                inputs = '\n'.join(utts)

            order = self.get_internal_speaker_order(convo, speakers, norm)
            output = self.format_output(convo, order)
            formatted_instances.append({
                'input': inputs, 
                'output': output,
                'context': self.context(),
                'question': self.decision(),
                'demographics': self.demographics(chronology, norm),
                'instance_id': hashlib.md5(inputs.encode('utf-8')).hexdigest()})
            
        random.seed(0)
        random.shuffle(formatted_instances)
        self.formatted_instances = formatted_instances
    
    def demographics(self, chronology, norm):
        raise NotImplementedError('Abstract class has unimplemented method: demographics')

    def context(self):
        raise NotImplementedError('Abstract class has unimplemented method: context')
    
    def decision(self):
        raise NotImplementedError('Abstract class has unimplemented method: decision')
    
    def drop_label(self, utts):
        raise NotImplementedError('Abstract class has unimplemented method: drop_label')

    def format_utt(self, utt):
        raise NotImplementedError('Abstract class has unimplemented method: format_utt')
    
    def format_output(self, convo):
        raise NotImplementedError('Abstract class has unimplemented method: format_output')
    
    def get_internal_speaker_id(self, utt):
        raise NotImplementedError('Abstract class has unimplemented method: get_internal_speaker_id')
    
    def get_internal_speaker_order(self, convo, speakers, norm):
        normed_speakers = [norm.index(speaker) for speaker in speakers]
        interal_speakers = []
        for utt in self.drop_label(self.get_chronology(convo)):
            try:
                isid = self.get_internal_speaker_id(utt)
            except NotImplementedError:
                return None
            interal_speakers.append(isid)
        binding = {norm : internal for norm, internal in 
            zip(normed_speakers, interal_speakers)}
        return [binding[k] for k in sorted(binding.keys())]
    
    def get_custom_chronology(self, convo):
        # bad convo timestamps, use id to sort
        ids = convo.get_utterance_ids()
        chronology = []
        for i in natsort(ids):
            chronology.append(convo.get_utterance(i))
        return chronology
    
    def get_chronology(self, convo):
        try:
            return convo.get_chronological_utterance_list()
        except ValueError:
            return self.get_custom_chronology(convo)

from abc import ABC, abstractmethod
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

'''
OCTIS code: https://github.com/MIND-Lab/OCTIS/tree/master
'''

em_topic_diversity = r"""
@article{DBLP:journals/corr/abs-1907-04907,
  author    = {Adji B. Dieng and
               Francisco J. R. Ruiz and
               David M. Blei},
  title     = {Topic Modeling in Embedding Spaces},
  journal   = {CoRR},
  volume    = {abs/1907.04907},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.04907},
  archivePrefix = {arXiv},
  eprint    = {1907.04907},
  timestamp = {Wed, 17 Jul 2019 10:27:36 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1907-04907.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

em_coherence = r"""
@inproceedings{DBLP:conf/wsdm/RoderBH15,
  author    = {Michael R{\"{o}}der and
               Andreas Both and
               Alexander Hinneburg},
  editor    = {Xueqi Cheng and
               Hang Li and
               Evgeniy Gabrilovich and
               Jie Tang},
  title     = {Exploring the Space of Topic Coherence Measures},
  booktitle = {Proceedings of the Eighth {ACM} International Conference on Web Search
               and Data Mining, {WSDM} 2015, Shanghai, China, February 2-6, 2015},
  pages     = {399--408},
  publisher = {{ACM}},
  year      = {2015},
  url       = {https://doi.org/10.1145/2684822.2685324},
  doi       = {10.1145/2684822.2685324},
  timestamp = {Tue, 21 May 2019 11:38:33 +0200},
  biburl    = {https://dblp.org/rec/conf/wsdm/RoderBH15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

class AbstractMetric(ABC):
    """
    Class structure of a generic metric implementation
    """

    def __init__(self):
        """
        init metric
        """
        pass

    @abstractmethod
    def score(self, model_output):
        """
        Retrieves the score of the metric

        :param model_output: output of a topic model in the form of a dictionary. See model for details on
        the model output
        :type model_output: dict
        """
        pass

    def get_params(self):
        return [att for att in dir(self) if not att.startswith("_") and att != 'info' and att != 'score' and
                att != 'get_params']
    
class TopicDiversity(AbstractMetric):
    def __init__(self, topk=10):
        """
        Initialize metric

        Parameters
        ----------
        topk: top k words on which the topic diversity will be computed
        """
        AbstractMetric.__init__(self)
        self.topk = topk

    def info(self):
        return {
            "citation": em_topic_diversity,
            "name": "Topic diversity"
        }

    def score(self, model_output):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        td : score
        """
        topics = model_output["topics"]
        if topics is None:
            return 0
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than ' + str(self.topk))
        else:
            unique_words = set()
            for topic in topics:
                unique_words = unique_words.union(set(topic[:self.topk]))
            td = len(unique_words) / (self.topk * len(topics))
            return td
        

class Coherence(AbstractMetric):
    def __init__(self, texts=None, topk=10, processes=1, measure='c_npmi'):
        """
        Initialize metric

        Parameters
        ----------
        texts : list of documents (list of lists of strings)
        topk : how many most likely words to consider in
        the evaluation
        measure : (default 'c_npmi') measure to use.
        processes: number of processes
        other measures: 'u_mass', 'c_v', 'c_uci', 'c_npmi'
        """
        super().__init__()
        if texts is None:
            raise NotImplementedError('_load_default_texts not implemented')
            #self._texts = _load_default_texts()
        else:
            self._texts = texts
        self._dictionary = Dictionary(self._texts)
        self.topk = topk
        self.processes = processes
        self.measure = measure

    def info(self):
        return {
            "citation": em_coherence,
            "name": "Coherence"
        }

    def score(self, model_output):
        """
        Retrieve the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        score : coherence score
        """
        topics = model_output["topics"]
        if topics is None:
            return -1
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            npmi = CoherenceModel(
                topics=topics,
                texts=self._texts,
                dictionary=self._dictionary,
                coherence=self.measure,
                processes=self.processes,
                topn=self.topk)
            return npmi.get_coherence()
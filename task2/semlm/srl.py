from pathlib import Path
from shutil import copyfile, copytree
import os

PATHLSTM_ROOT = '/cluster/home/psarlin/NLP-Adventures/task2/semlm/PathLSTM'


def Srl(annotation, **config):
    if annotation == 'propbank':
        return PropbankSrl(**config)
    if annotation == 'framenet':
        return FramenetSrl(**config)
    else:
        raise ValueError


class PropbankSrl:
    def __init__(self):
        from ccg_nlpy import local_pipeline
        self.pipeline = local_pipeline.LocalPipeline()

    def parse(self, text):
        assert isinstance(text, str)
        doc = self.pipeline.doc(text)
        srl = doc.get_srl_verb
        predicates = srl.get_predicates()

        index_predicate_token = []
        for p in predicates:
            index_predicate_token.append((
                p['start'],
                self._predicate_properties_to_name(p['properties']),
                p['tokens']))

        return index_predicate_token

    def _predicate_properties_to_name(self, p):
        return p['predicate'] + '.' + p['SenseNumber']


class FramenetSrl:
    corenlp_root = 'lib/stanford-corenlp-full-2018-02-27'
    corenlp_libs = ['stanford-corenlp-3.9.1.jar', 'stanford-corenlp-3.9.1-models.jar']
    other_java_libs = ['lib/anna-3.3.jar', 'lib/pathlstm.jar']

    srl_model_path = 'models/srl-ICCG16-stanford-eng.model'
    framenet_path = 'framenet/'

    def __init__(self, copy_locally=False):
        java_libs = [Path(self.corenlp_root, l) for l in self.corenlp_libs]
        java_libs += [Path(l) for l in self.other_java_libs]
        java_libs = [Path(PATHLSTM_ROOT, l).as_posix() for l in java_libs]

        import jnius_config
        jnius_config.add_options('-Xmx60g')
        jnius_config.add_classpath(*java_libs)

        from jnius import autoclass
        self.JString = autoclass('java.lang.String')
        self.PipelineOptions = autoclass(
                'se.lth.cs.srl.options.CompletePipelineCMDLineOptions')
        self.CompletePipeline = autoclass('se.lth.cs.srl.CompletePipeline')
        self.Predicate = autoclass('se.lth.cs.srl.corpus.Predicate')

        model_path = Path(PATHLSTM_ROOT, self.srl_model_path)
        framenet_path = Path(PATHLSTM_ROOT, self.framenet_path)
        # if copy_locally:
            # print('Making local copies of model and framenet data')
            # model_local_path = Path(os.environ['TMPDIR'], 'model')
            # copyfile(model_path.as_posix(), model_local_path.as_posix())
            # model_path = model_local_path
            # framenet_local_path = Path(os.environ['TMPDIR'], 'framenet')
            # copytree(framenet_path.as_posix(), framenet_local_path.as_posix())
            # framenet_path = framenet_local_path

        args = 'fnet -reranker -externalNNs -globalFeats -tokenize -stanford' \
               ' -srl {srl} -framenet {fn}'.format(
                       srl=model_path.as_posix(),
                       fn=framenet_path.as_posix())
        options = self.PipelineOptions()
        options.parseCmdLineArgs(args.split())
        self.pipeline = self.CompletePipeline.getCompletePipeline(options)

    def parse(self, text):
        assert isinstance(text, str)
        prediction = self.pipeline.parse(self.JString(text))

        index_predicate_token = []
        index_pos_token = []
        for i in range(prediction.size()):
            w = prediction.get(i)
            index_pos_token.append((i, w.getPOS(), w.getForm()))
            if isinstance(w, self.Predicate):
                index_predicate_token.append((i, w.getSense(), w.getForm()))

        return index_predicate_token, index_pos_token

install_gpu:
	pip3 install -r full_requirements.txt
	pip3 install tensorflow-gpu==1.6
	python3 -m spacy download en
	python3 -m spacy download en_core_web_lg
	wget -P task2/data/ https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz

light_install:
	pip3 install -r light_requirements.txt

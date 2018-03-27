install_gpu:
	pip3 install -r requirements.txt
	pip3 install tensorflow-gpu==1.6

install_cpu:
	pip3 install -r requirements.txt
	pip3 install tensorflow==1.6

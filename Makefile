init:
	conda env create -f sorting_analysis_env.yaml
	conda activate sorting_analysis
	jupyter labextension install @jupyter-widgets/jupyterlab-manager
	jupyter labextension install jupyter-matplotlib

test:
	pytest tests
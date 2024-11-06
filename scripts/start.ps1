docker run -it -p 8888:8888 --rm `
-v "${PWD}":/home/jovyan/work `
quay.io/jupyter/tensorflow-notebook:2024-10-23 `
start-notebook.py --ip='*' --NotebookApp.token='' --NotebookApp.password=''
required modules and libraries:

- sys
- pandas
- sklearn
- matplotlib.pyplot
- seaborn
- random
- numpy

language: python 3.x

dataset :
you will need dataset to be in same directory. you can download dataset from this url : https://archive.ics.uci.edu/ml/datasets/abalone
since dataset size is small, i included in my submission, so you don't need to downloaded. just included in same directory as main.py file

how to run:

python3 main.py  all  or python3 main.py    => to run comparison between different models
python3 main.py  dt                         => to run decision tree experiment with different hyper parameters
python3 main.py  nn                         => to run neural network experiment with different hyper parameters
python3 main.py  bdt                        => to run boosted decision tree experiment with different hyper parameters
python3 main.py  svm                        => to run support vector machine experiment with different hyper parameters
python3 main.py  knn                        => to run K Nearest Neighbors experiment with different hyper parameters
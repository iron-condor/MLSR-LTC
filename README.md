# Machine Learning Symbolic Regression Lattice Thermal Conductivity

#### [Christian Loftis](https://github.com/iron-condor), Kunpeng Yuan, Yong Zhao, Ming Hu, [Jianjun Hu](https://github.com/usccolumbia)
#### Department of Computer Science & Engineering, University of South Carolina

This repository contains code used to replicate the experiments described in [this paper.](https://pubs.acs.org/doi/10.1021/acs.jpca.0c08103)

If you like our paper, please cite:

```Loftis, Christian, Kunpeng Yuan, Yong Zhao, Ming Hu, and Jianjun Hu. "Lattice Thermal Conductivity Prediction Using Symbolic Regression and Machine Learning." The Journal of Physical Chemistry A (2020).```

## Getting Started
To get started, clone the repository to an empty folder, and enter it

```bash
git clone https://github.com/iron-condor/MLSR-LTC/
cd MLSR-LTC
```

The four subfolders contained inside each represent an experiment (GP1, GP2, mlp, rfr). Choose one, and enter it.

```bash
cd GP1
```

If you've chosen one of the Symbolic Regression experiments (either GP1 or GP2), there is an additional step. Please skip this step if you are running the mlp or rfr experiments.
```bash
source bin/activate
```

Choose a cross validation method to use. `generate_hybrid.py` generates models using the hybrid cross validation approach described in [the paper](https://arxiv.org/abs/2008.03670), while `generate_kfold.py` uses a 10-fold cross validation approach.

If you choose to run the hybrid cross validation experiment, be sure to choose a testing set. Testing sets can be listed with `python generate_hybrid.py -h` or `python generate_hybrid.py --help`.

```bash
python generate_kfold.py
#Or, alternatively
python generate_hybrid.py --middle-testing
```

If you've done everything right, you should see a short message in the terminal with the text "Running..."
After a few moments (perhaps a bit longer depending on your dataset), you will see intermittent updates that reflect the model's performance on the fold of the dataset it is being evaluated upon. If you are using a hybrid experiment, you will get 5 of these updates. If you are using the kfold experiment, you will see 10 of these.

After the model has finished training, you will see the model's final evaluation scores, along with a brief comparison to the Slack formula's performance on the same set. Then, the program will exit.

To view your results in greater depth, you can run the accompanying evaluation program. If you ran the hybrid experiment before, run the following code (and fill in the correct testing set).

```bash
python eval_hybrid.py --middle-testing -i
```

If you ran the kfold experiment before, run the following code instead.

```bash
python eval_kfold.py -i
```

You should see a list of models, indexed by the date in which they were created. There are three columns. The first indicates the model's index. The second shows the model's mean absolute error, and the third shows the model's R<sup>2</sup> score. By default, these metrics indicate the model's performance over the entire set, and not just over the testing set -- though this can be changed by specifying the -t flag.

We recommend you take a look at the help menu for the evaluation program, which can be accessed as follows

```bash
python eval_hybrid.py -h
#Or, alternatively
python eval_kfold.py -h
```

Once the list finishes loading, the program will tell you how the Slack model performed on the same set. At this stage, you can enter the index of the model that you want to view. This will bring up a parity plot demonstrating the model's performance. You can also type "slack" to view the Slack model's performance in a similar parity plot, or "quit" to exit the program.

## Acknowledgements

This research was supported, in part, by a grant from the Magellan Scholar program, from the Department of Undergraduate Research at the University of South Carolina, Columbia. Research reported in this publication was also partially supported by the National Science Foundation under grant numbers: 1940099, 1905775, OIA-1655740 (via SC EPSCoR/IDeA 20-SA05) and by DOE under grant number DE-SC0020272.

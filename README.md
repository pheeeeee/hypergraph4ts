# Hyper-Graphical Models for Time-Series Analysis

This repository is the official implementation of [Hyper-Graphical Models for Time-Series Analysis]. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...


## Training

To run the model(s) in the paper, run this command:

```train
python train.py --data_name 'data_name' --node None --degree None --percentage 1 --print_computing_time True --print_inference_time True --draw_plot True --l 1440
```

Note. This only estimates <u>one</u> target time sereies. 

>📋 data_name should be one of the following strings: 'caiso', 'traffic', 'electricity', 'weather', 'etth1', 'ettm1', 'solar', 'wind', 'exchange'. The preprocessed data we used for experiments will be used for training. Details about the preprocessing procedure are provided in Section 7 and Appendix A.3.

>📋 node and degree are hyperparameters of our model. In our actual code, the node can be set as a list of bin thresholds according to the user's specifications. However, here the node parameter must be either an integer or None. If the hyperparameter node and degree are chosen to be None, it automatically uses the node and degrees we used for our experiment in the paper.

>📋 percentage : The percentage of used data for training from available time windows. (If zero, it is zero-shot learning(Only use the input lookback window to generate the target.))

>📋 print_computint_time :True if you want to print training time for each time window

>📋 print_inference_time : True if you want to print inference time

>📋 l : The lookback window size for the inference.

>📋 draw_plot : If True, the plot like the below is saved as 'plot.png'

<a href="url"><img src="/assets/prediction_target_plot.jpg" align="center" height="600" width="800" style="float:left; padding-right:15px" ></a>




## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).


## Experiments with fewer data



## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 

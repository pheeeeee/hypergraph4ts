import argparse
import os
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import pickle
from codes.counting import *
from codes.application import *
from codes.utils import *
from codes.metric import *
from codes.dataloader import *



def train_model(data, node=None, degree=None, print_computing_time=True):
    print(f"Starting training with:")
    print(f"Input data name: {data}")
    
    obj = application(data, bin=node, n_tail=degree, last=degree, parallel=False,timer=print_computing_time,print_computing_time=print_computing_time)
    print("Training (Computing) Done.")
    
    return obj
    
    """if to_graph is "hypergraph":
        G = obj.G
        graph = G.tograph(type = 'hypergraph')"""


def visualize(lookback, target, prediction, save_plot=None):
    lookback_window_range = np.arange(len(lookback))
    target_range = np.arange(len(lookback), len(lookback) + len(target))
    pred_range = np.arange(len(lookback), len(lookback) + len(prediction))

    plt.figure(figsize=(12, 6))
    plt.plot(lookback_window_range, lookback, label='Lookback Window')
    plt.plot(target_range, target, label='Ground Truth',  color='blue')
    plt.plot(pred_range, prediction, label='Predictions', color='red')
    # Shading the prediction area
    plt.axvspan(target_range[0], target_range[-1], color='lightgray', alpha=0.5, label='Prediction Area')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title('Lookback Window, Ground Truth, and Predictions with Shaded Background')
    plt.legend()
    plt.grid(True)
    if save_plot is not None:
        plt.savefig(save_plot)
    

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument('--data_name', type=str, required=True, help="Name of Data.")
    parser.add_argument('--node', type=float, required=True, help="The number of nodes for partion.")
    parser.add_argument('--degree', type=float, required=True, help="The degree of the model.")
    parser.add_argument('--percentage', type=float, required=True, help='The percentage of used data for training from available time windows. (If zero, it is zero-shot learning(Only use the input lookback window to generate the target.))')
    parser.add_argument('--print_computing_time', type=bool, required=True, help="True if you want to print training time for each time window")
    parser.add_argument('--print_inference_time', type=bool, required=True, help="True if you want to print inference time")
    parser.add_argument('--draw_plot', required=True, help='True if you want to visualize the plot of target data and generated data.')
    parser.add_argument('--l', type=int, required=True, help="Size of Lookback window to use. Choose among 1440, 720, 336, 192, 168, 96. (Default is 1440)" )
    
    # Parse the arguments
    args = parser.parse_args()

    print("Start loading data")
    if type(args.data_name) is string:
        assert args.data_name in ['caiso','traffic','electricity','weather','etth1','ettm1','solar','wind', 'exchange'], "Mentioned data does not exist in loaded dataset."
        with open('datasets/data.pkl', 'rb') as file:
            ts1data = pickle.load(file)
    
    train_data = ts1data[args.data_name][args.l]['train_data']
    prediction_data = ts1data[args.data_name][args.l]['test_data']
    n_windows = len(train_data)
    
    target_index = random.randint(0, n_windows)
    target = prediction_data[target_index]
    lookback_data = train_data[target_index]
    
    indices = list(range(n_windows))
    indices.remove(target_index)
    n_train = round(args.percentage * (n_windows-1))
    train_index = np.random.choice(indices, n_train)
    
    train = []
    for i in train_index:
        train.append(np.concatenate((train_data[i],prediction_data[i])))
    train.append(lookback_data)
    print('Data Loading Done.')

    # Call the training function with the provided arguments
    obj = train_model(args.input_data, node=args.node, degree=args.degree, print_computing_time=args.print_computing_time)
    lookback_window = application(lookback_data, bin=args.node, n_tail=args.degree, last=args.degree, parallel=False, timer=False)
    generated_data = obj.generate(n=len(target), initialnodes=lookback_window.lastnodes, how='uniform', inference_time=args.print_inference_time)

    loss = mean_absolute_error(generated_data, target)
    print(f"MAE of G4TS using {args.percentage *100}% of available training data {args.data_name} with lookback window size {args.l} is {loss}.")

    if args.draw_plot is True:
        visualize(lookback_data, target=target, prediction=generated_data, save_plot='plot.png')
        
    

if __name__ == "__main__":
    main()

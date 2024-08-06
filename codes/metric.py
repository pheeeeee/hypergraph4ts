

import matplotlib.pyplot as plt
from multiprocessing import Pool
from codes.counting import *
from codes.application import *
from codes.utils import *

import time

def one2one_metrictable(trainingdata, validationdata, node, degree, champ, metric, timer):
    """_summary_

    Args:
        trainingdata (seq): one data
        validationdata (seq): one data
        node (list): list of number of node
        degree (list): list of n_tails
        champ (int): How many data to generate for robustness
        metric (function): _description_

    Returns:
        dataframe: _description_
    """
    assert not is_list_of_sequences(validationdata)  ; "This is one to one"
            
    with Pool() as pool:
        args = [(trainingdata, validationdata, n, d, len(validationdata), champ, metric, timer) for d in degree for n in node]
        pooled_results = pool.map(worker, args)
        results = [result for result, _ in pooled_results]
        best_guesses = [guess for _, guess in pooled_results]
        metricframe = np.array(results).reshape(len(degree), len(node))
        best_setting = np.argmin(metricframe)
        best_guesses = best_guesses[best_setting]
        best_setting = np.unravel_index(best_setting, metricframe.shape)

    # Organize results into a structured format
    return metricframe, best_setting, best_guesses


def is_list_of_sequences(dataset):
    # Check if the dataset itself is a list
    if not isinstance(dataset, Iterable):
        return False
    
    # Check each element in the list to confirm it's a sequence
    for sequence in dataset:
        # Ensure the element is an iterable (list, tuple, etc.) but not string-like
        if not isinstance(sequence, (list, tuple, np.ndarray)) or isinstance(sequence, str):
            return False
    
    # If all elements are sequences, return True
    return True


def generate_metrics(trainingdata, validationdata, node, n_tail, length, champ, metric, timer=False):
    """
    trainingdata is ONE dataset.
    validationdata is ONE dataset corresponding to the trainingdata.
    """
    obj = application(data=trainingdata, bin = node, n_tail=n_tail, last=n_tail,  parallel=True, timer=timer)
    if timer:
        duration = obj.timer
    else:
        duration = None
    gen = [obj.generate(n=length, initialnodes=obj.lastnodes, how='average') for _ in range(champ)]
    if metric == crps_pwm:
        met = np.empty(length)
        for i in range(length):
            pred = np.array([g[i] for g in gen])
            met[i] = metric(validationdata[i], pred)
        best_guess = gen[0]
    else:
        met = [metric(validationdata, g) for g in gen]
        best_guess = gen[np.argmin(met)]
    return np.average(met), best_guess, duration


def worker(args):
    return generate_metrics(*args)


def fun_metrictable(train_data, target, nodes, degrees, champ, metric, timer=False):
    """_summary_

    Args:
        champ (int): How many data to generate for robustness
        metric (function): _description_

    Returns:
        dataframe: _description_
    """
    if is_list_of_sequences(target):
        assert len(train_data)==len(target), "The number of training dataset must equal the number of validation dataset"
        # Creating a pool of workers to handle computations
        with Pool() as pool:
            args = [(train_data[i], vali, n, d, len(vali), champ, metric,timer) for d in degrees for n in nodes for i, vali in enumerate(target)]
            pooled_results = pool.map(worker, args)
            results = [result for result, _ , __ in pooled_results]
            best_guesses = [guess for _, guess, __ in pooled_results]
            timer_record = [dura for _, __, dura in pooled_results]
            metricframe = np.array(results).reshape(len(degrees), len(nodes), len(target))
            timer_record = np.array(timer_record).reshape(len(degrees), len(nodes), len(target))
            timer_record = timer_record.sum(axis=2)
            best_setting = np.argmin(metricframe)
            best_guesses = best_guesses[best_setting]
            best_setting = np.unravel_index(best_setting, metricframe.shape)
            metricframe = np.mean(metricframe, axis=2)
            
    else:
        with Pool() as pool:
            args = [(train_data, target, n, d, len(target), champ, metric, timer) for d in degrees for n in nodes]
            pooled_results = pool.map(worker, args)
            results = [result for result, _ , __ in pooled_results]
            best_guesses = [guess for _, guess,__  in pooled_results]
            timer_record = [dura for _, __, dura in pooled_results]
            metricframe = np.array(results).reshape(len(degrees), len(nodes))
            timer_record = np.array(timer_record).reshape(len(degrees), len(nodes))
            best_setting = np.argmin(metricframe)
            best_guesses = best_guesses[best_setting]
            best_setting = np.unravel_index(best_setting, metricframe.shape)
    return metricframe, best_setting, best_guesses, timer_record


training_classification_reports = {}
validation_classification_reports = {}
testing_classification_reports = {}
training_accuracies = {}
testing_accuracies = {}
validation_accuracies = {}
training_times = {}

import time

start_time = None

def start_timer():
    global start_time
    start_time = time.time()

def stop_timer(key):
    global start_time
    if start_time is None:
        raise ValueError("Timer has not been started.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    start_time = None  # Reset the start time for future use
    training_times[key] = elapsed_time
    return elapsed_time

def save_accuracies(key,training_acc,validation_acc,testing_acc,training_clf_rep,validation_clf_rep,test_clf_rep):
    training_accuracies[key] = training_acc
    validation_accuracies[key] = validation_acc
    testing_accuracies[key] = testing_acc
    training_classification_reports[key] = training_clf_rep
    validation_classification_reports[key] = validation_clf_rep
    testing_classification_reports[key] = test_clf_rep
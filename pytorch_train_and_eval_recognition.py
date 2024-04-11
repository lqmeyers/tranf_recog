import torch
import sys
# sys.path.insert(1, '/home/lmeyers/beeid_clean_luke/PYTORCH_CODE/')
# sys.path.insert(2, '/home/lmeyers/beeid_clean_luke/KEY_CODE_FILES/')
# sys.path.insert(3,'/home/lmeyers/paintid')

#from pytorch_resnet50_conv3 import resnet50_convstage3
from pytorch_data import prepare_for_triplet_loss
from recognition_models import RecogModel
from pytorch_data import Flowerpatch_Pairs
import pickle 

import numpy as np
import pandas as pd
#from PIL import Image as Image2
import matplotlib.pyplot as plt
#import random 
import os
from tqdm import tqdm

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
#from torchvision.io import read_image
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from sklearn.metrics import confusion_matrix
import sys 
import argparse

import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb
from datetime import datetime
import yaml
import time 

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


print("finished imports")


# DEVICE INFO NEEDED TO KNOW WHERE WE WILL PERFORM COMPUTATIONS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################## Function for getting embeddings of an entire dataset ##########################

def get_embeddings(model, dataloader):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch['image'].to(device))
            labels += list(batch['label'].detach().cpu().numpy())
            embeddings.append(outputs.detach().cpu().numpy())
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    return embeddings, labels



####################################### Training Handle ###########################################

def train_and_eval(config_file):
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        model_config = config['model_settings'] # settings for model building
        train_config = config['train_settings'] # settings for model training
        data_config = config['data_settings'] # settings for data loading
        eval_config = config['eval_settings'] # settings for evaluation
        torch_seed = config['torch_seed']
        verbose = config['verbose']
    except Exception as e:
        print('ERROR - unable to open experiment config file. Terminating.')
        print('Exception msg:',e)
        return -1

    ############################# Build Datasets and Dataloaders ###########################################

    image_size = data_config['input_size']
    bs = data_config['batch_size']
    num_epochs = train_config['num_epochs']
    
    train_pos_prob = train_config['percent_pos']

    if data_config['datafiles']['valid'] != None: 
        ### Train dataset and dataloader
        train_df = pd.read_csv(data_config['datafiles']['train'])
        train_df = prepare_for_triplet_loss(train_df, data_config['label_col'], data_config['fname_col']) #deprecated function only renames columns

        train_dataset = Flowerpatch_Pairs(train_df, 'filename', 'label',image_size,'train',pos_pair_prob=train_pos_prob)
        train_dataloader =  DataLoader(train_dataset, batch_size=bs, shuffle=True)

        ### Valid dataset and dataloader
        valid_df = pd.read_csv(data_config['datafiles']['valid'])
        valid_df = prepare_for_triplet_loss(valid_df, data_config['label_col'], data_config['fname_col']) #deprecated function standardizes col names

        valid_dataset = Flowerpatch_Pairs(valid_df, 'filename', 'label',image_size,'valid',pos_pair_prob=train_pos_prob)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False)
    else:
        #if no valid dataset, sample from training set
        train_df = pd.read_csv(data_config['datafiles']['train'])

        valid_num_rows = round(data_config['percent_valid']*len(train_df))
        valid_rows = train_df.sample(n=valid_num_rows)
        
        train_df = train_df.drop(valid_rows.index)
        train_num = len(train_df)
        valid_df = valid_rows

        print(f"Using {valid_num_rows} samples for validation set")
        print(f"{train_num} total training samples")

        train_df = prepare_for_triplet_loss(train_df, data_config['label_col'], data_config['fname_col']) #deprecated function only renames columns
    
        train_dataset = Flowerpatch_Pairs(train_df, 'filename', 'label',image_size,'train',pos_pair_prob=train_pos_prob)
        train_dataloader =  DataLoader(train_dataset, batch_size=bs, shuffle=True)

        valid_df = prepare_for_triplet_loss(valid_df, data_config['label_col'], data_config['fname_col'])  #deprecated function standardizes col names

        valid_dataset = Flowerpatch_Pairs(valid_df, 'filename', 'label',image_size,'valid',pos_pair_prob=train_pos_prob)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False)

    #### Test dataset and dataloader
    test_df = pd.read_csv(data_config['datafiles']['test'])

    num_test_ids = len(np.unique(test_df[data_config['label_col']]))
    pos_prob = 1/num_test_ids #set test_dataloader to reflect structure of real world 

    test_df = prepare_for_triplet_loss(test_df, data_config['label_col'], data_config['fname_col'])

    # Build test dataset and dataloader
    test_dataset = Flowerpatch_Pairs(test_df, 'filename', 'label',image_size,'test',pos_pair_prob=pos_prob)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)
     
    resume_training = train_config['wandb_resume']
    #initialize wandb logging
    if resume_training == True: 
        experiment = wandb.init(project= train_config["wandb_project_name"],entity=train_config['wandb_entity_name'],resume=True,id=train_config['wandb_run_id'],dir=train_config['wandb_dir_path'])
    else:
        experiment = wandb.init(project= train_config["wandb_project_name"],entity=train_config['wandb_entity_name'],dir=train_config['wandb_dir_path'])
    
    if verbose:
        now = datetime.now() # current date and time
        dt = now.strftime("%y-%m-%d %H:%M")
        print(f'Date and time when this experiment was started: {dt}')
        print("Data Settings:")
        print(data_config)
        print("Train Settings:")
        print(train_config)
        print("Model Settings:")
        print(model_config)
    
    #SET GPU TO USE
    os.environ["CUDA_VISIBLE_DEVICES"]=str(train_config['gpu'])
    if verbose:
        print('Using GPU',train_config['gpu'])

    ############## Load and build models ##########################################33
    model = RecogModel() #recognition model
    
    emb_path = model_config['embeddor_path'] #pretrained embeddor
    model_name = os.path.basename(emb_path)
    embedder = torch.load(emb_path) 

    #TODO look into graph mode for pytorch, embeddor being duplicated 
    #make all params unupdateable? 
    for param in embedder.parameters():
        param.requires_grad = False

    #send to CUDA 
    model.to(device)
    embedder.to(device)

    print("Initilizing run started at ",datetime.now())
    print('Using embeddings from',model_name)

    # Define the cross-entropy loss function
    loss_fn = nn.BCEWithLogitsLoss() #More stable, 

    # Define the optimizer (e.g., SGD or Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])

    # Scheduler for reduing learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,factor=.5,min_lr=1e-5)

    # Initialize early stopping variables
    best_valid_acc = 0.0
    best_model = model
    valid_loss = 'Null'
    best_epoch_loss = 1000.0
    num_epochs_no_improvement = 0
    check_for_early_stopping = train_config['early_stopping']
    consecutive_epochs = train_config['early_stop_consecutive_epochs']
    stop_early = False


    # Train the model
    if verbose:
        print('Training model...')
    print_k = train_config['print_k']
    start = time.time()
    # Training loop for classifier
    for epoch in range(train_config["num_epochs"]):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_dataloader):
            
            #TODO probably should move embedding in model overall
            #each batch has 3 tensor arrays
            features = data['image']
            pairs = data['pair'] #TODO rename pair image to image2 
            labels = data['label'].to(device)

            with torch.no_grad(): 
                # get embeddings of img pair
                reference = embedder(features.to(device)) 
                query = embedder(pairs.to(device))
            
            #apply stop gradients, for extra security
            reference = reference.detach()
            query = query.detach()

            # Pass through model
            outputs = model(reference,query) #TODO change outputs to more explicit name, output_logit
            outputs = outputs.squeeze()

            # Calculate loss
            loss = loss_fn(outputs, labels) 

            # Backpropagate 
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            # calc avg loss
            loss_value = loss.item()
            if batch_idx % 10 == 0: 
                print("[",epoch," ",batch_idx,"loss:",loss_value,"]")
            running_loss += loss_value
        
        # Log the average loss for the epoch
        epoch_loss = running_loss / len(train_dataloader)
        wandb.log({"Loss": epoch_loss, "Learning Rate":train_config["learning_rate"]}, step=epoch)
        print('Epoch',epoch,'Loss:',epoch_loss)
        
        # Perform Validation
        if (epoch+1)%train_config["perform_validation"] == 0:
            with torch.no_grad():
                valid_preds = np.empty((0,)) 
                valid_truth = np.empty((0,))
            for valid_batch_idx, data in enumerate(valid_dataloader):
                    valid_features = data['image'] 
                    valid_pairs = data['pair']
                    valid_labels = data['label'].to(device)

                    # get embeddings of img pair
                    valid_reference = embedder(valid_features.to(device))
                    valid_query = embedder(valid_pairs.to(device))
                
                    # Pass through model
                    valid_outputs = model(valid_reference,valid_query) 
                    valid_outputs = torch.sigmoid(valid_outputs)
                    #TODO logits -> predicts labeling 
                    valid_outputs = valid_outputs.squeeze()
                

                    # Calculate the predicition loss 
                    valid_loss = loss_fn(valid_outputs, valid_labels)

                    #logits is the class probabilities`
                    labels_add = valid_labels.detach().cpu().numpy()
                    valid_truth = np.concatenate((valid_truth,labels_add)) #may not be super efficient, due to continuous array space allocation (look into), concatenating all at once? 
                    
                    preds_add = valid_outputs.detach().cpu().numpy()
                    
                    #filter against .5, change to 0 if remove sigmoid 
                    preds_add = (preds_add>=0.5).astype(int)
                    
                    valid_preds = np.concatenate((valid_preds,preds_add))               

            # Get the predicted labels (indices with maximum probability)
            valid_accuracy = accuracy_score(valid_truth, valid_preds)
            print(f'Validation accuracy {epoch+1}/{num_epochs}, Acc: {valid_accuracy}')
            wandb.log({"Validation accuracy":valid_accuracy},step=epoch)

            if train_config['early_stopping_metric'] == "accuracy":
                # Check if validation accuracy has improved
                if valid_accuracy > best_valid_acc:
                    best_valid_acc = valid_accuracy
                    best_model = model
                    num_epochs_no_improvement = 0
                    num_epoch_saved = epoch
                else:
                    num_epochs_no_improvement += train_config["perform_validation"]
                
            # Check if early stopping condition is met
            if check_for_early_stopping:
                if num_epochs_no_improvement >= consecutive_epochs:
                    print(f'Early stopping at epoch {epoch+1} due to no improvement in validation loss for {consecutive_epochs} consecutive epochs')
                    stop_epoch = epoch+1
                    stop_early = True 

        # Adjust learning rate based on validation accuracy
        #scheduler.step(accuracy)
                    
        if train_config['early_stopping_metric'] == 'loss':
            if epoch_loss < best_epoch_loss: #Uses loss for early stopping instead of valid accuracy 
                best_epoch_loss = epoch_loss
                best_model = model
                num_epochs_no_improvement = 0
                num_epoch_saved = epoch
            else:
                num_epochs_no_improvement += 1

        #Breaks Epoch iteration to stop training early
        # will only be true if checking for early stopping is enabled                     
        if stop_early == True:
            break

    #---- Perform eval with best model---------
    model = best_model
    stop_epoch = epoch+1
    stop = time.time()
    duration = (stop-start)/60
    print(f'Total train time: {duration}min')
    print('Using model from epoch ',num_epoch_saved)

    #---- perform evaluation

    model.eval()

    print(f'Evaluating on Test Set')
    print(f'Positive pair probability = 1/{num_test_ids}')
    print(round(pos_prob*len(test_df)),'True Positives |',round(len(test_df)-(pos_prob*len(test_df))),'True Negatives')
    
        # Forward pass through the classifier
    with torch.no_grad():
        test_binary_preds = np.empty((0,))
        test_logits_preds = np.empty((0,))
        test_truth = np.empty((0,))
        for test_batch_idx, data in enumerate(test_dataloader):
            test_features = data['image']
            test_pairs = data['pair']
            test_labels = data['label'].to(device)

            # get embeddings of img pair
            test_reference = embedder(test_features.to(device))
            test_query = embedder(test_pairs.to(device))
        
            # Pass through model
            test_outputs = model(test_reference,test_query) #calling model vs calling forward (check understanding)
            test_logits = test_outputs.squeeze()

            test_outputs = torch.sigmoid(test_outputs)
            test_outputs = test_outputs.squeeze() #
            
            # Calculate the predicition loss
            test_loss = loss_fn(test_outputs, test_labels)
        
            #____ test outputs (i forgot the word)
            test_outputs = test_outputs.detach().cpu().numpy()
            test_logits = test_logits.detach().cpu().numpy()
            test_outputs = (test_outputs>=0.5).astype(int)

            test_truth = np.concatenate((test_truth, test_labels.detach().cpu().numpy()))
            test_binary_preds = np.concatenate((test_binary_preds,test_outputs)) #TODO fix this one too to prevent truncation
            test_logits_preds = np.concatenate((test_logits_preds,test_logits))

    # Convert to PyTorch tensors (if not already)
    conf = confusion_matrix(test_truth,test_binary_preds)

    # Calculate eval metrics 
    accuracy = accuracy_score(test_truth,test_binary_preds)
    precision = precision_score(test_truth,test_binary_preds)
    recall = recall_score(test_truth,test_binary_preds)
        
    # print(f"Accuracy: {accuracy:.2f}")
    # print(f"Precision: {precision:.2f}")
    # print(f"Recall: {recall:.2f}")

    # Calculate roc_curve and auc 
    fpr, tpr, _ = roc_curve(test_truth,test_logits_preds)
    roc_auc = auc(fpr, tpr)
    
    #------ Code for plotting ROC curve with FP and TP
    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc="lower right")
    # curve_filepath = wandb_dir+"Roc_curve.png"
    # plt.savefig(curve_filepath,)
    
        
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    results = {}
    
    # Add total training loss to results 
    results['train_num'] = train_num
    results['train_loss'] = epoch_loss
    results['train_positive_prob'] = train_pos_prob
    results['eval_positive_prob'] = pos_prob

 
    # Adding other metrics to results to pass to csv
    results['valid_accuracy'] = valid_accuracy
    results['wandb_id'] = experiment.id
    print(experiment.id)
    results['start_time'] = experiment.start_time
    results['train_time'] = duration
    results['stop_epoch'] = stop_epoch


    results['accuracy'] = accuracy
    results['precision'] = precision
    results['recall'] = recall
    results['roc_auc'] = roc_auc

    with open(eval_config['pickle_file'],'wb') as fi:
        pickle.dump(results,fi)
        print("Results saved to pickle file")

    if model_config['model_path'] is not None:
        print('Saving model...')
        torch.save(model, model_config['model_path'])
    else:
        print('model_path not provided. Not saving model')
    print('Finished')

    #finish wandb run 
    wandb.finish()

    # Save the pre-trained classifier for later use with the generator
    torch.save(model.state_dict(),model_config["model_path"])

print("beginning execution")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="yaml file with experiment settings", type=str)
    args = parser.parse_args()
    train_and_eval(args.config_file)

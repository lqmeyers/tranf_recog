import numpy as np
import pandas as pd
from PIL import Image as Image2
import matplotlib.pyplot as plt
import time
import argparse
import yaml
import pickle
from datetime import datetime
import gc
import os
import wandb
import pickle
import sys 
sys.path.insert(0,"../")
# sys.path.insert(1, '/home/lmeyers/beeid_clean_luke/PYTORCH_CODE/')
# sys.path.insert(2, '/home/lmeyers/beeid_clean_luke/KEY_CODE_FILES/')
# sys.path.insert(3,'/home/lmeyers/paintid')

# from pytorch_resnet50_conv3 import resnet50_convstage3
# from data import prepare_for_triplet_loss


import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.io import read_image
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from sklearn.metrics import confusion_matrix
import sys 

import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


import torch
from transformers import ViTFeatureExtractor
from pytorch_data import *
from recognition_models import *
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


print("finished imports")

##########################################################################################
# FUNCTION TO GET EMBEDDINGS AND LABELS FOR EVALUATING MODEL
def get_embeddings(model, dataloader, loss_fn, miner, device, feature_extractor=None):
    embeddings = []
    all_labels = []
    loss = 0.0
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            if feature_extractor is None:
                images = batch['image'].to(device)
            else:
                images = [transforms.functional.to_pil_image(x) for x in batch['image']]
                images = np.concatenate([feature_extractor(x)['pixel_values'] for x in images])
                images = torch.tensor(images, dtype=torch.float).to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            hard_pairs = miner(outputs, labels)
            loss += loss_fn(outputs, labels, hard_pairs).detach().cpu().numpy()
            embeddings.append(outputs.detach().cpu().numpy())
            all_labels += list(labels.detach().cpu().numpy())
    embeddings = np.vstack(embeddings)
    all_labels = np.array(all_labels)
    loss/=k
    return embeddings, all_labels, loss
##########################################################################################


########################## Function for getting embeddings of an entire dataset ##########
def get_embeddings_w_track(model, dataloader,device):
    model.eval()
    embeddings = []
    labels = []
    tracks = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch['image'].to(device))
            labels += list(batch['label'].detach().cpu().numpy())
            tracks += list(batch['track'].detach().cpu().numpy())
            embeddings.append(outputs) #keep in tensor format #.detach().cpu().numpy())
    embeddings = torch.vstack(embeddings) #variable stack? 
    labels = np.array(labels)
    tracks = np.array(tracks)
    return embeddings, labels, tracks
########################################################################################

##########################################################################################
# FUNCTION TO AGGREGATE TEST EMBEDDINGS AND LABELS FOR EVALUATING MODEL
def get_loss(model, dataloader, loss_fn, miner, device, feature_extractor=None):
    embeddings = []
    all_labels = []
    loss = 0.0
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            tracks = batch['track_embeddings']
            labels = batch['id'].to(device)
            outputs = model(tracks)
            hard_pairs = miner(outputs, labels)
            loss += loss_fn(outputs, labels, hard_pairs).detach().cpu().numpy()
            embeddings.append(outputs.detach().cpu().numpy())
            all_labels += list(labels.detach().cpu().numpy())
    embeddings = np.vstack(embeddings)
    all_labels = np.array(all_labels)
    loss/=k
    return embeddings, all_labels, loss
##########################################################################################

#########################################################################################
# FUNCTION TO PERFORM KNN EVALUATION
#
def knn_evaluation(train_images, train_labels, test_images, test_labels, n_neighbors, per_class=True, conf_matrix=True):
    # BUILD KNN MODEL AND PREDICT
    results = {}
    print(f"Training kNN classifier with k=1")
    my_knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    my_knn.fit(train_images, train_labels)
    knn_pred = my_knn.predict(test_images)
    knn_acc = np.round(np.sum([1 for pred, label in zip(knn_pred, test_labels) if pred == label])/test_labels.shape[0],4)
    print(f"1NN test accuracy: {knn_acc}")
    # store results
    results['1NN_acc'] = knn_acc

    print(f"Training kNN classifier with k=3")
    my_knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
    my_knn.fit(train_images, train_labels)
    knn_pred = my_knn.predict(test_images)
    knn_acc = np.round(np.sum([1 for pred, label in zip(knn_pred, test_labels) if pred == label])/test_labels.shape[0],4)
    print(f'3NN test accuracy: {knn_acc}')
    # store results
    results['3NN_acc'] = knn_acc

    label_list = np.unique(train_labels)
    results['label_list'] = label_list
    if per_class:
        knn_class = np.zeros(len(label_list))
        print(f'\nPer label {n_neighbors}NN test accuracy:')
        for k, label in enumerate(label_list):
            mask = test_labels == label
            knn_class[k] = np.round(np.sum(knn_pred[mask]==test_labels[mask])/np.sum(mask),4)
            print(f'{label}\t{knn_class[k]:.2f}')
        # store results
        results['knn_class'] = knn_class
    if conf_matrix:
        knn_conf = confusion_matrix(test_labels, knn_pred)
        results['knn_conf'] = knn_conf
        print('\nPrinting Confusion Matrix:')
        print(results['knn_conf'])
    return results
#########################################################################################


########################################################################################
#Function to return dataloaders for train and validation set for precomputing embeddings 
#
def get_train_valid_dataloaders(data_config):
    """returns dataloaders for train and validation set"""
    
    #Load some utils
    image_size = data_config['input_size']
    images_per_track = data_config['images_per_track']
    bs = data_config['embeddor_batch_size']

    #If valid df not already made, sample it
    if data_config['datafiles']['valid'] != None: 
        df = pd.read_csv(data_config['datafiles']['train'])
        print("Reading dataframe at ",data_config['datafiles']['train'])
        train_dataset = Flowerpatch_w_Track_and_Filter(df,'new_filepath',
                                                       'ID','track',image_size,'train',imgs_per_track=images_per_track)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
       
        ### Valid dataset and dataloader
        valid_df = pd.read_csv(data_config['datafiles']['valid'])
        valid_dataset = Flowerpatch_w_Track_and_Filter(valid_df,'new_filepath',
                                                       'ID','track',image_size,'test',imgs_per_track=images_per_track)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False)
    else:
        #if no valid dataset, sample from training set
        train_df = pd.read_csv(data_config['datafiles']['train'])
    
        unique_tracks = train_df['track'].unique() # Calculate the number of unique tracks
        total_tracks = len(unique_tracks)
        valid_num_tracks = round(data_config['percent_valid'] * total_tracks) #num tracks to sample 
        valid_tracks = np.random.choice(unique_tracks, size=valid_num_tracks, replace=False)  # Randomly select tracks 
       
        valid_df = train_df[train_df['track'].isin(valid_tracks)] #build valid df 
        train_df = train_df[~train_df['track'].isin(valid_tracks)] #remove valid tracks from train set

        train_num = len(train_df)
        valid_num_rows = len(valid_df)

        print(f"Using {valid_num_rows} samples for validation set")
        print(f"{train_num} total training samples")

        # Build pytroch datasets and dataloaders
        train_dataset = Flowerpatch_w_Track_and_Filter(train_df,'new_filepath',
                                                       'ID','track',image_size,'train',imgs_per_track=images_per_track)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=False)

        valid_dataset = Flowerpatch_w_Track_and_Filter(valid_df,'new_filepath',
                                                       'ID','track',image_size,'test',imgs_per_track=images_per_track)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False)
    
    return train_dataloader, valid_dataloader
####################################################################################


#######################################################################################
#
# Function to return pytorch dataloader for test and reference set
def get_test_ref_dataloaders(data_config):
    """Returns pytorch dataloaders for train and reference set with subsamping"""
    
    #Load Miscl Utils 
    image_size = data_config['input_size']
    images_per_track = data_config['images_per_track']
    bs = data_config['embeddor_batch_size']

    # Build test dataloader and subsample for reference set
    test_df = pd.read_csv(data_config['datafiles']['test'])
    grouped = test_df.groupby(['ID','track']).filter(lambda x: len(x) >= images_per_track) #only sample long enough tracks
    random_tracks_indices = grouped.groupby('ID').apply(lambda x: np.random.choice(x.index)).values #select random tracks
    ref_df = pd.DataFrame()

    for index in random_tracks_indices:
        id_track_id = test_df.loc[index, 'ID']
        id_track_track = test_df.loc[index,'track']
        id_track_df = test_df[(test_df['ID'] == id_track_id) & (test_df['track'] == id_track_track)]
        selected_images = id_track_df.sample(n=images_per_track)  # Sample multiple images for each selected ID and track combination
        ref_df = pd.concat([ref_df, selected_images], axis=0)
       
        #remove other images from test
        test_df = test_df[~((test_df['track'] == id_track_track) & (test_df['ID'] == id_track_id))]

    ref_num_samples = len(ref_df)
    test_num = len(test_df)
    print(f"Using {ref_num_samples} samples for the reference set")
    print(f"{test_num} total test samples")

    #set different batch size for small reference set
    aggr_bs = data_config['aggregator_batch_size']
    if aggr_bs > ref_num_samples:
        ref_bs = ref_num_samples
    else:
        ref_bs = aggr_bs
    
    # Build pytroch datasets and dataloaders
    test_dataset = Flowerpatch_w_Track_and_Filter(test_df,'new_filepath',
                                                'ID','track',image_size,'test',imgs_per_track=images_per_track)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)
   
    ref_dataset = Flowerpatch_w_Track(ref_df,'new_filepath','ID','track',image_size,'test')
    ref_dataloader = torch.utils.data.DataLoader(ref_dataset, batch_size=ref_bs, shuffle=False)

    return test_dataloader, ref_dataloader
####################################################################################


####################################################################################
#
# A Function to precompute and preliminarily eval quality of embeddings
def precompute_embeddings(emb_path,data_config,device):
    """precompute and return embeddings and labels for all datasets for training of feature agglomerator"""
    
    #Load embedder 
    model_name = os.path.basename(emb_path)
    embedder = torch.load(emb_path)
    embedder.eval() 

    ## Load dataloaders for feeding images through embedder 
    train_dataloader, valid_dataloader = get_train_valid_dataloaders(data_config)
    test_dataloader, ref_dataloader = get_test_ref_dataloaders(data_config)

    # Compute Embeddings
    train_embeddings, train_labels, train_tracks = get_embeddings_w_track(embedder,train_dataloader,device)
    print('Train embeddings made with',model_name,"shape:",train_embeddings.size())

    valid_embeddings, valid_labels, valid_tracks = get_embeddings_w_track(embedder,valid_dataloader,device)
    print('Valid embeddings made with',model_name,"shape:",valid_embeddings.size())

    test_embeddings, test_labels, test_tracks = get_embeddings_w_track(embedder,test_dataloader,device)
    print('Test embeddings made with',model_name,"shape:",test_embeddings.size())
    
    ref_embeddings, ref_labels, ref_tracks = get_embeddings_w_track(embedder,ref_dataloader,device)
    print('Reference embeddings made with',model_name,"shape:",ref_embeddings.size())

    ## Evaluate quality of embeddings using KNN
    print("KNN evaluation before multi-image agglomeration training")
    print("")
    print("KNN within train embeddings (initilized on valid set:)")
    naive1 = knn_evaluation(valid_embeddings.cpu().numpy(),valid_labels,train_embeddings.cpu().numpy(),train_labels,1,False,False) #TODO Swap
    print("")
    print("KNN within test embeddings (initilized on ref set:)")
    naive1 = knn_evaluation(ref_embeddings.cpu().numpy(),ref_labels,test_embeddings.cpu().numpy(),test_labels,1,False,False)
    print("")
    print("KNN evaluation open set (initilized batch 1 evaluated on batch 2)")
    naive2 = knn_evaluation(train_embeddings.cpu().numpy(), train_labels, test_embeddings.cpu().numpy(), test_labels, 1,per_class=False)

    return [train_embeddings,train_labels,train_tracks],[valid_embeddings,valid_labels,valid_tracks],[test_embeddings,test_labels,test_tracks],[ref_embeddings,ref_labels,ref_tracks]
####################################################################################

####################################################################################
#
#Get model returns the pytorch model to be trained
def get_model(model_config):
    if model_config['aggregator_type'] == 'attention':
        model = AttentionAggregator(model_config['images_per_track'],model_config['latent_dim']) #recognition model
    elif model_config['aggregator_type'] == 'fixed':
        model = FixedAggregator(model_config['images_per_track'],model_config['latent_dim'])
    elif model_config['aggregator_type'] == 'mean':
        model = MeanAggregator(model_config['images_per_track'],model_config['latent_dim'])
    return model

############################################### Training and Eval Handle###########################################
#
#
def train_and_eval(config_file):
    """A training and evaluation handle for multifeature agglomerator using precompueted embeddings"""
    ##### Open and read config 
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
    
    resume_training = train_config['wandb_resume']
   
    ###### Initialize wandb logging
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
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f'Found device: {device}')

    ##### Set Random seeds 
    torch.manual_seed(torch_seed)
    np.random.seed(torch_seed)
    
    ##### Define some Miscl Utils
    image_size = data_config['input_size']
    images_per_track = data_config['images_per_track']
    model_config['images_per_track'] = images_per_track
    bs = data_config['embeddor_batch_size']
    aggr_bs = data_config['aggregator_batch_size']
   

    ##### Precompute Embeddings
    emb_path = model_config['embeddor_path']
    train_set,valid_set,test_set,ref_set = precompute_embeddings(emb_path,data_config,device) #return as list [embeddings,labels,track_ids]

    #May need to update batch size params for small ref set 
    if aggr_bs > len(ref_set[0]):
        ref_bs = len(ref_set[0])
    else:
        ref_bs = aggr_bs
    

    ##### Build Dataloaders of precomputed embeddings 
    # Now on recieving packets of frame embedding
    train_dataset = Flowerpatch_Embeddings_v2(train_set[0],train_set[1],train_set[2],images_per_track)
    train_dataloader =  DataLoader(train_dataset, batch_size=aggr_bs, shuffle=True)

    valid_dataset = Flowerpatch_Embeddings_v2(valid_set[0],valid_set[1],valid_set[2],images_per_track)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=aggr_bs, shuffle=False)

    test_dataset = Flowerpatch_Embeddings_v2(test_set[0],test_set[1],test_set[2],images_per_track)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=aggr_bs, shuffle=False)

    ref_dataset = Flowerpatch_Embeddings_v2(ref_set[0],ref_set[1],ref_set[2],images_per_track)
    ref_dataloader = torch.utils.data.DataLoader(ref_dataset, batch_size=ref_bs, shuffle=False)

 
    #### Build Aggregator model
    if verbose:
        print('Building model....')
    model = get_model(model_config)
    #### Send to CUDA 
    model.to(device)
   

    # load latest saved checkpoint if resuming a failed run
    if resume_training == True: 
        saved = os.listdir(os.path.dirname(model_config['model_path'])+r'/checkpoints/')
        check_array = []
        for f in saved:
            check_array.append(f[:-4])
        check_array = np.array(check_array,dtype=np.int64)
        #most_recent_epoch = np.max(check_array) #find most recent checkpoint
        most_recent_epoch = train_config['checkpoint_to_load'] # make this part of yml
        print(f'Resuming training from saved epoch: {most_recent_epoch}')
        most_recent_model = os.path.dirname(model_config['model_path'])+r'/checkpoints/'+str(most_recent_epoch)+'.pth'
        print(f'Loading saved checkpoint model {most_recent_model}')
        model = torch.load(most_recent_model)
    

    ##### Define Optimizer, Miner and other training features 

    #optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    optimizer = optim.SGD(model.parameters(), lr=train_config['learning_rate'])
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.75, verbose=True,min_lr = 1e-5)
    #miner = miners.MultiSimilarityMiner()
    miner = miners.TripletMarginMiner(margin=train_config['margin'], type_of_triplets="semihard", distance = CosineSimilarity())
    miner_type = "semihard"
    loss_fn = losses.TripletMarginLoss(train_config['margin'], distance = CosineSimilarity())
    if verbose:
        print('Loss:',loss_fn)


    #### If resuming training set epoch number
    if resume_training == True:
        epoch_range = range(train_config['num_epochs'])[most_recent_epoch:]
        stop_epoch = 0
    else:
        epoch_range = range(train_config['num_epochs'])
        stop_epoch = 0

    ##### Initialize early stopping variables
    best_valid_loss = float('inf')
    best_model = model
    valid_loss = 'Null'
    num_epochs_no_improvement = 0
    check_for_early_stopping = train_config['early_stopping']
    consecutive_epochs = train_config['early_stop_consecutive_epochs']
    stop_early = False

    ##### Train the model
    if verbose:
        print('Training model...')
    print_k = train_config['print_k']

    start = time.time()
    for epoch in epoch_range: 
        running_loss = 0.0
        for k, data in enumerate(train_dataloader):
            # Load data for minibatch
            packets_of_frame_embeddings = data['track_embeddings'] #size [batch_size, img_count, latent_dim]
            labels = data['id'].to(device) 
            optimizer.zero_grad()

            outputs = model(packets_of_frame_embeddings)

            # get semi-hard triplets
            triplet_pairs = miner(outputs, labels)

            # Calculate loss
            loss = loss_fn(outputs, labels, triplet_pairs)
            loss.backward()
            
            # Update model params 
            optimizer.step()
            running_loss += loss.item()

            # Log with wandb 
            experiment.log({
                'train loss': loss.item(),
                'epoch': epoch,
                #'learning rate' : lr
                'triplet_num': torch.numel(triplet_pairs[0])
            })

        ### Save Checkpoints if correct epoch 
        if epoch % train_config['save_checkpoint_freq'] == 0 or (epoch+1) == train_config['num_epochs']: 
                if os.path.dirname(model_config['model_path']) is not None:
                    print('Saving checkpoint',epoch)
                    if not os.path.exists(os.path.dirname(model_config['model_path'])+r'/checkpoints/'):
                        os.mkdir(os.path.dirname(model_config['model_path'])+r'/checkpoints/')
                    torch.save(model,(os.path.dirname(model_config['model_path'])+r'/checkpoints/'+str(epoch)+".pth"))

        ### calculate validation metrics             
        with torch.no_grad():
            valid_outputs, valid_labels, valid_loss = get_loss(model, valid_dataloader, loss_fn, miner, device)
            print(f'[{epoch + 1}, {k + 1:5d}] train_loss: {running_loss/print_k:.4f} | val_loss: {valid_loss:.4f}')
            #scheduler.step(valid_loss)
            #current_lr = optimizer.param_groups[0]['lr']
            experiment.log({'valid loss': valid_loss, })
                           # 'learning rate': current_lr})

            # Check if validation loss has improved
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = model
                num_epochs_no_improvement = 0
            else:
                num_epochs_no_improvement += 1

            # Check if early stopping condition is met
            if check_for_early_stopping == True:
                if num_epochs_no_improvement >= consecutive_epochs:
                    print(f'Early stopping at epoch {epoch+1} due to no improvement in validation loss for {consecutive_epochs} consecutive epochs')
                    stop_epoch = epoch+1
                    stop_early = True 

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

    # evaluate on test set using KNN
    if verbose:
        print('Evaluating model...')
    model.eval()
      
    #### First calculate baseline metrics
    baseline_metrics = eval_baselines(model_config,test_dataloader,ref_dataloader,loss_fn,miner,device,eval_config)
    
    ##### Calculate loss and get aggregated features from trained model
    # 4/25 test knn init on non averaged vals
    # reference_embeddings = ref_set[0]
    # reference_labels = ref_set[1]
    
    reference_embeddings, reference_labels, reference_loss = get_loss(model, ref_dataloader, loss_fn, miner, device)
    test_embeddings, test_labels, test_loss = get_loss(model, test_dataloader, loss_fn, miner, device)   
    
    print(f'Reference (or Train) Loss: {reference_loss:.4f}')
    print('Reference size:',reference_embeddings.shape)
    print(f'Test (or Query) Loss: {test_loss:.4f}')
    print('Test (or Query) size:',test_embeddings.shape)

    #### Evaluate aggregated features
    results = knn_evaluation(reference_embeddings, reference_labels, test_embeddings, test_labels, 
                            eval_config['n_neighbors'], eval_config['per_class'], eval_config['conf_matrix'])
    
    #### add baselines for saving 
    results = results | baseline_metrics

    ##### Save results in pickle file and save trained model
    # Add total training loss to results 
    results['train_loss'] = running_loss
    print(results)

    # Adding other metrics to results to pass to csv
    results['valid_loss'] = valid_loss
    results['wandb_id'] = experiment.id
    results['images_per_track'] = images_per_track
    results['start_time'] = experiment.start_time
    results['train_time'] = duration
    results['stop_epoch'] = stop_epoch
    results['total_testing_images'] = len(test_embeddings)
    results['total_reference_images'] = len(reference_embeddings)
    results['model_type'] = model_config['aggregator_type']

    # if not os.path.exists(eval_config['pickle_file']):
    #     with open(eval_config['pickle_file'],'ab'):
    #         os.utime(eval_config['pickle_file'], None)

    # Save results to temporary file
    with open(eval_config['pickle_file'],'wb') as fi:
        pickle.dump(results,fi)
        print("Results saved to pickle file")

    #### Save trained model
    if model_config['model_path'] is not None:
        print('Saving model...')
        torch.save(model, model_config['model_path'])
    else:
        print('model_path not provided. Not saving model')
    print('Finished')
    wandb.finish()


#################################################################################
#
# Eval only baselines with premade embedding loaders
def eval_baselines(model_config,test_dataloader,ref_dataloader,loss_fn,miner,device,eval_config):
    baselines = {}
    for m in ['fixed','mean']:
        model_config['aggregator_type'] = m
        model = get_model(model_config)
        ##### Calculate loss and get aggregated features from trained model
        reference_embeddings, reference_labels, reference_loss = get_loss(model, ref_dataloader, loss_fn, miner, device)
        test_embeddings, test_labels, test_loss = get_loss(model, test_dataloader, loss_fn, miner, device)   
        
        print(f'Reference (or Train) Loss: {reference_loss:.4f}')
        print('Reference size:',reference_embeddings.shape)
        print(f'Test (or Query) Loss: {test_loss:.4f}')
        print('Test (or Query) size:',test_embeddings.shape)

        #### Evaluate aggregated features
        print(f"Evaluating baseline model {m}")
        results = knn_evaluation(reference_embeddings, reference_labels, test_embeddings, test_labels, 
                                eval_config['n_neighbors'], eval_config['per_class'], eval_config['conf_matrix'])
        baselines[f'1NN_{m}'] = results['1NN_acc']
        baselines[f"3NN_{m}"] = results['3NN_acc']
    return baselines



############################################### \Eval Handle###########################################
#
# Eval only
##############################################################################33
def eval_only(config_file,save_model=False,pretrained=False):
    """A evaluation handle only for multifeature agglomerator using precompueted embeddings"""
    ##### Open and read config 
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
     
    
    if verbose:
            now = datetime.now() # current date and time
            dt = now.strftime("%y-%m-%d %H:%M")
            print(f'Date and time when this experiment was started: {dt}')
            print("Data Settings:")
            print(data_config)
            print("Model Settings:")
            print(model_config)
    
    #SET GPU TO USE
    os.environ["CUDA_VISIBLE_DEVICES"]=str(train_config['gpu'])
    if verbose:
        print('Using GPU',train_config['gpu'])
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f'Found device: {device}')

    ##### Set Random seeds 
    torch.manual_seed(torch_seed)
    np.random.seed(torch_seed)
    
    ##### Define some Miscl Utils
    image_size = data_config['input_size']
    images_per_track = data_config['images_per_track']
    model_config['images_per_track'] = images_per_track
    bs = data_config['embeddor_batch_size']
    aggr_bs = data_config['aggregator_batch_size']
   

    ##### Precompute Embeddings
    emb_path = model_config['embeddor_path']
    train_set,valid_set,test_set,ref_set = precompute_embeddings(emb_path,data_config,device) #return as list [embeddings,labels,track_ids]

    #May need to update batch size params for small ref set 
    if aggr_bs > len(ref_set[0]):
        ref_bs = len(ref_set[0])
    

    test_dataset = Flowerpatch_Embeddings_v2(test_set[0],test_set[1],test_set[2],images_per_track)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=aggr_bs, shuffle=False)

    ref_dataset = Flowerpatch_Embeddings_v2(ref_set[0],ref_set[1],ref_set[2],images_per_track)
    ref_dataloader = torch.utils.data.DataLoader(ref_dataset, batch_size=ref_bs, shuffle=False)

 
    #### Build Aggregator model
    if verbose:
        print('Building model....')
    if pretrained== False:
        model = get_model(model_config)
    else:
        print("loading model:",model_config['model_path'])
        model = torch.load(model_config["model_path"])
    #### Send to CUDA 
    model.to(device)
    
    miner = miners.TripletMarginMiner(margin=train_config['margin'], type_of_triplets="semihard", distance = CosineSimilarity())
    miner_type = "semihard"
    loss_fn = losses.TripletMarginLoss(train_config['margin'], distance = CosineSimilarity())

    # evaluate on test set using KNN
    if verbose:
        print('Evaluating model...')
    model.eval()
           
    #### First calculate baseline metrics
    print("calculating baselines")
    baseline_metrics = eval_baselines(model_config,test_dataloader,ref_dataloader,loss_fn,miner,device,eval_config)
    print(baseline_metrics)
          
    ##### Calculate loss and get aggregated features from trained model
    reference_embeddings, reference_labels, reference_loss = get_loss(model, ref_dataloader, loss_fn, miner, device)
    test_embeddings, test_labels, test_loss = get_loss(model, test_dataloader, loss_fn, miner, device)   
    
    print(f'Reference (or Train) Loss: {reference_loss:.4f}')
    print('Reference size:',reference_embeddings.shape)
    print(f'Test (or Query) Loss: {test_loss:.4f}')
    print('Test (or Query) size:',test_embeddings.shape)

    #### Evaluate aggregated features
    results = knn_evaluation(reference_embeddings, reference_labels, test_embeddings, test_labels, 
                            eval_config['n_neighbors'], eval_config['per_class'], eval_config['conf_matrix'])
    
    #### add baselines for saving 
    results = results | baseline_metrics
    
    ##### Save results in pickle file and save trained model
    # Add total training loss to results 
    results['train_loss'] = None
    print(results)

    # Adding other metrics to results to pass to csv
    results['valid_loss'] = None
    results['wandb_id'] = None
    results['images_per_track'] = images_per_track
    results['start_time'] = None
    results['train_time'] = None
    results['stop_epoch'] = None
    results['total_testing_images'] = len(test_embeddings)
    results['total_reference_images'] = len(reference_embeddings)
    results['model_type'] = model_config['aggregator_type']

    # Save results to temporary file
    with open(eval_config['pickle_file'],'wb') as fi:
        pickle.dump(results,fi)
        print("Results saved to pickle file")
    if save_model == True:
        #### Save trained model
        if model_config['model_path'] is not None:
            print('Saving model...')
            torch.save(model, model_config['model_path'])
        else:
            print('model_path not provided. Not saving model')
        print('Finished')



print("beginning execution")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="yaml file with experiment settings", type=str)
    args = parser.parse_args()
    train_and_eval(args.config_file)

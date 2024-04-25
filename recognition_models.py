################################## Model store file for recognition modules ##################################
# Created 3/18/24 
# Luke Meyers 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


################# Simple Neural Network Model #############################
    
#------ Recog model = super simple mlp to test training handle and data loader, 1 hidden layer 256n
class RecogModel(nn.Module): 
    def __init__(self, input_size=128):
        super(RecogModel, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, 256)  # Two input vectors concatenated
        self.fc2 = nn.Linear(256, 1)  # Output layer with one neuron

    def forward(self, x1, x2):
        # Concatenate the two input vectors along the feature dimension
        x = torch.cat((x1, x2), dim=1)
        
        # Pass through fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
    
        x = self.fc2(x)
        # Sigmoid for binary output 
        #x = torch.sigmoid(self.fc2(x)) #remove sigmoid from recog model cuase logits loss does it  
        return x
    
############################################## Code from Thomas Atkins github##########################

# #An FC layer with no bias that takes in the feature
# #vector and the mean of all vectors, and outputs an
# #attention matrix.
# class VectorMeanLayer(Layer):
#     def __init__(self, k):
#         super(VectorMeanLayer, self).__init__()
        
#     def build(self, input_shape):
#         self.w = self.add_weight(
#             name='w',
#             shape=(input_shape[-1]*2, input_shape[-1]),
#             initializer="random_normal",
#             trainable=True,
#         )
        
#     def call(self, inputs):
#         means = tf.math.reduce_mean(inputs, axis=1)
#         #these next two lines just take a matrix of size
#         #batch_size*latent_dim and repeat the means to form
#         #a tensor of size batch_size*track_size*latent_dim
#         means = tf.expand_dims(means, axis=1)
#         means = tf.repeat(means, repeats=[inputs.shape[1]], axis=1)
#         #now we stick the means to the original values
#         full_inputs = tf.keras.layers.concatenate([inputs, means], axis=2)
#         return tf.matmul(full_inputs, self.w)

################# A pytorch translation of Thomas Atkins code, tbd on functionality
#from recognition_models import AttentionAggregator

class VectorMeanLayer(nn.Module):
    def __init__(self, k,verbose=False,latent_dim=128):
        super(VectorMeanLayer, self).__init__()
        self.verbose = verbose
        self.idx = 0 
        #build weight matrix to compare all features by each other, and the mean
        self.w = nn.Parameter(1-(torch.randn(latent_dim * 2, latent_dim)* 0.99),requires_grad = True)  # Initialize weights close to 0
        self.bias = nn.Parameter(torch.tensor(1.0), requires_grad=False)
    def forward(self, inputs):
        if self.idx != 0:
            self.verbose = False

        batch_size, track_size, latent_dim = inputs.size()
        if self.verbose == True: 
            print("initilial input size", inputs.size())

        # Calculate mean along the track dimension
        means = torch.mean(inputs, dim=1, keepdim=True)
        if self.verbose == True: 
            print("Size of means of features",means.size())
       

        # Repeat means to match the original input shape
        means = means.expand(-1, track_size, -1)
        if self.verbose == True: 
            print("expand means",means.size())
          

        # Concatenate means with original input along the last dimension
        full_inputs = torch.cat([inputs, means], dim=2)
        if self.verbose == True: 
            print("full inputs size",full_inputs.size())
            print("weight matrix size",self.w.size())

        # Perform matrix multiplication with weights
        output = torch.matmul(full_inputs,(self.w.to(full_inputs.device)+ self.bias.to(full_inputs.device)))
        if self.verbose == True: 
            print("Outputed matrix size", output.size())
        
        self.idx += 1 
        return output 

class AttentionAggregator(nn.Module):
    def __init__(self,img_count=5,latent_dim=128) -> None:
        super(AttentionAggregator,self).__init__()
        #self.bs = batch_size
        self.img_count = img_count
        self.latet_dim = latent_dim
        
        #Build and Normalize attention matrix re Thomas Atkins code
        self.Generate_A = VectorMeanLayer(k=1,verbose=True,latent_dim=self.latet_dim)
        self.Normalize_A = nn.Softmax(dim=1)
        self.ElementWise = torch.mul
        self.idx = 0 
        #If batch_size = 256
        # Un-normalized Attention Matrix torch.Size([256, 2, 128])
        # Normalized Attention Matrix torch.Size([256, 2, 128])
        # Features scaled by attention weights torch.Size([256, 2, 128])
        # Summed fraction features torch.Size([256, 128])
        # Agglomerated features after L2 norm torch.Size([256, 128])
                    
    def forward(self,x):
        A = self.Generate_A(x)
        
        if self.idx == 0: 
            print("Un-normalized Attention Matrix",A.size())
        
        #normalize the matrix by columns, this is now a valid attention matrix
        A = self.Normalize_A(A)
        
        if self.idx == 0: 
            print("Normalized Attention Matrix",A.size())
        
        #element-wise multiply by the features
        x = self.ElementWise(x, A)
        
        if self.idx == 0: 
            print("Features scaled by attention weights",x.size())
        
        #sum the columns to get an unormalized feature vector
        x = torch.sum(x, axis=1)
        
        if self.idx == 0: 
            print("Summed fraction features",x.size())
        
        #normalize the output
        x = F.normalize(x, p=2, dim=1)
        
        if self.idx == 0: 
            print("Agglomerated features after L2 norm",x.size())

        self.idx += 1
        return x



    
# #Use the layers as first-class obejcts for simplification
# #Gotta love Python
# attention_subnets = {"simple"      : ConstantLayer,
#                      "vector"      : VectorLayer,
#                      "vector_mean" : VectorMeanLayer,
#                      "vector_window" : VectorWindowLayer,
#                      "vector_mean_window": VectorMeanWindowLayer}    
    
# class TemporalContrastiveLearning(tf.keras.Model):
    
#     def __init__(self, base_model, temperature=0.1, subnet="simple", k=1, window_size=1):
#         super(TemporalContrastiveLearning, self).__init__()
#         self.backbone = base_model
#         self.loss_tracker = tf.keras.metrics.Mean(name="loss")
#         self.valid_loss_tracker = tf.keras.metrics.Mean(name="valid_loss")
#         self.temperature = temperature
#         self.time_model = tf.keras.Sequential()
#         self.time_model.add(tf.keras.layers.TimeDistributed(base_model))
#         self.Generate_A = attention_subnets[subnet](k=k)
#         self.Normalize_A = Softmax(axis=1)
#         self.ElementWise = Multiply()
#         self.model_name = "TemporalConstrastiveLearning"
        
#     def call(self, data):
#         x = data
#         x = self.time_model(x)
#         #generate our attention matrix A by whatever method we have
#         A = self.Generate_A(x)
#         #normalize the matrix by columns, this is now a valid
#         #attention matrix
#         A = self.Normalize_A(A)
#         #element-wise multiply by the features
#         x = self.ElementWise([x, A])
#         #sum the columns to get an unormalized feature vector
#         x = tf.math.reduce_sum(x, axis=1)
#         #normalize the output
#         x = tf.math.l2_normalize(x, axis=1)
#         return x

#     def train_step(self, data):
#         x1, x2, y = data
        
#         with tf.GradientTape() as tape:
#             x1 = self(x1, training=True)
#             x2 = self(x2, training=True)
            
#             sim_matrix1 = tf.matmul(x1, x2, transpose_b=True)/ self.temperature
#             sim_matrix2 = tf.matmul(x2, x1, transpose_b=True)/ self.temperature
            
#             loss1 = tfa.losses.npairs_loss(y_pred=sim_matrix1, y_true=y)
#             loss2 = tfa.losses.npairs_loss(y_pred=sim_matrix2, y_true=y)
#             loss = loss1 + loss2
        
#         trainable_vars = self.trainable_weights
#         gradients = tape.gradient(loss, trainable_vars)
#         self.loss_tracker.update_state(loss)

#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         return {"loss": self.loss_tracker.result()}
    
#     def test_step(self, data):
#         x1, x2, y = data
        
#         x1 = self(x1, training=False)
#         x2 = self(x2, training=False)
            
#         sim_matrix1 = tf.matmul(x1, x2, transpose_b=True)/ self.temperature
#         sim_matrix2 = tf.matmul(x2, x1, transpose_b=True)/ self.temperature
            
#         loss1 = tfa.losses.npairs_loss(y_pred=sim_matrix1, y_true=y)
#         loss2 = tfa.losses.npairs_loss(y_pred=sim_matrix2, y_true=y)
#         loss = loss1 + loss2
        
#         self.valid_loss_tracker.update_state(loss)
        
#         return {"loss": self.valid_loss_tracker.result()}
    
#     @property
#     def metrics(self):
#         # We list our `Metric` objects here so that `reset_states()` can be
#         # called automatically at the start of each epoch
#         # or at the start of `evaluate()`.
#         # If you don't implement this property, you have to call
#         # `reset_states()` yourself at the time of your choosing.
#         return [self.loss_tracker, self.valid_loss_tracker]
    
#     #this is just so that we can plot and save the model using Keras
#     def model(self, track_size, input_size_1, input_size_2):
#         x = Input(shape=(track_size, input_size_1, input_size_2, 3))
#         return tf.keras.Model(inputs=[x], outputs=self.call(x))

######################### Simple feed through model to return the mean in track dimension########################
class MeanAggregator(nn.Module):
    def __init__(self,img_count=5,latent_dim=128) -> None:
        super(MeanAggregator,self).__init__()
        self.img_count = img_count
        self.latet_dim = latent_dim
   
    def forward(self,inputs):
        batch_size, track_size, latent_dim = inputs.size()
        # Select only the latent dimension from the tensor
        output = torch.mean(inputs, dim=1)
        return output 
        
######################### Simple feed through model to return only the first image in a track########################
class FixedAggregator(nn.Module):
    def __init__(self,img_count=5,latent_dim=128) -> None:
        super(FixedAggregator,self).__init__()
        self.img_count = img_count
        self.latet_dim = latent_dim
   
    def forward(self,inputs):
        batch_size, track_size, latent_dim = inputs.size()
        # Select only the latent dimension from the tensor
        output = inputs[:, 0, :]
        return output 
        


############################# Pytroch implementation of attention based aggregator and recog ######################
    
class AttentionTrackRecog(nn.Module):
    def __init__(self,batch_size=32,img_count=5,latent_dim=128,emb_path= "/home/gsantiago/ReID_model_training/new_auto_train_eval/models_trained/summer_bee_dataset_open_train_bee_64_ids_batch1_sample_num_64/wandb/run-20231106_004425-yida7voj/files/summer_bee_dataset_open_train_bee_64_ids_batch1_sample_num_64.pth") -> None:
        super(AttentionTrackRecog,self).__init__()
        self.bs = batch_size
        self.img_count = img_count
        self.latet_dim = latent_dim
        self.model_name = os.path.basename(emb_path)
        self.embedder = torch.load(emb_path) 
        #make all params unupdateable?
        for param in self.embedder.parameters():
            param.requires_grad = False

        #Build and Normalize attention matrix re Thomas Atkins code
        self.Generate_A = VectorMeanLayer(k=1)
        self.Normalize_A = nn.Softmax(dim=1)
        self.ElementWise = torch.mul
        
        #fully connected layers for decision at end 
        self.fc1 = nn.Linear(latent_dim * 2, 256)  # Two input vectors concatenated
        self.fc2 = nn.Linear(256, 1)  # Output layer with one neuron
    
    
    def pass_through_attention(self,x):
        A = self.Generate_A(x)
        #normalize the matrix by columns, this is now a valid
        #attention matrix
        A = self.Normalize_A(A)
        #element-wise multiply by the features
        x = self.ElementWise([x, A])
        #sum the columns to get an unormalized feature vector
        x = torch.sum(x, axis=1)
        #normalize the output
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self,x1,x2):

        # #resize batches to embed images
        # x1_flat = torch.reshape(x1, (-1,self.img_count))
        # x2_flat = torch.reshape(x2, (-1,self.img_count))

        # with torch.no_grad(): 
        #     # get embeddings of img pair
        #     x1_emb = self.embedder(x1_flat)
        #     x2_emb = self.embedder(x2_flat)
        
        # #apply stop gradients, for extra security
        # x1_emb = x1_emb.detach()
        # x2_emb = x2_emb.detach()

        # #reshape back to original
        # x1 = torch.reshape(x1, (self.bs, self.img_count, -1))
        # x2 = torch.reshape(x2, (self.bs, self.img_count, -1))

        #x1,x2 current shape = [batchsize,img_count,latent_dim]

        #pass through attention 
        x1 = self.pass_through_attention(x1)
        x1 = self.pass_through_attention(x2)

        # Concatenate the two input tensors along the feature dimension
        x = torch.cat((x1, x2), dim=1)
        
        # Pass through fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
    
        x = self.fc2(x)
        # Sigmoid for binary output 
        #x = torch.sigmoid(self.fc2(x)) #remove sigmoid from recog model cuase logits loss does it  
        return x

  
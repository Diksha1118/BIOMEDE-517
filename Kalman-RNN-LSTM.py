##RNN with batches
import torch.optim as optim
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor
from torch import nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats
from tqdm import tqdm
import wandb
import random
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
torch.autograd.set_detect_anomaly(True)


def mse(y_true, y_pred): 
    # Ensure y_true and y_pred are numpy arrays to simplify the calculation.
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    # Compute MSE.
    mse_x1 = np.mean((y_true[:,0] - y_pred[:,0]) ** 2)
    mse_x2 = np.mean((y_true[:,1] - y_pred[:,1]) ** 2)
    mse_x3 = np.mean((y_true[:,2] - y_pred[:,2]) ** 2)
    mse_x4 = np.mean((y_true[:,3] - y_pred[:,3]) ** 2)
    mse_final = np.array([mse_x1, mse_x2, mse_x3, mse_x4]) #combining into array
    return mse_final

def corr(y_true, y_pred):
    # Ensure y_true and y_pred are numpy arrays.
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()  # Ensure conversion from PyTorch tensor to NumPy array
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()  # Ensure conversion from PyTorch tensor to NumPy array

    # Initialize a list to store correlation results.
    corr_results = []
    
    # Calculate Pearson correlation coefficient for each state variable.
    for i in range(4):  # Assuming there are 4 state variables.
        if np.std(y_true[:, i]) == 0 or np.std(y_pred[:, i]) == 0:
            # If either y_true or y_pred for the state variable is constant, append nan values.
            corr_results.append((np.nan, np.nan))
        else:
            # Calculate and append the correlation coefficient and p-value if inputs are not constant.
            corr = scipy.stats.pearsonr(y_true[:, i], y_pred[:, i])
            corr_results.append(corr)
    
    # Convert the list of tuples into a NumPy array for easy handling.
    corr_final = np.array(corr_results)
    return corr_final

class LSTMs(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, gain_size):
        super(LSTMs, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, gain_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.batch_size= batch_size

    def initialize_hidden_state(self,x):
        #batch_size=100
        
        batch_size1 = x.size(0)  #fetches the size of the first dimension whoich is the batchsize
        self.h0 = torch.zeros(self.num_layers,batch_size1, self.hidden_size)
        #cell state for long term  memory
        self.c0 = torch.zeros(self.num_layers,batch_size1, self.hidden_size)
       
        
    #forward method defines how the input data flows through the network. 
    def forward(self, x):    
        # Forward propagate the RNN
        (out,(self.h0,self.c0))  = self.lstm(x, (self.h0,self.c0))
        #out = out.view(-1, out.size(2))
        #out = out[:, -1, :]
        out=out.squeeze(0).detach()
        out = self.fc(out)  # Use last time step's output
        print(out.size())
        return out  

    
df_regression = pd.read_csv("regression_dataset.csv")
y = df_regression.iloc[:, :4].values  # States: Position and Velocity
X = df_regression.iloc[:, 4:].values  # Observations: Channel Data

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print('Xtrain:',X_train_tensor.size())
print('Ytrain:',y_train_tensor.size())
# Create data loaders
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)    

##KALMAN CLASS
class KalmanFilter():
  def __init__(self, rnn_model_instance,append_ones_y=False, device='cpu'):
    self.A, self.C, self.W, self.Q = None, None, None, None # state model, obs model, state noise, obs noise
    self.At, self.Ct = None, None # useful transposes
    self.append_ones_y = append_ones_y
    self.device = device
    self.rnn_model_instance = rnn_model_instance
    self.y_pred = torch.zeros(1, 4)
    self.prior= torch.zeros(1, 4)
    self.y_predictions1 = []
    
  def train(self, x, y): # x is input matrix of observations, y is input matrix of ground truth state
    if self.append_ones_y:
      y = torch.cat((y, torch.ones([y.shape[0], 1])), dim=1) # pads y with an extra column of ones if needed
    ytm = y[:-1, :] # all of y except the last row     (yt-1)prior state
    yt = y[1:, :] # all of y except the first row

    self.A = (yt.T @ ytm) @ torch.pinverse(ytm.T @ ytm) # calculates kinematic state model
    self.W = (yt - (ytm @ self.A.T)).T @ (yt - (ytm @ self.A.T)) / (yt.shape[0] - 1) # covariance/noise for state model
    self.C = (x.T @ y) @ torch.pinverse(y.T @ y) # calculates neural observation model
    self.Q = (x - (y @ self.C.T)).T @ (x - (y @ self.C.T)) / yt.shape[0] # covariance/noise for obs model

    self.At = self.A.T 
    self.Ct = self.C.T
  
  def predict(self, x, K_rnn,start_y=None, return_tensor=True): # actual heavy lifting of the model run
    y_predictions=[]
    if start_y is not None:
        self.y_pred = start_y
    elif hasattr(self, 'self.prior'):
        self.y_pred = self.prior
    
    for seq in tqdm(range(x.shape[0])): #100*380
        #K_rnn.unsqueeze(0)
        K_rnn1=K_rnn[seq]
        
        K_rnn1 = K_rnn1.view(4,95)#reshaping into matrix   (4*95)
        #x = x.view((x.shape[0], -1))
        
        
        yt = (self.y_pred).float() @ self.At          
        # Use RNN-predicted Kalman Gain for state update
        self.y_pred = (yt.T + K_rnn1 @ (x[seq:seq+1].mT - self.C @ yt.T)).T
        y_predictions.append(self.y_pred)
        #self.y_pred = self.y_pred.detach()
        
    self.prior=self.y_pred.detach()            
    y_predictions_tensor = torch.cat(y_predictions, dim=0)   
    # Ensure the function returns the accumulated predictions tensor
    if return_tensor:
        return y_predictions_tensor  # Returns the tensor of all predictions
    else:
        return y_predictions  # Returns the list of predictions if return_tensor is False  



rnn_model_instance = LSTMs(input_size=95, hidden_size=128, num_layers=2, gain_size=380)


kf = KalmanFilter(rnn_model_instance, append_ones_y=False)  
kf.train(X_train_tensor, y_train_tensor)

##RNN training
optimizer = optim.Adam(rnn_model_instance.parameters(), lr=0.0001)
criterion = nn.MSELoss()
num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0    
      
    for i, (inputs, targets) in enumerate(train_loader):
        inputs1=inputs.unsqueeze(0)
       # input_r = inputs1.repeat(100, 1, 1)
        #inputs2=inputs.unsqueeze(1)
        rnn_model_instance.initialize_hidden_state(inputs1)    
        print(f"Batch {i+1}/{len(train_loader)} with input size {inputs.size()} and target size {targets.size()}")
        optimizer.zero_grad() #reset the gradient
        # RNN predicts Kalman Gain based on current observations
        K_rnn = rnn_model_instance(inputs1).squeeze()
        pred_state = kf.predict(inputs,K_rnn)
        # Kalman Filter prediction and update steps
        loss = criterion(pred_state.T, targets.T)
        loss.backward(retain_graph=True) 
        torch.nn.utils.clip_grad_norm_(rnn_model_instance.parameters(), max_norm=1.0)
        optimizer.step()  
    total_loss = total_loss + loss.item()
          
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')


##RNN testing
rnn_model_instance.eval()  # Set the RNN to evaluation mode
total_test_loss = 0
all_predictions = []
with torch.no_grad():
    
    for i, (inputs,targets) in enumerate(test_loader):
        inputs1=inputs.unsqueeze(0)
       # input_r = inputs1.repeat(100, 1, 1)
        rnn_model_instance.initialize_hidden_state(inputs1)    
        print(f"Batch {i+1}/{len(test_loader)} with input size {inputs.size()} and target size {targets.size()}")
        K_rnn = rnn_model_instance(inputs1).squeeze()
        pred_state = kf.predict(inputs,K_rnn)  
        
        # Kalman Filter prediction and update steps
        loss = criterion(pred_state.T, targets.T)
        all_predictions.append(pred_state.T.cpu().numpy())   
    #print(len(all_predictions))
    total_loss = total_loss + loss.item()      
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(test_loader)}')  


all_predictions_array = np.concatenate(all_predictions, axis=1)
print(all_predictions_array.shape)
all_predictions_array=all_predictions_array.T
print(all_predictions_array.shape)
print('ytest',y_test)
# compute MSE between predictions and ground truth    
kf_mse = mse(y_test, all_predictions_array)
print("Kalman Filter MSE Possition X :",kf_mse[0])
print("Kalman Filter MSE Possition Y :",kf_mse[1])
print("Kalman Filter MSE Velocity X :",kf_mse[2])
print("Kalman Filter MSE Velocity Y :",kf_mse[3])

# compute pearson correlation
kf_corr = corr(y_test, all_predictions_array)
print("Kalman Filter Correlation Possition X :",kf_corr[0][0])
print("Kalman Filter Correlation Possition Y :",kf_corr[1][0])
print("Kalman Filter Correlation Velocity X :",kf_corr[2][0])
print("Kalman Filter Correlation Velocity Y :",kf_corr[3][0])
   
#Plotting
#actual positions
#predicted positions
def plot_all(truth, pred):
  fig, axs = plt.subplots(2, 2)
  axs = axs.flatten()
  for i in range(4):
    axs[i].plot(truth[:50, i], label=f'Ground truth')
    axs[i].plot(pred[:50, i], label=f'Prediction')
    axs[i].set_title(f'X_{i+1}')
    axs[i].legend()
  fig.suptitle(f"Ground Truth vs. Prediction for LSTM-RNN")
  plt.tight_layout()
  plt.show()

plot_all(y_test, all_predictions_array)    
   
   


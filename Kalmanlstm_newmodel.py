###THIS IS KALMAN LSTM

from torch import linalg as LA
import torch.optim as optim
import torch
torch.autograd.set_detect_anomaly(True)
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
import random
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import StepLR
torch.autograd.set_detect_anomaly(True)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
standard_scaler = StandardScaler()



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
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.observation_dim=95
        self.state_dim=4
        self.A = nn.Parameter(torch.ones(self.state_dim, self.state_dim))
        self.C = nn.Parameter(torch.ones(self.observation_dim, self.state_dim))
        self.K = nn.Parameter(torch.ones(self.observation_dim, self.state_dim))
        # Set shape attributes
        self.A_shape = self.A.shape  # Store the shape of A
        self.C_shape = self.C.shape  # Store the shape of C
        self.K_shape = self.K.shape  # Store the shape of C
        A_flat, C_flat,K_flat = self.A.view(-1), self.C.view(-1), self.K.view(-1)
        # Calculate combined dimensions for parameter processor
        self.combined_dim = A_flat.numel() + C_flat.numel() + K_flat.numel()
        self.hidden_dim = 2000
        self.num_layers = 4
        #self.param_processor = None  # This will be dynamically initialized
        self.lstm = nn.LSTM(input_size=95, 
                                hidden_size=self.hidden_dim,
                                num_layers=self.num_layers,
                                batch_first=True)
        # Fully connected layer that maps from hidden state space to output space
        self.fc = nn.Linear(self.hidden_dim, self.combined_dim)

    # Initialize LSTM
    def initialize_hidden_state(self,x):
        
        combinedinput = x
       
        inputs1=combinedinput.unsqueeze(0)
        
        # Calculate combined dimensions for parameter processor
        self.combinedinputsize = combinedinput.shape[1] 
        # Initialize hidden state with zeros
        self.h0 = torch.zeros(self.num_layers, inputs1.size(0), self.hidden_dim).to(inputs1.device)
        # Initialize cell state
        self.c0 = torch.zeros(self.num_layers, inputs1.size(0), self.hidden_dim).to(inputs1.device)

        
    def forward(self, x):

        combinedinput = x

        inputs1=combinedinput.unsqueeze(0)
        # Calculate combined dimensions for parameter processor
        self.combinedinputsize = combinedinput.shape[1] 
        out, (self.h0,self.c0) = self.lstm(inputs1, (self.h0.detach(), self.c0.detach()))
        out = self.fc(out)
        return out
    
df_regression = pd.read_csv("regression_dataset.csv")
y = df_regression.iloc[:, :4].values  # States: Position and Velocity
X = df_regression.iloc[:, 4:].values  # Observations: Channel Data
##yscaler
#position_scale = 10
#velocity_scale = 100
#scaling_factors = [position_scale, position_scale, velocity_scale, velocity_scale]
#y=y*scaling_factors

X_norm = np.empty_like(X)
# Apply NORM transformation to each column
for i in range(X.shape[1]):
    X_norm[:, i] = ((X[:, i]-np.mean(X[:, i]))/np.std(X[:, i]))
X_norm_tensor = torch.tensor(X_norm, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

if torch.isnan(X_norm_tensor).any() or torch.isinf(X_norm_tensor).any():
    raise ValueError("Input features contain NaN or Inf values.")

if torch.isnan(y_tensor).any() or torch.isinf(y_tensor).any():
    raise ValueError("Target features contain NaN or Inf values.")


# Split the data into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_norm, y, test_size=0.2,  random_state=None, shuffle=False)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train1, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train1, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test1, dtype=torch.float32)##add X_noisy here if wanna test with noise
y_test_tensor = torch.tensor(y_test1, dtype=torch.float32)

print('Xtrain:',X_train_tensor.size())
print('Ytrain:',y_train_tensor.size())
# Create data loaders
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
test_loader = DataLoader(test_data, batch_size=50, shuffle=False)

##KALMAN CLASS
class KalmanFilter():
  def __init__(self, append_ones_y=False, device='cpu'):
    self.A, self.C = None, None # state model, obs model, state noise, obs noise
    self.At, self.Ct = None, None # useful transposes
    self.append_ones_y = append_ones_y
    self.device = device
    #self.rnn_model_instance = rnn_model_instance
    self.y_pred = torch.zeros(1, 4)
    self.prior= torch.zeros(1, 4)
    

  def predict(self, x, y,kalmanparams, shapes, start_y=None, return_tensor=True):
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN values in x")
        if torch.isnan(kalmanparams).any():
            raise ValueError("Input contains NaN values in kalmanparams")
        
        y_predictions=[]    
        #x = x.view((x.shape[0], -1)) # reshapes x into a 2D tensor with the same number of rows    
        
        for t in range(kalmanparams.shape[0]): # iterate through every timestep
            #reshaping from flat to original shapes
            A_shape, C_shape, K_shape = shapes
            split_sizes = [np.prod(A_shape),  np.prod(C_shape),  np.prod(K_shape)]
            if sum(split_sizes) != kalmanparams.shape[1]:
                raise ValueError("Sum of split sizes does not match the total number of elements in kalmanparams.")
            A_flat, C_flat, K_flat = torch.split(kalmanparams[t], split_sizes)

            A = A_flat.view(A_shape)  
            At=A.T 
            C = C_flat.view(C_shape) 
            K = K_flat.view(K_shape) 
            K_scaled = K.clone()  # Make a copy of K
            K_scaled[:2] = K_scaled[:2]*100
            K_scaled[2:] = K_scaled[2:]*100
            xt = x[t].unsqueeze(1)  # Correctly reshape x[t] to column vector
            
            yt = (self.y_pred).float() @ At          
            # Use RNN-predicted Kalman Gain for state update
            #self.y_pred = (yt.T + K @ (x[t:t+1].mT - self.C @ yt.T)).T
            self.y_pred = yt.T + K_scaled.T @ (xt - C @ yt.T)
            y_predictions.append(self.y_pred)
            self.y_pred = self.y_pred.detach()
            self.y_pred = self.y_pred.T
            #print(f"Iteration {t}") 
            #print(f"K shape: {K_scaled.shape}, (xt - C @ yt.T).shape: {(xt - C @ yt.T).shape}")
            #self.y_pred = yt.T + K @ (xt - C @ yt.T) 
            #self.y_pred=self.y_pred.T
            #UPDATE
            #Pt =(torch.from_numpy(np.eye(Pt.shape[0])) - K_scaled @ C).float() @ Pt # update error covariance
            #y_predictions.append(self.y_pred)
          
       
        #self.prior=self.y_pred.detach()
        y_predictions_tensor = torch.cat(y_predictions, dim=1)
        if torch.isnan(y_predictions_tensor).any():
            raise ValueError("Input contains NaN values in predictions")

        # Ensure the function returns the accumulated predictions tensor
        if return_tensor:
            return y_predictions_tensor  # Returns the tensor of all predictions
        else:
            return y_predictions  # Returns the list of predictions if return_tensor is False


kf = KalmanFilter()
rnn_model_instance = NeuralNetwork()
##ANN training
#lr=0.0001 -WORKING
optimizer = optim.Adam(rnn_model_instance.parameters(), lr=0.0001)      
criterion = nn.MSELoss()
all_predictions11 = []
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0

    for i, (inputs, targets) in enumerate(train_loader):
        rnn_model_instance.initialize_hidden_state(inputs)  
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Batch {i+1}/{len(train_loader)}")
        optimizer.zero_grad() #reset the gradient

        kalmanparams = rnn_model_instance(inputs).squeeze()
        shapes = [rnn_model_instance.A_shape,  rnn_model_instance.C_shape,rnn_model_instance.K_shape]
        predicted_states = kf.predict(inputs,targets, kalmanparams, shapes)
        if torch.isnan(predicted_states).any():
            print("NaN values detected in predicted_states before loss calculation")
            continue
        predicted_states=predicted_states.T
        loss = criterion(predicted_states, targets)
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(rnn_model_instance.parameters(), max_norm=1.0)
        print(f'Loss: {loss}')
        optimizer.step()
        #all_predictions11.append(predicted_states.T.cpu().numpy())
        all_predictions11.append(predicted_states.detach().cpu().numpy())

#   scheduler.step()
    total_loss = total_loss + loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

### Save the model   
#torch.save(rnn_model_instance, 'NNKalman1.pth')
torch.save(rnn_model_instance, 'NNKalman1.pth')
print("Model saved to NNKalman1.pth")


##RNN testing
#model = NeuralNetwork()  # Instantiate your model
model_path = 'NNKalman1.pth'  
model = torch.load(model_path)

#model = torch.load('NNKalman1.pth')
model.eval()  # Set the RNN to evaluation mode
total_test_loss = 0
all_predictions = []
with torch.no_grad():

    for i, (inputs,targets) in enumerate(test_loader):

        print(f"Batch {i+1}/{len(test_loader)}")
        kalmanparams = rnn_model_instance(inputs).squeeze()
        shapes = [rnn_model_instance.A_shape,  rnn_model_instance.C_shape,rnn_model_instance.K_shape]
        predicted_states = kf.predict(inputs,targets, kalmanparams, shapes)
        
      
        # Kalman Filter prediction and update steps
        loss = criterion(predicted_states.T, targets)
        all_predictions.append(predicted_states.T.cpu().numpy())
    #print(len(all_predictions))
    total_test_loss = total_test_loss + loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_test_loss / len(test_loader)}')


all_predictions_array = np.concatenate(all_predictions, axis=0)
print(all_predictions_array.shape)
#all_predictions_array=all_predictions_array.T
print(all_predictions_array.shape)
print('ytest',y_test1.shape)
# compute MSE between predictions and ground truth
kf_mse = mse(y_test1, all_predictions_array)
print("Kalman Filter MSE Possition X :",kf_mse[0])
print("Kalman Filter MSE Possition Y :",kf_mse[1])
print("Kalman Filter MSE Velocity X :",kf_mse[2])
print("Kalman Filter MSE Velocity Y :",kf_mse[3])

# compute pearson correlation
kf_corr = corr(y_test1, all_predictions_array)
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
    axs[i].plot(truth[:200, i], label=f'Ground truth')
    axs[i].plot(pred[:200, i], label=f'Prediction')
    axs[i].set_title(f'X_{i+1}')
    axs[i].legend()
  fig.suptitle(f"Ground Truth vs. Prediction for Kalman-LSTM(Improvised Model)")
  plt.tight_layout()
  plt.show()

plot_all(y_test1, all_predictions_array)


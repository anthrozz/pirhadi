import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import datetime
mpl.use('TkAgg')
my_date = datetime.datetime.now()
current_day = my_date.day
current_hour = my_date.hour
current_min = my_date.minute
current_sec = my_date.second
strTimeofday = str(current_day) + "_" + str(current_hour) + "_" + str(current_min) + "_" + str(current_sec)
import torch.nn.functional as F
layer_node = 100
epochs = 2000
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(3, layer_node)
        self.fc2 = nn.Linear(layer_node, 2*layer_node)
        self.fc3 = nn.Linear(2*layer_node,  layer_node)
        self.fc4 = nn.Linear(layer_node, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)  # No activation for regression; use softmax for classification



import csv


def write_array_to_csv(array, filename="output.csv"):
    """
    Writes a two-dimensional array to a CSV file, ensuring each element is iterable.

    Parameters:
    - array: two-dimensional list (list of lists) or a list of floats to be written to the CSV file
    - filename: name of the CSV file (default is "output.csv")
    """
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            for row in array:
                # Ensuring each row is iterable
                if not isinstance(row, (list, tuple)):
                    row = [row]

                writer.writerow(row)

            print(f"The array has been successfully written to {filename}")

    except Exception as e:
        print(f"An error occurred while writing to {filename}: {e}")




def root_mean_square_error(y_true, y_pred):
    mse = torch.mean(torch.sum((y_true - y_pred) ** 2, axis=1) / 6)
    return torch.sqrt(mse)
    #mse = torch.mean((y_true - y_pred) ** 2)
    #return torch.sqrt(mse)


def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.sum(torch.abs(y_true - y_pred), axis=1) / 6)
    #return torch.mean(torch.abs(y_true - y_pred))


def mean_square_error(y_true, y_pred):
    #return torch.mean(torch.sum((y_true - y_pred) ** 2, axis=1) / 3)

    return torch.mean((y_true - y_pred) ** 2)





def train(modelFNN,epoch,train_loader,optimizer,train_losses):
    modelFNN.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        output = modelFNN(data)
        optimizer.zero_grad()

        #loss = mean_square_error(target,output )
        #loss= root_mean_square_error(target,output)
        loss=mean_absolute_error(target,output)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
#        if batch_idx % 500 == 0:
#            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}], Loss: {loss.item()}")

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    #print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}], Loss: {avg_train_loss}")
    return train_losses
def test(modelFNN,test_loader):
    modelFNN.eval()
    test_loss = 0
    true_values = []
    predictions = []
    test_load=[]
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = modelFNN(data)
            #test_loss += mean_square_error(outputs, targets).item()
            #test_loss += root_mean_square_error(targets, outputs).item()
            test_loss += mean_absolute_error(targets, outputs).item()
            test_load.append(test_loss)
            true_values.extend(targets.numpy().flatten())
            predictions.extend(outputs.numpy().flatten())

    test_loss /= len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}\n")
    print(f"\nTest set: Max loss: {np.max(test_load):.4f}\n")
    print(f"\nTest set: Min loss: {np.min(test_load):.4f}\n")
    print(test_load)
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, predictions, alpha=0.2)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red')  # Diagonal line
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Model Predictions vs. True Values')
    plt.show()
    return test_loss
def validate(modelFNN,val_loader, val_losses):
    modelFNN.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = modelFNN(data)
            val_loss += mean_square_error(target,output).item()  # Sum up batch loss
    # val_loss /= len(val_loader.dataset)

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)
    #print(f"\nValidation set: Average loss: {avg_val_loss:.4f}\n")
    return val_losses
csv_path1 = r"C:\Users\gursukarateke\OneDrive\Documentos\Ansoft\Project171222.aedtresults\regression\pirhadi\pirhadi_multilayer_results_corrected.csv"

df1 = pd.read_csv(csv_path1)
selected_columns = ['l1 [mm];l1_up [mm];l2 [mm];l2_up [mm];W [mm];W_up [mm];Freq [GHz];cang_deg(S(FloquetPort1:1', 'FloquetPort1:1)) [deg]']

data = df1[df1.columns].values

def createPrediction(modelFNN):
# Load dataset


    slopes = []
    groups = []
    for i in range(0, len(data), 5):
        group = data[i:i + 5]
        frequency = group[:, 6]
        phase_freq_0 = group[0, 7]
        phase_freq_1 = group[1, 7]
        phase_freq_2 = group[2, 7]
        #phase_f0 = group[1, 4]
        #reflection_phase = group[:, 4]
        dimensions = group[0, :6]
        #slope = linregress(frequency, reflection_phase)[0]

        new_member = [phase_freq_0, phase_freq_1,phase_freq_2]
        slopes.append(new_member)
        groups.append(dimensions)
    print(len(groups), len(slopes))  # 165
    #print(slopes)
    groups = np.array(groups)

    X_dataset = torch.tensor(slopes, dtype=torch.float32)
    y_dataset = torch.tensor(groups, dtype=torch.float32)
    # Split dataset
    dataset = TensorDataset(X_dataset, y_dataset)
    total_samples = len(dataset)
    train_size = int(0.85 * total_samples)
    #val_size = int(0.10 * total_samples)
    test_size = total_samples - train_size #- val_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)#
#    print(train_loader.dataset)
    #val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
#    print(val_loader.dataset)
    test_loader = DataLoader(test_dataset, batch_size=1)
#    print(test_loader.dataset)



    optimizer = optim.Adam(modelFNN.parameters(), lr=0.001, weight_decay=0.0005)
    #optimizer=optim.Adadelta(model.parameters(),lr=0.001,weight_decay=0.001)
    #optimizer=optim.RMSprop(model.parameters(),lr=0.0001,weight_decay=0.005)

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        train_losses=train(modelFNN,epoch,train_loader,optimizer,train_losses)
        #val_losses=validate(val_loader,val_losses)
    print(np.average(train_losses))
    print(np.min(train_losses))
    print(np.max(train_losses))
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.xlim(0,epochs)
    plt.ylim(0, 1)
#    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    write_array_to_csv(train_losses,strTimeofday+"_trainlosses.csv")

    if(test(modelFNN,test_loader)>2):
        print("test_fail")
        return []
    search_data = pd.read_csv(
        "C:\\Users\\gursukarateke\\OneDrive\\Documentos\\Ansoft\\three_layer.aedtresults\\new_data.csv")
    new_x = search_data[['Freq [GHz]', 'cang_deg(S(FloquetPort1:1,FloquetPort1:1)) [deg]']].values
    # groups_newdata = [search_data.iloc[i:i+3,:] for i in range(0, len(search_data), 3)]
    groups_newdata = []
    slopes_new = []
    predictions = []
    for i in range(0, len(new_x), 3):
        group = new_x[i:i + 3]
        frequency_new = group[:, 0]  # Select the first column as x
        reflection_phase_new = group[:, 1]  # Select the second column as y
        phase_f0_new = group[1, 1]
        phase_freq_0_new = group[0, 1]
        phase_freq_1_new = group[1, 1]
        phase_freq_2_new = group[2, 1]
        slope_new = linregress(frequency_new, reflection_phase_new)[0]
        new_member = [phase_freq_0_new, phase_freq_1_new,phase_freq_2_new]
        slopes_new.append(new_member)
    print(slopes_new)
    slopes_new = np.array(slopes_new)

    #    slopes_new = np.reshape(slopes_new, (slopes_new.shape[0], -1))
    modelFNN.eval()
    for i in range(0, len(slopes_new)):
        predictions.append(modelFNN(torch.tensor(slopes_new[i], dtype=torch.float32))[:6].detach().numpy())

    # new_predictions = predictions[:3].detach().numpy()
    length = len(predictions)
    a = np.array(predictions)
    print(a,length)

    if (np.any((a > 13.8) | (a<0.1))):
        print(np.where((a > 13.8)| (a<0.1)))
        print("fail")
        return []

    print (test_size)




    csv_file = strTimeofday + '_train_losses.csv'


    # Specify the CSV file path
    csv_file = strTimeofday + '_output.csv'
    model_save = strTimeofday + '_output_model'
    torch.save(modelFNN.state_dict(), model_save)

    # Write the data to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(['*', 'a1', 'a2', 'a3'])  # Modify column names as needed

        # Write the data rows
        for i, row in enumerate(predictions):
            modified_row = [str(value) + 'mm' for value in row]
            writer.writerow([i + 1] + modified_row)
    return a


trial=0

while (True):

    modelFNN=FNN()
    predictions = createPrediction(modelFNN)
    if(predictions==[]):
        trial+=1
        print("fail_trial:", trial)
        layer_node += (int)(0.1*layer_node)
        epochs+=(int)(0.05*epochs)
        print(layer_node,epochs)
        if (trial==20):
            print('epic fail')
            break
    else:
        print("success:",trial)
        break
# Assuming your 2D array is called 'data'

# Determine the length of the array


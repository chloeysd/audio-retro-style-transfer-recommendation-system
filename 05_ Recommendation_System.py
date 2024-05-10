import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms
from src.audio_folder_dataset import AudioFolder
from src.audio_folder_collate_fn import collate_audio_folder_batch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import pandas as pd

class M5(nn.Module):
    def __init__(self, n_input=2, n_output=11, stride=16, n_channel=32):
        super().__init__()
        #first convolutional layer
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        #batch normalization
        self.bn1 = nn.BatchNorm1d(n_channel)
        #max pooling
        self.pool1 = nn.MaxPool1d(4, 4)
        #second convolutional layer
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        #batch normalization
        self.bn2 = nn.BatchNorm1d(n_channel)
        #max pooling
        self.pool2 = nn.MaxPool1d(4)
        #third convolutional layer
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        #batch normalization
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        #max pooling
        self.pool3 = nn.MaxPool1d(4)
        #fully connected layer
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        #reLU activation function after batch normalization
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x)) 
        x = self.pool3(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        #squeeze the second dimension
        x = torch.squeeze(x, 1) 
        return x

#load the trained model
model = M5() 
model.load_state_dict(torch.load('best_audio_classifier.pt'))
#set the model to evaluation mode
model.eval() 

#load the trained model
model = M5() 
model.load_state_dict(torch.load('best_audio_classifier.pt'))
#set the model to evaluation mode
model.eval() 

#audio preprocessing function
def preprocess_audio(file_path):
    #load the audio file and get waveform and sample rate using torchaudio，ref:https://stackoverflow.com/questions/71108331/torchaudio-load-audio-with-specific-sampling-rate
    waveform, sample_rate = torchaudio.load(file_path)
    #perform appropriate transformations based on model requirements
    new_sample_rate = 8000
    transform = transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    waveform = transform(waveform)
    #ensure the waveform has two channels
    if waveform.shape[0] == 1:
        waveform = torch.cat([waveform, waveform], dim=0) 
    return waveform

def classify_audio(file_path):
    #disables gradient calculation to save memory and computations,ref:https://discuss.pytorch.org/t/question-about-use-of-torch-max-function-to-calculate-accuracy/187500
    with torch.no_grad():
        #initialize variables for correctness and total predictions
        total = 0
        #load and preprocess the audio file
        waveform = preprocess_audio(file_path)
        #add a batch dimension to match model input
        waveform = waveform.unsqueeze(0)
        #pass the processed waveform through the model
        outputs = model(waveform)
        #find the class with the highest probability
        _, predicted = torch.max(outputs, 1)
        #increment total predictions
        total += 1
        #return the predicted class index as an integer
        return predicted.item()

#define GUI functions,Ref:https://stackoverflow.com/questions/54785138/how-to-access-a-desired-path-with-filedialog-askopenfilename-in-tkinter
def audio_file():
    file_path = filedialog.askopenfilename()
    #set the text of the genre_label to display the selected file's path,ref:https://www.geeksforgeeks.org/how-to-change-the-tkinter-label-text/
    genre_label.config(text=f"Selected File: {file_path}")

#load the dataset
data = pd.read_csv('data/features_3_sec.csv')
#create a label encoder,ref:https://stackoverflow.com/questions/61467312/application-function-labelencoder-fit-transform-in-python?rq=3
label_encoder = LabelEncoder()
#encode the labels
data['label'] = label_encoder.fit_transform(data['label'])

#split data into features and labels,ref:https://www.kaggle.com/code/andradaolteanu/work-w-audio-data-visualise-classify-recommend/notebook#Introduction
y = data['label']
#features of the dataset, excluding the label column
X = data.loc[:, data.columns != 'label']
#assuming 'filename' is a column with audio file names
X = X.drop(columns=['filename']) 
#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#train a classifier,initialize the KNN classifier with 19 neighbors
knn = KNeighborsClassifier(n_neighbors=19)
#fit the classifier to the training data
knn.fit(X_train, y_train)

#load the dataset, setting 'filename' as the index column
data = pd.read_csv(f'data/features_30_sec.csv', index_col='filename')
#create a DataFrame with just the labels for each track
labels = data[['label']]
#remove 'length' and 'label' columns as they are not features
data = data.drop(columns=['length','label'])

#standardize the feature values
data_scaled=preprocessing.scale(data)
#compute similarity as a 2D numpy array
similarity = cosine_similarity(data_scaled)
#convert to DataFrame for easier manipulation
sim_df_labels = pd.DataFrame(similarity)
#set the row index to be the song filenames
sim_df_names = sim_df_labels.set_index(labels.index)
#set the column names to be the song filenames, making it easy to identify songs
sim_df_names.columns = labels.index

def find_similar_songs(name):
    # Find songs most similar to another song
    series = sim_df_names[name].sort_values(ascending=False)
    # Remove cosine similarity == 1 (songs will always have the best match with themselves)
    series = series.drop(name)
    # Return the 5 top matches as a list
    return series.head(5).index.tolist()

def classify_and_recommend():
    #retrieve the file path selected by the user in the GUI,ref:https://stackoverflow.com/questions/63871376/tkinter-widget-cget-variable
    file_path = genre_label.cget("text").split(": ")[1]
     #check if a file has been selected  
    if file_path: 
        #classify the selected audio file's genre using the pre-trained model
        predicted_genre_index = classify_audio(file_path)
        #use LabelEncoder to reverse transform the predicted index to get the genre name,ref:https://stackoverflow.com/questions/52870022/inverse-transform-method-labelencoder
        predicted_genre_label = label_encoder.inverse_transform([predicted_genre_index])[0]  
        #update the GUI to display the predicted genre
        predicted_genre.config(text=f"Predicted Genre: {predicted_genre_label}")  

        #extract the song name from the file path for finding similar songs
        song_name = file_path.split('/')[-1] 
        #find and return a list of similar songs based on the song name
        similar_songs = find_similar_songs(song_name) 
        #display the list of recommended similar songs in a new window
        show(root, similar_songs) 

## Class for displaying recommendations dialog,ref:https://stackoverflow.com/questions/37219191/how-to-return-the-selected-items-of-a-listbox-when-using-wait-window-in-tkinte
class MyDialog(object):
    def __init__(self, parent, similar_songs):
        #initialize a top-level window for recommendations
        self.toplevel = tk.Toplevel(parent)
        #set the title of the top-level window
        self.toplevel.title("Similar Song Recommendations")

        #create a listbox to display recommended songs
        self.listbox = tk.Listbox(self.toplevel)
        #pack the listbox to fill and expand within the window
        self.listbox.pack(side="top", fill="x")

        #insert similar songs into the listbox
        for song in similar_songs:
            #insert each song at the end of the listbox
            self.listbox.insert(tk.END, song)

#function to display recommendations
def show(parent, similar_songs):
    #create RecommendationsDialog instance with parent window and similar songs
    MyDialog(parent, similar_songs)

if __name__ == "__main__":
    #create the main GUI window
    root = tk.Tk()
    #set the title of the main window
    root.title("Music Genre Classifier & Recommender")

    #button to browse files
    browse_button = tk.Button(root, text="Browse", command=audio_file)
    #pack the browse button into the main window
    browse_button.pack()

    #label to display selected file
    genre_label = tk.Label(root, text="Selected Audio: ")
    #pack the label to display selected file into the main window
    genre_label.pack()

    #button to classify and recommend
    classify_button = tk.Button(root, text="Classify & Recommend", command=lambda: classify_and_recommend())
    #pack the classify and recommend button into the main window
    classify_button.pack()

    #label to display predicted genre
    predicted_genre = tk.Label(root, text="Predicted Genre: ")
    #pack the label to display predicted genre into the main window
    predicted_genre.pack()

    #run the GUI
    root.mainloop()

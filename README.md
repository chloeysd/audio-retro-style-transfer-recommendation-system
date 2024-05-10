# Project Overview
This project explores audio style transfer and develops a CNN-based genre recognition music recommendation system. It leverages the RAVE pre-trained model to generate stylized audio and employs a CNN to transfer styles to user-uploaded audio. The project investigates the feasibility of audio style transfer and evaluates the characteristics of audio segments processed by the RAVE model versus those subjected to style transfer. Various parameterizations and optimization strategies were explored. The M5 model was trained with an audio dataset to implement genre recognition and develop a music recommendation system, all accessible through a GUI.

## Environment Setup 
Ensure the following requirements are met to run this project:
- **Python 3.x:** Main programming language.
- **torch and torchaudio:** For neural network model training and audio processing.
- **pandas:** For data manipulation and analysis.
- **sklearn:** For machine learning utilities.
- **matplotlib and seaborn:** For data visualization.
- **Tkinter:** For the graphical user interface.

## Project Structure 
The project directory structure is as follows:
- `rave_models/`: Stores the required pre-trained models.
- `src/`: Contains source code files and utility scripts.
- `generated/`: Includes audio files generated by the model.
  - `Jazz.wav`: Original audio uploaded.
  - `generated_audio.wav`: Vintage style audio using the RAVE model.
  - `modified_audio.wav`: Audio processed for vintage timbre using the RAVE model.
  - `output_audio_1.wav`: Style transfer audio using MSE & Adam.
  - `output_audio_2.wav`: Style transfer audio using MAE & SGD.
- `data/`: Contains datasets and data tables from Kaggle. Download the GTZAN Dataset - Music Genre Classification from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and move the `genres_original` folder into this directory before running the notebooks.

## Jupyter Notebooks
- `01_RAVE_Vintage_Audio_Generate_and_Characteristics.ipynb`: Generates and characterizes vintage-style audio using the RAVE model.
- `02_CNN_Audio_Style_Transform.ipynb`: Applies a CNN for audio style transformation.
- `03_RAVE_Vintage_Audio_Processing_and_Characteristics.ipynb`: Processes retro audio and compares features with the RAVE model.
- `04_CNN_Audio_Classification_model.ipynb`: Implements a CNN model for audio genre classification.
- `05_Recommendation_System.py`: Script for creating a music recommendation system based on the audio genre recognition model.

## Model Files
- `best_audio_classifier.pt`: A PyTorch model file of the audio classifier, trained with a fresh dataset.

## Running the Project
- **Step 1:** Generate vintage audio and analyze Mel spectrograms, spectral centroids, and chromagrams.
- **Step 2:** Perform audio style transfer using different loss functions and optimizers.
- **Step 3:** Process original audio with a vintage effect and compare feature maps.
- **Step 4:** Train the M5 model for genre recognition and test its performance.
- **Step 5:** Develop a music recommendation system based on genre recognition with a GUI for user interaction.

## Acknowledgments 
This code was adapted from classroom content, online resources, and consultations with a large language model for error resolution. Proper citations are provided within the code comments.

Project Organization

Inspired by: https://mengdong.github.io/2018/05/28/Python-Machine-Learning-Project-Template/

image
CRISP-DM Process Model

https://web.archive.org/web/20220401041957/https://www.the-modeling-agency.com/crisp-dm.pdf
Business Understanding
Objectives

Create an automatic music genre recognition (MGR) system and web portal for it.

    Build a dataset containg song metadata and their various genres and spectrograph info.
    Develop a pipeline to import audio clips from datasets
    Create a web app front-end (can run on desktop).
    Host the program as a web server.
    Develop a program to run a user-submitted audio clip against the model and print results witha ccuracy metrics
    Content based recommender system for music similar to audio clip
        Train a neural network

Situation

The user will enter a song clip, then receive a formatted top-n list of genres sorted by confidence value in descending order.
Data Mining Goals
Technologies, Libraries, Tools

    Poetry: Project Dependency Management Tool
    Pytorch: Machine Learning
    Librosa: Audio and Music Proccessing
    Matplotlib: Data Visualization
    Numpy: General purpose array-processing
    Pandas: Data analysis and manipulation tool
    Morgan: logger
    Pytest: Test framework

Data Understanding
Datasets

    GTZAN Genre Collection by G. Tzanetakis and P. Cook
    Million Song Dataset by LabROSA and The Echonest

Data Description
Exploratory Data Analysis
Data Quality Analysis
Data Preparation
Select Data
Clean Data
Construct Data
Integrate Data
Format Data
Modeling
Select Modeling Techniques
Generate Test Design
Build Model
Assess Model
Evaluation
Evaluate Results
Review Process
Next Steps
Deployment
Plan Deployment
Plan Monitoring and Maintenance
Produce Final Report
Review Project

# 🧠 Intelligent Anomaly Detection in Network Traffic
This project detects anomalous patterns in network traffic using machine learning techniques, with the goal of identifying potential threats such as intrusions and unusual flows in real-time and act upon it.

## 🚀 Features
- 📥 Ingests network flow data (e.g., from Zeek, pcap, NetFlow)
- 🧮 Extracts statistical features from flows or sessions
- 🤖 Applies machine learning models (Isolation Forest [unsupervised deep learning], Autoencoder) to detect anomalies & take action
- 🔔 Flags high-risk traffic for alerting or review or even block network traffic
- 📊 Web dashboard for real-time anomaly monitoring and feedback

## 📈 How It Works
- Capture or load network traffic data from dataset.
- Preprocess and extract features like bytes, packets, durations, etc.
- Normalise data to ensure consistent scale.
- Run anomaly detection models on feature data.
- Score traffic and flag anything above the threshold as an anomaly.
- Display results in a web dashboard or write to logs for review.
- Prompt the user on how to take action 

## 📂 Project Structure
├── data/               # Raw and preprocessed flow data

├── models/             # Trained ML models

├── src/

│   ├── ingest.py       # Network data parsing & loading

│   ├── features.py     # Feature extraction

│   ├── detect.py       # Anomaly detection logic

│   └── train.py        # Model training script

├── dashboard/          # Frontend code 

├── README.md

└── requirements.txt

## 📊 Sample Output
Timestamp	Src IP	Dst IP	Protocol	Score	Anomaly

2025-08-05 14:00	10.0.0.2	192.168.1.5	TCP	0.97	✅ Yes

2025-08-05 14:01	10.0.0.3	192.168.1.10	UDP	0.23	❌ No

## 📚 Datasets
- CIC-IDS2017: https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Set up
ensure you have git lfs on your system

if you're on mac, in the terminal run 

brew install git-lfs

run git lfs install 

git lfs pull

## Flowchart link: https://app.eraser.io/workspace/83NC47DnjOiRg5gxlERZ?origin=share
Animal NER + Image Classification Pipeline
 Project Overview
This project builds a machine learning pipeline that combines:

NER (Named Entity Recognition): extracts animal names from text.

Image Classification: identifies animals in images.

Pipeline: compares the text and image results, returning True if they match and False otherwise.

The solution includes training scripts, inference scripts, a final pipeline, and a Jupyter Notebook demo with EDA and edge cases.

 Project Structure
Код
project/
│
├── animal_data/                # Dataset with ≥10 animal classes
│   ├── Cat/
│   ├── Dog/
│   ├── Bear/
│   └── ...
│
├── ner_model/                  # Saved NER model
├── classifier_model/           # Saved image classifier (ResNet18)
│   ├── animal_classifier.pth
│   ├── classes.txt
│
├── train_ner.py                # Train NER model
├── infer_ner.py                # Inference with NER
├── train_classifier.py         # Train image classifier
├── infer_classifier.py         # Inference with classifier
├── pipeline.py                 # Final pipeline script
└── Demo.ipynb                  # Jupyter Notebook demo
 Requirements
Install dependencies from requirements.txt:

bash
pip install -r requirements.txt
requirements.txt (pinned versions)
txt
numpy==1.26.4
pandas==2.2.1
matplotlib==3.8.3
seaborn==0.13.2

torch==2.2.1
torchvision==0.17.1
torchaudio==2.2.1

transformers==4.39.3
tokenizers==0.15.2
huggingface-hub==0.22.2

jupyter==1.0.0
notebook==7.1.2
ipykernel==6.29.3
ipywidgets==8.1.2
tqdm==4.66.2

Pillow==10.2.0
scikit-learn==1.4.1.post1
 If you use GPU with CUDA, install the correct PyTorch build from pytorch.org  (pytorch.org in Bing).

🚀 Usage
1. Train NER
bash
py train_ner.py --train_file data/train.txt --val_file data/val.txt --output_dir ner_model --epochs 5 --batch_size 16
2. Inference with NER
bash
py infer_ner.py --model_dir ner_model --text "There is a cat in the picture."
3. Train Image Classifier (ResNet18)
bash
py train_classifier.py --data_dir animal_data --output_dir classifier_model --epochs 10 --batch_size 32
4. Inference with Classifier
bash
py infer_classifier.py --model_dir classifier_model --image_path test_images/Cat_1_2.jpg
5. Run the Final Pipeline
bash
py pipeline.py --ner_model_dir "C:/Users/Artem/ner_model" --clf_model_dir "C:/Users/Artem/classifier_model" --text "There is a cat in the picture." --image_path "C:/Users/Artem/test_images/Cat_1_2.jpg"
Output:

Код
Text animal: cat
Image animal: Cat
Pipeline result: True
 Demo Notebook
Open Demo.ipynb in Jupyter Notebook to see:

Dataset EDA (class distribution, sample images).

NER examples.

Image classifier examples.

Pipeline demonstration.

Edge cases:

Text without animals → False.

Text and image mismatch → False.

Multiple animals in text → configurable logic (True if at least one matches).

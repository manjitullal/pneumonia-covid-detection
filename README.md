# pneumonia-covid-detection
Project on detecting Covid/Pneumonia using Xray Images

Each year, pneumonia affects about 450 million people globally (7% of the population) and results in about 4 million deaths. A chest X-ray is often used to diagnose pneumonia. 
We propose a Deep Learning model trained on X ray images which will aid the doctors in the diagnosis.

Dataset 

The main dataset we would like to use for this project is the CovidX dataset, which combines multiple open source Covid datasets.

https://github.com/lindawangg/COVID-Net1

We also plan on using a second, larger dataset: MIMIC-CXR or NIH Chest X-Ray

https://mimic-cxr.mit.edu/about/data/

Modeling 

a)  Train a Convolutional Neural Network (CNN) on COVIDX dataset.
    Use transfer learning from popular architectures such as ResNet, Inception Net.
    
b)  Use CNN visualization techniques to determine how CNNs are making decisions.

c)  Train a CNN on the larger dataset, and use transfer learning to finetune this model to COVID-19 detection.

d)  Text summarization/Report generation for the X ray image.

c)  Localization of problematic areas.


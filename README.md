# Applied Machine Learning Systems ELEC0134 (22/23) -  Assignment

## The organization of this project is as follows:

- AMLS_22-23_SN12345678
  - A1
    - gender_detetion.py
    - gender_detection_model_0.pth
    - ...
    - gender_detection_model_9.pth
  - A2
    - emotion_detection.py
    - emotion_detection_model_0.pth
    - ...
    - emotion_detection_model_9.pth
  - B1
    - face_shape_recognition.py
    - face_shape_recognition_model_0.pth
    - ...
    - face_shape_recognition_model_9.pth
  - B2
    - eye_color_recognition.py
    - eye_color_recognition_model_0.pth
    - ...
    - eye_color_recognition_model_9.pth
  - Datasets
    - cartoon_set
    - celeba
    - cartoon_set_test
    - celeba_test
  - utils.py
  - main.py
  - requirements.txt
  - README.md

## The role of each file is as follows:

- The requirements.txt file provides the packages required to run your code.The installation process is as follows:

```
#  Step 1，Create a new Conda environment using Python 3.9:
conda create -n cv python=3.9

#  Step 2， Activate the newly created environment:
conda activate cv

# Step 3: Install the libraries specified in requirements.txt:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

- The utils.py file provides common definitions for the " .py" files under the ‘A1’, ‘A2’, ‘B1’, and ‘B2’ folders, such as loading data, image preprocessing, defining, training, and evaluating models.
- The " .py" files under the ‘A1’, ‘A2’, ‘B1’, and ‘B2’ folders are mainly used for training models and saving model parameters. Each model is trained 10 times, and the trained parameters are saved to the respective ".pth" files under each folder.
- The main.py file directly loads the pre-trained models and outputs the corresponding prediction results for each task.
- The Datasets folder is used to store training and testing images.
  
  ## The packages required to run the code are as follows:
- os
- pandas
- Image
- Dataset
- torch
- torch.nn
- torch.optim
- transforms
- models
- numpy
- DataLoader
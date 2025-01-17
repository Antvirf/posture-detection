# Webcam-based slouch detection
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Antvirf_posture-detection&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Antvirf_posture-detection)
[![CodeQL](https://github.com/Antvirf/posture-detection/actions/workflows/codeql.yml/badge.svg)](https://github.com/Antvirf/posture-detection/actions/workflows/codeql.yml)

Have had the idea to do this for a long time, and found a good start from dcstang's repo based on openCV. Since that initial fork, I have changed the approach from haarcascade (detecting a face) to Mediapipe (detecting body landmarks) and implemented an ML model to classify the detected pose as either straight/slouching with the appropriate alerts in case of the latter.

# Installation/usage
As a basic pre-requisite, run this on a recent macOS. It would be best to have [anaconda/conda](https://docs.anaconda.com/anaconda/install/mac-os/) installed for package management. Open a terminal and cd to a directory of your choice, then follow the instructions below.

    # Install required packages
    pip install numpy pandas opencv-python mediapipe
    
    # Clone the repo
    git clone https://github.com/Antvirf/posture-detection
    
    # Once installation is complete, run the code
    python main.py

## Latest version
Landmark/pose capture still based on Mediapipe, but added a new PostureCriteriaML class inspired by [this  notebook/tutorial](https://github.com/nicknochnack/Body-Language-Decoder/blob/main/Body%20Language%20Decoder%20Tutorial.ipynb). Now, the code runs different ML models, picks the best performing one, and uses that to do predictions instead. The breach check was also changed to a 'proportion X of frames slouched over Y seconds' instead of consecutive frames to ensure that a single frame error doesn't reset the counter.

Trained the model at 3 FPS for 5 minutes slouching, 5 minutes sitting straight. The accuracy with my setup is very good at ~97% in test which makes it usable for my case. Note that the model is *probably* dependent on your setup, camera angle, and the physical size of your body - so may not work as well if you take the model as provided.


### Misc
* Original author: dcstang
* License: This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
Not to be misused for illegal, maleficient purposes.


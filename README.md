# posture-detection
# Antti's notes
Have had the idea to do this for a long time, and found a good start from dcstang's repo based on openCV. Their code implemented facial recognition based on the webcam, and the 'first pass' built directly on top of that. The 'second pass' saw a change of apprach/library to Mediapipe, which allows more precise tracking of body parts for future developments.


## Third pass
Landmark/pose capture still based on Mediapipe, but added a new PostureCriteriaML class inspired by [this  notebook/tutorial](https://github.com/nicknochnack/Body-Language-Decoder/blob/main/Body%20Language%20Decoder%20Tutorial.ipynb). Now, the code runs different ML models, picks the best performing one, and uses that to do predictions instead. Using the current settings, the model's performance in real life use is not yet that great, need to perhaps consider longer periods of training and/or adding more or different landmarks. Left some debt to the non-ML version where the functions don't work quite the same in all cases and data collection is a bit different.

Trained the model at 3 FPS for 5 minutes slouching, 5 minutes sitting straight. The accuracy with my setup is very good at ~97% in test which makes it usable for my case. Note that the model is *probably* dependent on your setup, camera angle, and the physical size of your body - so may not work as well if you take the model as provided.

## Second pass (now in main_simple.py)
This iteration is based on Google's mediapipe and loses almost all common elements with the previous code. Instead of alerting based on facial area, it now alerts based on distance between chosen body parts. The selection of criteria can be made with the class of PostureCriteria, where you can specify 2 Mediapipe bodies (as identified by their IDs) to track for each criterion, as well as specify whether a breach is considered going above or below the calibrated value.
Current setup includes:
* eye separation: Similar to area from above, can track whether your head moves closer to the screen
* shoulder separation: Same as for eyes, but shoulder based. Tracked separately as it is possible to just slouch your head (while keeping shoulders back normally)
* nose-shoulder separation: Not perfect at the moment, and set as a value where a breach is considered going below the threshold (i.e. if your nose and shoulder are closer, this would indicate worse posture). Too dependent on distance from the screen, hence the to-do including a scaling set of points

## First pass (now in main_depr.py)
* Computing 'area' of the face detected, simply taking the width/height of the rectangle
* Slouching is simply defined as when the currently detected area > threshold area, i.e. if the user's face is larger (=closer) to the camera than before
* 'Calibrating' the threshold area over the first few seconds the program is ran, by asking the user to slouch at the start
* 'Monitoring', alerting the user via macOS alert + sound if the limit is consecutively breached for over 3 seconds





### Misc
Original author: dcstang

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
Not to be misused for illegal, maleficient purposes.


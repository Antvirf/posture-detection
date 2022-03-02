import cv2
import os
import time
import math
import PoseModule as pm
# import logging
import pickle

# ML imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


"""
Mediapipe landmarks of interest
More info https://google.github.io/mediapipe/solutions/pose.html
O. nose
2. left_eye
5. right_eye
9. mouth left
10. mouth_right
11. left_shoulder
12. right_shoulder
"""


def say_stuff(stuff):
    os.system(
        """say "{}" """.format(stuff)
    )


def notify(title, text, sound, sound_only=False):
    if sound_only:
        os.system("afplay /System/Library/Sounds/Submarine.aiff")
    else:
        os.system("""
            osascript -e 'display notification "{}" with title "{}" sound name "{}"'""".format(text, title, sound))


class PostureCriterion():
    def __init__(self, in_name, in_message, in_landmarks, in_direction="above", scaler=False):
        self.name = in_name
        self.message = in_message
        self.landmarks = in_landmarks
        self.calibration_data = []  # list of dicts
        self.threshold = 0.0
        self.breach_direction = in_direction  # One of 'above' or 'below': breach when value is ABOVE or BELOW threshold

    def compute_point_distance(self, data):
        # Assume input data is a dict with just the two points
        assert isinstance(data, dict)
        x1, x2 = data[self.landmarks[0]][0], data[self.landmarks[1]][0]
        y1, y2 = data[self.landmarks[0]][1], data[self.landmarks[1]][1]
        dist = math.sqrt((
            x2 - x1)**2 + (
            y2 - y1)**2)
        return dist

    def add_calibration_data(self, data):
        assert isinstance(data, dict)
        assert isinstance(data[0], list)
        # Assume data is just a list of all the positions, not filtered - here, we only take the data points for the 2 landmarks we care about
        data_to_append = {}
        data_to_append[self.landmarks[0]] = data[self.landmarks[0]]
        data_to_append[self.landmarks[1]] = data[self.landmarks[1]]
        # Append new data
        self.calibration_data.append(data_to_append)

    def calibrate(self):
        print(self.calibration_data)
        running_total = 0.0
        divisor = len(self.calibration_data) / 1.0
        # Compute average
        for data_point in self.calibration_data:
            running_total += self.compute_point_distance(data_point)
        average = running_total / divisor

        # Set threshold distance to average across distance between landmarks in calibration data
        self.threshold = average

    def check_breach(self, data, output=False):
        # Input data is of the form dict[id] = [x, y]
        assert isinstance(data, dict)
        assert isinstance(data[0], list)
        dist = self.compute_point_distance(data)
        if output:
            print(("\t{} ({})\tCur: {}\t Thresh:{}").format(
                self.name,
                self.breach_direction,
                dist,
                self.threshold
            ))
        return self.check_threshold_breach(dist, output)

    def check_threshold_breach(self, value, output):
        if self.breach_direction == "above" and value > self.threshold:
            return True
        elif self.breach_direction == "below" and value < self.threshold:
            return True
        else:
            return False


class PostureCriterionML(PostureCriterion):
    def __init__(self, in_name, in_message, in_landmarks):
        self.name = in_name
        self.message = in_message
        self.landmarks = in_landmarks
        self.calibration_data = []  # list of dicts
        self.threshold = 0.0
        self.model = "Model not loaded"
        self.latest_breach = False

    def read_model(self, path="model.pkl"):
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def calibrate(self):
        # Training code
        # Convert data to a df
        df = pd.DataFrame(self.calibration_data)
        print(df)

        x = df.drop('pose', axis=1)  # input values - everything except pose
        y = df['pose']  # output value - just the pose

        # Split data to tests, trains
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

        pipelines = {
            'lr': make_pipeline(StandardScaler(), LogisticRegression()),
            'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
            'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
            'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        }
        fit_models = {}
        for algo, pipeline in pipelines.items():
            model = pipeline.fit(x_train, y_train)
            fit_models[algo] = model

        # Test models, pick the best
        top = 0,
        for algo, model in fit_models.items():
            yhat = model.predict(x_test)
            score = accuracy_score(y_test, yhat)
            print(algo, score)
            if score >= top:
                self.model = model
                top = score

        # Dump model
        with open('model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def check_breach(self, data, output=False):
        assert isinstance(data, dict)
        df = pd.DataFrame([data])

        # Make prediction based on the model
        pred = self.model.predict(df)[0]
        prob = self.model.predict_proba(df)[0]
        self.latest_prediction = pred
        self.latest_prediction_prob = prob
        if pred == 'slouch':
            self.latest_breach = True
            return True
        else:
            self.latest_breach = False
            return False

    def get_prediction(self):
        return self.latest_prediction, self.latest_prediction_prob

    def add_calibration_data(self, data, pose):
        assert isinstance(data, dict)

        # Assume data is just a list of all the positions, not filtered - here, we only take the data points for the landmarks we care about
        data['pose'] = pose
        # Append new data
        self.calibration_data.append(data)


if __name__ == '__main__':
    # Taking posedetector component from the PoseModule and starting video capture
    poser = pm.poseDetector()
    cap = cv2.VideoCapture(0)

    # Defining posture criteria
    criteria = [
        PostureCriterionML("ML model", "get back", [0, 2, 5, 11, 12]),
    ]

    # Main Parameters
    frame_rate = 30
    ignore_fps = False
    train = False  # If false, only monitor. Set to true if you wish to train then model
    calibration_period_seconds = 300  # If train - how long to calibrate (= how long to spend in each pose to train the model)

    # Breach definition - proportion of breach-frames in a given period
    duration = 10  # Time window, seconds
    breach_threshold_percent = 0.5  # The percentage of frames classified as slouching within given time period

    try:
        # Background variables
        prev, capture_time = 0, 0

        start_time = time.time()

        # Breach background calculations
        breach_frames = duration * frame_rate  # The number of frames
        breach_frames_limit = int(breach_frames * float(breach_threshold_percent))
        cur_breaches = 0
        print("Breach condition is {} percentage breaches over {} frames.".format(breach_threshold_percent, breach_frames_limit))
        if train:
            print("Program started, please slouch for {} seconds to calibrate...".format(calibration_period_seconds))
            first_calibration = False
            second_calibration = False
        else:
            print("Monitoring started.")
            first_calibration = True
            second_calibration = True

            # Load models
            [crit.read_model() for crit in criteria]

        while True:
            time_elapsed = time.time() - prev
            time_elapsed_total = time.time() - start_time
            # print("\tElapsed:", time_elapsed_total)
            res, img = cap.read()

            # Process and get positions
            img = poser.findPose(img, True)
            positions, img = poser.getPositionArrayByIds(img, [15, 16])

            if not positions:
                continue  # Skip if issue

            # Check whether calibration is complete and should switch to monitoring
            if time_elapsed_total > calibration_period_seconds and not first_calibration:
                # Compute average
                print("Slouch calibration complete, please sit up straight")

                # Set flag
                first_calibration = True

            elif time_elapsed_total > calibration_period_seconds * 2 and not second_calibration:
                # Compute average
                print("Straight calibration complete, training the model")

                # Calibration
                [crit.calibrate() for crit in criteria]

                # Set flag
                print("Calibration 2 complete, monitoring active")
                second_calibration = True

            # Whether to process a frame or pass, as limited by FPS
            if time_elapsed > 1. / frame_rate or not ignore_fps:
                prev = time.time()
                # Calibration
                if not first_calibration:  # Calibration ongoing, provide data to each criteria
                    [crit.add_calibration_data(positions, "slouch") for crit in criteria]
                elif not second_calibration:
                    [crit.add_calibration_data(positions, "straight") for crit in criteria]

                # Monitoring
                else:
                    criteria[0].check_breach(positions)
                    prediction, probability = criteria[0].get_prediction()
                    probability = str(round(max(list(probability)), 2))
                    print(prediction, probability)

                    # Draw prediction on the image
                    scaler = 2
                    if prediction == 'straight':
                        color_tuple = (0, 125, 0)
                    else:
                        color_tuple = (0, 0, 125)
                    text_color = (255, 255, 255)

                    cv2.rectangle(img, (0, 0), (250 * scaler, 60 * scaler), color_tuple, -1)
                    cv2.putText(img, 'Pose', (95 * scaler, 12 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scaler, text_color, 1, cv2.LINE_AA)
                    cv2.putText(img, str(prediction), (90 * scaler, 40 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1 * scaler, text_color, 2, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(img, '%', (15 * scaler, 12 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scaler, text_color, 1, cv2.LINE_AA)
                    cv2.putText(img, str(probability), (10 * scaler, 40 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1 * scaler, text_color, 2, cv2.LINE_AA)

                    if any([crit.check_breach(positions) for crit in criteria]):
                        breaching_criteria = [crit.name for crit in criteria if crit.latest_breach]
                        cur_breaches = min(cur_breaches + 1, breach_frames_limit + 1)

                        print('Breach found due to {}, cur_breaches:'.format(
                            ", ".join(breaching_criteria)
                        ), cur_breaches)
                        # if False and (consecutive_breaches > frame_rate * 3): # Threshold set to 3 seconds
                        if cur_breaches > breach_frames_limit:
                            # messages = [crit.message for crit in criteria if crit.check_breach(positions)]
                            # output = " and ".join(messages)
                            # notify("Posture Alert", "Stop slouching", "submarine")
                            print("\a", end="")

                    else:
                        cur_breaches = max(0, cur_breaches - 1)

            cv2.imshow('Pose detection', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        # When everything done, release the capture
        print("\nRequested to stop, stopping")
        cap.release()
        cv2.destroyAllWindows()

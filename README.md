# posture-detection
# Antti's notes
Have had the idea to do this for a long time, and found a good start from dcstang's repo based on openCV. Their code implemented facial recognition based on the webcam, and what I built on top of that was simply:
* Computing 'area' of the face detected, simply taking the width/height of the rectangle
* Slouching is simply defined as when the currently detected area > threshold area, i.e. if the user's face is larger (=closer) to the camera than before
* 'Calibrating' the threshold area over the first few seconds the program is ran, by asking the user to slouch at the start
* 'Monitoring', alerting the user via macOS alert + sound if the limit is consecutively breached for over 3 seconds


# Potential to-do
* Head rotation alerts - it is possible to sit far back and have head tilted down, which is still bad posture
* Shoulder-based alerts - ensure shoulders are kept back, perhaps by measuring distance between shoulders scaled to width of the face



## Original Author

* dcstang

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
Not to be misused for illegal, maleficient purposes.


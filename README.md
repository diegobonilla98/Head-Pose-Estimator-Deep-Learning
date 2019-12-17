# Head-Pose-Estimator-Deep-Learning
A deep learning approach of head pose estimator using convolutional neural networks and transfer learning.

Using the Cambridge dataset of images (download here: http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html) and a CNN with transfer learning, a neural network learns to estimate the tilt and the pan of a head.
Also the faced library is used to state-of-the-art facial recognition (https://github.com/iitzco/faced).

The model uses 6 convnets, 1 fully connected layer and 1 output fully connected. All with ReLu activations and l2 + Dropout regularizations.
The model reached and accuracy of +90% and a loss (using Mean Squared Error loss function) around 0.1 in only 12 epochs (1h and 40min +- on 8th gen i5 CPU)

The test file takes the saved model (not able to upload it for size limitations) and the webcam image to predict the head pose. The angle of the head is then draw with lines using some trigonometry and the rotation matrix (here is explain better with the formulas: https://en.wikipedia.org/wiki/Rotation_of_axes)

The image is a test (not calibrated and the angle of axis not translated correcty yet) and the tilt and pan values seem to work very fine (despite of the lines...)

[!Test Image](test_webcam.png)

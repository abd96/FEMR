# Facial Expression Music Recommendation 
The idea is simple. Given an image of a person with some facial expression, recommend a song from spotify based on
the predicted emotion of the facial expression.

## Facial Expression Detection
#### Approach 1 : Using data from [VISGRAF faces database](http://app.visgraf.impa.br/database/faces)

Each image was labeled with an expression from 0 to 13. I chose to learn only 7 expressions : 

| Label  | Expression  |
|---|---|
| 0  | natural/normal |
|1   | happy|
| 2  | sad  |
| 3 | surprised|
| 4|  angry |
|5|disgust|
| 6 | fear  |

Learning with these data was difficult, as i had minimal knowledge in Computer Vision one year ago.
Using [Haarcascades Frontalface Default Model](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) from the Opencv Computer Vision Library 
i was able to crop training images and extract mouth, right_eyebrow, left_eyebrow, right_eye, left_eye, nose and jaw areas.

These areas were then combined to form the Features Matrix, that represents the features to be learnt.
For the Model i used CNNs to train on these data, but i was unable to reach accuracy higher than 0.2.
I panicked watching the accuracy stuck at 0.2, screamed, cried and then broke my computer....... no just kidding, go ahead and read approach 2.
#### Approach 2 : Using data from [Facial Expression Recognition Kaggle Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/overview)
After some further searches i found these data from Kaggle. Data was labeled and saved as csv. This felt 
like heaven.

Training on these data was a little bit easier. For the Model i used the following Model structure : 

```_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 46, 46, 32)        320       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 44, 44, 64)        18496     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 22, 22, 64)        0         
_________________________________________________________________
batch_normalization (BatchNo (None, 22, 22, 64)        256       
_________________________________________________________________
dropout (Dropout)            (None, 22, 22, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 30976)             0         
_________________________________________________________________
dense (Dense)                (None, 64)                1982528   
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 7)                 455       
=================================================================
Total params: 2,002,055
Trainable params: 2,001,927
Non-trainable params: 128
_________________________________________________________________

```

After 80 Epochs the model reached an accuracy of 0.66, a great improvement compared to approach 1. 
But still not perfect.


## Music Recommendation 
This part is not yet implemented.
#### Using Spotify's Music Features 
TODO 


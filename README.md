 ## Face_Recognition
 We're going to build a face recognition system. Many of the ideas presented here are from [FaceNet](https://arxiv.org/pdf/1503.03832.pdf). 

`Face recognition` problems commonly fall into one of two categories:

**Face Verification** "Is this the claimed person?" For example, at some airports, you can pass through customs by letting a system scan your passport and then verifying that you (the person carrying the passport) are the correct person. A mobile phone that unlocks using your face is also using face verification. This is a 1:1 matching problem.

**Face Recognition** "Who is this person?" For example, when employees entering the office without needing their identity card to identify themselves. This is a 1:K matching problem.

`FaceNet` learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, you can then determine if two pictures are of the same person.

**Channels-last notation:

Here we'll be using a pre-trained model which represents ConvNet activations using a "channels last" convention.

In other words, a batch of images will be of shape <img src="https://render.githubusercontent.com/render/math?math=(m,n_H,n_W,n_C)">.

 ## 1- Naive Face Verification
 
In Face Verification, you're given two images and you have to determine if they are of the same person. The simplest way to do this is to compare the two images pixel-by-pixel. If the distance between the raw images is below a chosen threshold, it may be the same person!

<p align="center">
  <img width="500" src="https://github.com/ShafieCoder/Face_Recognition/blob/master/images/pixel_comparison.png" alt="face verification">
</p>

Of course, this algorithm performs poorly, since the pixel values change dramatically due to variations in lighting, orientation of the person's face, minor changes in head position, and so on.

We'll see that rather than using the raw image, we can learn an encoding, <img src="https://render.githubusercontent.com/render/math?math=f(img)">.

By using an encoding for each image, an element-wise comparison produces a more accurate judgement as to whether two pictures are of the same person.

## 2- Encoding Face Images into a 128-Dimensional Vector

#### 2.1 - Using a ConvNet to Compute Encodings
The FaceNet model takes a lot of data and a long time to train. So following the common practice in applied deep learning, we'll load weights that someone else has already trained. The network architecture follows the Inception model from [Szegedy et al.](https://arxiv.org/abs/1409.4842). Moreover, An Inception network implementation has been provided in the file [`inception_blocks_v2.py`](https://github.com/ShafieCoder/Face_Recognition/blob/master/inception_blocks_v2.py) to get a closer look at how it is implemented.

The key things to be aware of are:

* This network uses 160x160 dimensional RGB images as its input. Specifically, a face image (or batch of  𝑚  face images) as a tensor of shape  <img src="https://render.githubusercontent.com/render/math?math=(m,n_H,n_W,n_C) = (m,160,160,3)"> 
* The input images are originally of shape 96x96, thus, you need to scale them to 160x160. This is done in the `img_to_encoding()` function.
* The output is a matrix of shape  (𝑚,128)  that encodes each input face image into a 128-dimensional vector

By using a 128-neuron fully connected layer as its last layer, the model ensures that the output is an encoding vector of size 128. We then use the encodings to compare two 
face images. 

So, an encoding is a good one if:

* The encodings of two images of the same person are quite similar to each other.
* The encodings of two images of different persons are very different.

The triplet loss function formalizes this, and tries to "push" the encodings of two images of the same person (Anchor and Positive) closer together, while "pulling" the encodings of two images of different persons (Anchor, Negative) further apart.

#### 2.2- The Triplet Loss

For an image  𝑥 , its encoding is denoted as  𝑓(𝑥) , where  𝑓  is the function computed by the neural network.
images as follows:
<p align="center">
  <img width="500" src="https://github.com/ShafieCoder/Face_Recognition/blob/master/images/f_x.png" alt="encoding">
</p>

Training will use triplets of images  (𝐴,𝑃,𝑁) :

* A is an "Anchor" image--a picture of a person.
* P is a "Positive" image--a picture of the same person as the Anchor image.
* N is a "Negative" image--a picture of a different person than the Anchor image.
 
These triplets are picked from the training dataset. <img src="https://render.githubusercontent.com/render/math?math=(A(i),P(i),N(i))"> is used here to denote the  𝑖 -th training example.

We'd like to make sure that an image <img src="https://render.githubusercontent.com/render/math?math=A(i)">

 of an individual is closer to the Positive <img src="https://render.githubusercontent.com/render/math?math=P(i)"> than to the Negative image <img src="https://render.githubusercontent.com/render/math?math=N(i)"> ) by at least a margin <img src="https://render.githubusercontent.com/render/math?math=\alpha"> :
 
 <img src="https://render.githubusercontent.com/render/math?math=\Vert f(A^{(i)})- f(P^{(i)})\Vert_{2}^{2} \dotplus  \alpha < \Vert f(A^{(i)})- f(N^{(i)})\Vert_{2}^{2} ">
 
WE would thus like to minimize the following `"triplet cost"`:

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{J}=\sum_{i=1}^{m}\left[\underbrace{\Vert f(A^{(i)})-f(P^{(i)} )\Vert_{2}^{2} }_{(1)}-\underbrace{ \Vert f(A^{(i)}) -f(N^{(i)}) \Vert_{2}^{2} }_{(2)} \dotplus \alpha \right]_{\dotplus}">

Here, the notation " <img src="https://render.githubusercontent.com/render/math?math=\left[z \right]_{\dotplus}"> " is used to denote  𝑚𝑎𝑥(𝑧,0).

**Notes**:

* The term (1) is the squared distance between the anchor "A" and the positive "P" for a given triplet; you want this to be small.
* The term (2) is the squared distance between the anchor "A" and the negative "N" for a given triplet, you want this to be relatively large. It has a minus sign preceding it because minimizing the negative of the term is the same as maximizing that term.
* 𝛼  is called the margin. It's a hyperparameter that you pick manually. You'll use  𝛼=0.2 .
Most implementations also rescale the encoding vectors to haven L2 norm equal to one (i.e., <img src="https://render.githubusercontent.com/render/math?math=\Vert f(img) \Vert_{2} = 1">); 


## 3 - Loading the Pre-trained Model
FaceNet is trained by minimizing the triplet loss. But since training requires a lot of data and a lot of computation,
 we won't train it from scratch here. Instead, we'll load a previously trained
model in our code.

## 4 - Applying the Model
 We're building a system for an office building where the building manager would like to offer facial recognition to allow the employees to enter the building.

We'd like to build a face verification system that gives access to a list of people. To be admitted, each person has to swipe an identification card at the entrance. The face recognition system then verifies that they are who they claim to be.

#### 4.1 - Face Verification
Now we'll build a database containing one encoding vector for each person who is allowed to enter the office. To generate the encoding, we'll use img_to_encoding(image_path, model), which runs the forward propagation of the model on the specified image.

This database maps each person's name to a 128-dimensional encoding of their face.

When someone shows up at your front door and swipes their ID card (thus giving us their name), we can look up their encoding in the database, and use it to check if the person standing at the front door matches the name on the ID.

To implement the verify() function, which checks if the front-door camera picture (image_path) is actually the person called "identity". We will have to go through the following steps:

* Compute the encoding of the image from image_path.
* Compute the distance between this encoding and the encoding of the identity image stored in the database.
* Open the door if the distance is less than 0.7, else do not open it.

**Note:** In this implementation, we should use the L2 distance np.linalg.norm and compare the L2 distance, not the square of the L2 distance, to the threshold 0.7.

*Hints*:

* identity is a string that is also a key in the database dictionary.
* img_to_encoding has two parameters: the image_path and model.

#### 4.2 - Face Recognition
Our face verification system is mostly working. But since Kian got his ID card stolen, when he came back to the office the next day he couldn't get in!

To solve this, you'd like to change your face verification system to a face recognition system. This way, no one has to carry an ID card anymore. An authorized person can just walk up to the building, and the door will unlock for them!

You'll implement a face recognition system that takes as input an image, and figures out if it is one of the authorized persons (and if so, who). Unlike the previous face verification system, you will no longer get a person's name as one of the inputs.


Exercise 3 - who_is_it
Implement who_is_it() with the following steps:

Compute the target encoding of the image from image_path
Find the encoding from the database that has smallest distance with the target encoding.
Initialize the min_dist variable to a large enough number (100). This helps you keep track of the closest encoding to the input's encoding.
Loop over the database dictionary's names and encodings. To loop use for (name, db_enc) in database.items().
Compute the L2 distance between the target "encoding" and the current "encoding" from the database. If this distance is less than the min_dist, then set min_dist to dist, and identity to name.

Ways to improve your facial recognition model:

Although you won't implement these here, here are some ways to further improve the algorithm:

Put more images of each person (under different lighting conditions, taken on different days, etc.) into the database. Then, given a new image, compare the new face to multiple pictures of the person. This would increase accuracy.

Crop the images to contain just the face, and less of the "border" 
region around the face. This preprocessing removes some of the irrelevant pixels
 around the face, and also makes the algorithm more robust.



 6 - References
1. Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). 
FaceNet: A Unified Embedding for Face Recognition and Clustering

2. Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). DeepFace
: Closing the gap to human-level performance in face verification

3. This implementation also took a lot of inspiration from the official FaceNet
 github repository: https://github.com/davidsandberg/facenet

4. Further inspiration was found here: 
https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/

5. And here: https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/tf_to_keras.ipynb



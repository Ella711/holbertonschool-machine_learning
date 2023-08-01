# 0x04. Data Augmentation
## Details
 By: Alexa Orrico, Software Engineer at Holberton School Weight: 1Project will startJul 31, 2023 12:00 AM, must end byAug 2, 2023 12:00 AMManual QA review must be done(request it when you are done with the project)## Resources
Read or watch :
* [Data Augmentation | How to use Deep Learning when you have Limited Data — Part 2](https://intranet.hbtn.io/rltoken/2jHe5oim91wZro4SRdoB1w) 

* [tf.image](https://intranet.hbtn.io/rltoken/SJyavhXgzGprWBKoU0p2Ow) 

* [tf.keras.preprocessing.image](https://intranet.hbtn.io/rltoken/gezEJPsqC-6-mGpI0B38Wg) 

* [Automating Data Augmentation: Practice, Theory and New Direction](https://intranet.hbtn.io/rltoken/wboFN7gUujerC1dfHX24PQ) 

## Learning Objectives
At the end of this project, you are expected to be able to  [explain to anyone](https://intranet.hbtn.io/rltoken/zOatgAKBN76XGs6z4iJ0rQ) 
 ,  without the help of Google :
### General
* What is data augmentation?
* When should you perform data augmentation?
* What are the benefits of using data augmentation?
* What are the various ways to perform data augmentation?
* How can you use ML to automate data augmentation?
## Requirements
### General
* Allowed editors:  ` vi ` ,  ` vim ` ,  ` emacs ` 
* All your files will be interpreted/compiled on Ubuntu 16.04 LTS using  ` python3 `  (version 3.6.12)
* Your files will be executed with  ` numpy `  (version 1.16) and  ` tensorflow `  (version 1.15)
* All your files should end with a new line
* The first line of all your files should be exactly  ` #!/usr/bin/env python3 ` 
* All of your files must be executable
* A  ` README.md `  file, at the root of the folder of the project, is mandatory
* Your code should follow the  ` pycodestyle `  style (version 2.4)
* All your modules should have documentation ( ` python3 -c 'print(__import__("my_module").__doc__)' ` )
* All your classes should have documentation ( ` python3 -c 'print(__import__("my_module").MyClass.__doc__)' ` )
* All your functions (inside and outside a class) should have documentation ( ` python3 -c 'print(__import__("my_module").my_function.__doc__)' `  and  ` python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)' ` )
* Unless otherwise stated, you cannot import any module except  ` import tensorflow as tf ` 
## Download TF Datasets
 ` pip install --user tensorflow-datasets
 ` ## Tasks
### 0. Flip
          mandatory         Progress vs Score  Task Body Write a function   ` def flip_image(image): `   that flips an image horizontally:
*  ` image `  is a 3D  ` tf.Tensor `  containing the image to flip
* Returns the flipped image
```bash
$ cat 0-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
flip_image = __import__('0-flip').flip_image

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(0)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(flip_image(image))
    plt.show()
$ ./0-main.py

```
 ![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/10/3c70d4fb24140e583ec2cc640bba178f090c3829.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230801%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230801T143152Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=59bd4cb56bc7bf2e73ec1601f8d4c8ed422a9d3a24471a0010c6401acb62a2f9) 

 Task URLs  Technical information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` pipeline/0x04-data_augmentation ` 
* File:  ` 0-flip.py ` 
Checker Docker image:
 Self-paced manual review  Panel footer - Controls 
### 1. Crop
          mandatory         Progress vs Score  Task Body Write a function   ` def crop_image(image, size): `   that performs a random crop of an image:
*  ` image `  is a 3D  ` tf.Tensor `  containing the image to crop
*  ` size `  is a tuple containing the size of the crop
* Returns the cropped image
```bash
$ cat 1-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
crop_image = __import__('1-crop').crop_image

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(1)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(crop_image(image, (200, 200, 3)))
    plt.show()
$ ./1-main.py

```
 ![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/10/e3b06484b6d2c0dcdd99a447fb2e83e2975b758a.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230801%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230801T143152Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2b7fca0cb1db0dd74d0985f1f931252dcb4b174af885021f43e79a27c7781169) 

 Task URLs  Technical information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` pipeline/0x04-data_augmentation ` 
* File:  ` 1-crop.py ` 
Checker Docker image:
 Self-paced manual review  Panel footer - Controls 
### 2. Rotate
          mandatory         Progress vs Score  Task Body Write a function   ` def rotate_image(image): `   that rotates an image by 90 degrees counter-clockwise:
*  ` image `  is a 3D  ` tf.Tensor `  containing the image to rotate
* Returns the rotated image
```bash
$ cat 2-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
rotate_image = __import__('2-rotate').rotate_image

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(2)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(rotate_image(image))
    plt.show()
$ ./2-main.py

```
 ![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/10/670106424f5b215f33b4c0f39699ae1ffe89dbb3.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230801%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230801T143152Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=dd3e0b990f7294620d9fe9d8201190531dbde10deef829b403737b71602987a4) 

 Task URLs  Technical information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` pipeline/0x04-data_augmentation ` 
* File:  ` 2-rotate.py ` 
Checker Docker image:
 Self-paced manual review  Panel footer - Controls 
### 3. Shear
          mandatory         Progress vs Score  Task Body Write a function   ` def shear_image(image, intensity): `   that randomly shears an image:
*  ` image `  is a 3D  ` tf.Tensor `  containing the image to shear
*  ` intensity `  is the intensity with which the image should be sheared
* Returns the sheared image
```bash
$ cat 3-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
shear_image = __import__('3-shear').shear_image

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(3)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(shear_image(image, 50))
    plt.show()
$ ./3-main.py

```
 ![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/10/cd5148646829e2f9b540cea1833d34d5f89faf2c.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230801%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230801T143152Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=d0a7d0878b018b9b98c1a839a428b7643e8d0b0810db14fe519bbc995b6f1dc8) 

 Task URLs  Technical information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` pipeline/0x04-data_augmentation ` 
* File:  ` 3-shear.py ` 
Checker Docker image:
 Self-paced manual review  Panel footer - Controls 
### 4. Brightness
          mandatory         Progress vs Score  Task Body Write a function   ` def change_brightness(image, max_delta): `   that randomly changes the brightness of an image:
*  ` image `  is a 3D  ` tf.Tensor `  containing the image to change
*  ` max_delta `  is the maximum amount the image should be brightened (or darkened)
* Returns the altered image
```bash
$ cat 4-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
change_brightness = __import__('4-brightness').change_brightness

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(4)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(change_brightness(image, 0.3))
    plt.show()
$ ./4-main.py

```
 ![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/10/3001edca791b04ccde934a44fe3095b1e544a425.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230801%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230801T143152Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2416edd77db43c4da4e9c8abccf8a952e410b711d168d98c6a2f72f8b2214606) 

 Task URLs  Technical information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` pipeline/0x04-data_augmentation ` 
* File:  ` 4-brightness.py ` 
Checker Docker image:
 Self-paced manual review  Panel footer - Controls 
### 5. Hue
          mandatory         Progress vs Score  Task Body Write a function   ` def change_hue(image, delta): `   that changes the hue of an image:
*  ` image `  is a 3D  ` tf.Tensor `  containing the image to change
*  ` delta `  is the amount the hue should change
* Returns the altered image
```bash
$ cat 5-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
change_hue = __import__('5-hue').change_hue

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(5)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(change_hue(image, -0.5))
    plt.show()
$ ./5-main.py

```
 ![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/10/a1e9035f2000dbb5649032ac424d1ebe980e8a07.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230801%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230801T143152Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=98c94108a45d23ea361db77a819f8eaec69dfa1148e33a4c23852a9618e92791) 

 Task URLs  Technical information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` pipeline/0x04-data_augmentation ` 
* File:  ` 5-hue.py ` 
Checker Docker image:
 Self-paced manual review  Panel footer - Controls 
### 6. Automation
          mandatory         Progress vs Score  Task Body Write a blog post describing step by step how to perform automated data augmentation. Try to explain every step you know of, and give examples. A total beginner should understand what you have written.
* Have at least one picture, at the top of the blog post
* Publish your blog post on Medium or LinkedIn
* Share your blog post at least on LinkedIn
* Write professionally and intelligibly
* Please, remember that these blogs must be written in English to further your technical ability in a variety of settings
Remember, future employers will see your articles; take this seriously, and produce something that will be an asset to your future
When done, please add all urls below (blog post, LinkedIn post, etc.)
 Task URLs #### Add URLs here:
                Save               Technical information Checker Docker image:
 Self-paced manual review  Panel footer - Controls 
### 7. PCA Color Augmentation
          #advanced         Progress vs Score  Task Body Write a function   ` def pca_color(image, alphas): `   that performs PCA color augmentation as described in the  [AlexNet](https://intranet.hbtn.io/rltoken/zEzc_8giX0XkuUTiQsnqXA) 
  paper:
*  ` image `  is a 3D  ` tf.Tensor `  containing the image to change
*  ` alphas `  a tuple of length 3 containing the amount that each channel should change
* Returns the augmented image
```bash
$ cat 100-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
pca_color = __import__('100-pca').pca_color

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(100)
np.random.seed(100)
doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    alphas = np.random.normal(0, 0.1, 3)
    plt.imshow(pca_color(image, alphas))
    plt.show()
$ ./100-main.py

```
 ![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/10/73ecbc2cf5ac920e1e022178c420eb7c7585c345.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230801%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230801T143152Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=40f477b1c989a1b8731b87f961ae682db78228126d928098196cfc2dcf6fd0e2) 

 Task URLs  Technical information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` pipeline/0x04-data_augmentation ` 
* File:  ` 100-pca.py ` 
Checker Docker image:
 Self-paced manual review  Panel footer - Controls 
Ready for a  manual review
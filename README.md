# TensorFlow Project 🧱
- I used a Lego bricks dataset from Kaggle that can be found [here](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images?select=LEGO+brick+images+v1)
    - There are 15 classes and a total of 401 photos in each, meaning there are a total of 6,015 images
    - This is realtively small dataset because I didn't want to overload my GPU and CPU with too much
- [Here](https://drive.google.com/drive/folders/1-7wCbWscN4LC7hc9h0iuMPi7Fn-WLZ48?usp=share_link) is a Google drive folder with calculations for both my good and bad CNN
## Features 🔦
- Most recent run results: `Epoch 100/100 - 396/396 - 7s - loss: 0.0384 - accuracy: 0.9876 - val_loss: 0.3467 - val_accuracy: 0.8994 - 7s/epoch - 17ms/step`
### CNN Calculations 🔢
- Image sizes were all square
    - Original was 200x200x3, but I inputted it as 200x200x1
    - Chose to grayscale images to quicken run time because they were gray, but still in RGB format
- Played around with the placement of certain functions
    - Chose to use alternate `Conv2D()` and `MaxPool2D()` similar to the example code because it helped improve my CNN's accuracy
    - Other variations including using `Conv2D()` a lot didn't work that well; maxpooling every other performed much better
        - I uploaded this calculation and the good CNN to Drive
    - Didn't use `Dropout()` because it reduced my accuracy
### Live Feed 📹
- Grayscaled input
- Didn't implement a background/none class, so that is my next step to improvement
  - Would try to classify non-Lego objects as Legos (including myself) instead of saying background/none
- Worked better after implemented data augmentation, but still not that good
### Data Augmentation 🎞
- Used `flip_left_right()` and `adjust_brightness()` successfully
- 386 steps with data augmentation; 132 without data augmentation
- Slightly improved my accuracy
    - Was `~90%` and now consistently `~98%`

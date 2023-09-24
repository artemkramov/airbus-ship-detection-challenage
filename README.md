# Airbus Ship Detection Challenge
This repository contains notebooks and results obtained during the work with the [**Airbus Ship Detection Challenge**](https://www.kaggle.com/competitions/airbus-ship-detection/).

## Description of the task
Airbus is excited to challenge Kagglers to build a model that detects all ships in satellite images as quickly as possible. Can you find them even in imagery with clouds or haze?

Here’s the backstory: Shipping traffic is growing fast. More ships increase the chances of infractions at sea like environmentally devastating ship accidents, piracy, illegal fishing, drug trafficking, and illegal cargo movement. This has compelled many organizations, from environmental protection agencies to insurance companies and national government authorities, to have a closer watch over the open seas.

Airbus offers comprehensive maritime monitoring services by building a meaningful solution for wide coverage, fine details, intensive monitoring, premium reactivity and interpretation response. Combining its proprietary-data with highly-trained analysts, they help to support the maritime industry to increase knowledge, anticipate threats, trigger alerts, and improve efficiency at sea.

A lot of work has been done over the last 10 years to automatically extract objects from satellite images with significative results but no effective operational effects. Now Airbus is turning to Kagglers to increase the accuracy and speed of automatic ship detection.

## Project structure

All calculations were performed using Google Colab notebooks. 
Such a choice can be explained by the necessity to use GPU for training of a big amount of data (Kaggle zip file is ~30GB).
Thus, it was decided to rely on the Colab environment during the worj with the task.
The folder `notebooks` contains the following Google Colab Notebooks:
- `Airbus_challenge_(dataset).ipynb` is a notebook to analyze the dataset
- `Airbus_challenge_(training).ipynb` is a notebook to train the semantic segmentation model
- `Airbus_challenge_(fine_tune_classifier).ipynb` is a notebook to train a binary classifier to distinguish between images with ships and without them (it's explained later)
- `Airbus_challenge_evaluate.ipynb` is a notebook to evaluate the performance of the segmentation model on the test set
- `Airbus_challenge_(inference).ipynb` is a notebook to run both segmentation and classification in order to get masks for the samples of the test set (with visualization)

The folder `results` contains 2 CSV files that contains RLE-based predictions of the images of the test set. These files were passed to the Kaggle panel to get results:
- `submission_v1.csv` corresponds to a stand-alone segmentation model
- `submission_v2.csv` corresponds to a segmentation model with a classifier

The requested dependencies are listed in the `requirements.txt` file.

## Research steps

### Dataset analysis
The main takeaways of the dataset analysis (see `Airbus_challenge_(dataset).ipynb`) are the following:
- all images are of the same shape with a good picture quality; width and height of images are the same
- the dataset is imbalanced: images without ships represent a majority class (~78% of samples)
- in most of cases the images with ships contain 1 or 2 ships
- segmentation maps are pretty sparse

Thus, in order to deal with the problem of the imbalance data, it was decided to upsample the dataset with augmented samples.
Augmented samples were generated from the minority class (pictures with ships) using non-destructive transformations (horizontal flip, rotation, etc.) 
till the moment when a number of images with ships equals to a number of images without ships.
Moreorver, maybe it would be better to add destructive transformations (e.g., scaling).

### Model architecture
MaskFormer segmentation model with a Swin transformer as a backbone was selected to be used. 
Namely, it was decided to take a model `facebook/maskformer-swin-base-ade` that was pre-trained on the  ADE20k semantic segmentation dataset.
This model was introduced in the paper `Per-Pixel Classification is Not All You Need for Semantic Segmentation`. The motivation of the usage of the model is the following:
- It can be a more efficient way to fine-tune already trained segmentation model on a smaller amount of samples instead of the training of a model from scratch
- Transformer-based models showed good performance on different datasets (ADE20k, COCO) and outperformed 
traditional CNN-only based model as Mask R-CNN or FCN (according to `paperswithcode` stats)
- MaskFormer approach implies the ability to deal with any number of labels during the classification of the binary masks. 
It can be trained for semantic, instance, and panoptic segmentation with the same logic.
Also, taking into account the results that are reported in the aforementioned paper, it was decided to use the MaskFormer model.

It has to be said that U-Net model can potentially be used for this dataset: this model showed good results for medicine data
that are pretty much sparse as the current dataset. However, it was decided to try to use more modern approach that 
can be extended later to perform more complex instance and panoptic segmentation.

### Training of the segmentation model
The training (see `Airbus_challenge_(training).ipynb`) was performed using the `Adam` optimizer. The loss function was used according to the implementation in the paper.
Namely, the loss function was calculated by the weighted sum of the label cross-entropy loss, binary mask loss, and the dice loss.
The quality of the training was tracked by the calculation of a mean IoU value for the validation set.
The training was performed during almost 2 epochs till the moment when the training loss curve finished to converge.
The checkpoint with the highest mean IoU value (`0.8197`) was saved.

### Evaluation (step 1)
The evaluation (see `Airbus_challenge_evaluate.ipynb`) was performed on the test set. Namely, 
all binary masks were predicted by the segmentation model; then labeled binary masks (i.e., non-zero values were divided into classes where each class represents a separate instance) 
were transformed into RLE-format. Retrieved data (`sumbission_v1.csv`) were pushed to Kaggle for further estimation.
Kaggle results for the first submission `submission_v1.csv` were the following:

*   Private Score `0.67972`
*   Public Score `0.52359`

As can be seen, the scores are pretty low. After the analysis of the results, it seemed that such a low scores can be explained by the following factors:


*   Test set is highly imbalanced in terms of images without ships (majority class). Such an assumption is made after passing a file with empty masks to Kaggle: the private score is `0.76566`.
*   Model is predicting too many false positives.

In order to verify such a hypothesis, it was decided to train a binary classifier that would be able to distinguish between images with ships and without them.

### Binary classifier
The same previously considered balanced dataset was used for the training of the classifier. 
It was decided to fine-tune the `google/vit-base-patch16-224-in21k` model that was previously trained
on the huge ImageNet dataset. The model is based on the Vision Transformer that can be too heavy to be used for the 
segmentation task, but it can be suitable for the image classification. 
The fine-tuning process is implemented in the notebook `Airbus_challenge_(fine_tune_classifier).ipynb`.

### Evaluation (step 2)
The next step of the evaluation (see `Airbus_challenge_evaluate.ipynb`) was implemented together with a pre-trained classifier.
Namely, only images that were classified by the classifier as images with ships were passed to the input of the segmentation model.
Than a new file `sumbission_v2.csv` was passed to Kaggle platform.
The usage of the classifier allowed achieving much better results:

*   Private Score `0.79862`
*   Public Score `0.62482`

However, the scores are still pretty low. According to manual inspection, it seems that the problem of false positives still remains unsolved
and required more investigation.

### Inference
The weights of both models (the segmentation model and the classifier) were uploaded to Hugging Face hub.
The notebook `Airbus_challenge_(inference).ipynb` contains an example how to use them in order to get binary masks 
of the images from the test subset. This notebook can be uploaded to Google Colab and be executed without any external dependencies
as far as models' weights are loaded from the hub.
 
 



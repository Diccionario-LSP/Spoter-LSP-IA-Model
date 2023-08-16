![Top Banner](https://data.matsworld.io/signlanguagerecognition/GitHub_banner.png)

> by **[Matyáš Boháček](https://github.com/matyasbohacek)** and **[Marek Hrúz](https://github.com/mhruz)**, University of West Bohemia <br>
> Should you have any questions or inquiries, feel free to contact us [here](mailto:matyas.bohacek@matsworld.io).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sign-pose-based-transformer-for-word-level/sign-language-recognition-on-lsa64)](https://paperswithcode.com/sota/sign-language-recognition-on-lsa64?p=sign-pose-based-transformer-for-word-level)


This is a forked version of the original repository created by Matyáš Boháček and is based on the research paper titled "[Sign Pose-based Transformer for Word-level Sign Language Recognition](https://openaccess.thecvf.com/content/WACV2022W/HADCV/html/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.html)." 

## Research Contributions

In the course of our research using Spoter, we have made several contributions, including the following papers and extended abstract:

- **[Impact of Pose Estimation Models for Landmark-based Sign Language Recognition](https://research.latinxinai.org/papers/neurips/2022/pdf/18_CameraReady.pdf)**

In this paper, we delve into the significance of pose estimation models for landmark-based sign language recognition. We specifically explore the utilization of 29 and 71 landmarks from Mediapipe, Openpose, and RHnet models to input into both the Spoter and a graph-based model. Through our analysis, we conclude that the Mediapipe model in combination with the Spoter model exhibits better compatibility with our dataset. Interestingly, we observe that employing 71 points yields positive outcomes, but subsequent experiments led us to discover that using 54 points actually yields superior results.

- **[Less is More: Techniques to Reduce Overfitting in your Transformer Model for Sign Language Recognition](https://research.latinxinai.org/papers/cvpr/2023/pdf/Joe_Huamani.pdf)**

The strategies detailed in this paper are designed to counter overfitting by making modifications to both the data and the training process. Our findings underscore the effectiveness of employing a combination of the AEC and PUCP305 techniques, which yield notable improvements in our results. Additionally, we highlight the significance of data augmentation, label smoothing, and model complexity reduction in enhancing model generalization. These insights have led us to make certain parameter adjustments to the Spoter model to better balance complexity and performance. 

- **[Impact of Video Length Reduction due to Missing Landmarks on Sign Language Recognition Model](https://research.latinxinai.org/papers/cvpr/2023/pdf/Carlos_Vasquez.pdf)**

This study investigates how shortening videos due to missing landmarks affects the performance of our sign language recognition model. Our research reveals that while reducing video length does result in a slight drop in model performance, we've chosen to retain videos with missing parts in our dataset.


Please feel free to explore this repository and the associated papers to gain a deeper understanding of our research and its outcomes.

For any questions, comments, or collaborations, please don't hesitate to get in touch!

## Get Started

First, make sure to install all necessary dependencies using:

```shell
pip install -r requirements.txt
```

Create an account on [Weights & Biases](https://wandb.ai/) to facilitate experiment tracking and reproducibility. Then please set up your `WANDB_API_KEY` in your environment.

```
export WANDB_API_KEY=your_api_key_here
```

To train the model, simply specify the hyperparameters and run the following:

```
python -m train
  --experiment_name [str; name of the experiment to name the output logs and plots in WandB]
  
  --epochs [int; number of epochs]
  --lr [float; learning rate]
  
  --training_set_path [str; path to the H5 file with training set's skeletal data]
  --validation_set_path [str; path to the H5 file with validation set's skeletal data]
```

The hyperparameter modifications made during our research are hardcoded in the repository, so you can directly experiment with the provided hyperparameters.

## License

The **code** is published under the [Apache License 2.0](https://github.com/matyasbohacek/spoter/blob/main/LICENSE) which allows for both academic and commercial use as presented in the original repository of Matyáš Boháček.
# multimodal-emotions

Some code to do feature extraction and affect classification (stress vs. amusement vs. baseline) based on the original paper, [Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection](https://dl.acm.org/doi/10.1145/3242969.3242985). We use the following data modalities collected by a chest strap: 3-axis acceleration, ECG, EDA, EMG, respiration, and temperature. There are 15 subjects.

First we can take a look at the experimental states for each subject (we only care about stress, amusement, and baseline, and we'll ignore the others):

<img width="722" alt="study_condition_labels" src="https://github.com/calderast/multimodal-emotions/assets/70605721/e76a2ab6-ddd4-4a4d-927f-f98ab5583ebc">

Let's also take a look at the raw sensor data for this subject:

<img width="984" alt="raw_sensor_data" src="https://github.com/calderast/multimodal-emotions/assets/70605721/f7724351-1f1b-49d4-8da8-04c50982d482">

Now it's time to extract some features! We use a default window size of 60 seconds with a window shift of .25 seconds. As an example, the extracted acceleration features for subject 2 look like this:

<img width="758" alt="condition_labels" src="https://github.com/calderast/multimodal-emotions/assets/70605721/f5f02e4a-085a-47ac-9d55-bc1b15738054">

Here, the grey dashed lines indicate boundaries between different study conditions (corresponding to the study conditions shown above):

<img width="891" alt="ACC_features" src="https://github.com/calderast/multimodal-emotions/assets/70605721/c613a0dd-035e-4748-88a9-8763f4a4bc6a">



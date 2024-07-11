# Self-Supervised Depth Completion
Mobiltech-Gachon PJ for the month

## Basic DenseLiDAR Architecture
<img width="797" alt="image" src="https://github.com/9-coding/Mobiltech-Gachon/assets/127665166/7b973b56-38af-4e79-b602-c5f0443db6ce">
<hr>

## Our Variation Samples
### 1. Add SAM from Basic DenseLiDAR
<img width="710" alt="image" src="https://github.com/9-coding/Mobiltech-Gachon/assets/127665166/a8ea5c67-c4e8-4996-b656-bdecba63f98a">

### 2. SAM + Depth Anything V2
- Add SAM for image guidance and Depth Anything V2 for self-supervised Learning
<img width="930" alt="image" src="https://github.com/9-coding/Mobiltech-Gachon/assets/127665166/c0ba2cfb-1f5a-4538-8622-497974cde69b">

### 3. Raw LiDAR + SAM + Depth Anything V2
- Remove IP_Basic and rectify_depth
- Add SAM for image guidance and Depth Anything V2 for self-supervised Learning
<img width="479" alt="image" src="https://github.com/9-coding/Mobiltech-Gachon/assets/127665166/0e6e97fd-4685-47ca-a8d9-6730f360a94e">

### 4. Variation sample #2 + 3 DCUs
- For using raw image input and SAM result simultaneously.
- `dcu(dcu(guided_LiDAR + raw_image) + dcu(raw_LiDAR + guided_image)) `

<img width="804" alt="image" src="https://github.com/9-coding/Mobiltech-Gachon/assets/127665166/4f470a07-081c-481b-8ab0-d293ad47e962">

### 5. Simple version using Depth Anything V2 without SAM

<img width="866" alt="image" src="https://github.com/9-coding/Mobiltech-Gachon/assets/127665166/eabf4a70-6acc-4f8b-8be8-c7254bb24988">

### 6. Using DeepLiDAR

<img width="1053" alt="image" src="https://github.com/9-coding/Mobiltech-Gachon/assets/127665166/d756783f-4b0d-4607-98e0-1669b1a8ffb2">


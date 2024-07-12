# Self-Supervised Depth Completion
Mobiltech-Gachon PJ for the month

## Basic DenseLiDAR Architecture
<img width="797" alt="image" src="https://github.com/9-coding/Mobiltech-Gachon/assets/127665166/7b973b56-38af-4e79-b602-c5f0443db6ce">

## Target Architecture
- For using raw image input and SAM result simultaneously.
- `dcu(dcu(guided_LiDAR + raw_image) + dcu(raw_LiDAR + guided_image)) `
![Screenshot from 2024-07-12 16-50-21](https://github.com/user-attachments/assets/e74a0bcd-2ff6-43cc-aed6-4878f5390f74)


## Our Variation Samples
### 1. Add SAM from Basic DenseLiDAR
<img width="710" alt="image" src="https://github.com/9-coding/Mobiltech-Gachon/assets/127665166/a8ea5c67-c4e8-4996-b656-bdecba63f98a">

### 2. SAM + Depth Anything V2
- Add SAM for image guidance and Depth Anything V2 for self-supervised Learning
![Screenshot from 2024-07-12 17-50-04](https://github.com/user-attachments/assets/94636523-ea4e-4aa4-b473-5bf415438fef)


### 3. Raw LiDAR + SAM + Depth Anything V2
- Remove IP_Basic and rectify_depth
- Add SAM for image guidance and Depth Anything V2 for self-supervised Learning
![Screenshot from 2024-07-12 17-54-27](https://github.com/user-attachments/assets/9ae6414d-4b4e-4817-b5a2-8940025fed46)


### 4. Variation sample #2 + 3 DCUs
- For using raw image input and SAM result simultaneously.
- `dcu(dcu(guided_LiDAR + raw_image) + dcu(raw_LiDAR + guided_image)) `

![Screenshot from 2024-07-12 16-50-21](https://github.com/user-attachments/assets/e74a0bcd-2ff6-43cc-aed6-4878f5390f74)

### 5. Simple version using Depth Anything V2 without SAM
![Screenshot from 2024-07-12 18-00-00](https://github.com/user-attachments/assets/536b1c2f-6fea-4031-8eef-69112166d69c)



### 6. Using DeepLiDAR
![Screenshot from 2024-07-12 17-58-35](https://github.com/user-attachments/assets/f0918591-435f-4bf8-9725-8098ef20213d)



# Computer Vision

This project consists of several tasks related to computer vision, including boundary detection, face detection and recognition, feature detection and image matching, image processing, and image segmentation. Each section below provides images and explanations of the tasks implemented in the project.

## Sections
- [Image Preprocessing](#image-preprocessing)
- [Boundary Detection](#boundary-detection)
- [Features Detection and Image Matching](#features-detection-and-image-matching)
- [Image Segmentation](#image-segmentation)
- [Face Detection and Recognition](#face-detection-and-recognition)


---

## Image Preprocessing

This is the initial step in the image processing chain, where raw image data is enhanced to improve its quality and remove distortions or artifacts. It prepares the image for further analysis by applying various correction techniques.

### I. Noise Generation 
Users can manipulate images by introducing various
types of noise using a convenient combo box interface. This feature allows for experimentation with different noise models, providing insights into the impact of noise on image quality and subsequent processing algorithms.

### &nbsp;&nbsp;&nbsp;1. Uniform Noise

![Unifrom_noise](https://github.com/user-attachments/assets/0c3918ac-a74f-4f34-95ca-2dbd7a42bee8)

### &nbsp;&nbsp;&nbsp;2. Gaussian Noise

![gaussian_noise](https://github.com/user-attachments/assets/1a29ec69-0752-4e75-9047-394a49ca21e4)

### &nbsp;&nbsp;&nbsp;3. Salt & Pepper Noise

![saltandpaper](https://github.com/user-attachments/assets/1b48b73b-3393-4963-a6b0-6534cf4845ea)


### II. Image Filtering
We implemented various filters to enhance image quality and reduce noise *(e.g. `average`, `gaussian`, `median filter`)*. We experimented with different kernel sizes, specifically 3x3, 5x5, and 7x7, to observe their effects on the filtration process.

### &nbsp;&nbsp;&nbsp;1. Average Filter Applied on Uniform Noise

![Average](https://github.com/user-attachments/assets/daf02bfe-0038-4488-8c07-e11fe3d7b0bc)

### &nbsp;&nbsp;&nbsp;1. Gaussian Filter Applied on Uniform Noise

![gaussian](https://github.com/user-attachments/assets/55d583bd-218c-48b8-a867-27ef32d3682c)

### &nbsp;&nbsp;&nbsp;1. Median Filter Applied on Uniform Noise

![median](https://github.com/user-attachments/assets/bd8acc1e-69c4-42f8-bb68-50e1f8d3066d)


### III. Edge Detection
Identifying the boundaries or outlines of objects in the image by detecting changes in intensity.

### &nbsp;&nbsp;&nbsp;1. Sobel

![sobel](https://github.com/user-attachments/assets/83df83ae-89ce-4d44-912b-c98c9c246cdd)

### &nbsp;&nbsp;&nbsp;2. Roberts

![roberts](https://github.com/user-attachments/assets/0a9d4b9c-68e9-4b9e-a85e-11c515f6c099)

### &nbsp;&nbsp;&nbsp;3. Prewitt

![prewitt](https://github.com/user-attachments/assets/f23d9561-eee5-4ce8-a20f-76a95f365000)

### &nbsp;&nbsp;&nbsp;4. Canny

![canny](https://github.com/user-attachments/assets/454bab41-98b8-4df5-882f-950b0eccebd3)


### IV. Histograms
The histogram of an image gives a graphical representation of the distribution of pixel intensities within that image. It plots the frequency of occurrence of each intensity value along the x-axis, with the corresponding number of pixels having that intensity value along the y-axis. This allows us to visualize how light or dark an image is overall.

- ### For a darker image
Observation: The histogram shows a higher concentration of pixels with
lower intensity values, indicating a predominance of dark areas.

![darker image](https://github.com/user-attachments/assets/9d2cbf70-a6fd-44c6-800f-eb44a4ad955a)

The entire GUI is displayed which contains a `histogram`, `distribution curve`, `RGB histogram`, `cumulative curve`

![histogram](https://github.com/user-attachments/assets/2ea016e6-90cd-4db7-abc4-e9d54b8c11bb)


- ### For a brighter image
Observation: We notice a more balanced distribution of intensity values,
with a broader spread across the intensity axis. This indicates a wider range of brightness levels present in the image.


![brighter](https://github.com/user-attachments/assets/cb1bcf55-06b1-4c8a-a532-6b7d2b9c70d4)

Also the entire GUI is displayed which contains a `histogram`, `distribution curve`, `RGB histogram`, `cumulative curve`

![brighter histo](https://github.com/user-attachments/assets/356626f4-6126-4e77-a416-631f0d58f37a)

- ### Distribution Curve
The distribution curve represents the frequency distribution of pixel intensities across the entire image. Peaks in the distribution curve indicate dominant intensity levels, while valleys suggest less
common intensity levels.

Observation: We noticed that the curve was skewed towards the higher end of the intensity axis,
indicating that the image contains a significant number of bright pixels.

![distribution](https://github.com/user-attachments/assets/90fcbc35-f780-4a81-b915-111c9442fd0a)

- ### Histogram equalization
Histogram equalization is a technique used to improve the contrast in an image. It operates by effectively spreading out the most frequent intensity values, i.e., ‘stretching out’ the
intensity range of the image. This method usually increases the global contrast of images when its usable data is represented by close contrast values.

### &nbsp;&nbsp;&nbsp;1. With `YCrCb color space`
Observation: The image appears slightly desaturated and contrasted.

![YCrCb color space](https://github.com/user-attachments/assets/e3ce147e-2ddd-4507-b1b5-ef68347d3d97)

### &nbsp;&nbsp;&nbsp;2. With `HSV color space`
Observation: The image appears more vibrant and
bright. The colors are more distinct and saturated, which might make certain details stand out
more.

![HSV color space](https://github.com/user-attachments/assets/cb6c0b67-ea03-40d3-8e14-09ae55589c79)


### V. Normalization
Adjusting the pixel values of an image so that they fall within a
specific range or distribution.
Normalization helps reduce the effect of variations in illumination, enhancing the contrast of
images, and improving the performance of algorithms that operate on image data.

![normalization](https://github.com/user-attachments/assets/b0067c51-4c8f-44a2-83ca-32e9d520dd74)

### VI. Thresholding
### &nbsp;&nbsp;&nbsp;1. Global Thresholding
It is a simple and widely used technique in image processing for segmenting an image into two regions: `foreground` (object of interest) and `background`. The goal is to find a `single threshold value` that separates these two regions based on pixel intensity.

Global Thresholding Parameter:
- `Threshold value` = 109.0

![global](https://github.com/user-attachments/assets/0a0032b6-384a-48aa-9372-409d156daa7e)

### &nbsp;&nbsp;&nbsp;2. Local Thresholding
It takes a more nuanced approach. Instead of using a single threshold for the entire image, it computes different thresholds for different regions based on the local characteristics of each pixel’s neighborhood.

Observation: Local thresholding tends to preserve fine details

Global Thresholding Parameter:
- `Window Size` = 40px

![local](https://github.com/user-attachments/assets/12afc8cd-aeb6-4631-b2f5-ed6777423619)


### VII. Frequency Domain Filter
Finding the frequency components of an image through
Fourier analysis. The frequency domain provides information about the spatial frequencies present in an image, such as low-frequency components representing smooth areas and high-frequency components representing edges or fine details.

### &nbsp;&nbsp;&nbsp;1. Low-pass Filter

![low pass](https://github.com/user-attachments/assets/391e303e-580c-421f-857f-324727491c18)

### &nbsp;&nbsp;&nbsp;2. High-pass Filter

![final high pass](https://github.com/user-attachments/assets/4a2dc697-cfec-4e8f-bf4b-f06a6a456ba5)


### VIII. Hybrid Image
- `Low Frequency Image`: Marilyn image
- `High Frequency Image`: Einstein image

![hybrid image](https://github.com/user-attachments/assets/b5d666ea-bfc2-402a-bb2a-653bae9e8baa)

---


## Boundary Detection

### Description
Boundary detection is the task of identifying the edges and boundaries within an image. It is commonly used in image processing to delineate objects or regions within a scene.

### I. Edge Detection
Identifying edges within the images.
### Case 1
Parameters:
- `Sigma` = 1

![sigma1](https://github.com/user-attachments/assets/94ea546f-1eae-4590-999a-ff16c917f74a)

- `Sigma` = 50

![sigma50 again](https://github.com/user-attachments/assets/291751e1-458d-4245-935f-4ce256d0f8d1)

### Observation
For higher sigma value, the image becomes more blurred (more
smoothness) and loss of details.

### Case 2
Parameters:
- `T_low` = 0.05
- `T_high` = 0.09

![lenda1](https://github.com/user-attachments/assets/55788b5c-7bfe-4ae5-b18b-829dd8a031c4)

- `T_low` = 0.1
- `T_high` = 0.3

![lenda2](https://github.com/user-attachments/assets/94b30d97-6dab-4a04-9ed7-5bbe428e90ae)

### Observation
- For `lower` values `T_low` and `T_high`: The weak edges appear clearly.
- For `higher` values `T_low` and `T_high`: The strongest edges appear clearly and most of the details are lost.

### I. Shape Detection
### &nbsp;&nbsp;&nbsp;1. Line Detection
### Case 1
Parameters:
- `Number of Lines` = 4
- `Neighbor Size ` = 10

Observation: The line detection algorithm was applied with a small
`number of peaks = 4`, resulting in a reduced number of detected lines
compared to the original lines in the image.


![num of lines4](https://github.com/user-attachments/assets/dff32790-7bbf-41d2-b3a9-a81014d4a311)

- `Number of Lines` = 8
- `Neighbor Size ` = 10

Observation: The line detection algorithm was applied with a
moderate `number of peaks = 8`, resulting in a balanced representation
of lines in the image.

![8lines](https://github.com/user-attachments/assets/5ff2b26c-8ab3-49bd-bc78-7ca09f39e695)

- `Number of Lines` = 27
- `Neighbor Size ` = 10

Observation: The line detection algorithm was applied with a large
`number of peaks = 27`, resulting in the detection of one line being
represented as multiple lines, leading to a dense cluster of lines
plotted over it.

![27lines](https://github.com/user-attachments/assets/96b5ffb9-82f4-4714-827e-46ed73b0d3db)

### Case 2
Parameters:
- `Neighbor Size ` = 1
- `Number of Lines` = 7

Observation: Using a small `neighboring size = 1`, results in detected lines that are closely aligned with the original ones in the image.

![1neighbor](https://github.com/user-attachments/assets/68c8b2d5-28af-473d-b16e-0a6c30623239)

- `Neighbor Size ` = 35
- `Number of Lines` = 7

Observation: Using a large `neighboring size = 35`, results in a more generalized representation of detected lines compared to the original ones in the image.

![35neigh](https://github.com/user-attachments/assets/0726d880-4612-4240-8258-304e741aa95b)

### &nbsp;&nbsp;&nbsp;2. Circle Detection
### Case 1: Narrow Range

- `Min Radius` = 7
- `Max Radius` = 50
- `Bin Threshold` = 0.6
- `Pixel Threshold` = 20

![circle1](https://github.com/user-attachments/assets/5192b363-5298-4dd2-9329-2967190aa307)

### Case 2: Wide Range
- `Min Radius` = 64
- `Max Radius` = 160
- `Bin Threshold` = 0.6
- `Pixel Threshold` = 20

![wide circle](https://github.com/user-attachments/assets/23b36b76-a6df-4d96-b1bc-447245d6168e)

- `Min Radius` = 17
- `Max Radius` = 100
- `Bin Threshold` = 0.6
- `Pixel Threshold` = 20

![circle case2](https://github.com/user-attachments/assets/021815e9-4423-44fc-943a-9abeee0af1d2)

### Case 3: Bin Threshold
- `Min Radius` = 7
- `Max Radius` = 81
- `Bin Threshold` = 1.0
- `Pixel Threshold` = 100

![case3](https://github.com/user-attachments/assets/ab51b6ab-189c-4a8c-bcce-12b7e12bfab7)

### &nbsp;&nbsp;&nbsp;3. Ellipses Detection

### &nbsp;&nbsp;&nbsp;4. Active Contour Model (Snakes)
It is a powerful technique used in image processing for tasks such as object tracking and segmentation. The model works by iteratively adjusting a contour to fit the edges in an image, based on an energy minimization
process.

### Chain code
The chain code is a compact representation used in image processing and
computer vision to describe the shape of a contour. It simplifies the contour by encoding the direction of transitions between adjacent points along the contour.

### The perimeter of the contour
It refers to the total length of its boundary. It represents the
distance around the shape.

###  The area of the contour
It represents the total surface area enclosed by the contour.

### Parameters

- `Square Contour`
- `Alpha` = 3
- `Beta` = 96
- `Gamma` = 100
- `Iterations` = 95

![snake1](https://github.com/user-attachments/assets/11fa21a3-ab84-478d-988b-fa4da22afff5)

`Chain code 8`: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 0, 7, 0, 7, 0, 1, 7, 0, 0, 7, 0, 0, 0, 7,
7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 4, 5, 4, 4, 5, 4, 4, 5, 4, 3, 5, 4, 4, 5, 4, 4, 4, 4, 3, 7,
0, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1]

`Chain code 4`: [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 1, 1, 1, 1, 1]

`Contour perimeter`: 625.86

`Contour area`: 28925.50 square units

---

- `Square Contour`
- `Alpha` = 3
- `Beta` = 96
- `Gamma` = 92
- `Iterations` = 95

![snake2](https://github.com/user-attachments/assets/ca96eb75-5b7e-4ebe-a364-28b4d685e3b2)

`Chain code 8`: [2, 1, 5, 1, 1, 1, 1, 1, 1, 7, 7, 7, 0, 1, 1, 1, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7,
7, 7, 7, 3, 2, 3, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 7, 5, 6, 6, 6, 7, 6, 6, 5, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 0,
4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 4, 4, 4, 0, 0, 4, 4, 0,
0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 6, 1, 2, 1, 1, 1, 2, 1,
3, 2, 1, 2, 1, 2, 1, 1, 1]

`Chain code 4`: [1, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0, 2,
2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 2,
2, 2, 1, 1, 1, 3, 1, 1, 1, 1, 1]

`Contour perimeter`: 665.76

`Contour area`: 22442.50 square units

---

- `Circle Contour`
- `Alpha` = 3
- `Beta` = 96
- `Gamma` = 92
- `Iterations` = 95

![snake3](https://github.com/user-attachments/assets/724c0e2b-640f-4e2e-b1bd-a455a408737c)

`Chain code 8`: [6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 2]

`Chain code 4`: [3, 2, 2, 2, 1, 1, 0, 3, 3, 1]

`Contour perimeter`: 663.46

`Contour area`: 33908.00 square units

---

## Features Detection and Image Matching
Feature detection involves identifying key points in images that are used to match similar images or track objects. 

### I. Feature Extraction
### &nbsp;&nbsp;&nbsp;1. Harris Operator
### Parameters:
- `Very Low Threshold` = 0.1

### Observation: 
The algorithm detects a large number of corners, resulting in an over-detection scenario.

![0 1threshold](https://github.com/user-attachments/assets/44a331a7-02a0-4823-b4c8-021e51a2ca3c)

- `High Threshold` = 0.8

### Observation: 
Applying a high threshold value of 0.8 to the Harris response significantly reduces the number of detected corners to 103, with only a sparse set identified in regions with pronounced intensity variations.

![0 8threshold](https://github.com/user-attachments/assets/aaeb2e68-d2a4-439e-9667-ce14b0700d7b)

- `Medium Threshold` = 0.4

### Observation: 
A balanced distribution of corners is identified, covering both prominent features and subtle intensity variations.

![medthreshold](https://github.com/user-attachments/assets/98892a20-abc7-4ee7-8c37-b1edd7cb4be1)

### &nbsp;&nbsp;&nbsp;2. Lambda Minus
### Parameters:
- `Threshold` = 0.04
- `block_size` = 2
- ` k_size` = 3

![lambda1](https://github.com/user-attachments/assets/2ed74176-120d-4a5b-9d71-5fe48a03227e)

Computation time: 0.004015207290649414

![lambda2](https://github.com/user-attachments/assets/4c685e97-57b6-428e-b68c-69167f73fdfe)
Computation time: 0.03145337104797363

- `Threshold` = 0.04
- `block_size` = 2
- ` k_size` = 5

![lambda3](https://github.com/user-attachments/assets/1a8d30a1-d69e-4eb0-b788-d72cce318338)
Computation time: 0.004987001419067383

- `Threshold` = 0.04
- `block_size` = 2
- ` k_size` = 7

![lambda5](https://github.com/user-attachments/assets/41a2979a-3d74-4285-adee-149b2e158ce3)
Computation time: 0.00398564338684082

---

- `Threshold` = 0.04
- `block_size` = 3
- ` k_size` = 7

![lambda6](https://github.com/user-attachments/assets/80577339-301b-47af-923e-db951f7f9574)
Computation time: 0.0030677318572998047

### Observation:
- If you increase the `blockSize`, the window of the neighborhood considered for corner detection becomes larger.
- If you increase the `ksize`, the Sobel operator becomes more sensitive to larger, more prominent edges and less sensitive to smaller, finer details. This could result in fewer corners being detected, but those detected may be more robust and significant.


### II. Feature Descriptors Generation
- ### The Scale-Invariant Feature Transform (SIFT)  
It is a powerful method for detecting and describing local features in images. It is widely used in computer vision tasks such as object recognition, image stitching, and 3D reconstruction. SIFT features are invariant to image scale, and rotation, and partially invariant to changes in illumination and viewpoint.

![f](https://github.com/user-attachments/assets/3a320f73-d6be-4c7b-8480-7e0fb83ca576)

### Observation: 
What we see in the above images are are the Keypoints with their orientation.

Computation_time: 129.2920205593109 sec

### III. Feature Matching
### &nbsp;&nbsp;&nbsp;1. Using KNN

![maching using KNN](https://github.com/user-attachments/assets/d20172db-cbbd-46b5-8013-d68a0cabf930)

### &nbsp;&nbsp;&nbsp;2. Using NCC

![NCC](https://github.com/user-attachments/assets/302eb6ba-f9d5-401f-8917-65fb5fa565db)

### &nbsp;&nbsp;&nbsp;3. Using SSD

![SSD](https://github.com/user-attachments/assets/559de31a-2d71-4500-8762-48d2f38d3e81)

### &nbsp;&nbsp;&nbsp;4. Detecting Objects Using NCC

![detecting objects using NCC](https://github.com/user-attachments/assets/b97cb176-1ce0-48e1-9d8c-cd5325467610)


---

## Face Detection and Recognition

### Description
Face detection involves identifying human faces in digital images. Face recognition goes a step further by identifying or verifying individuals based on facial features.


---

## Image Segmentation

### Description
Image segmentation is the process of partitioning an image into multiple segments to simplify its analysis. It is commonly used to isolate objects or regions of interest in an image.


---


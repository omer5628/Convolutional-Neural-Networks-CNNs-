# Convolutional-Neural-Networks-CNNs-

## Summary
This README provides a summary of the implementation and evaluation of a Convolutional Neural Network (CNN) for classifying images from the Fashion-MNIST dataset. The dataset consists of 70,000 grayscale images across 10 categories, each of size 28x28 pixels. The objective was to design, train, and evaluate a CNN model using standard deep learning techniques while incorporating various experiments and optimizations to achieve high classification accuracy.

### Key Steps

#### Data Preparation
- The dataset was preprocessed by normalizing images to have values between -1 and 1.
- Images were resized to 32x32 pixels to accommodate the input size for the CNN.
- Data augmentation techniques, including random horizontal flips and rotations, were applied to increase dataset variability.

#### Model Architecture
- The CNN architecture included:  
  - Two convolutional layers with ReLU activation and max-pooling.  
  - Fully connected layers with a dropout layer to mitigate overfitting.  
  - Hyperparameters tuned through grid search.

#### Training
- The model was trained for 10 epochs using the Adam optimizer.
- A batch size of 64 and a learning rate of 0.001 were found to be optimal.
- Training and validation losses were monitored for convergence.

#### Evaluation
- The final model achieved a test accuracy of 91.73%.
- Performance was further analyzed using precision, recall, and F1-scores for each class.
- Visualizations of feature maps and filters were generated for interpretability.

---

## Analysis

### Model Performance
- The model performed well, achieving a test accuracy of 91.73%, demonstrating its capability to effectively classify Fashion-MNIST images.
- Precision and recall scores indicated strong performance across most categories, though the "Shirt" class exhibited relatively lower accuracy (F1-score: 0.76), suggesting potential challenges in distinguishing it from similar classes like "T-shirt/top" or "Coat".
- The confusion matrix highlighted common misclassifications, providing insights into areas for improvement.

### Training Observations
- Both training and validation losses steadily decreased, indicating good convergence.
- The dropout layer (p=0.3) effectively reduced overfitting, as evidenced by the close alignment of training and validation losses.
- Hyperparameter tuning showed that a dropout rate of 0.5 and the Adam optimizer yielded the best results.

### Visualization of Filters and Feature Maps
#### Filters
- The filters in the first convolutional layer captured basic edge and texture features, such as horizontal and vertical lines.
- Deeper layers extracted more complex patterns, including shapes and high-level abstractions specific to clothing items.

#### Feature Maps
- Feature maps from the first convolutional layer demonstrated clear edge detection and texture patterns.
- The second layer feature maps highlighted more abstract patterns and specific features of clothing items, such as contours and regions of interest.
- Visualization of feature maps provided insight into how the network progressively learns hierarchical features from raw pixel data.

### Challenges Encountered
1. **Class Imbalance**:  
   - The "Shirt" class was harder to classify accurately, likely due to similarities with other classes. Techniques such as class weighting or additional augmentation specific to this class could help.

2. **Limited Dataset Variability**:  
   - While data augmentation improved performance, the dataset's inherent grayscale nature posed challenges for distinguishing subtle differences between classes.

3. **Computational Constraints**:  
   - Training on larger architectures with more convolutional layers was limited by computational resources, necessitating a balance between complexity and efficiency.

---

## Conclusion
The implemented CNN model effectively classifies Fashion-MNIST images with high accuracy, leveraging data augmentation and dropout to enhance performance. Future improvements could include experimenting with deeper architectures, fine-tuning class-specific augmentations, and integrating advanced regularization techniques such as batch normalization. The visualizations of filters and feature maps provided valuable insights into the network's learning process, confirming its ability to extract meaningful features at multiple levels of abstraction.

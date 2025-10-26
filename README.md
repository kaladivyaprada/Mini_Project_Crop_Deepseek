Project plan
## 🎯 **PHASE 1: Project Setup & Basic App**
**Credit: 1 | Goal: Working Streamlit skeleton**

### Prompt 1.1 - Basic App Structure:
```
Create a complete Streamlit app for satellite image analysis with the following features:

1. A main title "Crop Classification AI"
2. Sidebar with file uploader for JPG/PNG images
3. Display uploaded image in main area
4. Two buttons: "NDVI Analysis" and "CNN Classification" 
5. Basic layout with columns for results

Make it a single file app.py that I can run immediately. Include all necessary imports and error handling.
```

### Prompt 1.2 - Requirements File:
```
Generate a requirements.txt file for a crop classification project with these libraries:
- Streamlit for web interface
- OpenCV and Pillow for image processing
- NumPy for calculations
- Matplotlib for visualization
- PyTorch for deep learning

Make sure versions are compatible and it's ready to install with pip.
```

### Prompt 1.3 - Image Loading:
```
Create a function `load_and_preprocess_image(uploaded_file)` that:
- Takes Streamlit uploaded file
- Converts to OpenCV/numpy format
- Resizes to 224x224 pixels
- Normalizes pixel values to 0-1
- Returns processed image ready for analysis

Include error handling for different file types.
```

---

## 🎯 **PHASE 2: NDVI Implementation**
**Credit: 1 | Goal: Working NDVI analysis**

### Prompt 2.1 - NDVI Calculation:
```
Write a function `calculate_ndvi(image)` for crop classification that:

1. Takes RGB image (numpy array, 0-1 normalized)
2. Simulates NIR band by using: NIR = (green + red) / 2  
3. Calculates NDVI = (NIR - Red) / (NIR + Red + 1e-8)
4. Returns NDVI map and classified result:
   - NDVI > 0.6: "Healthy Crop"
   - NDVI 0.3-0.6: "Moderate Crop" 
   - NDVI < 0.3: "Soil/Non-Vegetated"

Also create visualization function to show NDVI as heatmap.
```

### Prompt 2.2 - NDVI Integration:
```
Integrate the NDVI function into my Streamlit app:

1. When "NDVI Analysis" button is clicked:
   - Process uploaded image
   - Calculate NDVI
   - Display original image, NDVI heatmap, and classification result side by side
   - Show percentage of vegetation coverage

2. Add a download button to save NDVI results

Use Streamlit columns for clean layout.
```

### Prompt 2.3 - Visualization:
```
Create a function `plot_ndvi_results(original_img, ndvi_map, classification)` that:

1. Creates a 2x2 matplotlib subplot:
   - Top-left: Original image
   - Top-right: NDVI heatmap (green to brown colormap)
   - Bottom-left: Classification mask
   - Bottom-right: Statistics (vegetation percentage, average NDVI)

2. Converts matplotlib figure to Streamlit-compatible format

Return the figure for display in Streamlit.
```

---

## 🎯 **PHASE 3: CNN Model Implementation**
**Credit: 1 | Goal: Working CNN classifier**

### Prompt 3.1 - Simple CNN Model:
```
Create a complete PyTorch CNN model for crop classification with:

1. Model Architecture:
   - Input: 3-channel RGB images (224x224)
   - 2 convolutional layers with ReLU and MaxPooling
   - Fully connected layers
   - Output: 3 classes (Healthy Crop, Moderate Crop, Soil)

2. Include model definition, forward pass, and prediction function

3. Create a function `classify_with_cnn(image)` that:
   - Preprocesses image for CNN
   - Runs inference
   - Returns class label and confidence score

Make it work without training data by using a pre-trained pattern.
```

### Prompt 3.2 - Model Integration:
```
Integrate the CNN classifier into my Streamlit app:

1. Add "CNN Classification" button functionality
2. When clicked:
   - Show loading spinner
   - Run CNN classification
   - Display: Original image, CNN prediction, Confidence score
   - Compare with NDVI results

3. Add a section to compare NDVI vs CNN results

Handle cases where model might not be trained (use mock predictions for demo).
```

### Prompt 3.3 - Mock Training:
```
Create a function `mock_cnn_training()` that:

1. Generates synthetic training data (random images with labels)
2. "Trains" the CNN model for 2-3 epochs
3. Saves the model to a file
4. Returns training accuracy history

This is for demonstration purposes since we don't have real training data in bolt.new.

Also create a function to load the saved model for inference.
```

---

## 🎯 **PHASE 4: Dashboard Enhancement**
**Credit: 1 | Goal: Professional interface**

### Prompt 4.1 - UI Improvement:
```
Enhance my Streamlit crop classification app with:

1. Better styling (custom CSS for professional look)
2. Progress bars for processing
3. Results summary card showing:
   - Analysis method used
   - Processing time
   - Confidence scores
   - Vegetation percentage

4. Sidebar sections for:
   - File upload
   - Analysis options
   - Model settings

Make it look like a professional agricultural tool.
```

### Prompt 4.2 - Comparison Feature:
```
Add a comprehensive comparison feature between NDVI and CNN methods:

1. "Compare Methods" button that runs both analyses
2. Side-by-side comparison showing:
   - NDVI results vs CNN results
   - Agreement/disagreement between methods
   - Processing time comparison
   - Confidence levels

3. Add metrics: Accuracy estimate, Processing speed, Resource usage

Use Streamlit expanders to organize the comparison results.
```

### Prompt 4.3 - Export Functionality:
```
Create export functionality for the app:

1. Generate PDF report with:
   - Original image
   - Analysis results from both methods
   - Statistics and metrics
   - Timestamp and analysis parameters

2. Add download buttons for:
   - Processed images
   - Results data (CSV)
   - PDF report

Use ReportLab or similar for PDF generation within Streamlit.
```

---

## 🎯 **PHASE 5: Testing & Final Polish**
**Credit: 1 | Goal: Complete project**

### Prompt 5.1 - Testing Functions:
```
Create comprehensive test functions for the crop classification app:

1. `test_ndvi_calculation()` - Test NDVI with sample images
2. `test_cnn_inference()` - Test CNN with mock data
3. `test_app_workflow()` - Test complete Streamlit workflow
4. `test_image_processing()` - Test image loading and preprocessing

Each test should print PASS/FAIL and return success rate.

Also create a demo mode that runs all tests when app starts.
```

### Prompt 5.2 - Performance Optimization:
```
Optimize the app for better performance:

1. Add caching for expensive functions using `@st.cache_data`
2. Implement lazy loading for images
3. Add progress indicators for long operations
4. Optimize image processing pipelines

Make sure the app runs smoothly within bolt.new's resource limits.
```

### Prompt 5.3 - Final Integration:
```
Create the final complete version of the crop classification app with:

1. All previous features integrated
2. Error handling for all possible cases
3. User instructions and tooltips
4. Sample data for demonstration
5. Professional documentation within the app

Make it production-ready with proper code organization and comments.

Also add a "Demo Mode" that runs with sample images if no file is uploaded.
```


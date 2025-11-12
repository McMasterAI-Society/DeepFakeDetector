*Please refer to the document titled ModelArchitectureFlow.pdf to follow along with the explanation provided below.*

**Fusion Model (see page 11 of *ModelArchitectureFlow.pdf*) - Explanation**

1.  Parallel processing of the image in three different streams:
    
    *  CNN (refer to page 1 of *ModelArchitectureFlow.pdf*) identifies spatial features (edges, textures, and high-level face distortions)
        
        -  Local features, e.g., eyes, mouth, and texture inconsistencies
            
        -  Artifacts related to lighting and skin appearance / smoothness
            
    *  ViT (refer to page 5 of *ModelArchitectureFlow.pdf*) handles images by splitting them into patches and extracting dependencies between those patches (e.g., spatial relationships between facial features)
        
        -  Learns global semantic relationship
            
        -  Makes use of face-specific attention mechanisms
            
    *  FFT (refer to page 8 of *ModelArchitectureFlow.pdf*) focuses on frequency-domain features, capturing high-frequency artifacts that are notable in deepfakes, e.g.:
        
        -  Periodicity
            
        -  Unnatural smoothness
            
        -  Blurring
            
2.  Feature fusion
    
    *  Concatenate the final 1x512 feature vectors that are outputted by each of the models into a single combined 1x1536 feature vector
        
        -  Allows the model to use the full range of info from each model type
            
3.  Learned weighted fusion
    
    *  As the model trains, a learned weight for each branch’s output can be used to allow the model to focus more on the CNN, ViT, or FFT features, depending on which one seems to best separate the classes (fake vs real).
        
4.  Final classification
    
    *  A fully connected dense layer is used to make the final decision
        
    *  Passed through a sigmoid activation function for binary classification
        
5.  Heatmap generation for visual interpretation
    
    *  Methods like Grad-CAM can be used to highlight the parts of the image that influenced the decision
        
    *  Can be done on each individual’s feature map
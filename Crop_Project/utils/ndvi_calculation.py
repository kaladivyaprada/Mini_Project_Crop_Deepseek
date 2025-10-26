import numpy as np
import cv2
from sklearn.cluster import KMeans

class NDVIAnalyzer:
    @staticmethod
    def calculate_ndvi(rgb_image):
        """
        Calculate NDVI from RGB image using simulated NIR band
        NIR is simulated as (Green + Red) / 2
        """
        try:
            # Extract channels
            red = rgb_image[:, :, 0]
            green = rgb_image[:, :, 1]
            blue = rgb_image[:, :, 2]
            
            # Simulate NIR band
            nir = (green + red) / 2
            
            # Calculate NDVI
            denominator = nir + red + 1e-8  # Avoid division by zero
            ndvi = (nir - red) / denominator
            
            return ndvi
            
        except Exception as e:
            raise ValueError(f"Error calculating NDVI: {str(e)}")
    
    @staticmethod
    def classify_ndvi(ndvi_map):
        """Classify NDVI values into categories"""
        try:
            # Create classification mask
            healthy_mask = ndvi_map > 0.6
            moderate_mask = (ndvi_map >= 0.3) & (ndvi_map <= 0.6)
            soil_mask = ndvi_map < 0.3
            
            # Create classified image
            classified = np.zeros((*ndvi_map.shape, 3), dtype=np.float32)
            classified[healthy_mask] = [0, 1, 0]  # Green for healthy
            classified[moderate_mask] = [1, 1, 0]  # Yellow for moderate
            classified[soil_mask] = [0.6, 0.3, 0.1]  # Brown for soil
            
            # Calculate statistics
            total_pixels = ndvi_map.size
            healthy_percentage = np.sum(healthy_mask) / total_pixels * 100
            moderate_percentage = np.sum(moderate_mask) / total_pixels * 100
            soil_percentage = np.sum(soil_mask) / total_pixels * 100
            average_ndvi = np.mean(ndvi_map)
            
            stats = {
                'healthy_percentage': round(healthy_percentage, 2),
                'moderate_percentage': round(moderate_percentage, 2),
                'soil_percentage': round(soil_percentage, 2),
                'average_ndvi': round(average_ndvi, 3),
                'vegetation_coverage': round(healthy_percentage + moderate_percentage, 2)
            }
            
            return classified, stats
            
        except Exception as e:
            raise ValueError(f"Error classifying NDVI: {str(e)}")
    
    @staticmethod
    def create_ndvi_heatmap(ndvi_map):
        """Create a colored heatmap from NDVI values"""
        try:
            # Normalize NDVI to 0-1 for colormap
            ndvi_normalized = (ndvi_map + 1) / 2  # NDVI ranges from -1 to 1
            
            # Apply colormap (green to brown)
            heatmap = np.zeros((*ndvi_map.shape, 3), dtype=np.float32)
            
            # Create custom colormap: brown -> yellow -> green
            for i in range(ndvi_map.shape[0]):
                for j in range(ndvi_map.shape[1]):
                    ndvi_val = ndvi_normalized[i, j]
                    if ndvi_val < 0.5:
                        # Brown to Yellow
                        r = 0.6 + (1 - 0.6) * (ndvi_val * 2)
                        g = 0.3 + (1 - 0.3) * (ndvi_val * 2)
                        b = 0.1 * (1 - ndvi_val * 2)
                    else:
                        # Yellow to Green
                        r = 1 - (ndvi_val - 0.5) * 2
                        g = 1
                        b = 0
                    
                    heatmap[i, j] = [r, g, b]
            
            return heatmap
            
        except Exception as e:
            raise ValueError(f"Error creating NDVI heatmap: {str(e)}")
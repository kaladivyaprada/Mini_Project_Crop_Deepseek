import matplotlib.pyplot as plt
import numpy as np
import io
import base64

class ResultVisualizer:
    @staticmethod
    def create_comparison_plot(original_img, ndvi_heatmap, classified_img, stats):
        """Create a comprehensive visualization of results"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Crop Classification Analysis Results', fontsize=16, fontweight='bold')
            
            # Original image
            axes[0, 0].imshow(original_img)
            axes[0, 0].set_title('Original Satellite Image', fontweight='bold')
            axes[0, 0].axis('off')
            
            # NDVI Heatmap
            im = axes[0, 1].imshow(ndvi_heatmap)
            axes[0, 1].set_title('NDVI Heatmap', fontweight='bold')
            axes[0, 1].axis('off')
            plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            # Classification result
            axes[1, 0].imshow(classified_img)
            axes[1, 0].set_title('Classification Result', fontweight='bold')
            axes[1, 0].axis('off')
            
            # Statistics
            axes[1, 1].axis('off')
            stats_text = (
                f"Vegetation Coverage: {stats['vegetation_coverage']}%\n"
                f"Healthy Crop: {stats['healthy_percentage']}%\n"
                f"Moderate Crop: {stats['moderate_percentage']}%\n"
                f"Soil/Non-Vegetated: {stats['soil_percentage']}%\n"
                f"Average NDVI: {stats['average_ndvi']}"
            )
            axes[1, 1].text(0.1, 0.9, stats_text, fontsize=12, va='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            raise ValueError(f"Error creating visualization: {str(e)}")
    
    @staticmethod
    def create_confidence_chart(probabilities):
        """Create a bar chart of class probabilities"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            classes = list(probabilities.keys())
            probs = list(probabilities.values())
            
            bars = ax.bar(classes, probs, color=['green', 'orange', 'brown'])
            ax.set_ylabel('Probability')
            ax.set_title('CNN Classification Confidence')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            raise ValueError(f"Error creating confidence chart: {str(e)}")
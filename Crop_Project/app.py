from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import time
import json
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import io

from config import Config
from utils.image_processing import ImageProcessor
from utils.ndvi_calculation import NDVIAnalyzer
from utils.visualization import ResultVisualizer
from models.cnn_model import CNNClassifier

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)  # Enable CORS for all routes

# Initialize components
Config.init_app(app)
cnn_classifier = CNNClassifier(app.config['MODEL_PATH'])

@app.route('/')
def home():
    return jsonify({
        "message": "Crop Classification AI API",
        "version": "1.0.0",
        "endpoints": {
            "analyze_ndvi": "/api/analyze/ndvi",
            "analyze_cnn": "/api/analyze/cnn",
            "compare_methods": "/api/analyze/compare",
            "train_model": "/api/train",
            "export_results": "/api/export/pdf"
        }
    })

@app.route('/api/analyze/ndvi', methods=['POST'])
def analyze_ndvi():
    """Endpoint for NDVI analysis"""
    try:
        start_time = time.time()
        
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        # Check if file is allowed
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not ImageProcessor.allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Process image
        original_image = ImageProcessor.load_and_preprocess_image(file)
        
        # Calculate NDVI
        ndvi_map = NDVIAnalyzer.calculate_ndvi(original_image)
        
        # Classify NDVI
        classified_img, stats = NDVIAnalyzer.classify_ndvi(ndvi_map)
        
        # Create heatmap
        ndvi_heatmap = NDVIAnalyzer.create_ndvi_heatmap(ndvi_map)
        
        # Create visualization
        visualization = ResultVisualizer.create_comparison_plot(
            original_image, ndvi_heatmap, classified_img, stats
        )
        
        processing_time = round(time.time() - start_time, 2)
        
        # Prepare response
        response = {
            "success": True,
            "method": "NDVI Analysis",
            "processing_time": processing_time,
            "statistics": stats,
            "visualization": visualization,
            "original_image": ImageProcessor.numpy_to_base64(original_image),
            "ndvi_heatmap": ImageProcessor.numpy_to_base64(ndvi_heatmap),
            "classified_image": ImageProcessor.numpy_to_base64(classified_img)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze/cnn', methods=['POST'])
def analyze_cnn():
    """Endpoint for CNN classification"""
    try:
        start_time = time.time()
        
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        # Check if file is allowed
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not ImageProcessor.allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Process image
        original_image = ImageProcessor.load_and_preprocess_image(file)
        
        # Run CNN classification
        cnn_results = cnn_classifier.predict(original_image)
        
        # Create confidence chart
        confidence_chart = ResultVisualizer.create_confidence_chart(
            cnn_results['all_probabilities']
        )
        
        processing_time = round(time.time() - start_time, 2)
        
        # Prepare response
        response = {
            "success": True,
            "method": "CNN Classification",
            "processing_time": processing_time,
            "prediction": cnn_results['predicted_class'],
            "confidence": cnn_results['confidence'],
            "all_probabilities": cnn_results['all_probabilities'],
            "original_image": ImageProcessor.numpy_to_base64(original_image),
            "confidence_chart": confidence_chart
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze/compare', methods=['POST'])
def compare_methods():
    """Endpoint to compare both NDVI and CNN methods"""
    try:
        start_time = time.time()
        
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        # Check if file is allowed
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not ImageProcessor.allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Process image
        original_image = ImageProcessor.load_and_preprocess_image(file)
        
        # Run NDVI analysis
        ndvi_start = time.time()
        ndvi_map = NDVIAnalyzer.calculate_ndvi(original_image)
        classified_img, ndvi_stats = NDVIAnalyzer.classify_ndvi(ndvi_map)
        ndvi_time = round(time.time() - ndvi_start, 2)
        
        # Run CNN classification
        cnn_start = time.time()
        cnn_results = cnn_classifier.predict(original_image)
        cnn_time = round(time.time() - cnn_start, 2)
        
        # Determine agreement
        ndvi_primary = "Healthy Crop" if ndvi_stats['healthy_percentage'] > 50 else \
                      "Moderate Crop" if ndvi_stats['moderate_percentage'] > 30 else "Soil/Non-Vegetated"
        
        agreement = "High" if ndvi_primary == cnn_results['predicted_class'] else "Medium" if \
                   (ndvi_primary in ["Healthy Crop", "Moderate Crop"] and 
                    cnn_results['predicted_class'] in ["Healthy Crop", "Moderate Crop"]) else "Low"
        
        total_time = round(time.time() - start_time, 2)
        
        # Prepare comparison response
        response = {
            "success": True,
            "method": "Comparison Analysis",
            "total_processing_time": total_time,
            "comparison": {
                "agreement": agreement,
                "ndvi_primary_class": ndvi_primary,
                "cnn_primary_class": cnn_results['predicted_class'],
                "processing_times": {
                    "ndvi": ndvi_time,
                    "cnn": cnn_time
                },
                "confidence_levels": {
                    "ndvi_vegetation_coverage": ndvi_stats['vegetation_coverage'],
                    "cnn_confidence": cnn_results['confidence']
                }
            },
            "ndvi_results": {
                "statistics": ndvi_stats,
                "processing_time": ndvi_time
            },
            "cnn_results": {
                "prediction": cnn_results['predicted_class'],
                "confidence": cnn_results['confidence'],
                "all_probabilities": cnn_results['all_probabilities'],
                "processing_time": cnn_time
            },
            "original_image": ImageProcessor.numpy_to_base64(original_image)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Endpoint for mock model training"""
    try:
        data = request.get_json()
        epochs = data.get('epochs', 3) if data else 3
        
        # Run mock training
        training_history = cnn_classifier.mock_training(epochs=epochs)
        
        response = {
            "success": True,
            "message": f"Model training completed for {epochs} epochs",
            "training_history": training_history,
            "final_accuracy": training_history['accuracy'][-1],
            "final_loss": training_history['loss'][-1]
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/export/pdf', methods=['POST'])
def export_pdf():
    """Endpoint to generate PDF report"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Create PDF in memory
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        
        # Add title
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(100, 750, "Crop Classification Analysis Report")
        
        # Add timestamp
        pdf.setFont("Helvetica", 10)
        pdf.drawString(100, 730, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add analysis results
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(100, 700, "Analysis Results:")
        
        y_position = 680
        pdf.setFont("Helvetica", 10)
        
        if 'ndvi_results' in data:
            pdf.drawString(100, y_position, "NDVI Analysis:")
            y_position -= 15
            for key, value in data['ndvi_results'].items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        pdf.drawString(120, y_position, f"{k}: {v}")
                        y_position -= 15
                else:
                    pdf.drawString(120, y_position, f"{key}: {value}")
                    y_position -= 15
        
        if 'cnn_results' in data:
            pdf.drawString(100, y_position, "CNN Analysis:")
            y_position -= 15
            for key, value in data['cnn_results'].items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        pdf.drawString(120, y_position, f"{k}: {v}")
                        y_position -= 15
                else:
                    pdf.drawString(120, y_position, f"{key}: {value}")
                    y_position -= 15
        
        pdf.save()
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"crop_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
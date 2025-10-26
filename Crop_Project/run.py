from app import app

if __name__ == '__main__':
    print("Starting Crop Classification AI Backend...")
    print("Available endpoints:")
    print("  - GET  / : API information")
    print("  - POST /api/analyze/ndvi : NDVI analysis")
    print("  - POST /api/analyze/cnn : CNN classification")
    print("  - POST /api/analyze/compare : Compare methods")
    print("  - POST /api/train : Mock model training")
    print("  - POST /api/export/pdf : Generate PDF report")
    print("  - GET  /api/health : Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
import os
import logging
import time
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from ml_model import predict_gear_image
from models import db, GearClassification

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the function to configure routes when the app is created
def configure_routes(app):
    # Configure middleware
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    
    # Configure app settings
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
    
    @app.route('/')
    def index():
        # Get recent classifications from the database
        recent_classifications = GearClassification.query.order_by(
            GearClassification.timestamp.desc()).limit(5).all()
        return render_template('index.html', recent_classifications=recent_classifications)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submits an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image with ML model
            try:
                # Now returns a tuple of (result, confidence)
                result, confidence = predict_gear_image(filepath)
                
                # Save the classification result to the database with confidence
                classification = GearClassification(
                    filename=filename,
                    result=result,
                    confidence=confidence
                )
                db.session.add(classification)
                db.session.commit()
                
                # Format confidence as percentage for display
                confidence_pct = f"{confidence * 100:.1f}%"
                
                return jsonify({
                    'success': True,
                    'result': result,
                    'confidence': confidence,
                    'confidence_pct': confidence_pct,
                    'filename': filename,
                    'id': classification.id
                })
            except Exception as e:
                logging.error(f"Error during prediction: {str(e)}")
                return jsonify({'error': 'Error processing image: ' + str(e)}), 500
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    
    @app.route('/history')
    def history():
        # Get all classifications from the database
        classifications = GearClassification.query.order_by(
            GearClassification.timestamp.desc()).all()
        return render_template('history.html', classifications=classifications)
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({'error': 'File too large (max 16MB)'}), 413
    
    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({'error': 'Server error occurred'}), 500
    
    # Clean up the uploads folder periodically
    @app.before_request
    def cleanup_old_uploads():
        try:
            current_time = time.time()
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                # Remove files older than 1 hour
                if os.path.isfile(filepath) and current_time - os.path.getmtime(filepath) > 3600:
                    os.remove(filepath)
        except Exception as e:
            logging.error(f"Error cleaning up uploads: {str(e)}")
            # Don't fail the request if cleanup fails

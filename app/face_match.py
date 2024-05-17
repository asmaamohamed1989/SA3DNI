import os
import logging
from flask import Flask, request, jsonify
from deepface import DeepFace
import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin.exceptions import FirebaseError
from io import BytesIO

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate(r"./sa3edny-b7978-firebase-adminsdk-yt5ha-8a3a7205e5.json")  # Replace with your service account key path
firebase_admin.initialize_app(cred, {'storageBucket': 'sa3edny-b7978.appspot.com'})

# Set environment variable to disable TensorRT (if needed)
os.environ['TF_DISABLE_TENSORRT'] = '1'

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set desired logging level

# Load the model during application startup
model = DeepFace.build_model("Facenet")

def verify_face(photo_bytes, photo2_path):
    result = model.verify(photo_bytes, photo2_path, enforce_detection=False)
    return result['verified']

def process_photo(photo_bytes):
    matched_photos = []

    # Get a reference to the Firebase Cloud Storage bucket
    bucket = storage.bucket()

    try:
        # List all blobs (files) in the bucket
        blobs = bucket.list_blobs()

        for blob in blobs:
            if "image" not in blob.content_type:
                continue 
            
            pic_bytes = blob.download_as_bytes()
            
            verified = verify_face(pic_bytes, photo_bytes)
            
            if verified:
                matched_photos.append({'name': blob.name, 'url': blob.public_url})

        if not matched_photos:
            return {'message': 'No matching photos found'}
        else:
            return {'matched_photos': matched_photos}

    except FirebaseError as e:
        logging.error(f"Error downloading photo from Firebase Storage: {str(e)}")
        return {'error': 'Error downloading photo from Firebase Storage'}
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        return {'error': f'An unexpected error occurred: {e}'}

@app.route('/match-face', methods=['POST'])
def match_face():
    try:
        photo_bytes = request.data
        if not photo_bytes:
            return jsonify({'error': 'No photo provided'}), 400

        matched_photos = process_photo(photo_bytes)

        response_data = {
            'matched_photos': matched_photos  # Include matched photo details in the response
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Default port is 10000 if not set
    app.run(host='0.0.0.0', port=port)

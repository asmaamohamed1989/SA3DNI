import os
import logging
from flask import Flask, request, jsonify
from deepface import DeepFace
import tempfile
import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin.exceptions import FirebaseError
from google.cloud.storage.blob import Blob
print("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:", os.getenv('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'))

app = Flask(__name__)
#app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Initialize Firebase
cred = credentials.Certificate(r"./sa3edny-b7978-firebase-adminsdk-yt5ha-8a3a7205e5.json")  # Replace with your service account key path
firebase_admin.initialize_app(cred, {'storageBucket': 'sa3edny-b7978.appspot.com'})

# Set environment variable to disable TensorRT (if needed)
os.environ['TF_DISABLE_TENSORRT'] = '1'

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set desired logging level


def verify_face(photo_path, photo2_path):
    result = DeepFace.verify(photo_path, photo2_path, model_name='Facenet', enforce_detection=False)
    return result['verified']


def process_photo(photo_path):
    matched_photos = []

    # Get a reference to the Firebase Cloud Storage bucket
    bucket = storage.bucket()

    try:
        # List all blobs (files) in the bucket
        blobs = bucket.list_blobs()

        blob: Blob
        counter = 0
        filename = f"./pic{counter}.jpg"
        for blob in blobs:
            print(blob.content_type)
            if "image" not in blob.content_type:
                continue 
            
            pic_bytes = blob.download_as_bytes(checksum=None)
            
            with open(filename, "wb") as file:
                file.write(pic_bytes)

            counter += 1
            #blob.download_to_filename(temp_file)
            # Perform face verification
            
            verified = verify_face(filename, photo_path)
            
            if verified:
                matched_photos.append({'name': blob.name, 'url': blob.public_url})

            # Explicitly delete temporary file after use
            #os.remove(temp_file.name)

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
        print(request.headers)
        photo_path = request.json.get('photo_path')
        if not photo_path:
            return jsonify({'error': 'No photo_path provided'}), 400

        matched_photos = process_photo(photo_path)

        response_data = {
            'matched_photos': matched_photos  # Include matched photo details in the response
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Default port is 10000 if not set
    app.run(host='0.0.0.0', port=port)

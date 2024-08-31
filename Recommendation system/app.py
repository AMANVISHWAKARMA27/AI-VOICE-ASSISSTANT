from flask import Flask, request, jsonify
import numpy as np
import requests
import cv2
from models.inference import recommend, recommend_from_history

app = Flask(__name__)

def preprocess_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download image. Status code: {response.status_code}")
        image = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode the image")
        return img
    except Exception as e:
        raise ValueError(f"Error processing image from URL: {str(e)}")

@app.route('/recommend', methods=['POST'])
def recommend_endpoint():
    content = request.json
    
    try:
        if 'product_name' in content and 'image' in content:
            product_name = content['product_name']
            image_url = content['image']
            
            # Process single product recommendation
            image = preprocess_image_from_url(image_url)
            recommendations = recommend(product_name, image_url, top_n=5)
            recommendations_list = recommendations.to_dict(orient='records')
            
            response = {
                "recommendations": recommendations_list
            }
        
        elif 'purchase_history' in content or 'search_history' in content:
            purchase_history = content.get('purchase_history', [])
            search_history = content.get('search_history', [])
            
            # Process recommendation based on history
            recommendations = recommend_from_history(purchase_history=purchase_history, search_history=search_history, top_n=5)
            recommendations_list = recommendations.to_dict(orient='records')
            
            response = {
                "recommendations": recommendations_list
            }
        
        else:
            return jsonify({"error": "Invalid request format"}), 400

    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Recommendation failed: {str(e)}"}), 500

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

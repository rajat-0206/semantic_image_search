import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import logging
from image_finder import ImageFinder

app = Flask(__name__)
CORS(app)

finder = None
INDEX_PATH = ".cache"

def initialize_finder():
    global finder
    if finder is None:
        finder = ImageFinder()
        if os.path.exists(f"{INDEX_PATH}/index.index"):
            try:
                finder.load_index(INDEX_PATH)
                print("Loaded existing index")
            except Exception as e:
                print(f"Failed to load index: {e}")
                finder = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    global finder
    
    try:
        initialize_finder()
        if finder is None:
            return jsonify({'error': 'No index available. Please build index first.'}), 400
        
        data = request.get_json()
        query = data.get('query')
        top_k = data.get('top_k', 10)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        results = finder.search_by_text(query, top_k)
        
        formatted_results = []
        for path, score in results:
            rel_path = os.path.relpath(path, os.getcwd())
            formatted_results.append({
                'path': rel_path,
                'full_path': path,
                'score': round(score, 4)
            })
        
        return jsonify({
            'query': query,
            'results': formatted_results,
            'total_found': len(formatted_results)
        })
        
    except Exception as e:
        print(f"Error searching: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5501) 
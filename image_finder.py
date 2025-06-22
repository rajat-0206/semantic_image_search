import os
import torch
import clip
import faiss
import numpy as np
from PIL import Image
from typing import List, Tuple
import argparse
import pickle
from tqdm import tqdm

TOTAL_IMAGES_TO_INDEX = 3000

class ImageFinder:

    MODEL_NAME = "ViT-B/32"
    
    def __init__(self):
        self.device = "cpu"        
        self.model, self.preprocess = clip.load(self.MODEL_NAME, device=self.device)        
        self.index = None
        self.image_paths = []
        self.image_embeddings = []
        self.metadata = {}
        
    def extract_image_embeddings(self, image_paths: List[str]):

        embeddings = []
        valid_paths = []
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                image = Image.open(img_path).convert('RGB')
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    embedding = image_features.cpu().numpy().flatten()
                    embeddings.append(embedding)
                    valid_paths.append(img_path)
                    
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
                continue
        
        if not embeddings:
            raise ValueError("No valid images found!")
            
        return np.array(embeddings), valid_paths
    
    def build_index(self, image_directory: str):
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        image_paths = []
        image_count = 0
        for root, dirs, files in os.walk(image_directory):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    image_paths.append(os.path.join(root, file))
                    image_count += 1
                    if image_count > TOTAL_IMAGES_TO_INDEX:
                        break

        if not image_paths:
            raise ValueError(f"No images found in {image_directory}")
        
        embeddings, valid_paths = self.extract_image_embeddings(image_paths)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.image_paths = valid_paths
        self.image_embeddings = embeddings
        
        print(f"Built FAISS index with {len(valid_paths)} images")
        
    def search_by_text(self, text_query: str, top_k: int = 10):
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        text_tokens = clip.tokenize([text_query]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_embedding = text_features.cpu().numpy().flatten()
        
        faiss.normalize_L2(text_embedding.reshape(1, -1))
        
        similarities, indices = self.index.search(
            text_embedding.reshape(1, -1).astype('float32'), 
            min(top_k, len(self.image_paths))
        )
        
        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx < len(self.image_paths):
                results.append((self.image_paths[idx], float(similarity)))
        
        return results
    
    def save_index(self, save_path: str):
        if self.index is None:
            raise ValueError("No index to save. Call build_index() first.")
        
        os.makedirs(save_path, exist_ok=True)
        
        faiss.write_index(self.index, f"{save_path}/index.index")
        
        # Save metadata
        metadata = {
            'image_paths': self.image_paths,
            'image_embeddings': self.image_embeddings.tolist() if self.image_embeddings is not None else None
        }
        
        with open(f"{save_path}/.metadata", 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Saved index to {save_path}")
    
    def load_index(self, load_path: str):
        self.index = faiss.read_index(f"{load_path}/index.index")
        with open(f"{load_path}/.metadata", 'rb') as f:
            metadata = pickle.load(f)
        
        self.image_paths = metadata['image_paths']
        self.image_embeddings = np.array(metadata['image_embeddings']) if metadata['image_embeddings'] else None
        
        print(f"Loaded index from {load_path} with {len(self.image_paths)} images")


def main():
    parser = argparse.ArgumentParser(description="Image Finder using CLIP and FAISS")
    parser.add_argument("--mode", choices=["build", "search"], required=True,
                       help="Mode: build index or search")
    parser.add_argument("--query", type=str, help="Text query for search")
    args = parser.parse_args()

    mode = args.mode if args.mode else "search"
    img_dir = "images"
    cache_dir = ".cache"
    
    finder = ImageFinder()
    if mode == "build":
        print(f"Building index from {img_dir}")
        finder.build_index(img_dir)
        finder.save_index(cache_dir)
        
    elif mode == "search":
        finder.load_index(cache_dir)
        if args.query:
            print(f"Searching for: {args.query}")
            results = finder.search_by_text(args.query)
            print(f"\nTop {len(results)} results for '{args.query}':")
            for i, (path, score) in enumerate(results, 1):
                print(f"{i}. {path} (similarity: {score:.4f})")            

if __name__ == "__main__":
    main()

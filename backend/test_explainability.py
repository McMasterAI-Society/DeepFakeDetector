"""Test script for explainability features."""
import asyncio
import traceback
import numpy as np
from PIL import Image
import io

async def main():
    from app.services.model_registry import get_model_registry
    from app.core.config import settings
    
    registry = get_model_registry()
    
    # Load models from fusion repo
    print("Loading models...")
    await registry.load_from_fusion_repo(settings.HF_FUSION_REPO_ID)
    print("Models loaded!")
    
    # Create a test image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_bytes = buf.getvalue()
    
    # Test each model
    models = ['cnn-transfer', 'gradfield-cnn', 'vit-base', 'deit-distilled']
    
    for model_name in models:
        print(f"\nTesting {model_name}...")
        try:
            model = registry.get_submodel(model_name)
            result = model.predict(image_bytes=img_bytes, explain=True)
            has_heatmap = 'heatmap_base64' in result
            print(f"  Success! pred={result['pred']}, has_heatmap={has_heatmap}")
            if has_heatmap:
                # Check heatmap is valid base64
                import base64
                decoded = base64.b64decode(result['heatmap_base64'])
                print(f"  Heatmap size: {len(decoded)} bytes")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

import torch
import sys
import os

sys.path.append('src')

try:
    print("Loading weights from best_model.pth...")
    video_state = torch.load('models/best_model.pth', map_location='cpu', weights_only=True)
    
    from model import SpatialModel
    spatial = SpatialModel()
    spatial_state = spatial.state_dict()
    
    # Map feature_extractor
    for k in video_state.keys():
        if k.startswith('feature_extractor.'):
            spatial_state[k] = video_state[k]
        elif k.startswith('frame_classifier.'):
            # Map frame_classifier to classifier
            new_k = k.replace('frame_classifier.', 'classifier.')
            spatial_state[new_k] = video_state[k]

    spatial.load_state_dict(spatial_state)
    torch.save(spatial_state, 'models/best_image_model.pth')
    print("Weight transfer to best_image_model.pth complete!")

except Exception as e:
    print("Error:", e)


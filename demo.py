from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import torch
from ultralytics import YOLO
from PIL import Image
import base64
import matplotlib.pyplot as plt
import io

def initialize_models(device='cuda'):
    """Initialize the YOLO and caption models."""
    # Initialize YOLO model
    som_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
    som_model.to(device)
    print('model to {}'.format(device))
    
    # Initialize caption model (choose one of the two options)
    # Option 1: BLIP2
    # caption_model_processor = get_caption_model_processor(
    #     model_name="blip2",
    #     model_name_or_path="weights/icon_caption_blip2",
    #     device=device
    # )
    
    # Option 2: Florence2
    caption_model_processor = get_caption_model_processor(
        model_name="florence2",
        model_name_or_path="weights/icon_caption_florence",
        device=device
    )
    
    return som_model, caption_model_processor

def process_image(image_path, som_model, caption_model_processor, box_threshold=0.03):
    """Process an image and return labeled results."""
    # Configuration for drawing bounding boxes
    draw_bbox_config = {
        'text_scale': 0.8,
        'text_thickness': 2,
        'text_padding': 3,
        'thickness': 3,
    }
    
    # Open and convert image
    image = Image.open(image_path)
    image_rgb = image.convert('RGB')
    
    # Perform OCR
    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_path,
        display_img=False,
        output_bb_format='xyxy',
        goal_filtering=None,
        easyocr_args={'paragraph': False, 'text_threshold': 0.9},
        use_paddleocr=True
    )
    text, ocr_bbox = ocr_bbox_rslt
    
    # Get labeled image and results
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_path,
        som_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=False,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        use_local_semantics=True,
        iou_threshold=0.1
    )
    
    return dino_labled_img, label_coordinates, parsed_content_list

def display_results(dino_labled_img, parsed_content_list):
    """Display the labeled image and print content list."""
    # Display the labeled image
    plt.figure(figsize=(12, 12))
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    plt.axis('off')
    plt.imshow(image)
    plt.show()
    
    # Print parsed content
    for content in parsed_content_list:
        print(content)

def main():
    # Initialize device and models
    device = 'cuda'
    som_model, caption_model_processor = initialize_models(device)
    
    # Process image
    image_path = 'OmniParser/imgs/windows_multitab.png'  # You can change this path as needed
    dino_labled_img, label_coordinates, parsed_content_list = process_image(
        image_path,
        som_model,
        caption_model_processor
    )
    
    # Display results
    display_results(dino_labled_img, parsed_content_list)

if __name__ == "__main__":
    main()
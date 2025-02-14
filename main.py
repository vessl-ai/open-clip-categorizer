import argparse
import os

import pandas as pd

from product_categorizer import ProductCategorizer, evaluate_predictions
from utils import display_sample_predictions, visualize_predictions_grid



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Product Categorization with CLIP')
    parser.add_argument('--h5_path', type=str, default='/dataset/image_dataset_small.h5',
                      help='Path to H5 dataset file')
    parser.add_argument('--csv_path', type=str, default='/dataset/gsshop_fashion_sample_train_small_revision.csv',
                      help='Path to CSV file containing category information')
    parser.add_argument('--model_name', type=str, default='ViT-L-14',
                      help='Name of the CLIP model to use')
    parser.add_argument('--pretrained', type=str, default='/model/open_clip_pytorch_model.bin',
                      help='Path to pretrained model weights')
    parser.add_argument('--image_weight', type=float, default=0.7,
                      help='Weight for image features in classification')
    parser.add_argument('--text_weight', type=float, default=0.3,
                      help='Weight for text features in classification')
    parser.add_argument('--translate_client', type=str, help='Naver translate client id')
    parser.add_argument('--translate_secret', type=str, help='Naver translate secret')
    parser.add_argument('--output_path', type=str, default='/output',
                      help='Path to save output files')
    
    args = parser.parse_args()

    # Initialize categorizer
    categorizer = ProductCategorizer(
        model_name=args.model_name, 
        pretrained=args.pretrained,
        image_weight=args.image_weight, 
        text_weight=args.text_weight,
        translate_client=args.translate_client,
        translate_secret=args.translate_secret
    )
    categorizer.load_categories(args.csv_path)

    # Load dataset
    df = pd.read_csv(args.csv_path, sep='\t')

    # Get predictions
    predictions = categorizer.process_dataset(args.h5_path, df)

    # Evaluate
    evaluate_predictions(predictions, df, categorizer)

    # Show sample predictions
    display_sample_predictions(predictions, df)

    # Visualize results
    output_file = os.path.join(args.output_path, 'predictions_grid.png')
    visualize_predictions_grid(df, predictions, args.h5_path, output_file=output_file)

if __name__ == "__main__":
    main()
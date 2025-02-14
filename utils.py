import h5py
import matplotlib.font_manager as fm 
import matplotlib.pyplot as plt


def display_sample_predictions(predictions, df, num_samples=5):
    """Display detailed predictions for sample items"""
    print("\nSample Predictions:")
    for i in range(min(num_samples, len(predictions))):
        print(f"\nItem {i+1}:")
        print(f"Product: {df.iloc[i]['doc']}")
        print(f"True category: {df.iloc[i]['label_name']}")
        
        print("\nPredictions by level:")
        for level, level_preds in predictions[i]['level_predictions'].items():
            print(f"\n{level.title()}:")
            for pred, conf in level_preds:
                print(f"  {pred}: {conf:.2f}%")
        
        print("\nFull path predictions:")
        for path, conf in predictions[i]['full_path'][:3]:
            print(f"  {path}: {conf:.2f}%")


def visualize_predictions_grid(df, predictions, h5_path, ncols=4, output_file='predictions_grid.png'):
    """Visualize predictions with images and classification results"""
    # Set font for Korean text
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['font.sans-serif'] = ['NanumGothic']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Calculate grid dimensions
    n_items = len(predictions)
    nrows = (n_items + ncols - 1) // ncols
    
    # Create figure
    fig = plt.figure(figsize=(6*ncols, 10*nrows))
    
    # Open H5 file
    with h5py.File(h5_path, 'r') as h5f:
        for idx, (pred_dict, (_, row)) in enumerate(zip(predictions, df.iterrows())):
            try:
                # Try different ways to access the image
                if 'images' in h5f:
                    img_data = h5f['images'][idx]
                else:
                    img_key = list(h5f.keys())[idx]
                    img_data = h5f[img_key][:]
                
                # Get predictions
                top_pred = pred_dict['full_path'][0][0]
                true_path = row['label_name']
                is_correct = top_pred == true_path
                confidence = pred_dict['full_path'][0][1]
                
                # Create subplot
                ax = fig.add_subplot(nrows, ncols, idx + 1)
                
                # Display image
                ax.imshow(img_data)
                ax.axis('off')
                
                # Format paths for better readability
                true_path_formatted = true_path.replace('>', ' → ')
                pred_path_formatted = top_pred.replace('>', ' → ')
                
                # Get product description
                doc = row['doc']
                max_chars = 100
                if len(doc) > max_chars:
                    doc = doc[:max_chars] + "..."
                
                # Create title
                title_color = 'green' if is_correct else 'red'
                title = f"Product: {doc}\n\n"
                title += f"True:\n{true_path_formatted}\n\n"
                title += f"Pred:\n{pred_path_formatted}\n\n"
                title += f"Conf: {confidence:.1f}%"
                
                ax.set_title(title, color=title_color, fontsize=9, fontproperties=font_prop, 
                            pad=10, wrap=True)
                
            except Exception as e:
                print(f"Error processing index {idx}: {str(e)}")
                continue
    
    # Calculate and display overall accuracy
    correct_predictions = sum(1 for pred, (_, row) in zip(predictions, df.iterrows())
                            if pred['full_path'][0][0] == row['label_name'])
    accuracy = correct_predictions / len(predictions) * 100
    plt.suptitle(f'Overall Accuracy: {accuracy:.1f}%', fontsize=14, y=1.02, fontproperties=font_prop)
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.show()
import base64
import io
import json
import re
import urllib.parse

import h5py
import open_clip
import pandas as pd
from PIL import Image
from soynlp.tokenizer import LTokenizer
import torch


class ProductCategorizer:
    def __init__(self, model_name="ViT-L-14-quickgelu", pretrained="openai", image_weight=0.7, text_weight=0.3, translate_client=None, translate_secret=None):
        # Initialize CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.image_weight = image_weight
        self.text_weight = text_weight
        self.tokenizer_ko = LTokenizer()
        self.translate_client = translate_client
        self.translate_secret = translate_secret

        # Add length detection related attributes
        self.length_prompts = {
            'mini': [
                "a mini length dress above the knee",
                "short length dress or skirt",
                "above knee length clothing",
                "미니 길이 의류",
                "무릎 위 길이"
            ],
            'midi': [
                "a midi length dress below the knee",
                "mid-calf length dress or skirt",
                "knee to ankle length clothing",
                "미디 길이 의류",
                "종아리 길이"
            ],
            'long': [
                "a long maxi dress covering ankles",
                "full length dress or skirt",
                "ankle length clothing",
                "롱 길이 의류",
                "발목 길이"
            ]
        }
        
        # Length keywords
        self.length_keywords = {
            'mini': ['미니', '숏', '쇼트', '미니멀'],
            'midi': ['미디', '미디엄', '미들'],
            'long': ['롱', '맥시', '롱한', '긴', '풀']
        }
        
        # Categories that need length detection
        self.length_categories = ['원피스', '스커트', '코트']
        
        # Pre-compute length embeddings
        self._compute_length_embeddings()

    def _compute_length_embeddings(self):
        """Pre-compute embeddings for length classification"""
        self.length_embeddings = {}
        with torch.no_grad():
            for length, prompts in self.length_prompts.items():
                tokens = self.tokenizer(prompts).to(self.device)
                features = self.model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
                self.length_embeddings[length] = features.mean(dim=0)

    def detect_length(self, image_features, title, product_type):
        """Detect product length from both title and image"""
        if not any(cat in product_type for cat in self.length_categories):
            return None, 0.0

        # First check title for explicit length mentions
        title_lower = title.lower()
        for length, keywords in self.length_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                return length, 1.0

        # If no length found in title, use image analysis
        with torch.no_grad():
            similarities = {}
            for length, embedding in self.length_embeddings.items():
                similarity = (100.0 * image_features @ embedding.T).item()
                similarities[length] = similarity
            
            best_length = max(similarities.items(), key=lambda x: x[1])
            return best_length

    
    def translate_to_english(self, text):
        """Translate Korean text to English using Papago API"""
        try:
            # Skip translation if text is empty or already in English
            if not text or all(ord(c) < 128 for c in text):
                return text
                
            client_id = self.translate_client
            client_secret = self.translate_secret
            
            # URL encode the text
            encText = urllib.parse.quote(text)
            data = "source=ko&target=en&text=" + encText
            url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
            
            # Create request
            request = urllib.request.Request(url)
            request.add_header("X-NCP-APIGW-API-KEY-ID", client_id)
            request.add_header("X-NCP-APIGW-API-KEY", client_secret)
            
            # Send request and get response
            response = urllib.request.urlopen(request, data=data.encode("utf-8"))
            rescode = response.getcode()
            
            if rescode == 200:
                # Parse JSON response
                response_body = response.read().decode('utf-8')
                translated_text = json.loads(response_body)['message']['result']['translatedText']
                return translated_text
            else:
                print(f"Translation error (code: {rescode}) for text: {text}")
                return text
            
        except Exception as e:
            print(f"Translation failed for text '{text}': {str(e)}")
            return text  # Return original text if translation fails

        
    def process_categories(self, label_name):
        """Process four-level category labels"""
        # Split hierarchical categories
        categories = label_name.split('>')
        department = categories[0].strip()
        category = categories[1].strip() if len(categories) > 1 else ""
        subcategory = categories[2].strip() if len(categories) > 2 else ""
        product_type = categories[3].strip() if len(categories) > 3 else ""

        # Translate each category level
        dept_en = self.translate_to_english(department)
        cat_en = self.translate_to_english(category)
        subcat_en = self.translate_to_english(subcategory)
        prod_en = self.translate_to_english(product_type)
        
        # Create prompts in both languages
        prompts = [
            # English prompts
            f"a fashion product in {dept_en}, {cat_en}, {subcat_en}, specifically a {prod_en}",
            f"a {prod_en} which is a {subcat_en} in the {cat_en} section of {dept_en}",
            f"this is a {prod_en} from {subcat_en}",
            # Korean prompts
            f"패션 제품: {department} > {category} > {subcategory} > {product_type}",
            f"{department}의 {category} 섹션에 있는 {subcategory} 중 {product_type}",
            f"{product_type} 제품"
        ]

        print(f"prompts: ", prompts)
        return prompts

    def get_category_levels(self, label_name):
        """Extract all category levels"""
        parts = label_name.split('>')
        return {
            'department': parts[0].strip(),
            'category': parts[1].strip() if len(parts) > 1 else "",
            'subcategory': parts[2].strip() if len(parts) > 2 else "",
            'product_type': parts[3].strip() if len(parts) > 3 else ""
        }
    
    def load_categories(self, train_csv, val_csv=None):
        # Load data
        self.train_df = pd.read_csv(train_csv, sep='\t')
        if val_csv:
            self.val_df = pd.read_csv(val_csv, sep='\t')
        
        # Get unique categories at each level
        self.category_levels = {
            'department': set(),
            'category': set(),
            'subcategory': set(),
            'product_type': set()
        }

        # Store translations
        self.translations = {}

        print("Translating categories... This might take a moment.")
        
        # Process all categories
        self.categories = self.train_df['label_name'].unique()
        self.category_prompts = {}
        
        for full_category in self.categories:
            levels = self.get_category_levels(full_category)
            for level, value in levels.items():
                if value and value not in self.translations:
                    self.translations[value] = self.translate_to_english(value)
                if value:  # Only add non-empty values
                    self.category_levels[level].add(value)
            self.category_prompts[full_category] = self.process_categories(full_category)
        
        # Pre-compute text embeddings for each level and full path
        self.text_features = {}
        with torch.no_grad():
            # Embed full paths
            for category, prompts in self.category_prompts.items():
                tokens = self.tokenizer(prompts).to(self.device)
                text_features = self.model.encode_text(tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                self.text_features[category] = text_features.mean(dim=0)
            
            # Embed individual levels
            for level, values in self.category_levels.items():
                self.text_features[f"level_{level}"] = {}
                for value in values:
                    prompts = [
                        f"a {level}: {value}",
                        f"this belongs to {level} {value}",
                        f"{value} {level}"
                    ]
                    tokens = self.tokenizer(prompts).to(self.device)
                    text_features = self.model.encode_text(tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    self.text_features[f"level_{level}"][value] = text_features.mean(dim=0)
        
        print("\nCategory statistics:")
        for level, values in self.category_levels.items():
            print(f"{level.capitalize()}: {len(values)} unique values")

    def process_image(self, image):
        """Process a single image for CLIP model"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply CLIP preprocessing
        processed = self.preprocess(image)
        return processed.unsqueeze(0).to(self.device)

    
    def predict_single(self, image, title):
        """Predict for a single image with title, including level-wise predictions"""
        image_input = self.process_image(image)
        
        with torch.no_grad():
            # Get image features
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # First get preliminary category to check if length detection is needed
            basic_prompts = self.process_title(title)
            basic_tokens = self.tokenizer(basic_prompts).to(self.device)
            basic_features = self.model.encode_text(basic_tokens)
            basic_features = basic_features.mean(dim=0)
            basic_features /= basic_features.norm(dim=-1, keepdim=True)
            
            # Get preliminary product type
            prelim_combined = (image_features * self.image_weight + basic_features * self.text_weight)
            prelim_combined /= prelim_combined.norm(dim=-1, keepdim=True)
            
            # Get preliminary category predictions
            prelim_level_predictions = {}
            for level in ['department', 'category', 'subcategory', 'product_type']:
                level_similarities = {}
                for value, text_features in self.text_features[f"level_{level}"].items():
                    similarity = (100.0 * prelim_combined @ text_features.T).item()
                    level_similarities[value] = similarity
                prelim_level_predictions[level] = sorted(level_similarities.items(), 
                                                       key=lambda x: x[1], reverse=True)[:1]
            
            # Check if product type needs length detection
            product_type = prelim_level_predictions['product_type'][0][0]
            
            # Detect length and enhance title if needed
            if any(cat in product_type for cat in ['원피스', '스커트', '코트']):
                length, confidence = self.detect_length(image_features.squeeze(0), title, product_type)
                if length and confidence > 0.5:
                    length_term = {
                        'mini': '미니',
                        'midi': '미디',
                        'long': '롱'
                    }.get(length, length)
                    # Enhance title with length information
                    title = f"{length_term} {title}"
            
            # Get enhanced title features
            title_prompts = self.process_title(title)
            print(f"title: {title_prompts}")
            title_tokens = self.tokenizer(title_prompts).to(self.device)
            title_features = self.model.encode_text(title_tokens)
            title_features = title_features.mean(dim=0)
            title_features /= title_features.norm(dim=-1, keepdim=True)
            
            # Combine features
            combined_features = (image_features * self.image_weight + title_features * self.text_weight)
            combined_features /= combined_features.norm(dim=-1, keepdim=True)
            
            # Get predictions for full paths
            full_path_predictions = {}
            for category, text_features in self.text_features.items():
                if not category.startswith('level_'):
                    similarity = (100.0 * combined_features @ text_features.T).item()
                    full_path_predictions[category] = similarity
            
            # Get predictions for each level
            level_predictions = {}
            for level in ['department', 'category', 'subcategory', 'product_type']:
                level_similarities = {}
                for value, text_features in self.text_features[f"level_{level}"].items():
                    similarity = (100.0 * combined_features @ text_features.T).item()
                    level_similarities[value] = similarity
                level_predictions[level] = sorted(level_similarities.items(), 
                                               key=lambda x: x[1], reverse=True)[:3]
            
            # Sort full path predictions
            sorted_full_predictions = sorted(full_path_predictions.items(), 
                                          key=lambda x: x[1], reverse=True)[:5]
            
            # Add length information to the results if detected
            result = {
                'full_path': sorted_full_predictions,
                'level_predictions': level_predictions
            }
            
            if any(cat in product_type for cat in ['원피스', '스커트', '코트']):
                result['length_info'] = {
                    'detected_length': length,
                    'confidence': confidence
                }
            
            return result
    

    def process_dataset(self, h5_path, df, is_training=True):
        """Process entire dataset and return predictions"""
        predictions_list = []
        batch_size = 32
        
        with h5py.File(h5_path, 'r') as file:
            images = file['images']
            total_images = len(df)
            
            for i in range(0, total_images, batch_size):
                batch_indices = range(i, min(i + batch_size, total_images))
                batch_predictions = []
                
                for idx in batch_indices:
                    # Get image and title
                    image = Image.fromarray(images[idx])
                    title = df.iloc[idx]['doc']
                    
                    # Get predictions for this item
                    predictions = self.predict_single(image, title)
                    batch_predictions.append(predictions)
                
                predictions_list.extend(batch_predictions)
                
                # Progress update
                current_count = min(i + batch_size, total_images)
                print(f"{'Training' if is_training else 'Validation'}: "
                      f"Processed {current_count}/{total_images} images "
                      f"({(current_count/total_images)*100:.1f}%)")
                
                # Optional: add a small batch of examples
                if i == 0:
                    print("\nSample predictions from first batch:")
                    for j in range(min(3, len(batch_predictions))):
                        print(f"\nItem {j+1}:")
                        print(f"Product: {df.iloc[j]['doc']}")
                        print(f"True category: {df.iloc[j]['label_name']}")
                        print("Top predictions:")
                        for path, conf in batch_predictions[j]['full_path'][:3]:
                            print(f"  {path}: {conf:.2f}%")
                    print("\nContinuing processing...\n")
        
        return predictions_list
        
    def separate_korean_words(self, text):
        """Separate concatenated Korean words with expanded patterns from dataset"""
        import re
        
        # Expanded fashion patterns based on the dataset
        fashion_patterns = {
            # Tops patterns
            r'(긴팔|반팔|민소매|하프|롱|크롭)(티셔츠|님티|맨투맨|티|셔츠|블라우스|탑)',
            r'(라운드|브이넥|터틀넥|하이넥|오프숄더)(티셔츠|니트|맨투맨|블라우스)',
            r'(오버|슬림|루즈|베이직|캐주얼)(핏|사이즈)',
            
            # Outerwear patterns
            r'(캐주얼|데일리|베이직|오버|슬림)(자켓|코트|점퍼|베스트)',
            r'(기모|누빔|퀼팅|패딩|플리스|양털|무스탕)(자켓|조끼|점퍼|코트|베스트)',
            r'(트렌치|더블|싱글|테일러드)(코트|자켓)',
            r'(윈터|리버서블|데님|체크|스트라이프)(자켓|코트|점퍼)',
            
            # Bottoms patterns
            r'(슬림|와이드|스트레이트|부츠컷|조거|밴딩)(팬츠|진|슬랙스|바지)',
            r'(미니|미디|롱|맥시|플레어|에이라인|플리츠)(스커트|원피스)',
            r'(하이|미드|로우)(웨이스트|라이즈)',
            r'(데님|코튼|린넨|울|레더)(팬츠|스커트|원피스)',
            
            # Dress patterns
            r'(미니|미디|롱|맥시)(원피스|드레스)',
            r'(셔츠|랩|점프|후드)(원피스|수트)',
            r'(플리츠|플레어|타이트|루즈|프릴)(원피스|드레스)',
            
            # Material/Style patterns
            r'(코듀로이|데님|니트|레더|스웨이드|트위드)(원피스|자켓|스커트|팬츠)',
            r'(캐시미어|울|앙고라|폴라|기모)(니트|코트|자켓)',
            r'(체크|도트|스트라이프|플라워|프린트)(패턴|무늬)',
            
            # Season/Weather patterns
            r'(봄|여름|가을|겨울)(신상|시즌|컬렉션)',
            r'(썸머|윈터|스프링|올)(시즌|웨어|룩)',
            
            # Set items
            r'(투피스|세트|셋업)(상하의|수트|룩)',
            r'(상하|세트|투피스)(구성|세트)',
            
            # Size/Fit patterns
            r'(빅|스몰|라지|미들)(사이즈|픽)',
            r'(오버|루즈|슬림|베이직)(핏|스타일|사이즈)',
            
            # Additional common combinations
            r'(여성|남성|공용)(용|복|의류|패션)',
            r'(데일리|캐주얼|베이직|트렌디)(룩|스타일|아이템)',
            r'(프리|원|투|스리|포)(사이즈|컬러)'
        }
        
        # First, add spaces for known patterns
        for pattern in fashion_patterns:
            text = re.sub(pattern, r'\1 \2', text)
        
        # Separate numbers and English from Korean
        text = re.sub(r'([가-힣])([A-Za-z0-9])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z0-9])([가-힣])', r'\1 \2', text)
        
        # Use soynlp for remaining concatenated Korean words
        words = self.tokenizer_ko.tokenize(text)
        return ' '.join(words)

    
    def process_title(self, title):
        """Process product title with improved Korean word separation"""
        # Extract brand name from brackets (if exists)
        brand = ""
        brand_match = re.search(r'\[([^\]]+)\]', title)
        if brand_match:
            brand = brand_match.group(1)
        
        # Clean the title but preserve brand information
        cleaned_title = title
        cleaned_title = re.sub(r'\([^\)]*\)', '', cleaned_title)  # Remove (부가정보)
        cleaned_title = re.sub(r'_[0-9-]+$', '', cleaned_title)   # Remove _123-456
        cleaned_title = re.sub(r'\b(?<![\[\]])[A-Z0-9]+(?:[A-Z0-9-_.]+[A-Z0-9]+)?\b', '', cleaned_title)  # Remove product codes
        
        # Separate concatenated Korean words
        cleaned_title = self.separate_korean_words(cleaned_title)
        
        # Clean extra whitespace
        cleaned_title = ' '.join(cleaned_title.split())
        
        # Translate to English
        title_en = self.translate_to_english(cleaned_title)
        
        # If we have a brand, add it explicitly to the English translation if not already present
        if brand and brand.lower() not in title_en.lower():
            title_en = f"[{brand}] {title_en}"
        
        # Create bilingual title
        bilingual_title = f"{cleaned_title} | {title_en}"
        
        prompts = [
            # Bilingual prompts with brand context
            bilingual_title,
            f"fashion item: {bilingual_title}",
            f"{'brand clothing' if brand else 'clothing item'}: {bilingual_title}",
            
            # Original Korean with brand context
            f"패션 상품: {cleaned_title}",
            cleaned_title,
            
            # English translation
            f"fashion product: {title_en}",
            title_en
        ]
        return prompts

    def process_image(self, image):
        """Process a single image for CLIP model"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply CLIP preprocessing
        processed = self.preprocess(image)
        return processed.unsqueeze(0).to(self.device)

    def process_base64_image(self, base64_string: str):
        """Process a base64 encoded image"""
        try:
            # Remove header if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_bytes))
            return image
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {str(e)}")


def evaluate_predictions(predictions, df, categorizer):
    """Evaluate predictions at each level"""
    metrics = {
        'department': {'correct': 0, 'total': 0},
        'category': {'correct': 0, 'total': 0},
        'subcategory': {'correct': 0, 'total': 0},
        'product_type': {'correct': 0, 'total': 0},
        'full_path': {'correct': 0, 'total': 0}
    }
    
    for i, pred_dict in enumerate(predictions):
        true_label = df.iloc[i]['label_name']
        true_levels = categorizer.get_category_levels(true_label)
        
        # Evaluate full path
        top_full_pred = pred_dict['full_path'][0][0]
        metrics['full_path']['total'] += 1
        if top_full_pred == true_label:
            metrics['full_path']['correct'] += 1
        
        # Evaluate each level
        for level, true_value in true_levels.items():
            if true_value:  # Only evaluate if we have a true value
                metrics[level]['total'] += 1
                top_level_pred = pred_dict['level_predictions'][level][0][0]
                if top_level_pred == true_value:
                    metrics[level]['correct'] += 1
    
    # Print results
    print("\nPrediction Results:")
    for metric, values in metrics.items():
        if values['total'] > 0:
            accuracy = values['correct'] / values['total']
            print(f"{metric.replace('_', ' ').title()} Accuracy: {accuracy:.2%}")
            
    return metrics
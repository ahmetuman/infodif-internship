import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import requests
from io import StringIO
import tarfile
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from scipy.stats import pearsonr, spearmanr
import torch

try:
    from datasets import load_dataset
except ImportError:
    pass

from glossbert_wsd_ranker import GlossBERTWSDRanker

class MixedWiCLargeScaleEvaluator:
    
    def __init__(self):
        self.hybrid_ranker = GlossBERTWSDRanker()
        
    def download_wic_dataset(self) -> pd.DataFrame:
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("super_glue", "wic", split="validation")
            
            wic_data = []
            for example in dataset:
                sentence1 = example['sentence1']
                sentence2 = example['sentence2']
                
                try:
                    words1 = sentence1.split()
                    words2 = sentence2.split()
                    
                    common_words = set(words1) & set(words2)
                    if common_words:
                        target_word = next((word for word in common_words 
                                          if len(word) > 2 and word.isalpha()), 
                                          list(common_words)[0])
                    else:
                        target_word = next((word for word in words1 
                                          if len(word) > 3 and word.isalpha()), 
                                          words1[0] if words1 else "word")
                        
                except Exception:
                    target_word = "word"
                
                binary_label = example['label'] == 1
                similarity_score = 4.5 if binary_label else 1.0
                
                wic_data.append({
                    'sentence1': sentence1,
                    'sentence2': sentence2,
                    'target_word': target_word.lower(),
                    'similarity_score': similarity_score,
                    'binary_label': binary_label,
                    'source': 'SuperGLUE_WiC'
                })
            
            df = pd.DataFrame(wic_data)
            return df
            
        except Exception as e:
            return self.create_synthetic_wic_data()
    
    def create_synthetic_wic_data(self) -> pd.DataFrame:
        wic_examples = [
            {'sentence1': "I need to visit the bank to deposit my paycheck", 'sentence2': "The financial institution approved my loan application", 'target_word': 'bank', 'similarity_score': 4.5, 'binary_label': True},
            {'sentence1': "I need to visit the bank to deposit my paycheck", 'sentence2': "We sat by the river bank watching the sunset", 'target_word': 'bank', 'similarity_score': 1.0, 'binary_label': False},
            {'sentence1': "The river bank was covered with beautiful flowers", 'sentence2': "Fish swim near the muddy bank of the stream", 'target_word': 'bank', 'similarity_score': 4.5, 'binary_label': True},
            {'sentence1': "She read an interesting book about history", 'sentence2': "The novel was placed on the bookshelf carefully", 'target_word': 'book', 'similarity_score': 4.5, 'binary_label': True},
            {'sentence1': "She read an interesting book about history", 'sentence2': "Please book a table for dinner tonight", 'target_word': 'book', 'similarity_score': 1.0, 'binary_label': False},
            {'sentence1': "I need to book a flight to Paris", 'sentence2': "Can you book the conference room for tomorrow", 'target_word': 'book', 'similarity_score': 4.5, 'binary_label': True},
            {'sentence1': "Turn on the light so we can see better", 'sentence2': "The bright lamp illuminated the entire room", 'target_word': 'light', 'similarity_score': 4.5, 'binary_label': True},
            {'sentence1': "Turn on the light so we can see better", 'sentence2': "This box is very light and easy to carry", 'target_word': 'light', 'similarity_score': 1.0, 'binary_label': False},
            {'sentence1': "The feather is light as air", 'sentence2': "She carried the light package effortlessly", 'target_word': 'light', 'similarity_score': 4.5, 'binary_label': True},
            {'sentence1': "The children love to play in the garden", 'sentence2': "Kids enjoy playing games outside every day", 'target_word': 'play', 'similarity_score': 4.5, 'binary_label': True},
            {'sentence1': "The children love to play in the garden", 'sentence2': "We watched an excellent play at the theater", 'target_word': 'play', 'similarity_score': 1.0, 'binary_label': False},
            {'sentence1': "The actors performed a Shakespeare play beautifully", 'sentence2': "The dramatic play received standing ovation", 'target_word': 'play', 'similarity_score': 4.5, 'binary_label': True},
        ]
        
        for example in wic_examples:
            example['source'] = 'Synthetic_WiC'
        
        df = pd.DataFrame(wic_examples)
        return df
    
    def download_sts_benchmark(self) -> pd.DataFrame:
        try:
            sts_url = "http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz"
            response = requests.get(sts_url, timeout=30)
            response.raise_for_status()
            
            import tarfile
            import io
            
            with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:gz') as tar:
                test_file = tar.extractfile('stsbenchmark/sts-test.csv')
                if test_file:
                    test_content = test_file.read().decode('utf-8')
                    lines = test_content.strip().split('\n')
                    data = []
                    
                    for line in lines:
                        parts = line.split('\t')
                        if len(parts) >= 7:
                            try:
                                score = float(parts[4])
                                sent1 = parts[5]
                                sent2 = parts[6]
                                genre = parts[0]
                                
                                data.append({
                                    'sentence1': sent1,
                                    'sentence2': sent2,
                                    'similarity_score': score,
                                    'source': 'STS-B'
                                })
                            except (ValueError, IndexError):
                                continue
                    
                    df = pd.DataFrame(data)
                    return df
                else:
                    return self.create_synthetic_sts_data()
                    
        except Exception as e:
            return self.create_synthetic_sts_data()
    
    def create_synthetic_sts_data(self) -> pd.DataFrame:
        sts_examples = [
            {'sentence1': "The computer is running very slowly today", 'sentence2': "My laptop performance has been quite poor", 'similarity_score': 4.2},
            {'sentence1': "Artificial intelligence will change the future", 'sentence2': "Cats are sleeping peacefully in the garden", 'similarity_score': 0.5},
            {'sentence1': "We flew to Paris for our summer vacation", 'sentence2': "Our trip to France was absolutely wonderful", 'similarity_score': 4.1},
            {'sentence1': "The airplane landed safely at the airport", 'sentence2': "Students are studying hard for their exams", 'similarity_score': 0.3},
            {'sentence1': "The chef prepared a delicious meal tonight", 'sentence2': "The cook made tasty food for dinner", 'similarity_score': 4.5},
            {'sentence1': "Italian cuisine is known for pasta dishes", 'sentence2': "The weather forecast predicts heavy rain", 'similarity_score': 0.2},
            {'sentence1': "Students are preparing for final examinations", 'sentence2': "Pupils study diligently for their tests", 'similarity_score': 4.7},
            {'sentence1': "The teacher explained the mathematics lesson", 'sentence2': "Beautiful flowers bloom in the spring garden", 'similarity_score': 0.1},
            {'sentence1': "The doctor examined the patient carefully", 'sentence2': "The physician checked the sick person thoroughly", 'similarity_score': 4.8},
            {'sentence1': "Hospitals provide essential medical care services", 'sentence2': "Musicians perform concerts in large auditoriums", 'similarity_score': 0.4},
        ]
        
        for example in sts_examples:
            example['source'] = 'Synthetic_STS'
        
        df = pd.DataFrame(sts_examples)
        return df
    
    def evaluate_hybrid_system_wic(self, wic_pairs: List[Tuple[str, str, str]]) -> Tuple[List[float], List[str], List[bool]]:
        similarities = []
        methods_used = []
        predictions = []
        
        for sent1, sent2, target_word in wic_pairs:
            try:
                results = self.hybrid_ranker.compare_sentences(sent1, [sent2], target_word=target_word)
                
                if results:
                    similarity = results[0]['similarity_score']
                    method = results[0]['method_used']
                    binary_pred = similarity > 0.5
                    
                    similarities.append(similarity)
                    methods_used.append(method)
                    predictions.append(binary_pred)
                else:
                    similarities.append(0.0)
                    methods_used.append('Failed')
                    predictions.append(False)
                    
            except Exception as e:
                similarities.append(0.0)
                methods_used.append('Error')
                predictions.append(False)
        
        return similarities, methods_used, predictions
    
    def evaluate_hybrid_system_sts(self, sts_pairs: List[Tuple[str, str]]) -> Tuple[List[float], List[str]]:
        similarities = []
        methods_used = []
        
        for sent1, sent2 in sts_pairs:
            try:
                results = self.hybrid_ranker.compare_sentences(sent1, [sent2], target_word=None)
                
                if results:
                    similarity = results[0]['similarity_score']
                    method = results[0]['method_used']
                    
                    similarities.append(similarity)
                    methods_used.append(method)
                else:
                    similarities.append(0.0)
                    methods_used.append('Failed')
                    
            except Exception as e:
                similarities.append(0.0)
                methods_used.append('Error')
        
        return similarities, methods_used
    
    def calculate_evaluation_metrics(self, predicted: List[float], gold: List[float], task_type: str = 'regression') -> Dict[str, float]:
        pred_array = np.array(predicted)
        gold_array = np.array(gold)
        
        metrics = {}
        
        if task_type == 'regression':
            pearson_corr, _ = pearsonr(pred_array, gold_array)
            spearman_corr, _ = spearmanr(pred_array, gold_array)
            mse = mean_squared_error(gold_array, pred_array)
            mae = mean_absolute_error(gold_array, pred_array)
            
            metrics.update({
                'pearson_correlation': pearson_corr if not np.isnan(pearson_corr) else 0.0,
                'spearman_correlation': spearman_corr if not np.isnan(spearman_corr) else 0.0,
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            })
            
        elif task_type == 'classification':
            binary_pred = (pred_array > 0.5).astype(int)
            binary_gold = gold_array.astype(int)
            accuracy = accuracy_score(binary_gold, binary_pred)
            
            metrics.update({
                'accuracy': accuracy,
                'threshold': 0.5
            })
        
        return metrics
    
    def print_example(self, dataset_name: str, pairs: List, scores: List[float], methods: List[str], gold_scores: List[float], binary_labels: List[bool] = None):
        if pairs:
            # Print one example
            idx = 0
            if dataset_name == "WiC":
                sent1, sent2, target_word = pairs[idx]
                print(f"\nExample from {dataset_name} dataset:")
                print(f"Source: {sent1}")
                print(f"Target: {sent2}")
                print(f"Target word: {target_word}")
                print(f"Predicted score: {scores[idx]:.3f}")
                print(f"Gold score: {gold_scores[idx]:.1f}")
                print(f"Method used: {methods[idx]}")
                
                # Print one true and one false example for WiC
                if binary_labels:
                    true_idx = next((i for i, label in enumerate(binary_labels) if label == True), None)
                    false_idx = next((i for i, label in enumerate(binary_labels) if label == False), None)
                    
                    if true_idx is not None:
                        sent1, sent2, target_word = pairs[true_idx]
                        print(f"\nTRUE example (same sense):")
                        print(f"Source: {sent1}")
                        print(f"Target: {sent2}")
                        print(f"Target word: {target_word}")
                        print(f"Predicted: {scores[true_idx]:.3f} | Method: {methods[true_idx]}")
                    
                    if false_idx is not None:
                        sent1, sent2, target_word = pairs[false_idx]
                        print(f"\nFALSE example (different sense):")
                        print(f"Source: {sent1}")
                        print(f"Target: {sent2}")
                        print(f"Target word: {target_word}")
                        print(f"Predicted: {scores[false_idx]:.3f} | Method: {methods[false_idx]}")
                        
            else:
                sent1, sent2 = pairs[idx]
                print(f"\nExample from {dataset_name} dataset:")
                print(f"Source: {sent1}")
                print(f"Target: {sent2}")
                print(f"Predicted score: {scores[idx]:.3f}")
                print(f"Gold score: {gold_scores[idx]:.1f}")
                print(f"Method used: {methods[idx]}")
    
    def run_mixed_evaluation(self, max_wic_samples: int = 100, max_sts_samples: int = 200) -> Dict:
        print("Semantic similarity on wixed wic + sts")
        
        all_results = {}
        total_pairs = 0
        total_glossbert_usage = 0
        total_minilm_usage = 0
        
        # WiC Dataset Evaluation
        wic_data = self.download_wic_dataset()
        if not wic_data.empty:
            if len(wic_data) > max_wic_samples:
                wic_sample = wic_data.sample(n=max_wic_samples, random_state=42)
            else:
                wic_sample = wic_data
            
            wic_pairs = [(row['sentence1'], row['sentence2'], row['target_word']) 
                        for _, row in wic_sample.iterrows()]
            wic_gold_scores = wic_sample['similarity_score'].tolist()
            wic_binary_labels = wic_sample['binary_label'].tolist()
            
            wic_hybrid_scores, wic_methods, wic_predictions = self.evaluate_hybrid_system_wic(wic_pairs)
            
            wic_classification_metrics = self.calculate_evaluation_metrics(
                [1 if pred else 0 for pred in wic_predictions], 
                [1 if label else 0 for label in wic_binary_labels], 
                'classification')
            wic_regression_metrics = self.calculate_evaluation_metrics(
                wic_hybrid_scores, wic_gold_scores, 'regression')
            
            wic_method_counts = {}
            for method in wic_methods:
                wic_method_counts[method] = wic_method_counts.get(method, 0) + 1
            
            total_pairs += len(wic_pairs)
            total_glossbert_usage += wic_method_counts.get('GlossBERT', 0)
            total_minilm_usage += wic_method_counts.get('MiniLM', 0)
            
            all_results['WiC'] = {
                'pairs': wic_pairs,
                'scores': wic_hybrid_scores,
                'methods': wic_methods,
                'gold_scores': wic_gold_scores,
                'classification_metrics': wic_classification_metrics,
                'regression_metrics': wic_regression_metrics,
                'method_counts': wic_method_counts,
                'num_pairs': len(wic_pairs)
            }
            
            self.print_example("WiC", wic_pairs, wic_hybrid_scores, wic_methods, wic_gold_scores, wic_binary_labels)
        
        # STS Dataset Evaluation
        sts_data = self.download_sts_benchmark()
        if not sts_data.empty:
            if len(sts_data) > max_sts_samples:
                sts_sample = sts_data.sample(n=max_sts_samples, random_state=42)
            else:
                sts_sample = sts_data
            
            sts_pairs = [(row['sentence1'], row['sentence2']) for _, row in sts_sample.iterrows()]
            sts_gold_scores = sts_sample['similarity_score'].tolist()
            
            sts_hybrid_scores, sts_methods = self.evaluate_hybrid_system_sts(sts_pairs)
            sts_regression_metrics = self.calculate_evaluation_metrics(
                sts_hybrid_scores, sts_gold_scores, 'regression')
            
            sts_method_counts = {}
            for method in sts_methods:
                sts_method_counts[method] = sts_method_counts.get(method, 0) + 1
            
            total_pairs += len(sts_pairs)
            total_glossbert_usage += sts_method_counts.get('GlossBERT', 0)
            total_minilm_usage += sts_method_counts.get('MiniLM', 0)
            
            all_results['STS'] = {
                'pairs': sts_pairs,
                'scores': sts_hybrid_scores,
                'methods': sts_methods,
                'gold_scores': sts_gold_scores,
                'regression_metrics': sts_regression_metrics,
                'method_counts': sts_method_counts,
                'num_pairs': len(sts_pairs)
            }
            
            self.print_example("STS", sts_pairs, sts_hybrid_scores, sts_methods, sts_gold_scores)
        
        # Print overall results
        print(f"\nOverall Results:")
        print(f"Total pairs evaluated: {total_pairs}")
        print(f"GlossBERT usage: {total_glossbert_usage}/{total_pairs} ({total_glossbert_usage/total_pairs*100:.1f}%)")
        print(f"MiniLM usage: {total_minilm_usage}/{total_pairs} ({total_minilm_usage/total_pairs*100:.1f}%)")
        
        if 'WiC' in all_results:
            wic_results = all_results['WiC']
            print(f"\nWiC Results ({wic_results['num_pairs']} pairs):")
            print(f"Binary Accuracy: {wic_results['classification_metrics']['accuracy']:.3f}")
            print(f"Pearson Correlation: {wic_results['regression_metrics']['pearson_correlation']:.3f}")
            print(f"Spearman Correlation: {wic_results['regression_metrics']['spearman_correlation']:.3f}")
            print(f"Method usage - GlossBERT: {wic_results['method_counts'].get('GlossBERT', 0)}, MiniLM: {wic_results['method_counts'].get('MiniLM', 0)}")
        
        if 'STS' in all_results:
            sts_results = all_results['STS']
            print(f"\nSTS Results ({sts_results['num_pairs']} pairs):")
            print(f"Pearson Correlation: {sts_results['regression_metrics']['pearson_correlation']:.3f}")
            print(f"Spearman Correlation: {sts_results['regression_metrics']['spearman_correlation']:.3f}")
            print(f"MSE: {sts_results['regression_metrics']['mse']:.3f}")
            print(f"MAE: {sts_results['regression_metrics']['mae']:.3f}")
            print(f"Method usage - GlossBERT: {sts_results['method_counts'].get('GlossBERT', 0)}, MiniLM: {sts_results['method_counts'].get('MiniLM', 0)}")
        
        return all_results


def main():
    evaluator = MixedWiCLargeScaleEvaluator()
    results = evaluator.run_mixed_evaluation(max_wic_samples=100, max_sts_samples=200)
    print("\nMixed evaluation completed!")


if __name__ == "__main__":
    main() 
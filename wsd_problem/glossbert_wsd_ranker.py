import os
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import re

class GlossBERTWSDRanker:
    
    def __init__(self, model_path: str = "./GlossBERT/pretrained_models", wordnet_path: str = "./GlossBERT/wordnet/index.sense.gloss"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load GlossBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        
        # Load pretrained GlossBERT weights if available
        try:
            import os
            weights_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            pass
        
        # Load WordNet sense glosses
        self.wordnet_senses = {}
        self._load_wordnet_glosses(wordnet_path)
        
        # Load SentenceTransformer for fallback
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            self.sentence_model = None
        
        # Cache for word discrimination thresholds
        self.word_thresholds = {}
    
    def _load_wordnet_glosses(self, wordnet_path: str):
        try:
            with open(wordnet_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        sense_key = parts[0]
                        gloss = parts[1]
                        self.wordnet_senses[sense_key] = gloss
        except FileNotFoundError:
            pass
    
    def _test_word_meaningfulness(self, word: str) -> Tuple[str, bool]:
        """
        Test if a word is meaningful for semantic disambiguation
        Returns (reason, use_glossbert) - simplified approach focusing on word characteristics
        """
        
        # Check cache first
        if word in self.word_thresholds:
            cached_threshold, cached_meaningful = self.word_thresholds[word]
            reason = f"cached assessment for '{word}'"
            return reason, cached_meaningful
        
        # Since WordNet contains synset IDs instead of actual glosses,
        # we'll use linguistic heuristics to determine meaningfulness
        
        # Known homophones and polysemous words that benefit from disambiguation
        meaningful_words = {
            'bank', 'play', 'light', 'book', 'match', 'spring', 'plant', 'bark', 
            'bat', 'bow', 'can', 'cap', 'duck', 'fair', 'jam', 'left', 'lie', 
            'mine', 'nail', 'park', 'point', 'right', 'rock', 'rose', 'row', 
            'saw', 'seal', 'star', 'tie', 'wave', 'well', 'yard'
        }
        
        # Basic/function words that don't benefit from disambiguation
        basic_words = {
            'think', 'believe', 'know', 'see', 'say', 'get', 'make', 'go', 
            'come', 'take', 'give', 'use', 'work', 'call', 'try', 'ask',
            'people', 'person', 'time', 'way', 'day', 'man', 'thing', 'life'
        }
        
        is_meaningful = False
        reason = f"'{word}' is a basic word - semantic similarity more appropriate"
        
        if word in meaningful_words:
            is_meaningful = True
            reason = f"'{word}' is a known homophone/polysemous word - good for disambiguation"
        elif word in basic_words:
            is_meaningful = False
            reason = f"'{word}' is a basic/function word - semantic similarity more appropriate"
        elif len(word) <= 3:
            is_meaningful = False
            reason = f"'{word}' is too short/basic - semantic similarity more appropriate"
        else:
            # For unknown words, check if they have multiple senses in WordNet
            target_senses = [key for key in self.wordnet_senses.keys() if word in key.lower()]
            if len(target_senses) >= 5:  # Many senses suggest polysemy
                is_meaningful = True
                reason = f"'{word}' has multiple senses ({len(target_senses)}) - potentially meaningful for disambiguation"
            else:
                is_meaningful = False
                reason = f"'{word}' has few senses ({len(target_senses)}) - semantic similarity more appropriate"
        
        # Cache the result
        self.word_thresholds[word] = (0.5 if is_meaningful else 0.0, is_meaningful)
        
        return reason, is_meaningful
    
    def _encode_context_gloss_pair(self, context: str, target_word: str, gloss: str) -> torch.Tensor:
        input_text = f"{context} [SEP] {target_word} : {gloss}"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            cls_embedding = embeddings[:, 0, :]
        
        return cls_embedding
    
    def _encode_sentence_only(self, sentence: str) -> torch.Tensor:
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            cls_embedding = embeddings[:, 0, :]
        
        return cls_embedding
    
    def _calculate_glossbert_similarity(self, source_sentence: str, candidate_sentence: str, target_word: str) -> Tuple[float, str]:
        if not self.wordnet_senses:
            return 0.0, "No WordNet data"
        
        # Get all possible senses for the target word
        target_senses = [key for key in self.wordnet_senses.keys() if target_word in key.lower()]
        
        if not target_senses:
            return 0.0, "No senses found"
        
        best_similarity = 0.0
        best_sense = ""
        
        # Encode source sentence with each possible sense
        source_embeddings = []
        for sense_key in target_senses[:5]:  # Limit to top 5 senses for speed
            gloss = self.wordnet_senses[sense_key]
            embedding = self._encode_context_gloss_pair(source_sentence, target_word, gloss)
            source_embeddings.append(embedding)
        
        # Encode candidate sentence with each possible sense
        candidate_embeddings = []
        for sense_key in target_senses[:5]:
            gloss = self.wordnet_senses[sense_key]
            embedding = self._encode_context_gloss_pair(candidate_sentence, target_word, gloss)
            candidate_embeddings.append(embedding)
        
        # Find best matching sense pair
        for i, source_emb in enumerate(source_embeddings):
            for j, candidate_emb in enumerate(candidate_embeddings):
                similarity = torch.cosine_similarity(source_emb, candidate_emb, dim=1).item()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_sense = target_senses[i] if i < len(target_senses) else "unknown"
        
        return best_similarity, best_sense
    
    def _calculate_sentence_similarity_minilm(self, sentence1: str, sentence2: str) -> float:
        if self.sentence_model is None:
            return self._calculate_sentence_similarity_bert(sentence1, sentence2)
        
        try:
            embeddings = self.sentence_model.encode([sentence1, sentence2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            return self._calculate_sentence_similarity_bert(sentence1, sentence2)
    
    def _calculate_sentence_similarity_bert(self, sentence1: str, sentence2: str) -> float:
        try:
            emb1 = self._encode_sentence_only(sentence1)
            emb2 = self._encode_sentence_only(sentence2)
            similarity = torch.cosine_similarity(emb1, emb2, dim=1).item()
            return similarity
        except Exception as e:
            return 0.0
    
    def _find_common_words_between_two_sentences(self, sentence1: str, sentence2: str) -> List[str]:
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
                     'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 
                     'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        words1 = set(re.findall(r'\b\w+\b', sentence1.lower()))
        words2 = set(re.findall(r'\b\w+\b', sentence2.lower()))
        
        common = words1 & words2
        filtered_common = [word for word in common if word not in stop_words and len(word) > 2]
        
        return sorted(filtered_common, key=len, reverse=True)
    
    def _word_exists_in_sentence(self, word: str, sentence: str) -> bool:
        words_in_sentence = re.findall(r'\b\w+\b', sentence.lower())
        return word.lower() in words_in_sentence
    
    def compare_sentences(self, source_sentence: str, candidate_sentences: List[str], target_word: Optional[str] = None) -> List[Dict]:
        results = []
        
        # Check if WordNet data is available for GlossBERT
        wordnet_available = len(self.wordnet_senses) > 0
        
        for i, candidate_sentence in enumerate(candidate_sentences):
            # Determine which word to use for GlossBERT
            selected_word = None
            
            if target_word is not None:
                # Use explicit target word if provided
                if self._word_exists_in_sentence(target_word, source_sentence):
                    selected_word = target_word
            else:
                # Find common words between source and this specific candidate
                common_words = self._find_common_words_between_two_sentences(source_sentence, candidate_sentence)
                if common_words:
                    selected_word = common_words[0]  # Use the first (longest) common word
            
            method_used = "MiniLM"
            similarity_reason = "General sentence similarity"
            sense_info = "N/A"
            
            if selected_word and wordnet_available:
                # Test if this word is meaningful for disambiguation
                reason, use_glossbert = self._test_word_meaningfulness(selected_word)
                
                if use_glossbert:
                    # Use GlossBERT approach for meaningful words
                    similarity_score, sense_info = self._calculate_glossbert_similarity(
                        source_sentence, candidate_sentence, selected_word
                    )
                    method_used = "GlossBERT"
                    similarity_reason = f"GlossBERT for meaningful word: {reason}"
                    
                else:
                    # Word is basic/non-meaningful - use MiniLM
                    similarity_score = self._calculate_sentence_similarity_minilm(source_sentence, candidate_sentence)
                    similarity_reason = f"MiniLM fallback: {reason}"
                    
            else:
                # Use SentenceTransformer MiniLM approach
                similarity_score = self._calculate_sentence_similarity_minilm(source_sentence, candidate_sentence)
            
            result = {
                'candidate_id': i + 1,
                'candidate_sentence': candidate_sentence,
                'candidate_word': selected_word if selected_word else "auto-detected",
                'similarity_score': similarity_score,
                'similarity_reason': similarity_reason,
                'method_used': method_used,
                'sense_info': sense_info
            }
            
            results.append(result)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results 
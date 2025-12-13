"""
Dynamic Semantic Analysis for Data Augmentation
Inspired by Carnegie Mellon University (CMU) Language Technologies Institute approaches
Uses semantic similarity and contextual understanding for dynamic variation generation
"""
import json
from typing import List, Dict, Optional, Set
from collections import defaultdict
import os

# Try multiple HTTP clients
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import ollama
    HAS_OLLAMA_PKG = True
except ImportError:
    HAS_OLLAMA_PKG = False


class SemanticAnalyzer:
    """
    Dynamic semantic analysis for generating variations
    Inspired by CMU's semantic analysis approaches
    """
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434", model: str = "llama3:8b"):
        self.ollama_base_url = ollama_base_url
        self.model = model
        self._cache = {}  # Cache for semantic analysis results
    
    def _call_ollama(self, prompt: str, timeout: float = 15.0) -> Optional[str]:  # Reduced from 30s to 15s
        """Call Ollama API directly via HTTP or Python package"""
        # Try ollama Python package first (cleaner API)
        if HAS_OLLAMA_PKG:
            try:
                # Use ollama package if available
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 100  # Reduced from 200 for faster responses
                    }
                )
                return response.get("response", "").strip()
            except Exception as e:
                # Fall back to HTTP
                pass
        
        # Fall back to HTTP API
        try:
            if HAS_HTTPX:
                print(f"        [Ollama HTTP] Calling with {len(prompt)} char prompt...", end="", flush=True)
                response = httpx.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_predict": 100  # Reduced from 200 for faster responses
                        }
                    },
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(" Done")
                    return result.get("response", "").strip()
                else:
                    print(f" Failed: HTTP {response.status_code}")
            elif HAS_REQUESTS:
                print(f"        [Ollama HTTP] Calling with {len(prompt)} char prompt...", end="", flush=True)
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_predict": 100  # Reduced from 200 for faster responses
                        }
                    },
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(" Done")
                    return result.get("response", "").strip()
                else:
                    print(f" Failed: HTTP {response.status_code}")
        except Exception as e:
            print(f" Failed: {e}")
        
        return None
    
    def extract_semantic_verbs(self, text: str, category: str) -> List[str]:
        """
        Extract semantically similar verbs using Ollama
        Inspired by CMU's semantic role labeling approaches
        Cached per category (not per text) for performance
        """
        # Cache by category only, not per text (much more efficient)
        cache_key = f"verbs:{category}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Use category-level prompt (not text-specific) for better caching
        prompt = f"""For tasks in the '{category}' category, suggest 6-8 common action verbs that are semantically similar and could be used interchangeably.

Category: {category}

Return ONLY a JSON array of verbs, no explanation. Example: ["write", "draft", "compose", "create", "author"]"""
        
        response = self._call_ollama(prompt)
        if response:
            try:
                # Try to parse JSON from response
                # Sometimes Ollama adds extra text, so we extract JSON
                import re
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if json_match:
                    verbs = json.loads(json_match.group())
                    if isinstance(verbs, list) and len(verbs) > 0:
                        self._cache[cache_key] = verbs
                        return verbs
            except:
                pass
        
        # Fallback: return empty list
        return []
    
    def generate_semantic_prefixes(self, text: str, category: str) -> List[str]:
        """
        Generate semantically appropriate prefixes using contextual analysis
        Inspired by CMU's discourse analysis approaches
        Cached per category (not per text) for performance
        """
        cache_key = f"prefixes:{category}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Simplified prompt - category only (not text-specific) for better caching
        prompt = f"""For tasks in the '{category}' category, generate 8-10 natural ways to introduce task requests.

Category: {category}

Consider: politeness levels, personal perspectives (I need, Can you, Let me), contextual frames.

Return ONLY a JSON array of prefixes. Example: ["I need to", "Can you help me", "Please", "I want to"]"""
        
        response = self._call_ollama(prompt)
        if response:
            try:
                import re
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if json_match:
                    prefixes = json.loads(json_match.group())
                    if isinstance(prefixes, list) and len(prefixes) > 0:
                        self._cache[cache_key] = prefixes
                        return prefixes
            except:
                pass
        
        # Fallback: return default prefixes
        return [
            "I need to", "Can you help me", "Please", "I want to",
            "Help me", "I should", "Let me", "I'm going to"
        ]
    
    def generate_semantic_suffixes(self, text: str, category: str) -> List[str]:
        """
        Generate semantically appropriate suffixes using temporal and contextual analysis
        Cached per category (not per text) for performance
        """
        cache_key = f"suffixes:{category}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Simplified prompt - category only (not text-specific) for better caching
        prompt = f"""For tasks in the '{category}' category, generate 6-8 natural ways to add temporal or contextual information.

Category: {category}

Consider: time constraints, urgency levels, contextual markers.

Return ONLY a JSON array of suffixes. Example: [" today", " this week", " as soon as possible"]"""
        
        response = self._call_ollama(prompt)
        if response:
            try:
                import re
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if json_match:
                    suffixes = json.loads(json_match.group())
                    if isinstance(suffixes, list) and len(suffixes) > 0:
                        self._cache[cache_key] = suffixes
                        return suffixes
            except:
                pass
        
        # Fallback: return default suffixes
        return [
            "", " today", " this week", " as soon as possible",
            " when you have time", " - urgent", " - important"
        ]
    
    def generate_paraphrases(self, text: str, category: str, num_paraphrases: int = 3) -> List[str]:
        """
        Generate semantic paraphrases using Ollama
        Inspired by CMU's paraphrase generation approaches
        """
        prompt = f"""Generate {num_paraphrases} different natural ways to express this task request. Each should be semantically equivalent but use different wording:

Original: "{text}"
Category: {category}

Requirements:
- Keep the same meaning and intent
- Use natural, conversational language
- Vary sentence structure and word choice
- Each paraphrase should be a complete sentence

Return ONLY the paraphrases, one per line, without numbering or bullets."""
        
        response = self._call_ollama(prompt)
        if response:
            paraphrases = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove common prefixes
                for prefix in ['- ', '1. ', '2. ', '3. ', '4. ', '5. ', '* ', '• ']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                
                if line and line != text and len(line) > 10:
                    paraphrases.append(line)
            
            return paraphrases[:num_paraphrases]
        
        return []
    
    def analyze_semantic_structure(self, text: str) -> Dict:
        """
        Analyze semantic structure of text
        Inspired by CMU's semantic role labeling and dependency parsing
        """
        prompt = f"""Analyze the semantic structure of this task request:

"{text}"

Identify:
1. Main action verb
2. Object/noun phrase
3. Modifiers/adjectives
4. Temporal markers (if any)
5. Urgency indicators (if any)

Return a JSON object with these fields."""
        
        response = self._call_ollama(prompt)
        if response:
            try:
                import re
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
        
        # Fallback: simple analysis
        words = text.lower().split()
        return {
            "action_verb": words[0] if words else "unknown",
            "object": " ".join(words[1:]) if len(words) > 1 else "",
            "modifiers": [],
            "temporal": None,
            "urgency": None
        }
    
    def check_ollama_available(self) -> bool:
        """Check if Ollama is available"""
        # Try ollama package first
        if HAS_OLLAMA_PKG:
            try:
                ollama.list()  # This will raise if not available
                return True
            except:
                pass
        
        # Fall back to HTTP check
        try:
            if HAS_HTTPX:
                response = httpx.get(
                    f"{self.ollama_base_url}/api/tags",
                    timeout=5.0
                )
                return response.status_code == 200
            elif HAS_REQUESTS:
                response = requests.get(
                    f"{self.ollama_base_url}/api/tags",
                    timeout=5.0
                )
                return response.status_code == 200
        except:
            pass
        
        return False


def get_semantic_analyzer(
    ollama_base_url: Optional[str] = None,
    model: Optional[str] = None
) -> Optional[SemanticAnalyzer]:
    """
    Factory function to get semantic analyzer
    """
    base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = model or os.getenv("OLLAMA_MODEL", "llama3:8b")
    
    analyzer = SemanticAnalyzer(ollama_base_url=base_url, model=model_name)
    
    if analyzer.check_ollama_available():
        return analyzer
    else:
        print(f"⚠ Ollama not available at {base_url}")
        return None


try:
    from deepiri_modelkit.ml.semantic import get_semantic_analyzer
except ImportError:
    
    get_semantic_analyzer = None
try:
    from deepiri_modelkit.ml.semantic import SemanticAnalyzer, get_semantic_analyzer
except ImportError:
    SemanticAnalyzer = None
    get_semantic_analyzer = None
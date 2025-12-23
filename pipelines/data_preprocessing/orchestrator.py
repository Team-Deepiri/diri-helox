import time
from typing import Optional, List, Dict, Any
from .base import PreprocessingStage, ProcessedData, StageResult
import networkx as nx

class PipelineOrchestrator:
    def __init__(self, stages : Optional[List[PreprocessingStage]] = None):
        self.stages = stages or []
        self.dag: Optional[nx.DiGraph] = None  # Will hold the dependency graph
        self.execution_order = []
        self.checkpoints = {}
        
        
    def add_stage(self, stage: PreprocessingStage) -> None:  
        stage_name = stage.get_name()
        existing_names = [s.get_name() for s in self.stages]
        if stage_name in existing_names:
            raise ValueError(f"Stage with name '{stage_name}' already exists")
        
        self.stages.append(stage)
        
    def build_dag(self) -> None:
        self.dag = nx.DiGraph()
        for stage in self.stages:
            stage_name = stage.get_name()
            self.dag.add_node(stage_name, stage = stage)
            stage_dependencies = stage.get_dependencies()
            
            for dep_name in stage_dependencies:
                if dep_name not in self.dag.nodes():
                    raise ValueError(f"dependency {dep_name} not found")
                self.dag.add_edge(dep_name, stage_name)
                
                
        if not nx.is_directed_acyclic_graph(self.dag):
            cycles = list(nx.simple_cycles(self.dag))
            raise ValueError(
                "Circular dependencies "
                f"The pipeline contains a cycle : {cycles}")
        

        self.execution_order = list(nx.topological_sort(self.dag))
        
        
    def execute(self, initial_data: Any=None) -> StageResult:
        if self.dag is None:
            raise ValueError("DAG not built. Call build_dag() first.")
        if not self.execution_order:
            raise ValueError("Execution order not computed. Call build_dag() first.")
        
        current_data = initial_data
        for stage_name in self.execution_order:
            stage = self.dag.nodes[stage_name]['stage']
            
            start_time = time.time()
            result = stage.process(current_data)
            execution_time = time.time() - start_time
            
            if result.execution_time is None:
                result.execution_time = execution_time
      
            self.checkpoints[stage_name] = result
            
            if not result.success:
                return result
            if result.processed_data:
                current_data = result.processed_data.data
                
        return result
    
                
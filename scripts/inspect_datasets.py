#!/usr/bin/env python3
"""
Dataset Inspection Tool
Comprehensive analysis and visualization of training datasets
"""
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class DatasetInspector:
    """Inspect and analyze training datasets"""
    
    def __init__(self, data_dir: str = "app/train/data"):
        self.data_dir = Path(data_dir)
        self.label_mapping = self._load_label_mapping()
    
    def _load_label_mapping(self) -> Dict:
        """Load label mapping if available"""
        label_map_file = self.data_dir / "label_mapping.json"
        if label_map_file.exists():
            with open(label_map_file) as f:
                return json.load(f)
        return {}
    
    def inspect_file(self, file_path: Path) -> Dict:
        """Inspect a single dataset file"""
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        print(f"\n{'='*60}")
        print(f"Inspecting: {file_path.name}")
        print(f"{'='*60}")
        
        data = []
        label_counts = Counter()
        text_lengths = []
        errors = []
        
        with open(file_path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                    
                    # Extract label
                    label = item.get('label', item.get('label_id', 'unknown'))
                    if isinstance(label, int):
                        label_id = label
                        label_name = self._get_label_name(label_id)
                    elif isinstance(label, str):
                        label_name = label
                        label_id = self._get_label_id(label)
                    else:
                        label_id = 'unknown'
                        label_name = 'unknown'
                    
                    label_counts[label_id] += 1
                    
                    # Text length
                    text = item.get('text', '')
                    text_lengths.append(len(text))
                    
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: JSON error - {e}")
                except Exception as e:
                    errors.append(f"Line {line_num}: {e}")
        
        # Statistics
        stats = {
            "file": str(file_path),
            "total_examples": len(data),
            "label_distribution": dict(label_counts),
            "text_length": {
                "min": min(text_lengths) if text_lengths else 0,
                "max": max(text_lengths) if text_lengths else 0,
                "mean": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
                "median": sorted(text_lengths)[len(text_lengths)//2] if text_lengths else 0
            },
            "errors": errors,
            "error_count": len(errors)
        }
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Unique labels: {len(label_counts)}")
        print(f"  Errors: {len(errors)}")
        
        print(f"\n📏 Text Length Statistics:")
        print(f"  Min: {stats['text_length']['min']} chars")
        print(f"  Max: {stats['text_length']['max']} chars")
        print(f"  Mean: {stats['text_length']['mean']:.1f} chars")
        print(f"  Median: {stats['text_length']['median']:.1f} chars")
        
        print(f"\n🏷️  Label Distribution:")
        for label_id, count in label_counts.most_common():
            label_name = self._get_label_name(label_id)
            percentage = (count / len(data)) * 100
            print(f"  {label_id} ({label_name}): {count} ({percentage:.1f}%)")
        
        if errors:
            print(f"\n⚠️  Errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        
        # Sample examples
        print(f"\n📝 Sample Examples:")
        for i, item in enumerate(data[:5], 1):
            label = item.get('label', item.get('label_id', 'unknown'))
            text = item.get('text', '')[:80]
            print(f"  {i}. [{label}] {text}...")
        
        return stats
    
    def _get_label_name(self, label_id: int) -> str:
        """Get label name from ID"""
        if self.label_mapping and 'id2label' in self.label_mapping:
            return self.label_mapping['id2label'].get(str(label_id), f"label_{label_id}")
        return f"label_{label_id}"
    
    def _get_label_id(self, label_name: str) -> int:
        """Get label ID from name"""
        if self.label_mapping and 'label2id' in self.label_mapping:
            return self.label_mapping['label2id'].get(label_name, -1)
        return -1
    
    def compare_datasets(self, files: List[Path]) -> Dict:
        """Compare multiple dataset files"""
        print(f"\n{'='*60}")
        print("Dataset Comparison")
        print(f"{'='*60}")
        
        all_stats = {}
        for file_path in files:
            stats = self.inspect_file(file_path)
            all_stats[file_path.name] = stats
        
        # Compare label distributions
        print(f"\n{'='*60}")
        print("Label Distribution Comparison")
        print(f"{'='*60}")
        
        all_labels = set()
        for stats in all_stats.values():
            if 'label_distribution' in stats:
                all_labels.update(stats['label_distribution'].keys())
        
        print(f"\n{'Label':<15} " + " ".join(f"{name[:20]:<20}" for name in all_stats.keys()))
        for label in sorted(all_labels):
            row = f"{label:<15} "
            for name, stats in all_stats.items():
                count = stats.get('label_distribution', {}).get(label, 0)
                total = stats.get('total_examples', 1)
                percentage = (count / total) * 100
                row += f"{count:>4} ({percentage:>5.1f}%)  "
            print(row)
        
        return all_stats
    
    def check_quality(self, file_path: Path) -> Dict:
        """Check dataset quality metrics"""
        stats = self.inspect_file(file_path)
        
        quality_issues = []
        quality_score = 100
        
        # Check for class imbalance
        if 'label_distribution' in stats:
            counts = list(stats['label_distribution'].values())
            if counts:
                max_count = max(counts)
                min_count = min(counts)
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                if imbalance_ratio > 3:
                    quality_issues.append(f"Class imbalance: {imbalance_ratio:.1f}x difference")
                    quality_score -= 20
        
        # Check for empty texts
        if stats.get('text_length', {}).get('min', 0) == 0:
            quality_issues.append("Empty text examples found")
            quality_score -= 15
        
        # Check for very short texts
        if stats.get('text_length', {}).get('mean', 0) < 10:
            quality_issues.append("Average text length very short (<10 chars)")
            quality_score -= 10
        
        # Check for errors
        if stats.get('error_count', 0) > 0:
            quality_issues.append(f"{stats['error_count']} parsing errors")
            quality_score -= stats['error_count'] * 2
        
        quality_score = max(0, quality_score)
        
        print(f"\n{'='*60}")
        print("Quality Assessment")
        print(f"{'='*60}")
        print(f"Quality Score: {quality_score}/100")
        
        if quality_issues:
            print(f"\n⚠️  Issues Found:")
            for issue in quality_issues:
                print(f"  - {issue}")
        else:
            print(f"\n✅ No major quality issues detected")
        
        return {
            "quality_score": quality_score,
            "issues": quality_issues,
            "stats": stats
        }


def main():
    parser = argparse.ArgumentParser(description="Inspect training datasets")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="app/train/data",
        help="Data directory (default: app/train/data)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Specific file to inspect"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple files"
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Run quality checks"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Inspect all dataset files"
    )
    
    args = parser.parse_args()
    
    inspector = DatasetInspector(data_dir=args.data_dir)
    data_dir = Path(args.data_dir)
    
    if args.file:
        # Inspect specific file
        file_path = data_dir / args.file
        inspector.inspect_file(file_path)
        if args.quality:
            inspector.check_quality(file_path)
    
    elif args.compare:
        # Compare files
        files = [data_dir / f for f in args.compare]
        inspector.compare_datasets(files)
    
    elif args.all:
        # Inspect all dataset files
        dataset_files = [
            "classification_train.jsonl",
            "classification_val.jsonl",
            "classification_test.jsonl",
            "synthetic_classification_train.jsonl",
            "synthetic_classification_val.jsonl",
            "synthetic_classification_test.jsonl"
        ]
        
        found_files = []
        for filename in dataset_files:
            file_path = data_dir / filename
            if file_path.exists():
                found_files.append(file_path)
        
        if not found_files:
            print("No dataset files found!")
            return
        
        for file_path in found_files:
            inspector.inspect_file(file_path)
            if args.quality:
                inspector.check_quality(file_path)
    
    else:
        # Default: inspect train/val/test if they exist
        default_files = [
            data_dir / "classification_train.jsonl",
            data_dir / "classification_val.jsonl",
            data_dir / "classification_test.jsonl"
        ]
        
        found_files = [f for f in default_files if f.exists()]
        
        if not found_files:
            print("No dataset files found!")
            print(f"\nUsage:")
            print(f"  python {sys.argv[0]} --file <filename>")
            print(f"  python {sys.argv[0]} --all")
            print(f"  python {sys.argv[0]} --compare file1.jsonl file2.jsonl")
            return
        
        for file_path in found_files:
            inspector.inspect_file(file_path)
            if args.quality:
                inspector.check_quality(file_path)


if __name__ == "__main__":
    main()


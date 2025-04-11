import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import argparse
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import traceback


class CellCountComparison:
    """
    A class to compare manual and automated cell counts.
    """
    
    def __init__(self, 
                 manual_csv_dir: str, 
                 detection_results_dir: str,
                 output_dir: str = "comparison_results",
                 cell_types: Optional[List[str]] = None):
        """
        Initialize the comparison tool.
        
        Args:
            manual_csv_dir: Directory containing manual count CSV files
            detection_results_dir: Directory containing detection results
            output_dir: Directory to save comparison results
            cell_types: List of cell types to analyze (default: auto-detect)
        """
        self.manual_csv_dir = Path(manual_csv_dir)
        self.csv_file_mapping = self._create_csv_file_mapping()
        self.detection_results_dir = Path(detection_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Cell types to analyze
        self.cell_types = cell_types or self._detect_cell_types()
        
        # Data containers
        self.manual_counts = {}
        self.detection_counts = {}
        self.comparison_results = {}
        
        print(f"Initialized Cell Count Comparison Tool")
        print(f"Manual CSV directory: {self.manual_csv_dir}")
        print(f"Detection results directory: {self.detection_results_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Cell types to analyze: {self.cell_types}")
        
    def _create_csv_file_mapping(self) -> Dict[str, Path]:
        mapping = {}
        for file_path in self.manual_csv_dir.glob("*.csv"):
            mapping[file_path.name.lower()] = file_path
        return mapping
    
    def _detect_cell_types(self) -> List[str]:
        cell_types = []
        pattern = re.compile(r"Raw(\w+)\.csv", re.IGNORECASE)
        
        for file_path in self.manual_csv_dir.glob("*.csv"):
            match = pattern.match(file_path.name)
            if match:
                cell_type = match.group(1).lower() + "_processed"
                cell_types.append(cell_type)
        
        if not cell_types:
            print("Warning: No cell type CSV files found. Using default cell types.")
            return [
                "basophil_processed",
                "eosinophil_processed",
                "erythroblast_processed",
                "immunoglobulin_processed",
                "lymphocyte_processed",
                "monocyte_processed",
                "neutrophil_processed",
                "platelet_processed"
            ]
        
        return cell_types

    def load_manual_counts(self) -> None:
        """
        Load manual cell counts from CSV files.
        """
        print("\nLoading manual cell counts...")
    
        for cell_type in self.cell_types:
            # Extract base name
            base_name = cell_type.split('_')[0]
        
            # Try different case variations for the CSV filename
            variations = [
                f"Raw{base_name.capitalize()}.csv",  
                f"Raw{base_name}.csv",               
                f"raw{base_name.capitalize()}.csv",  
                f"raw{base_name}.csv"                
            ]
        
            # Find the actual CSV file using case-insensitive mapping
            csv_path = None
            for variant in variations:
                if variant.lower() in self.csv_file_mapping:
                    csv_path = self.csv_file_mapping[variant.lower()]
                    break
        
            if not csv_path:
                print(f"Warning: Manual count file not found for {cell_type} (tried variations: {variations})")
                continue
        
            try:
                # Read CSV with explicit data types
                df = pd.read_csv(csv_path)
            
                # Clean up column names (remove any extra spaces)
                df.columns = [col.strip() for col in df.columns]
            
                # Convert columns to numeric, forcing errors to NaN
                numeric_columns = ['red_blood_cells', 'white_blood cells', 'ambiguous']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            
                # Calculate total cells (sum of red, white, and ambiguous)
                df['total_cells'] = df['red_blood_cells'].fillna(0) + \
                                    df['white_blood cells'].fillna(0) + \
                                    df['ambiguous'].fillna(0)
            
                # Convert to dictionary with image name as key
                counts_dict = {}
                for _, row in df.iterrows():
                    if pd.isna(row['image_name']):
                        continue
                    
                    # Store the image name as-is from the CSV
                    img_name = row['image_name']
                
                    counts_dict[img_name] = {
                        'red_blood_cells': float(row['red_blood_cells']) if not pd.isna(row['red_blood_cells']) else 0,
                        'white_blood_cells': float(row['white_blood cells']) if not pd.isna(row['white_blood cells']) else 0,
                        'ambiguous': float(row['ambiguous']) if not pd.isna(row['ambiguous']) else 0,
                        'total_cells': float(row['total_cells'])
                    }
            
                self.manual_counts[cell_type] = counts_dict
                print(f"  ✓ Loaded {len(counts_dict)} manual counts for {cell_type} from {csv_path.name}")
            
            except Exception as e:
                print(f"  ✗ Error loading {csv_path}: {str(e)}")
    
    def load_detection_results(self) -> None:
        """
        Load automated cell detection results.
        """
        print("\nLoading automated detection results...")
        
        # Try to load from cell_counts_summary.txt first
        summary_path = self.detection_results_dir / "cell_counts_summary.txt"
        if summary_path.exists():
            try:
                self._load_from_summary(summary_path)
                return
            except Exception as e:
                print(f"  ✗ Error loading from summary file: {str(e)}")
                print("  Falling back to individual cell type directories...")
        
        # Fall back to checking individual directories
        for cell_type in self.cell_types:
            cell_type_dir = self.detection_results_dir / cell_type
            
            if not cell_type_dir.exists():
                print(f"  ✗ Detection results directory not found: {cell_type_dir}")
                continue
            
            counts_dict = {}
            
            # Look for individual result files
            for result_file in cell_type_dir.glob("*_detection.png"):
                # Extract image name from filename 
                match = re.match(r"(.+?)_detection\.png", result_file.name)
                if match:
                    img_name = f"{match.group(1)}.jpg"
                    
                    # Try to extract count from filename or look for accompanying text file
                    count = self._extract_count_from_image_or_file(result_file)
                    if count is not None:
                        counts_dict[img_name] = {'total_cells': count}
            
            if counts_dict:
                self.detection_counts[cell_type] = counts_dict
                print(f"  ✓ Loaded {len(counts_dict)} detection results for {cell_type}")
            else:
                print(f"  ✗ No detection results found for {cell_type}")
    
    def _load_from_summary(self, summary_path: Path) -> None:
        print(f"  Loading from summary file: {summary_path}")
    
        with open(summary_path, 'r') as f:
            content = f.read()
    
        # Initialize data structure
        current_cell_type = None
        current_image = None
    
        for line in content.split('\n'):
            line = line.strip()
        
            # Skip empty lines and headers
            if not line or line.startswith('Cell Detection Results') or line.startswith('=='):
                continue
        
            # Check for image info
            if line.startswith('Image:'):
                current_image = line.split(':', 1)[1].strip()
            elif line.startswith('Cell Type:'):
                cell_type = line.split(':', 1)[1].strip()
            
                # Standardize cell type name (handle case differences)
                base_name = cell_type.split('_')[0].lower()
                current_cell_type = f"{base_name}_processed"
            
                if current_cell_type not in self.detection_counts:
                    self.detection_counts[current_cell_type] = {}
            elif line.startswith('Cells Detected:'):
                if current_image and current_cell_type:
                    count = int(line.split(':', 1)[1].strip())
                    self.detection_counts[current_cell_type][current_image] = {'total_cells': count}
    
        # Print summary of loaded data
        for cell_type, counts in self.detection_counts.items():
            print(f"  ✓ Loaded {len(counts)} detection results for {cell_type}")
        
    def _extract_count_from_image_or_file(self, image_path: Path) -> Optional[int]:
        # First, check if count is in the filename
        match = re.search(r'Detected[_\s:]+(\d+)[_\s]?cells', image_path.name)
        if match:
            return int(match.group(1))
        
        # Next, check if there's an accompanying text file
        txt_path = image_path.with_suffix('.txt')
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                content = f.read()
                match = re.search(r'Detected[_\s:]+(\d+)[_\s]?cells', content)
                if match:
                    return int(match.group(1))
        return None
    
    def compare_counts(self) -> None:
        """
        Compare manual and automated cell counts.
        """
        print("\nComparing manual and automated counts...")
    
        # Container for overall metrics
        all_manual_counts = []
        all_detected_counts = []
    
        for cell_type in self.cell_types:
            if cell_type not in self.manual_counts or cell_type not in self.detection_counts:
                print(f"  ✗ Skipping {cell_type}: missing manual or detection counts")
                continue
        
            manual = self.manual_counts[cell_type]
            detected = self.detection_counts[cell_type]
        
            # Create case-insensitive mappings for matching
            manual_keys_lower = {k.lower(): k for k in manual.keys()}
            detected_keys_lower = {k.lower(): k for k in detected.keys()}
        
            # Find common images using case-insensitive matching
            common_images = []
        
            for det_key_lower, det_key in detected_keys_lower.items():
                if det_key_lower in manual_keys_lower:
                    manual_key = manual_keys_lower[det_key_lower]
                    common_images.append((manual_key, det_key))
        
            if not common_images:
                print(f"  ✗ No common images found for {cell_type}")
                continue
        
            # Prepare comparison data
            comparison = {
                'image_names': [],
                'manual_counts': [],
                'detected_counts': [],
                'differences': [],
                'percent_errors': []
            }
        
            for manual_key, detected_key in sorted(common_images, key=lambda x: x[0]):
                manual_count = manual[manual_key]['total_cells']
                detected_count = detected[detected_key]['total_cells']
                difference = detected_count - manual_count
            
                # Calculate percent error (avoiding division by zero)
                if manual_count > 0:
                    percent_error = (difference / manual_count) * 100
                else:
                    percent_error = float('inf') if difference > 0 else 0
            
                comparison['image_names'].append(manual_key)
                comparison['manual_counts'].append(manual_count)
                comparison['detected_counts'].append(detected_count)
                comparison['differences'].append(difference)
                comparison['percent_errors'].append(percent_error)
            
                # Add to overall metrics
                all_manual_counts.append(manual_count)
                all_detected_counts.append(detected_count)
        
            # Calculate statistics
            comparison['stats'] = {
                'count': len(common_images),
                'mae': mean_absolute_error(comparison['manual_counts'], comparison['detected_counts']),
                'rmse': np.sqrt(mean_squared_error(comparison['manual_counts'], comparison['detected_counts'])),
                'r2': r2_score(comparison['manual_counts'], comparison['detected_counts']),
                'mean_difference': np.mean(comparison['differences']),
                'mean_percent_error': np.mean([pe for pe in comparison['percent_errors'] if not np.isinf(pe)]),
                'median_percent_error': np.median([pe for pe in comparison['percent_errors'] if not np.isinf(pe)])
            }
        
            self.comparison_results[cell_type] = comparison
        
            print(f"  ✓ Compared {len(common_images)} images for {cell_type}")
            print(f"    MAE: {comparison['stats']['mae']:.2f}, RMSE: {comparison['stats']['rmse']:.2f}")
            print(f"    Mean difference: {comparison['stats']['mean_difference']:.2f} cells")
            print(f"    Mean percent error: {comparison['stats']['mean_percent_error']:.2f}%")
    
        # Calculate and display overall accuracy
        if all_manual_counts:
            print("\n=== OVERALL ERROR METRICS ===")
            print(f"Total images compared: {len(all_manual_counts)}")
            print(f"Mean Absolute Error: {mean_absolute_error(all_manual_counts, all_detected_counts):.2f} cells")
            print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(all_manual_counts, all_detected_counts)):.2f} cells")
            print(f"R² Score: {r2_score(all_manual_counts, all_detected_counts):.4f}")
            print("================================")
    
            # Add a summary table of cell type performance
            print("\n=== CELL TYPE PERFORMANCE SUMMARY ===")
            print("Cell Type               | Mean % Error | MAE (cells)")
            print("------------------------|--------------|-----------")
    
            # Collect performance data for sorting
            performance_data = []
            for cell_type, comparison in self.comparison_results.items():
                stats = comparison['stats']
                performance_data.append((
                    cell_type.split('_')[0],  
                    stats['mean_percent_error'],
                    stats['mae']
                ))
    
            # Sort by absolute percent error (best performing first)
            performance_data.sort(key=lambda x: abs(x[1]))
    
            # Print sorted table
            for cell_type, mpe, mae in performance_data:
                print(f"{cell_type.ljust(24)} | {mpe:+.2f}% | {mae:.2f}")
    
            print("================================")
    
    def generate_reports(self) -> None:
        """
        Generate comparison reports and visualizations.
        """
        print("\nGenerating reports and visualizations...")
        
        # Create summary report
        self._generate_summary_report()
        
        # Create detailed reports for each cell type
        for cell_type, comparison in self.comparison_results.items():
            self._generate_cell_type_report(cell_type, comparison)
        
        # Create visualizations
        self._generate_visualizations()
        
        print(f"  ✓ Reports and visualizations saved to {self.output_dir}")
    
    def _generate_summary_report(self) -> None:
        """
        Generate a summary report of all cell types.
        """
        summary_path = self.output_dir / "summary_report.csv"
        
        # Prepare summary data
        summary_data = []
        
        for cell_type, comparison in self.comparison_results.items():
            stats = comparison['stats']
            summary_data.append({
                'cell_type': cell_type,
                'images_compared': stats['count'],
                'mean_absolute_error': stats['mae'],
                'root_mean_squared_error': stats['rmse'],
                'r2_score': stats['r2'],
                'mean_difference': stats['mean_difference'],
                'mean_percent_error': stats['mean_percent_error'],
                'median_percent_error': stats['median_percent_error']
            })
        
        # Create DataFrame and save to CSV
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(summary_path, index=False)
            
            # Also save as formatted text for easy reading
            txt_path = self.output_dir / "summary_report.txt"
            with open(txt_path, 'w') as f:
                f.write("Cell Count Comparison Summary\n")
                f.write("=============================\n\n")
                
                for row in summary_data:
                    f.write(f"Cell Type: {row['cell_type']}\n")
                    f.write(f"  Images Compared: {row['images_compared']}\n")
                    f.write(f"  Mean Absolute Error: {row['mean_absolute_error']:.2f} cells\n")
                    f.write(f"  Root Mean Squared Error: {row['root_mean_squared_error']:.2f} cells\n")
                    f.write(f"  R² Score: {row['r2_score']:.4f}\n")
                    f.write(f"  Mean Difference: {row['mean_difference']:.2f} cells\n")
                    f.write(f"  Mean Percent Error: {row['mean_percent_error']:.2f}%\n")
                    f.write(f"  Median Percent Error: {row['median_percent_error']:.2f}%\n\n")
    
    def _generate_cell_type_report(self, cell_type: str, comparison: dict) -> None:
        """
        Generate a detailed report for a specific cell type.
        
        Args:
            cell_type: Cell type name
            comparison: Comparison data dictionary
        """
        # Create cell type directory
        cell_type_dir = self.output_dir / cell_type
        cell_type_dir.mkdir(exist_ok=True)
        
        # Save detailed CSV
        detailed_path = cell_type_dir / "detailed_comparison.csv"
        df = pd.DataFrame({
            'image_name': comparison['image_names'],
            'manual_count': comparison['manual_counts'],
            'detected_count': comparison['detected_counts'],
            'difference': comparison['differences'],
            'percent_error': comparison['percent_errors']
        })
        df.to_csv(detailed_path, index=False)
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(comparison['manual_counts'], comparison['detected_counts'], alpha=0.7)
        
        # Add perfect prediction line
        max_count = max(max(comparison['manual_counts']), max(comparison['detected_counts']))
        plt.plot([0, max_count], [0, max_count], 'r--', alpha=0.7)
        
        plt.title(f'Manual vs. Detected Cell Counts: {cell_type}')
        plt.xlabel('Manual Count')
        plt.ylabel('Detected Count')
        plt.grid(True, alpha=0.3)
        
        # Add stats annotation
        stats = comparison['stats']
        plt.annotate(f"MAE: {stats['mae']:.2f}\nRMSE: {stats['rmse']:.2f}\nR²: {stats['r2']:.4f}",
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(cell_type_dir / "scatter_plot.png", dpi=300)
        plt.close()
        
        # Create difference histogram
        plt.figure(figsize=(10, 6))
        plt.hist(comparison['differences'], bins=min(20, len(comparison['differences'])), alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        plt.title(f'Distribution of Count Differences: {cell_type}')
        plt.xlabel('Detected - Manual Count')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(cell_type_dir / "difference_histogram.png", dpi=300)
        plt.close()
    
    def _generate_visualizations(self) -> None:
        # Prepare data for comparison across cell types
        cell_types = []
        mae_values = []
        rmse_values = []
        mean_percent_errors = []
        
        for cell_type, comparison in self.comparison_results.items():
            stats = comparison['stats']
            cell_types.append(cell_type.split('_')[0])  
            mae_values.append(stats['mae'])
            rmse_values.append(stats['rmse'])
            mean_percent_errors.append(stats['mean_percent_error'])
        
        if not cell_types:
            return
        
        # Set style for better-looking plots
        sns.set(style="whitegrid")
        
        # Create comparative bar chart for MAE
        plt.figure(figsize=(12, 6))
        bars = plt.bar(cell_types, mae_values, alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{height:.2f}', ha='center', va='bottom')
        
        plt.title('Mean Absolute Error by Cell Type')
        plt.xlabel('Cell Type')
        plt.ylabel('Mean Absolute Error (cells)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "mae_comparison.png", dpi=300)
        plt.close()
        
        # Create comparative bar chart for percent error
        plt.figure(figsize=(12, 6))
        bars = plt.bar(cell_types, mean_percent_errors, alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{height:.2f}%', ha='center', va='bottom')
        
        plt.title('Mean Percent Error by Cell Type')
        plt.xlabel('Cell Type')
        plt.ylabel('Mean Percent Error (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "percent_error_comparison.png", dpi=300)
        plt.close()
        
        # Create box plot of differences by cell type
        plt.figure(figsize=(14, 8))
        
        # Prepare data for box plot
        box_data = []
        box_labels = []
        
        for cell_type, comparison in self.comparison_results.items():
            box_data.append(comparison['differences'])
            box_labels.append(cell_type.split('_')[0])
        
        # Use the updated parameter name for Matplotlib 3.9+
        try:
            plt.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        except TypeError:
            # Fallback for older Matplotlib versions
            plt.boxplot(box_data, labels=box_labels, patch_artist=True)
            
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.title('Distribution of Count Differences by Cell Type')
        plt.xlabel('Cell Type')
        plt.ylabel('Detected - Manual Count')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "difference_boxplot.png", dpi=300)
        plt.close()
    
    def run(self) -> None:
        """
        Run the complete comparison workflow.
        """
        self.load_manual_counts()
        self.load_detection_results()
        self.compare_counts()
        self.generate_reports()
        
        print("\nComparison completed successfully!")
        print(f"Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Compare manual and automated cell counts.')
    parser.add_argument('--manual_dir', type=str, required=True,
                        help='Directory containing manual count CSV files')
    parser.add_argument('--detection_dir', type=str, required=True,
                        help='Directory containing detection results')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help='Directory to save comparison results')
    parser.add_argument('--cell_types', type=str, nargs='+',
                        help='Cell types to analyze (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Run comparison
    try:
        comparison = CellCountComparison(
            manual_csv_dir=args.manual_dir,
            detection_results_dir=args.detection_dir,
            output_dir=args.output_dir,
            cell_types=args.cell_types
        )
        comparison.run()
    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
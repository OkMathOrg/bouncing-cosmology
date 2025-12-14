#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run comprehensive validation and generate all figures
Optional convenience script
"""

import os
import json
import numpy as np

def save_results_robust(results, filename):
    """Save results with correct serialization"""
    def convert_to_serializable(obj):
        """Recursively convert objects to serializable"""
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            # Try to convert to a string
            try:
                return str(obj)
            except:
                return None
    
    results_serializable = convert_to_serializable(results)
    
    os.makedirs('output', exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to {filename}")

def main():
    """Run all tests and generate output"""
    
    print("=" * 60)
    print("COMPREHENSIVE VALIDATION AND FIGURE GENERATION")
    print("=" * 60)
    
    # 1. Run comprehensive validation
    print("\n[1] RUNNING COMPREHENSIVE VALIDATION...")
    from bounce import run_comprehensive_validation
    results, bg = run_comprehensive_validation()
    
    # 2. Generate all figures
    print("\n[2] GENERATING ALL FIGURES...")
    from generate_figures import main as generate_figures_main
    generate_figures_main()
    
    # 3. Save results
    if results is not None:
        save_results_robust(results, 'output/final_results.json')
        
        # Save summary
        with open('output/results_summary.txt', 'w', encoding='utf-8') as f:
            f.write("FINAL MODEL VALIDATION RESULTS\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"1. BOUNCE MECHANISM:\n")
            f.write(f"   a_min = {results.get('a_min', 'N/A'):.6e}\n")
            f.write(f"   H(bounce) = {results.get('H_bounce', 'N/A'):.2e}\n")
            f.write(f"   N_post-bounce = {results.get('N_post', 'N/A'):.1f} e-folds\n\n")
            
            f.write(f"2. THEORETICAL CONSISTENCY:\n")
            f.write(f"   Friedmann constraint: max error = {results.get('max_constraint_error', 'N/A'):.2e}\n")
            f.write(f"   NEC satisfied: {results.get('NEC_ok', False)}\n")
            f.write(f"   WEC satisfied: {results.get('WEC_ok', False)}\n")
            f.write(f"   BKL compatible: {results.get('bkl_compatible', False)}\n")
            f.write(f"   Shear/curvature ratio: {results.get('shear_curvature_ratio', 'N/A'):.2e}\n\n")
            
            f.write(f"3. FLATNESS SOLUTION:\n")
            f.write(f"   Required e-folds for |Ω_k| < 0.001: N_required = {results.get('N_required', 'N/A'):.2f}\n")
            f.write(f"   Actual post-bounce e-folds: N_actual = {results.get('N_actual', 'N/A'):.2f}\n")
            f.write(f"   Flatness achieved: {results.get('flatness_achieved', False)}\n\n")
            
            f.write(f"4. OBSERVATIONAL PREDICTIONS (N=60):\n")
            f.write(f"   n_s = {results.get('n_s', 'N/A'):.4f}\n")
            f.write(f"   r = {results.get('r', 'N/A'):.4f}\n")
            f.write(f"   A_s = {results.get('A_s', 'N/A'):.4e}\n")
            f.write(f"   n_s consistent with Planck: {results.get('n_s_consistent', False)}\n")
            f.write(f"   r < 0.036: {results.get('r_consistent', False)}\n")
            f.write(f"   A_s ≈ 2e-9: {results.get('A_s_consistent', False)}\n")
        
        print("✓ Results summary saved to output/results_summary.txt")
    
    print("\n" + "=" * 60)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    if results is not None and all([
        results.get('bounce', False),
        results.get('friedmann_ok', False),
        results.get('NEC_ok', False),
        results.get('flatness_achieved', False),
        results.get('bkl_compatible', False),
        results.get('n_s_consistent', False),
        results.get('r_consistent', False)
    ]):
        print("🎉 ALL CRITICAL TESTS PASSED!")
        print("The model is fully validated and ready for publication.")
    else:
        print("⚠ Some tests failed or gave warnings.")
        print("Review the output above for details.")

if __name__ == "__main__":

    main()

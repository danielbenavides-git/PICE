# Missing Data Analysis for Imputation Strategy
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency

# Variables to analyze for missingness
variables_to_impute = [
    "Sector", "Clase de Factura", "Clase de Movimiento V", 
    "Centro de Coste", "Fecha Valor", "Ledger", "Cantidad", 
    "Centro", "Hora", "Clase", "División"
]

# ============================================
# 1. MISSINGNESS PATTERN ANALYSIS
# ============================================

def analyze_missingness_patterns(df, variables):
    """
    Analyze patterns of missingness across variables
    """
    print("=" * 70)
    print("MISSINGNESS PATTERN ANALYSIS")
    print("=" * 70)
    
    # Create missingness indicator matrix
    missing_matrix = df[variables].isnull().astype(int)
    
    # Count unique missingness patterns
    pattern_counts = missing_matrix.value_counts()
    
    print(f"\nNumber of unique missingness patterns: {len(pattern_counts)}")
    print(f"\nTop 10 most common patterns (1 = missing, 0 = present):")
    print("-" * 70)
    print(pattern_counts.head(10))
    
    # Correlation between missingness indicators
    print("\n\nCorrelation between missingness indicators:")
    print("-" * 70)
    missing_corr = missing_matrix.corr()
    
    # Find high correlations (excluding diagonal)
    high_corr_pairs = []
    for i in range(len(missing_corr.columns)):
        for j in range(i+1, len(missing_corr.columns)):
            corr_value = missing_corr.iloc[i, j]
            if abs(corr_value) > 0.3:  # Threshold for moderate correlation
                high_corr_pairs.append((
                    missing_corr.columns[i], 
                    missing_corr.columns[j], 
                    corr_value
                ))
    
    if high_corr_pairs:
        print("\nVariables with correlated missingness (|r| > 0.3):")
        for var1, var2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {var1} <-> {var2}: {corr:.3f}")
    else:
        print("\nNo strong correlations found between missingness patterns")
    
    return missing_matrix, missing_corr

# ============================================
# 2. TEST FOR MCAR (Little's MCAR Test Alternative)
# ============================================

def test_mcar_using_ttest(df, var_with_missing, numeric_vars):
    """
    Test if missingness in a variable is related to values in other variables
    Uses t-tests to compare means of numeric variables between missing/not-missing groups
    """
    print("\n" + "=" * 70)
    print(f"TESTING MCAR FOR: {var_with_missing}")
    print("=" * 70)
    print("Comparing numeric variable means between missing vs non-missing groups")
    print("-" * 70)
    
    missing_indicator = df[var_with_missing].isnull()
    significant_diffs = []
    
    for num_var in numeric_vars:
        if num_var == var_with_missing:
            continue
            
        # Split data into missing and non-missing groups
        group_missing = df.loc[missing_indicator, num_var].dropna()
        group_present = df.loc[~missing_indicator, num_var].dropna()
        
        if len(group_missing) > 0 and len(group_present) > 0:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(group_missing, group_present, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            mean_diff = group_missing.mean() - group_present.mean()
            pooled_std = np.sqrt((group_missing.std()**2 + group_present.std()**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
            
            if p_value < 0.05:
                significant_diffs.append({
                    'variable': num_var,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'mean_missing': group_missing.mean(),
                    'mean_present': group_present.mean()
                })
    
    if significant_diffs:
        print(f"\n  SIGNIFICANT DIFFERENCES FOUND (suggests NOT MCAR):")
        for diff in sorted(significant_diffs, key=lambda x: x['p_value']):
            print(f"\n  Variable: {diff['variable']}")
            print(f"    p-value: {diff['p_value']:.6f}")
            print(f"    Cohen's d: {diff['cohens_d']:.3f}")
            print(f"    Mean when {var_with_missing} missing: {diff['mean_missing']:.2f}")
            print(f"    Mean when {var_with_missing} present: {diff['mean_present']:.2f}")
        return "MAR or MNAR"
    else:
        print(f"\n No significant differences found (consistent with MCAR)")
        return "MCAR"

# ============================================
# 3. TEST FOR MAR vs MNAR
# ============================================

def test_mar_vs_mnar_categorical(df, var_with_missing, categorical_vars):
    """
    Test if missingness is related to categorical variables (MAR)
    """
    print("\n" + "=" * 70)
    print(f"TESTING MAR FOR: {var_with_missing}")
    print("=" * 70)
    print("Chi-square tests for association with categorical variables")
    print("-" * 70)
    
    missing_indicator = df[var_with_missing].isnull()
    significant_assoc = []
    
    for cat_var in categorical_vars:
        if cat_var == var_with_missing:
            continue
        
        # Create contingency table
        contingency_table = pd.crosstab(missing_indicator, df[cat_var])
        
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate Cramér's V for effect size
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            if p_value < 0.05 and cramers_v > 0.1:
                significant_assoc.append({
                    'variable': cat_var,
                    'p_value': p_value,
                    'cramers_v': cramers_v,
                    'chi2': chi2
                })
    
    if significant_assoc:
        print(f"\n  SIGNIFICANT ASSOCIATIONS FOUND (suggests MAR):")
        for assoc in sorted(significant_assoc, key=lambda x: x['p_value']):
            print(f"\n  Variable: {assoc['variable']}")
            print(f"    p-value: {assoc['p_value']:.6f}")
            print(f"    Cramér's V: {assoc['cramers_v']:.3f}")
            print(f"    Chi-square: {assoc['chi2']:.2f}")
        return "MAR"
    else:
        print(f"\n✓ No significant associations found")
        return "Inconclusive"

# ============================================
# 4. VISUALIZATION OF MISSINGNESS
# ============================================

def visualize_missingness(df, variables):
    """
    Create visualizations to understand missingness patterns
    """
    missing_matrix = df[variables].isnull().astype(int)
    
    # 1. Missingness heatmap
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Heatmap of missingness
    sns.heatmap(missing_matrix.T, cmap='RdYlGn_r', cbar_kws={'label': 'Missing'}, 
                ax=axes[0], yticklabels=True)
    axes[0].set_title('Missingness Pattern Across Observations (Sample)', fontsize=14, weight='bold')
    axes[0].set_xlabel('Observation Index')
    axes[0].set_ylabel('Variables')
    
    # Correlation heatmap of missingness
    missing_corr = missing_matrix.corr()
    sns.heatmap(missing_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                ax=axes[1], square=True, linewidths=0.5)
    axes[1].set_title('Correlation Between Missingness Indicators', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.show()

# ============================================
# 5. COMPREHENSIVE ANALYSIS FUNCTION
# ============================================

def comprehensive_missingness_analysis(df, variables_to_impute):
    """
    Run complete missingness analysis and provide recommendations
    """
    print("\n")
    print("*" * 70)
    print("COMPREHENSIVE MISSINGNESS ANALYSIS")
    print("*" * 70)
    
    # Identify numeric and categorical variables
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_vars = [col for col in df.columns if col not in numeric_vars]
    
    # Filter to only include variables present in df
    variables_present = [v for v in variables_to_impute if v in df.columns]
    
    # 1. Pattern analysis
    missing_matrix, missing_corr = analyze_missingness_patterns(df, variables_present)
    
    # 2. Test each variable for MCAR
    results = {}
    for var in variables_present:
        if df[var].isnull().sum() > 0:
            mcar_result = test_mcar_using_ttest(df, var, numeric_vars)
            mar_result = test_mar_vs_mnar_categorical(df, var, categorical_vars)
            results[var] = {'MCAR_test': mcar_result, 'MAR_test': mar_result}
    
    # 3. Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 70)
    visualize_missingness(df, variables_present)
    
    # 4. Summary and recommendations
    print("\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)
    
    for var, result in results.items():
        print(f"\n{var}:")
        print(f"  Missingness type: {result['MCAR_test']} / {result['MAR_test']}")
        
        # Recommend imputation method
        if result['MCAR_test'] == 'MCAR':
            print(f"   Recommendation: Simple imputation (mean/median/mode) or deletion acceptable")
        elif result['MAR_test'] == 'MAR':
            print(f"   Recommendation: Multiple Imputation (MICE) or KNN Imputation")
        else:
            print(f"    Recommendation: Advanced methods (Random Forest, Deep Learning) or domain knowledge")
    
    print("\n" + "=" * 70)
    print("GENERAL RECOMMENDATIONS:")
    print("=" * 70)
    print("""
    1. MCAR (Missing Completely At Random):
       - Simple imputation: mean, median, mode
       - Listwise deletion (if small proportion)
       
    2. MAR (Missing At Random):
       - Multiple Imputation by Chained Equations (MICE)
       - K-Nearest Neighbors (KNN) imputation
       - Regression-based imputation
       
    3. MNAR (Missing Not At Random):
       - Random Forest imputation
       - Deep learning-based imputation
       - Domain expert consultation
       - Consider keeping missingness as a feature
    """)
    
    return results
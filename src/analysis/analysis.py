import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
from src.constants import COMPANY_STOCK_CODES, COMPANY_NAMES

def _plot_correlation_heatmap(ax, df):
    sns.heatmap(df.astype(float), annot=True, cmap='RdBu_r', center=0,
                fmt='.3f', ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Stock Returns vs Fundamental Variables\nCorrelation Matrix', fontweight='bold')
    ax.set_xlabel('Fundamental Variables')
    ax.set_ylabel('Companies')
    ax.tick_params(axis='x', rotation=45)

def _plot_r2_scores(ax, r2_scores):
    companies = list(r2_scores.keys())
    r2_values = list(r2_scores.values())
    bars = ax.bar(companies, r2_values, color='skyblue', alpha=0.7)
    ax.set_title('Linear Regression R² by Company', fontweight='bold')
    ax.set_xlabel('Companies')
    ax.set_ylabel('R² Score')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    for bar, r2 in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')

def _plot_mse(ax, mse_scores):
    companies = list(mse_scores.keys())
    mse_values = list(mse_scores.values())
    ax.bar(companies, mse_values, color='lightcoral', alpha=0.7)
    ax.set_title('Mean Squared Error by Company', fontweight='bold')
    ax.set_xlabel('Companies')
    ax.set_ylabel('MSE')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

def _plot_avg_correlations(ax, corr_df):
    avg_corrs = corr_df.mean(axis=0)
    ax.bar(range(len(avg_corrs)), avg_corrs.values, color='lightgreen', alpha=0.7)
    ax.set_title('Average Correlations by Variable', fontweight='bold')
    ax.set_xlabel('Fundamental Variables')
    ax.set_ylabel('Average Correlation')
    ax.set_xticks(range(len(avg_corrs)))
    ax.set_xticklabels(avg_corrs.index, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

def _plot_significant_variables(ax, top_vars):
    if not top_vars:
        ax.text(0.5, 0.5, "No significant variables found", ha='center', va='center', fontsize=12)
        ax.set_title('Top Significant Variables', fontweight='bold')
        ax.axis('off')
        return

    var_names = [f"{v['variable']}\n({v['company']})" for v in top_vars]
    p_vals = [v['p_value'] for v in top_vars]
    
    ax.bar(range(len(var_names)), p_vals, color='gold', alpha=0.7)
    ax.set_title('Top 3 Most Significant Variables\n(Lower p-value = More Significant)', fontweight='bold')
    ax.set_xlabel('Variables')
    ax.set_ylabel('P-value')
    ax.set_xticks(range(len(var_names)))
    ax.set_xticklabels(var_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax.legend()

def _plot_summary_text(ax, reg_results, top_vars, corr_df):
    ax.axis('off')
    avg_r2 = np.mean([r['r2'] for r in reg_results.values()])
    total_vars = sum(len(r['features']) for r in reg_results.values())
    significant_count = len(top_vars)

    summary_text = (
        f"ANALYSIS SUMMARY\n\n"
        f"Companies Analyzed: {len(reg_results)}\n"
        f"Fundamental Variables: {len(corr_df.columns)}\n\n"
        f"Average R²: {avg_r2:.3f}\n"
        f"Significant Variables: {significant_count}/{total_vars}\n\n"
        f"Key Findings:\n"
        f"• Correlation analysis completed\n"
        f"• Linear regression models fitted\n"
        f"• Statistical significance tested\n"
        f"• Top variables identified"
    )
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

def _align_data(stock_data, fund_data, company_code, company_name):
    if company_code not in stock_data.columns:
        return None, None, None

    stock = stock_data[company_code].dropna()
    
    X_list, y_vals, features = [], [], []

    for metric, data in fund_data.items():
        if company_name in data.columns:
            fund = data[company_name].dropna()
            
            common_years = set(stock.index.year) & set(fund.index)
            
            if len(common_years) > 1:
                s_aligned = stock[stock.index.year.isin(common_years)]
                f_aligned = fund[fund.index.isin(common_years)]
                
                n = min(len(s_aligned), len(f_aligned))
                if n > 1:
                    X_list.append(f_aligned.iloc[:n].values)
                    y_vals = s_aligned.iloc[:n].values
                    features.append(metric)
    
    if not X_list or not isinstance(y_vals, np.ndarray):
        return None, None, None

    return np.column_stack(X_list), y_vals, features

def calculate_fundamental_metrics(fund_df):
    data = {}
    for field in fund_df['Field'].unique():
        temp = fund_df[fund_df['Field'] == field].copy()
        years = [c for c in temp.columns if isinstance(c, int)]
        pivot = temp.set_index('Company name')[years].T
        pivot = pivot.apply(pd.to_numeric, errors='coerce')
        data[field] = pivot
    
    metrics = {}
    if 'SALES' in data:
        sales = data['SALES']
        metrics['sales_growth'] = sales.pct_change() * 100
    if 'EBITDA' in data:
        ebitda = data['EBITDA']
        metrics['ebitda_growth'] = ebitda.pct_change() * 100
    if 'SALES' in data and 'EBITDA' in data:
        ebitda_margin = data['EBITDA'].div(data['SALES']) * 100
        metrics['ebitda_margin_change'] = ebitda_margin.pct_change() * 100
    if 'PAT' in data:
        pat = data['PAT']
        metrics['pat_growth'] = pat.pct_change() * 100
    if 'SALES' in data and 'PAT' in data:
        pat_margin = data['PAT'].div(data['SALES']) * 100
        metrics['pat_margin_change'] = pat_margin.pct_change() * 100
    
    return metrics

def run_correlation_analysis(stock_data, fund_metrics):
    corr_data = {comp: {metric: np.nan for metric in fund_metrics.keys()} for comp in COMPANY_NAMES}

    for code, company in COMPANY_STOCK_CODES.items():
        X, y, features = _align_data(stock_data, fund_metrics, code, company)
        if X is None:
            continue
            
        for i, feature in enumerate(features):
            corr = np.corrcoef(y, X[:, i])[0, 1]
            if not np.isnan(corr):
                corr_data[company][feature] = corr
    
    return pd.DataFrame(corr_data).T

def run_regression_analysis(stock_data, fund_metrics):
    results = {}
    all_significant_vars = []
    
    for code, company in COMPANY_STOCK_CODES.items():
        X, y, features = _align_data(stock_data, fund_metrics, code, company)
        if X is None:
            continue

        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        
        for i, feature in enumerate(features):
            if model.pvalues[i+1] < 0.05:
                all_significant_vars.append({
                    'company': company,
                    'variable': feature,
                    'coefficient': model.params[i+1],
                    'p_value': model.pvalues[i+1]
                })
        
        results[company] = {
            'r2': model.rsquared,
            'mse': model.mse_resid,
            'coeffs': model.params[1:],
            'p_values': model.pvalues[1:],
            'features': features,
            'n_obs': len(y)
        }
    
    all_significant_vars.sort(key=lambda x: x['p_value'])
    top_3_vars = all_significant_vars[:3]
    
    return results, top_3_vars

def create_visualizations(corr_df, reg_results, top_vars):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    if not corr_df.empty:
        _plot_correlation_heatmap(axes[0, 0], corr_df)
        _plot_avg_correlations(axes[1, 0], corr_df)

    if reg_results:
        r2_scores = {comp: res['r2'] for comp, res in reg_results.items()}
        _plot_r2_scores(axes[0, 1], r2_scores)

        mse_scores = {comp: res['mse'] for comp, res in reg_results.items()}
        _plot_mse(axes[0, 2], mse_scores)

        _plot_significant_variables(axes[1, 1], top_vars)
        _plot_summary_text(axes[1, 2], reg_results, top_vars, corr_df)
    
    plt.tight_layout()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_path = os.path.join(base_dir, 'output', 'complete_analysis_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
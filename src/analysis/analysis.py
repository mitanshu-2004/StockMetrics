import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression
from src.constants import COMPANY_STOCK_CODES, COMPANY_NAMES

def _plot_correlation_heatmap(ax, df):
    sns.heatmap(df.astype(float), annot=True, cmap='RdBu_r', center=0, fmt='.3f', ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Stock vs Fundamentals Correlation', fontweight='bold')
    ax.set_xlabel('Fundamental Variables')
    ax.set_ylabel('Companies')
    ax.tick_params(axis='x', rotation=45)

def _plot_r2_scores(ax, r2_scores):
    companies = list(r2_scores.keys())
    r2_values = list(r2_scores.values())
    bars = ax.bar(companies, r2_values, color='#4287f5', alpha=0.8)
    ax.set_title('R² by Company', fontweight='bold')
    ax.set_ylabel('R² Score')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, r2 in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{r2:.3f}', ha='center', va='bottom')

def _plot_mse(ax, mse_scores):
    companies = list(mse_scores.keys())
    mse_values = list(mse_scores.values())
    ax.bar(companies, mse_values, color='#e74c3c', alpha=0.7)
    ax.set_title('MSE by Company', fontweight='bold')
    ax.set_ylabel('MSE')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

def _plot_avg_correlations(ax, corr_df):
    avg_corrs = corr_df.mean(axis=0).sort_values(ascending=False)
    ax.bar(range(len(avg_corrs)), avg_corrs.values, color='#2ecc71')
    ax.set_title('Avg Correlations by Variable', fontweight='bold')
    ax.set_ylabel('Correlation')
    ax.set_xticks(range(len(avg_corrs)))
    ax.set_xticklabels(avg_corrs.index, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

def _plot_significant_variables(ax, top_vars):
    if not top_vars:
        ax.text(0.5, 0.5, "No significant variables found", ha='center')
        ax.set_title('Top Variables', fontweight='bold')
        ax.axis('off')
        return

    var_names = [f"{v['variable']}\n({v['company']})" for v in top_vars]
    p_vals = [v['p_value'] for v in top_vars]
    
    ax.bar(range(len(var_names)), p_vals, color='#f39c12')
    ax.set_title('Top 3 Significant Variables', fontweight='bold')
    ax.set_ylabel('P-value')
    ax.set_xticks(range(len(var_names)))
    ax.set_xticklabels(var_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
    ax.legend()

def _plot_summary_text(ax, reg_results, top_vars, corr_df):
    ax.axis('off')
    avg_r2 = np.mean([r['r2'] for r in reg_results.values()])
    total_vars = sum(len(r['features']) for r in reg_results.values())
    significant_count = len(top_vars)

    summary = (
        f"Analysis Summary\n\n"
        f"Companies: {len(reg_results)}\n"
        f"Variables: {len(corr_df.columns)}\n"
        f"Avg R²: {avg_r2:.3f}\n"
        f"Significant: {significant_count}/{total_vars}"
    )
    
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, 
            fontsize=12, va='top', family='sans-serif',
            bbox=dict(boxstyle="round", fc="whitesmoke", ec="lightgray"))

def _align_data(stock_data, fund_data, company_code, company_name):
    if company_code not in stock_data.columns:
        return None, None, None

    stock = stock_data[company_code].dropna()
    

    company_fund_data = {}
    for var_name, var_data in fund_data.items():
        if company_name in var_data.columns:
            company_fund_data[var_name] = var_data[company_name].dropna()
    
    if not company_fund_data:
        return None, None, None
    

    fund_df = pd.DataFrame(company_fund_data)
    

    common_dates = set(stock.index) & set(fund_df.index)
    if len(common_dates) < 10:
        return None, None, None
    
    common_dates = sorted(common_dates)
    aligned_stock = stock.loc[common_dates]
    aligned_fund = fund_df.loc[common_dates]
    
    return aligned_fund, aligned_stock, list(fund_df.columns)

def calculate_fundamental_metrics(fund_df):
    data = {}
    for field in fund_df['Field'].unique():
        temp = fund_df[fund_df['Field'] == field].copy()
        years = [c for c in temp.columns if isinstance(c, int)]
        pivot = temp.set_index('Company name')[years].T
        pivot = pivot.apply(pd.to_numeric, errors='coerce')
        

        pivot.index = pd.to_datetime(pivot.index.astype(str) + '-12-31')
        
        data[field] = pivot
    
    metrics = {}
    if 'SALES' in data:
        sales = data['SALES']
        metrics['sales_growth'] = sales.pct_change(fill_method='ffill') * 100
    if 'EBITDA' in data:
        ebitda = data['EBITDA']
        metrics['ebitda_growth'] = ebitda.pct_change(fill_method='ffill') * 100
    if 'SALES' in data and 'EBITDA' in data:
        ebitda_margin = data['EBITDA'].div(data['SALES']) * 100
        metrics['ebitda_margin_change'] = ebitda_margin.pct_change(fill_method='ffill') * 100
    if 'PAT' in data:
        pat = data['PAT']
        metrics['pat_growth'] = pat.pct_change(fill_method='ffill') * 100
    if 'SALES' in data and 'PAT' in data:
        pat_margin = data['PAT'].div(data['SALES']) * 100
        metrics['pat_margin_change'] = pat_margin.pct_change(fill_method='ffill') * 100
    
    return metrics

def run_correlation_analysis(stock_data, fund_metrics):
    corr_data = {comp: {metric: np.nan for metric in fund_metrics.keys()} for comp in COMPANY_NAMES}

    for code, company in COMPANY_STOCK_CODES.items():
        X, y, features = _align_data(stock_data, fund_metrics, code, company)
        if X is None:
            continue
            
        for i, feature in enumerate(features):
            corr = np.corrcoef(y, X.iloc[:, i].values)[0, 1]
            if not np.isnan(corr):
                corr_data[company][feature] = corr
    
    return pd.DataFrame(corr_data).T

def run_regression_analysis(stock_returns, fund_metrics):
    regression_results = {}
    all_significant_vars = []
    
    for code, name in zip(COMPANY_STOCK_CODES.keys(), COMPANY_NAMES):
        X, y, features = _align_data(stock_returns, fund_metrics, code, name)
        
        if X is None or y is None:
            continue
            

        model = LinearRegression()
        model.fit(X, y)
        

        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        

        _, p_values = f_regression(X, y)
        

        significant_vars = []
        for i, (feature, p_val) in enumerate(zip(features, p_values)):
            if p_val < 0.05:
                significant_vars.append((feature, p_val))
                all_significant_vars.append({
                    'company': name,
                    'variable': feature,
                    'coefficient': model.coef_[i],
                    'p_value': p_val
                })
        

        significant_vars.sort(key=lambda x: x[1])
        

        regression_results[name] = {
            'r2': r2,
            'mse': mse,
            'n_obs': len(y),
            'features': features,
            'coefficients': model.coef_.tolist(),
            'significant_vars': significant_vars
        }
    

    all_significant_vars.sort(key=lambda x: x['p_value'])
    

    top_3_vars = all_significant_vars[:3] if all_significant_vars else []
    

    top_vars_by_company = {}
    for company, results in regression_results.items():
        sig_vars = results['significant_vars']
        if sig_vars:
            top_vars_by_company[company] = [
                (var, results['coefficients'][results['features'].index(var)], p_val)
                for var, p_val in sig_vars[:3]
            ]
    
    return regression_results, top_3_vars, top_vars_by_company

def create_visualizations(corr_df, reg_results, top_vars):
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    
    _plot_correlation_heatmap(axes[0, 0], corr_df)
    
    r2_scores = {comp: res['r2'] for comp, res in reg_results.items()}
    _plot_r2_scores(axes[0, 1], r2_scores)
    
    mse_scores = {comp: res['mse'] for comp, res in reg_results.items()}
    _plot_mse(axes[1, 0], mse_scores)
    
    _plot_avg_correlations(axes[1, 1], corr_df)
    
    _plot_significant_variables(axes[2, 0], top_vars)
    
    _plot_summary_text(axes[2, 1], reg_results, top_vars, corr_df)
    
    plt.savefig(os.path.join(output_dir, 'analysis_results.png'), dpi=300)
    plt.close()
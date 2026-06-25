import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict

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

def _align_data(stock_data, fund_data, company_code, company_name, lag=0):
    """Join one firm's annual return to its fundamental-growth metrics on year-end dates.

    With lag=0 the year-t return is matched to the year-t fundamental growth: a
    same-year (contemporaneous) relationship. With lag=1 the fundamentals are shifted
    forward one year, so the year-t return is matched to the year-(t-1) fundamental
    growth. That second form uses only information available before the return period,
    which is what a real predictive test needs.
    """
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

    if lag:
        # Move each fundamental year forward by `lag` years so year-(t-lag) growth
        # lines up with the year-t return. The index stays on year-end dates, so the
        # set-intersection join below still matches.
        fund_df = fund_df.copy()
        fund_df.index = fund_df.index + pd.DateOffset(years=lag)


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
    
    # pandas 2.x removed `fill_method` from `pct_change`. Forward-fill first,
    # then compute percent change, which is equivalent to the prior behaviour.
    metrics = {}
    if 'SALES' in data:
        sales = data['SALES']
        metrics['sales_growth'] = sales.ffill().pct_change() * 100
    if 'EBITDA' in data:
        ebitda = data['EBITDA']
        metrics['ebitda_growth'] = ebitda.ffill().pct_change() * 100
    if 'SALES' in data and 'EBITDA' in data:
        ebitda_margin = data['EBITDA'].div(data['SALES']) * 100
        metrics['ebitda_margin_change'] = ebitda_margin.ffill().pct_change() * 100
    if 'PAT' in data:
        pat = data['PAT']
        metrics['pat_growth'] = pat.ffill().pct_change() * 100
    if 'SALES' in data and 'PAT' in data:
        pat_margin = data['PAT'].div(data['SALES']) * 100
        metrics['pat_margin_change'] = pat_margin.ffill().pct_change() * 100
    
    return metrics

def run_comovement_analysis(stock_returns, code_to_name, alpha=0.05):
    """Pairwise correlation of the firms' annual returns, with FDR correction.

    This is a positive control. The fundamentals-to-returns regression finds no
    robust signal, contemporaneously or one year ahead, on this small annual sample.
    These five firms are all in the same sector, so their returns do co-move. A
    pipeline that reported "nothing significant" everywhere would be suspect. Showing
    that the same FDR-corrected methodology detects this strong, real effect is
    evidence it discriminates signal from noise rather than always returning null.
    """
    import itertools
    from scipy.stats import pearsonr

    returns = stock_returns.rename(columns=code_to_name)
    companies = [c for c in code_to_name.values() if c in returns.columns]

    rows, raw_p = [], []
    for a, b in itertools.combinations(companies, 2):
        pair = returns[[a, b]].dropna()
        if len(pair) < 3:
            continue
        r, p = pearsonr(pair[a], pair[b])
        rows.append({'company_a': a, 'company_b': b,
                     'correlation': float(r), 'p_value': float(p),
                     'n_obs': int(len(pair))})
        raw_p.append(p)

    if raw_p:
        reject, p_adj, _, _ = multipletests(raw_p, alpha=alpha, method='fdr_bh')
        for row, rej, padj in zip(rows, reject, p_adj):
            row['p_value_fdr'] = float(padj)
            row['significant'] = bool(rej)

    rows.sort(key=lambda x: -x['correlation'])
    return rows


def run_correlation_analysis(stock_data, fund_metrics, code_to_name):
    names = list(code_to_name.values())
    corr_data = {comp: {metric: np.nan for metric in fund_metrics.keys()} for comp in names}

    for code, company in code_to_name.items():
        X, y, features = _align_data(stock_data, fund_metrics, code, company)
        if X is None:
            continue

        for i, feature in enumerate(features):
            corr = np.corrcoef(y, X.iloc[:, i].values)[0, 1]
            if not np.isnan(corr):
                corr_data[company][feature] = corr

    return pd.DataFrame(corr_data).T

def _permutation_pvalue(X_const, y, observed_r2, n_permutations=2000, seed=42):
    """Empirical p-value for an OLS R^2 under label permutation.

    Refits the model on shuffled targets `n_permutations` times and returns the
    fraction of shuffles (with the standard +1 smoothing) whose R^2 is at least the
    observed R^2. A high value means the observed fit is indistinguishable from noise.
    """
    rng = np.random.default_rng(seed)
    y_values = np.asarray(y, dtype=float)
    X_values = np.asarray(X_const, dtype=float)
    at_least = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(y_values)
        null_r2 = sm.OLS(permuted, X_values).fit().rsquared
        if null_r2 >= observed_r2:
            at_least += 1
    return (at_least + 1) / (n_permutations + 1)


def run_regression_analysis(stock_returns, fund_metrics, code_to_name, alpha=0.05, lag=0):
    # First pass: fit one OLS per company and collect every coefficient test.
    # lag=0 regresses the year-t return on year-t fundamental growth (a same-year,
    # contemporaneous fit). lag=1 regresses it on the prior year's growth (a one-year-
    # ahead predictive fit). Everything downstream is identical, so the two runs are
    # directly comparable.
    regression_results = {}
    flat_tests = []  # (company, feature, raw_p)

    for code, name in code_to_name.items():
        X, y, features = _align_data(stock_returns, fund_metrics, code, name, lag=lag)

        if X is None or y is None:
            continue

        # Multivariate OLS. Each p-value is a partial test that holds the other
        # regressors fixed, so it matches the jointly-fitted coefficient. The
        # earlier version reported f_regression p-values, which are univariate and
        # therefore inconsistent with the multivariate model whose coefficients it
        # paired them with.
        X_const = sm.add_constant(X)
        ols = sm.OLS(y, X_const).fit()
        coefs = ols.params.drop('const')
        p_values = ols.pvalues.drop('const')

        # In-sample fit is optimistic with 17 points and 5 predictors, so also
        # report a leave-one-out cross-validated RMSE that reflects out-of-sample
        # error.
        in_sample_mse = mean_squared_error(y, ols.predict(X_const))
        cv_pred = cross_val_predict(LinearRegression(), X.values, y.values,
                                    cv=LeaveOneOut())
        cv_rmse = np.sqrt(mean_squared_error(y, cv_pred))

        # Permutation test on R^2: with only ~17 points and 5 predictors, OLS fits a
        # sizeable R^2 even to noise. Shuffling y breaks any real X->y relationship,
        # so the distribution of R^2 over many shuffles is the null. The empirical
        # p-value is the share of shuffles whose R^2 is at least the observed one;
        # a large p-value means the in-sample fit is within what pure chance yields.
        perm_p = _permutation_pvalue(X_const, y, ols.rsquared)

        regression_results[name] = {
            'r2': float(ols.rsquared),
            'adj_r2': float(ols.rsquared_adj),
            'mse': float(in_sample_mse),
            'cv_rmse': float(cv_rmse),
            'perm_pvalue_r2': float(perm_p),
            'n_obs': int(len(y)),
            'features': features,
            'coefficients': [float(coefs[f]) for f in features],
            'p_values': {f: float(p_values[f]) for f in features},
            'significant_vars': [],
        }
        for f in features:
            flat_tests.append((name, f, float(p_values[f])))

    # Multiple-comparison control. There is one coefficient test per
    # (company, feature) pair, so the family is all of them together. Testing 25
    # coefficients at alpha=0.05 is expected to yield ~1 false positive by chance,
    # so we apply Benjamini-Hochberg to control the false discovery rate across the
    # whole family and report both the raw and adjusted p-values.
    fdr = {}
    if flat_tests:
        raw_p = [t[2] for t in flat_tests]
        reject, p_adj, _, _ = multipletests(raw_p, alpha=alpha, method='fdr_bh')
        for (company, feature, _), rej, padj in zip(flat_tests, reject, p_adj):
            fdr[(company, feature)] = (bool(rej), float(padj))

    # Second pass: a variable is "significant" only if it survives FDR correction.
    all_significant_vars = []
    for name, res in regression_results.items():
        for feature in res['features']:
            rejected, p_adj = fdr[(name, feature)]
            res.setdefault('p_values_fdr', {})[feature] = p_adj
            if rejected:
                res['significant_vars'].append((feature, res['p_values'][feature]))
                all_significant_vars.append({
                    'company': name,
                    'variable': feature,
                    'coefficient': res['coefficients'][res['features'].index(feature)],
                    'p_value': res['p_values'][feature],
                    'p_value_fdr': p_adj,
                })
        res['significant_vars'].sort(key=lambda x: x[1])

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
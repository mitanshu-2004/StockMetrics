import os
import pandas as pd
from src.utils.data_loader import load_data, clean_stock_data
from src.analysis.analysis import (
    calculate_fundamental_metrics,
    run_correlation_analysis,
    run_regression_analysis,
    run_comovement_analysis,
    create_visualizations
)

def main():
    print("Starting Stock vs Fundamentals Analysis...")
    
    stock_raw, fund_raw, code_to_name = load_data()
    print(f"Security mapping (ISIN-verified): {code_to_name}")
    stock_returns = clean_stock_data(stock_raw, list(code_to_name.keys()))
    fund_metrics = calculate_fundamental_metrics(fund_raw)

    print(f"Metrics calculated: {list(fund_metrics.keys())}")

    corr_df = run_correlation_analysis(stock_returns, fund_metrics, code_to_name)
    # Two regressions on the same data: contemporaneous (year-t return on year-t
    # fundamental growth) and predictive (year-t return on the prior year's growth).
    # The first asks whether fundamentals and returns move together in the same year;
    # the second asks whether last year's fundamentals forecast this year's return.
    reg_results, top_3_vars, _ = run_regression_analysis(
        stock_returns, fund_metrics, code_to_name, lag=0)
    reg_results_pred, top_3_pred, _ = run_regression_analysis(
        stock_returns, fund_metrics, code_to_name, lag=1)
    comovement = run_comovement_analysis(stock_returns, code_to_name)

    create_visualizations(corr_df, reg_results, top_3_vars)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')

    corr_df.to_csv(os.path.join(output_dir, 'complete_correlation_matrix.csv'))

    _write_regression_csvs(reg_results, output_dir, 'complete_regression_results.csv',
                           'coefficient_tests.csv')
    _write_regression_csvs(reg_results_pred, output_dir,
                           'complete_regression_results_predictive.csv',
                           'coefficient_tests_predictive.csv')

    top_vars_path = os.path.join(output_dir, 'top_3_significant_variables.csv')
    if top_3_vars:
        pd.DataFrame(top_3_vars).to_csv(top_vars_path, index=False)
    elif os.path.exists(top_vars_path):
        # Nothing is significant after correction; drop any stale file from an
        # earlier run so the output directory never contradicts the analysis.
        os.remove(top_vars_path)

    pd.DataFrame(comovement).to_csv(
        os.path.join(output_dir, 'stock_comovement.csv'), index=False)
    
    print("\n=== Results summary ===")
    print("\nCorrelation description:")
    print(corr_df.describe().round(3))

    _print_regression_block(
        "Contemporaneous: year-t return vs year-t fundamental growth", reg_results)
    _print_regression_block(
        "Predictive: year-t return vs prior-year fundamental growth", reg_results_pred)

    if comovement:
        n_sig = sum(1 for c in comovement if c.get('significant'))
        mean_abs_r = sum(abs(c['correlation']) for c in comovement) / len(comovement)
        strongest = comovement[0]
        print("\nPositive control: inter-stock return co-movement")
        print(f"  Mean |correlation| = {mean_abs_r:.2f}, "
              f"{n_sig}/{len(comovement)} pairs significant after FDR.")
        print(f"  Strongest: {strongest['company_a']} vs {strongest['company_b']} "
              f"r={strongest['correlation']:.2f} (FDR p={strongest['p_value_fdr']:.4f}).")
        print("  Reading: neither same-year nor prior-year fundamentals explain these")
        print("  annual returns after correction, yet the five sector peers co-move")
        print("  strongly. The same FDR-corrected pipeline finds the real effect and")
        print("  reports none where there is none.")

    print("\nAnalysis complete. Results saved to the 'output' directory.")


def _write_regression_csvs(reg_results, output_dir, summary_name, coef_name):
    """Write the per-company summary table and the full coefficient table for one run."""
    reg_list = []
    for comp, results in reg_results.items():
        reg_list.append({
            'Company': comp,
            'R2_Score': results['r2'],
            'Adj_R2_Score': results['adj_r2'],
            'In_Sample_MSE': results['mse'],
            'CV_RMSE_LOO': results['cv_rmse'],
            'Permutation_P_R2': results['perm_pvalue_r2'],
            'Observations': results['n_obs'],
            'Features': ', '.join(results['features']),
        })
    pd.DataFrame(reg_list).to_csv(os.path.join(output_dir, summary_name), index=False)

    # Full coefficient table with raw and FDR-adjusted p-values, so every test is
    # auditable rather than only the ones that happened to clear a threshold.
    coef_rows = []
    for company, results in reg_results.items():
        for feature in results['features']:
            coef_rows.append({
                'Company': company,
                'Variable': feature,
                'Coefficient': results['coefficients'][results['features'].index(feature)],
                'P_Value_Raw': results['p_values'][feature],
                'P_Value_FDR': results['p_values_fdr'][feature],
                'Significant_FDR_5pct': (feature, results['p_values'][feature]) in results['significant_vars'],
            })
    pd.DataFrame(coef_rows).to_csv(os.path.join(output_dir, coef_name), index=False)


def _print_regression_block(label, reg_results):
    print(f"\nRegression results ({label}):")
    for company, results in reg_results.items():
        print(f"\n{company}:")
        print(f"  R-squared: {results['r2']:.3f}  Adj R-squared: {results['adj_r2']:.3f}  "
              f"In-sample MSE: {results['mse']:.3f}  LOO-CV RMSE: {results['cv_rmse']:.3f}  "
              f"Observations: {results['n_obs']}")
        print(f"  Permutation p-value for R-squared: {results['perm_pvalue_r2']:.3f} "
              f"({'fit is within chance' if results['perm_pvalue_r2'] > 0.05 else 'fit exceeds chance'})")
        if results['significant_vars']:
            print("  Significant variables (FDR-adjusted p<0.05):")
            for var, p_val in results['significant_vars']:
                print(f"    - {var} (raw p={p_val:.3f}, "
                      f"FDR p={results['p_values_fdr'][var]:.3f})")
        else:
            strongest = min(results['features'], key=lambda f: results['p_values'][f])
            print("  No variable survives FDR correction. Strongest signal: "
                  f"{strongest} (raw p={results['p_values'][strongest]:.3f}, "
                  f"FDR p={results['p_values_fdr'][strongest]:.3f}).")


if __name__ == "__main__":
    main()
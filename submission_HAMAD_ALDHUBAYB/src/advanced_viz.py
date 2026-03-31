"""
Advanced Visualization Module
Creates publication-quality charts with sophisticated styling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)
sns.set_palette("husl")


def create_3d_style_surface(df, save_path='reports/advanced_3d_surface.png'):
    """
    Create a 3D-style surface visualization showing charges by age and BMI.
    Uses contourf to simulate 3D surface.
    """
    from scipy.interpolate import griddata
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Separate smokers and non-smokers
    smokers = df[df['smoker'] == 'yes']
    non_smokers = df[df['smoker'] == 'no']
    
    for idx, (data, title, ax) in enumerate([
        (non_smokers, 'Non-Smokers', axes[0]),
        (smokers, 'Smokers', axes[1])
    ]):
        # Create grid
        x = data['age'].values
        y = data['bmi'].values
        z = data['charges'].values
        
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate
        zi = griddata((x, y), z, (xi, yi), method='cubic')
        
        # Plot filled contours
        levels = np.linspace(z.min(), z.max(), 20)
        cmap = 'viridis' if idx == 0 else 'plasma'
        cf = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap, alpha=0.8)
        
        # Add contour lines
        cs = ax.contour(xi, yi, zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt='$%1.0f')
        
        # Scatter actual points
        scatter = ax.scatter(x, y, c=z, cmap=cmap, edgecolors='white', 
                           linewidths=0.5, s=50, alpha=0.9)
        
        ax.set_xlabel('Age (years)', fontweight='bold')
        ax.set_ylabel('BMI (kg/m²)', fontweight='bold')
        ax.set_title(f'{title}: Medical Charges by Age & BMI', fontweight='bold', fontsize=14)
        
        # Colorbar
        cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
        cbar.set_label('Charges ($)', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    return fig


def create_radar_chart_comparison(df, save_path='reports/advanced_radar.png'):
    """
    Create radar chart comparing risk profiles across regions.
    """
    from math import pi
    
    # Calculate metrics by region
    regions = df['region'].unique()
    metrics = ['Avg Age', 'Avg BMI', 'Smoker %', 'Avg Children', 'Avg Charges']
    
    # Normalize metrics to 0-1 scale
    data = {}
    for region in regions:
        region_df = df[df['region'] == region]
        data[region] = [
            region_df['age'].mean() / df['age'].max(),
            region_df['bmi'].mean() / df['bmi'].max(),
            (region_df['smoker'] == 'yes').mean(),
            region_df['children'].mean() / df['children'].max(),
            region_df['charges'].mean() / df['charges'].max()
        ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Number of variables
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Plot each region
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    for idx, (region, values) in enumerate(data.items()):
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=region.title(), color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax.set_title('Regional Risk Profile Comparison\n(Normalized Metrics)', 
                 fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    return fig


def create_sankey_style_flow(df, save_path='reports/advanced_sankey.png'):
    """
    Create a flow diagram showing distribution from demographics to charges.
    Uses stacked bars with connecting gradients.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)
    
    # Define charge categories
    df['charge_category'] = pd.cut(df['charges'], 
                                   bins=[0, 10000, 20000, 30000, 50000, float('inf')],
                                   labels=['<$10K', '$10-20K', '$20-30K', '$30-50K', '>$50K'])
    
    # Color scheme
    colors = ['#2ECC71', '#F39C12', '#E74C3C', '#9B59B6', '#3498DB']
    
    # Plot 1: Age groups to charges
    ax1 = fig.add_subplot(gs[0, 0])
    age_groups = pd.cut(df['age'], bins=[0, 30, 40, 50, 65], labels=['<30', '30-40', '40-50', '50+'])
    crosstab = pd.crosstab(age_groups, df['charge_category'], normalize='index') * 100
    crosstab.plot(kind='barh', stacked=True, ax=ax1, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_title('Age Group → Charge Distribution', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Percentage (%)')
    ax1.legend(title='Charges', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: BMI categories to charges
    ax2 = fig.add_subplot(gs[0, 1])
    bmi_groups = pd.cut(df['bmi'], bins=[0, 25, 30, 35, 55], labels=['Normal', 'Overweight', 'Obese I', 'Obese II+'])
    crosstab2 = pd.crosstab(bmi_groups, df['charge_category'], normalize='index') * 100
    crosstab2.plot(kind='barh', stacked=True, ax=ax2, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_title('BMI Category → Charge Distribution', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Percentage (%)')
    ax2.legend(title='Charges', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 3: Smoker status to charges
    ax3 = fig.add_subplot(gs[1, 0])
    crosstab3 = pd.crosstab(df['smoker'], df['charge_category'], normalize='index') * 100
    crosstab3.plot(kind='barh', stacked=True, ax=ax3, color=colors, edgecolor='white', linewidth=0.5)
    ax3.set_title('Smoker Status → Charge Distribution', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Percentage (%)')
    ax3.legend(title='Charges', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 4: Region to charges
    ax4 = fig.add_subplot(gs[1, 1])
    crosstab4 = pd.crosstab(df['region'], df['charge_category'], normalize='index') * 100
    crosstab4.plot(kind='barh', stacked=True, ax=ax4, color=colors, edgecolor='white', linewidth=0.5)
    ax4.set_title('Region → Charge Distribution', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Percentage (%)')
    ax4.legend(title='Charges', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 5: Overall summary heatmap (bottom spanning both columns)
    ax5 = fig.add_subplot(gs[2, :])
    
    # Create summary statistics
    summary_data = []
    for cat in df['charge_category'].cat.categories:
        cat_df = df[df['charge_category'] == cat]
        summary_data.append([
            len(cat_df),
            cat_df['age'].mean(),
            cat_df['bmi'].mean(),
            (cat_df['smoker'] == 'yes').mean() * 100
        ])
    
    summary_df = pd.DataFrame(summary_data, 
                              columns=['Count', 'Avg Age', 'Avg BMI', 'Smoker %'],
                              index=df['charge_category'].cat.categories)
    
    # Normalize for heatmap
    summary_norm = (summary_df - summary_df.min()) / (summary_df.max() - summary_df.min())
    
    im = ax5.imshow(summary_norm.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(len(summary_df.columns)):
        for j in range(len(summary_df)):
            text = ax5.text(j, i, f'{summary_df.iloc[j, i]:.1f}',
                           ha="center", va="center", color="black", fontweight='bold', fontsize=12)
    
    ax5.set_xticks(range(len(summary_df)))
    ax5.set_xticklabels(summary_df.index, fontsize=11)
    ax5.set_yticks(range(len(summary_df.columns)))
    ax5.set_yticklabels(summary_df.columns, fontsize=11)
    ax5.set_title('Charge Category Summary Statistics', fontweight='bold', fontsize=14, pad=15)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax5, shrink=0.6)
    cbar.set_label('Normalized Value', fontweight='bold')
    
    plt.suptitle('Demographic Flow to Medical Charges', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    return fig


def create_advanced_residual_analysis(y_true, y_pred, model_name='Gradient Boosting',
                                      save_path='reports/advanced_residuals.png'):
    """
    Create advanced residual analysis with confidence intervals and multiple diagnostics.
    """
    residuals = y_true - y_pred
    std_residuals = (residuals - residuals.mean()) / residuals.std()
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 0.05])
    
    # Plot 1: Residuals vs Fitted with confidence band
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, residuals, alpha=0.5, c='steelblue', edgecolors='white', s=40)
    
    # Add LOWESS smooth line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, y_pred, frac=0.3)
        ax1.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2, label='LOWESS trend')
    except:
        pass
    
    # Add horizontal reference lines
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.axhline(y=residuals.std(), color='red', linestyle='--', alpha=0.5, label='±1σ')
    ax1.axhline(y=-residuals.std(), color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Fitted Values ($)', fontweight='bold')
    ax1.set_ylabel('Residuals ($)', fontweight='bold')
    ax1.set_title('Residuals vs Fitted (with trend)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scale-Location (Sqrt of standardized residuals)
    ax2 = fig.add_subplot(gs[0, 1])
    sqrt_std_resid = np.sqrt(np.abs(std_residuals))
    ax2.scatter(y_pred, sqrt_std_resid, alpha=0.5, c='darkgreen', edgecolors='white', s=40)
    
    try:
        smoothed2 = lowess(sqrt_std_resid, y_pred, frac=0.3)
        ax2.plot(smoothed2[:, 0], smoothed2[:, 1], 'r-', linewidth=2)
    except:
        pass
    
    ax2.set_xlabel('Fitted Values ($)', fontweight='bold')
    ax2.set_ylabel('√|Standardized Residuals|', fontweight='bold')
    ax2.set_title('Scale-Location Plot', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Q-Q plot with reference line
    ax3 = fig.add_subplot(gs[1, 0])
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.get_lines()[0].set_markerfacecolor('steelblue')
    ax3.get_lines()[0].set_markersize(6)
    ax3.get_lines()[0].set_alpha(0.6)
    ax3.get_lines()[1].set_color('red')
    ax3.get_lines()[1].set_linewidth(2)
    ax3.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Histogram with KDE and normal fit
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(residuals, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='white')
    
    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    ax4.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
    
    # Normal fit
    mu, std = stats.norm.fit(residuals)
    ax4.plot(x_range, stats.norm.pdf(x_range, mu, std), 'r--', linewidth=2, label=f'Normal (μ={mu:.0f}, σ={std:.0f})')
    
    ax4.set_xlabel('Residual Value ($)', fontweight='bold')
    ax4.set_ylabel('Density', fontweight='bold')
    ax4.set_title('Residual Distribution', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Residuals vs Order (checking for autocorrelation)
    ax5 = fig.add_subplot(gs[2, 0])
    order = np.arange(len(residuals))
    ax5.scatter(order, residuals, alpha=0.4, c='purple', s=20)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Add rolling mean
    window = max(20, len(residuals) // 20)
    rolling_mean = pd.Series(residuals).rolling(window=window, center=True).mean()
    ax5.plot(order, rolling_mean, 'r-', linewidth=2, label=f'Rolling mean (w={window})')
    
    ax5.set_xlabel('Observation Order', fontweight='bold')
    ax5.set_ylabel('Residuals ($)', fontweight='bold')
    ax5.set_title('Residuals vs Order (Autocorrelation Check)', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Cooks distance (influence plot)
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Calculate leverage and Cook's distance approximation
    leverage = (std_residuals - std_residuals.mean()) ** 2
    leverage = leverage / leverage.sum()
    cooks_d = (std_residuals ** 2) * leverage / (1 - leverage + 0.001)
    
    ax6.scatter(leverage, std_residuals, c=cooks_d, cmap='YlOrRd', alpha=0.6, s=50)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.axhline(y=2, color='red', linestyle='--', alpha=0.5)
    ax6.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Leverage', fontweight='bold')
    ax6.set_ylabel('Standardized Residuals', fontweight='bold')
    ax6.set_title('Influence Plot (Leverage vs Residuals)', fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=cooks_d.min(), vmax=cooks_d.max()))
    sm.set_array([])
    cbar_ax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Cook's Distance", fontweight='bold')
    
    plt.suptitle(f'Advanced Residual Analysis: {model_name}', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    return fig


def create_interaction_heatmap(df, save_path='reports/advanced_interactions.png'):
    """
    Create advanced heatmap showing feature interactions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Age vs BMI with charges as color
    ax1 = axes[0, 0]
    pivot1 = df.pivot_table(values='charges', index=pd.cut(df['age'], bins=8), 
                            columns=pd.cut(df['bmi'], bins=8), aggfunc='mean')
    sns.heatmap(pivot1, annot=False, cmap='viridis', ax=ax1, cbar_kws={'label': 'Charges ($)'})
    ax1.set_title('Age × BMI → Charges', fontweight='bold', fontsize=13)
    ax1.set_xlabel('BMI Range')
    ax1.set_ylabel('Age Range')
    
    # Plot 2: Smoker vs Age with charges
    ax2 = axes[0, 1]
    pivot2 = df.pivot_table(values='charges', index='smoker', 
                            columns=pd.cut(df['age'], bins=6), aggfunc='mean')
    sns.heatmap(pivot2, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=ax2, 
                cbar_kws={'label': 'Charges ($)'})
    ax2.set_title('Smoker × Age → Charges', fontweight='bold', fontsize=13)
    ax2.set_xlabel('Age Range')
    ax2.set_ylabel('Smoker')
    
    # Plot 3: Children vs Region
    ax3 = axes[1, 0]
    pivot3 = df.pivot_table(values='charges', index='region', 
                            columns=pd.cut(df['children'], bins=[-0.5, 0.5, 1.5, 2.5, 5.5], 
                                          labels=['0', '1', '2', '3+']), aggfunc='mean')
    sns.heatmap(pivot3, annot=True, fmt='.0f', cmap='coolwarm', ax=ax3, 
                cbar_kws={'label': 'Charges ($)'})
    ax3.set_title('Region × Children → Charges', fontweight='bold', fontsize=13)
    ax3.set_xlabel('Number of Children')
    ax3.set_ylabel('Region')
    
    # Plot 4: Correlation matrix of engineered features
    ax4 = axes[1, 1]
    numeric_cols = ['age', 'bmi', 'children', 'charges', 'is_smoker', 'sex_male', 
                   'age_bmi_risk', 'family_size', 'bmi_smoker_interaction']
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, ax=ax4, square=True, cbar_kws={'label': 'Correlation'})
    ax4.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=13)
    
    plt.suptitle('Feature Interaction Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    return fig


if __name__ == '__main__':
    # Load data
    df = pd.read_csv('data/processed/insurance_engineered.csv')
    
    print("Creating advanced visualizations...")
    
    # Create all advanced plots
    create_3d_style_surface(df)
    create_radar_chart_comparison(df)
    create_sankey_style_flow(df)
    create_interaction_heatmap(df)
    
    # For residual analysis, we need predictions
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Quick model for residuals
    feature_cols = ['age', 'bmi', 'children', 'is_smoker', 'sex_male', 
                   'age_bmi_risk', 'family_size']
    X = df[feature_cols].fillna(0)
    y = df['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    create_advanced_residual_analysis(y_test, y_pred)
    
    print("\nAll advanced visualizations created!")

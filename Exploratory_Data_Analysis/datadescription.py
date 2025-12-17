#!/usr/bin/env python3
"""
Generate Statistics and Figures for Data Description Section

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
sns.set_style("whitegrid")

def load_data():
    """Load the datasets"""
    df_top = pd.read_csv('Top_Artists.csv')
    df_low = pd.read_csv('Low_Artists.csv')
    
    df_top['artist_tier'] = 'Top'
    df_low['artist_tier'] = 'Low'
    
    df = pd.concat([df_top, df_low], ignore_index=True)
    return df, df_top, df_low

def generate_table_1_dataset_composition(df, df_top, df_low):
    """Generate Table 1: Dataset Composition"""
    
    print("="*60)
    print("TABLE 1: DATASET COMPOSITION")
    print("="*60)
    
    # Calculate unique artists
    unique_top = df_top['artist'].nunique() if 'artist' in df_top.columns else 'N/A'
    unique_low = df_low['artist'].nunique() if 'artist' in df_low.columns else 'N/A'
    
    # Calculate songs with lyrics
    songs_with_lyrics = df['Lyrics'].notna().sum()
    songs_with_lyrics_pct = (songs_with_lyrics / len(df)) * 100
    
    # Check for English (this is after your cleaning, so should be 100%)
    # For now, we'll assume all are English after cleaning
    
    data = {
        'Dataset Component': [
            'Top Artists Songs',
            'Low Artists Songs',
            'Total Songs',
            'Unique Artists (Top)',
            'Unique Artists (Low)',
            'Songs with Complete Lyrics',
        ],
        'Count': [
            len(df_top),
            len(df_low),
            len(df),
            unique_top,
            unique_low,
            songs_with_lyrics,
        ],
        'Percentage': [
            f"{(len(df_top)/len(df))*100:.1f}%",
            f"{(len(df_low)/len(df))*100:.1f}%",
            "100%",
            "-",
            "-",
            f"{songs_with_lyrics_pct:.1f}%",
        ]
    }
    
    table1 = pd.DataFrame(data)
    print(table1.to_string(index=False))
    
    # Save as LaTeX
    with open('table1_composition.tex', 'w') as f:
        f.write(table1.to_latex(index=False, caption="Dataset Composition", label="tab:composition"))
    
    return table1

def generate_table_2_temporal_distribution(df):
    """Generate Table 2: Temporal Distribution"""
    
    print("\n" + "="*60)
    print("TABLE 2: TEMPORAL DISTRIBUTION")
    print("="*60)
    
    # Define time periods
    bins = [0, 2000, 2010, 2020, 2025]
    labels = ['Pre-2000', '2000-2009', '2010-2019', '2020-2024']
    
    df['period'] = pd.cut(df['Year'], bins=bins, labels=labels, right=False)
    
    # Create crosstab
    temporal_dist = pd.crosstab(df['period'], df['artist_tier'], margins=True)
    temporal_dist.columns = ['Low Artists', 'Top Artists', 'Total']
    
    print(temporal_dist)
    
    # Save as LaTeX
    with open('table2_temporal.tex', 'w') as f:
        f.write(temporal_dist.to_latex(caption="Temporal Distribution of Songs", label="tab:temporal"))
    
    return temporal_dist

def generate_table_3_playcount_statistics(df, df_top, df_low):
    """Generate Table 3: Play Count Statistics"""
    
    print("\n" + "="*60)
    print("TABLE 3: PLAY COUNT STATISTICS")
    print("="*60)
    
    def get_stats(data):
        return {
            'Minimum': f"{data.min():,.0f}",
            '25th Percentile': f"{data.quantile(0.25):,.0f}",
            'Median': f"{data.quantile(0.50):,.0f}",
            'Mean': f"{data.mean():,.0f}",
            '75th Percentile': f"{data.quantile(0.75):,.0f}",
            'Maximum': f"{data.max():,.0f}",
            'Std. Deviation': f"{data.std():,.0f}"
        }
    
    stats_data = pd.DataFrame({
        'Top Artists': get_stats(df_top['PlayCount']),
        'Low Artists': get_stats(df_low['PlayCount']),
        'Overall': get_stats(df['PlayCount'])
    })
    
    print(stats_data)
    
    # Save as LaTeX
    with open('table3_playcount.tex', 'w') as f:
        f.write(stats_data.to_latex(caption="Play Count Statistics", label="tab:playcount"))
    
    return stats_data

def generate_table_4_popularity_tiers(df):
    """Generate Table 4: Popularity Tier Distribution"""
    
    print("\n" + "="*60)
    print("TABLE 4: POPULARITY TIER DISTRIBUTION")
    print("="*60)
    
    # Create popularity tiers
    def assign_tier(count):
        if count < 1_000_000:
            return 'Underground'
        elif count < 10_000_000:
            return 'Emerging'
        elif count < 100_000_000:
            return 'Popular'
        elif count < 500_000_000:
            return 'Hit'
        else:
            return 'Mega-Hit'
    
    df['popularity_tier'] = df['PlayCount'].apply(assign_tier)
    
    # Calculate distributions
    tier_data = []
    tier_order = ['Underground', 'Emerging', 'Popular', 'Hit', 'Mega-Hit']
    ranges = ['< 1M', '1M - 10M', '10M - 100M', '100M - 500M', '> 500M']
    
    for tier, range_str in zip(tier_order, ranges):
        tier_df = df[df['popularity_tier'] == tier]
        top_count = sum((tier_df['artist_tier'] == 'Top'))
        low_count = sum((tier_df['artist_tier'] == 'Low'))
        total = len(tier_df)
        
        tier_data.append({
            'Tier': tier,
            'Play Count Range': range_str,
            'Top Artists': top_count,
            'Low Artists': low_count,
            'Total': total,
            'Percentage': f"{(total/len(df))*100:.1f}%"
        })
    
    table4 = pd.DataFrame(tier_data)
    print(table4.to_string(index=False))
    
    # Save as LaTeX
    with open('table4_tiers.tex', 'w') as f:
        f.write(table4.to_latex(index=False, caption="Popularity Tier Distribution", label="tab:tiers"))
    
    return table4, df

def generate_table_5_duration_statistics(df):
    """Generate Table 5: Song Duration Statistics"""
    
    print("\n" + "="*60)
    print("TABLE 5: SONG DURATION STATISTICS")
    print("="*60)
    
    # Convert duration to seconds (assuming milliseconds)
    df['duration_seconds'] = df['Duration'] / 1000
    df['duration_minutes'] = df['duration_seconds'] / 60
    df['is_radio_friendly'] = (df['duration_minutes'] >= 2.5) & (df['duration_minutes'] <= 4.5)
    
    # Calculate statistics
    top_duration = df[df['artist_tier'] == 'Top']['duration_seconds']
    low_duration = df[df['artist_tier'] == 'Low']['duration_seconds']
    
    # T-test
    t_stat, p_value = stats.ttest_ind(top_duration.dropna(), low_duration.dropna())
    
    # Radio-friendly percentage
    top_radio = df[df['artist_tier'] == 'Top']['is_radio_friendly'].mean() * 100
    low_radio = df[df['artist_tier'] == 'Low']['is_radio_friendly'].mean() * 100
    
    # Chi-square test
    contingency = pd.crosstab(df['artist_tier'], df['is_radio_friendly'])
    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
    
    duration_stats = pd.DataFrame({
        'Duration Metric': [
            'Mean (seconds)',
            'Median',
            'Radio-Friendly %',
            'Statistical Test'
        ],
        'Top Artists': [
            f"{top_duration.mean():.1f} ± {top_duration.std():.1f}",
            f"{top_duration.median():.1f}",
            f"{top_radio:.1f}%",
            '-'
        ],
        'Low Artists': [
            f"{low_duration.mean():.1f} ± {low_duration.std():.1f}",
            f"{low_duration.median():.1f}",
            f"{low_radio:.1f}%",
            '-'
        ],
        'Significant Difference?': [
            f"p = {p_value:.4f}",
            '-',
            f"χ² = {chi2:.2f}, p = {chi_p:.4f}",
            '-'
        ]
    })
    
    print(duration_stats.to_string(index=False))
    
    # Save as LaTeX
    # Save as LaTeX
    with open('table5_duration.tex', 'w', encoding='utf-8') as f:
        f.write(duration_stats.to_latex(index=False, caption="Song Duration Statistics", label="tab:duration"))

    
    return duration_stats, df

def generate_table_6_lyrics_statistics(df):
    """Generate Table 6: Lyrics Corpus Statistics"""
    
    print("\n" + "="*60)
    print("TABLE 6: LYRICS CORPUS STATISTICS")
    print("="*60)
    
    # Calculate text statistics
    df['word_count'] = df['Lyrics'].fillna('').str.split().str.len()
    df['line_count'] = df['Lyrics'].fillna('').str.count('\n') + 1
    
    # Check for structure markers
    df['has_structure'] = df['Lyrics'].fillna('').str.contains(r'\[Verse|\[Chorus|\[Bridge', case=False, regex=True)
    
    # Calculate overall statistics
    total_words = df['word_count'].sum()
    all_words = ' '.join(df['Lyrics'].fillna('').tolist()).lower().split()
    unique_vocab = len(set(all_words))
    
    corpus_stats = {
        'Metric': [
            'Total Words',
            'Unique Vocabulary',
            'Average Words per Song',
            'Average Lines per Song',
            'Songs with Structure Markers'
        ],
        'Value': [
            f"{total_words/1_000_000:.2f} million",
            f"{unique_vocab:,} words",
            f"{df['word_count'].mean():.0f} ± {df['word_count'].std():.0f}",
            f"{df['line_count'].mean():.0f} ± {df['line_count'].std():.0f}",
            f"{df['has_structure'].mean()*100:.1f}%"
        ]
    }
    
    table6 = pd.DataFrame(corpus_stats)
    print(table6.to_string(index=False))
    
    # Save as LaTeX
    with open('table6_lyrics.tex', 'w') as f:
        f.write(table6.to_latex(index=False, caption="Lyrics Corpus Statistics", label="tab:lyrics"))
    
    return table6, df

def generate_table_11_statistical_comparison(df):
    """Generate Table 11: Statistical Comparison of Artist Tiers"""
    
    print("\n" + "="*60)
    print("TABLE 11: STATISTICAL COMPARISON OF ARTIST TIERS")
    print("="*60)
    
    # Calculate various metrics
    df['log_play_count'] = np.log1p(df['PlayCount'])
    df['vocabulary_diversity'] = df['Lyrics'].fillna('').apply(
        lambda x: len(set(x.lower().split())) / max(len(x.split()), 1)
    )
    
    # Sentiment (simplified - you'd use TextBlob in practice)
    from textblob import TextBlob
    df['sentiment'] = df['Lyrics'].fillna('').apply(
        lambda x: TextBlob(x).sentiment.polarity if x else 0
    )
    
    metrics = ['log_play_count', 'word_count', 'sentiment', 'vocabulary_diversity']
    metric_names = ['Play Count (log)', 'Word Count', 'Sentiment', 'Vocabulary Diversity']
    
    comparison_data = []
    
    for metric, name in zip(metrics, metric_names):
        top_data = df[df['artist_tier'] == 'Top'][metric].dropna()
        low_data = df[df['artist_tier'] == 'Low'][metric].dropna()
        
        # T-test
        t_stat, p_value = stats.ttest_ind(top_data, low_data)
        
        # Cohen's d
        pooled_std = np.sqrt((top_data.std()**2 + low_data.std()**2) / 2)
        cohens_d = (top_data.mean() - low_data.mean()) / pooled_std if pooled_std > 0 else 0
        
        comparison_data.append({
            'Metric': name,
            'Top Artists (μ ± σ)': f"{top_data.mean():.3f} ± {top_data.std():.3f}",
            'Low Artists (μ ± σ)': f"{low_data.mean():.3f} ± {low_data.std():.3f}",
            't-statistic': f"{t_stat:.3f}",
            'p-value': f"{p_value:.4f}",
            "Cohen's d": f"{cohens_d:.3f}"
        })
    
    table11 = pd.DataFrame(comparison_data)
    print(table11.to_string(index=False))
    
    # Save as LaTeX
    with open('table11_comparison.tex', 'w',encoding='utf-8') as f:
        f.write(table11.to_latex(index=False, caption="Statistical Comparison of Artist Tiers", 
                                 label="tab:comparison"))
    
    return table11, df

def create_figure_1_playcount_distribution(df):
    """Create Figure 1: Distribution of Play Counts (Log Scale)"""
    
    print("\n" + "="*60)
    print("FIGURE 1: PLAY COUNT DISTRIBUTION")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Log-scale distribution
    df['log_play_count'] = np.log1p(df['PlayCount'])
    
    # Subplot 1: Overlapping histograms
    top_data = df[df['artist_tier'] == 'Top']['log_play_count']
    low_data = df[df['artist_tier'] == 'Low']['log_play_count']
    
    ax1.hist(top_data, bins=30, alpha=0.6, label='Top Artists', color='royalblue', edgecolor='black')
    ax1.hist(low_data, bins=30, alpha=0.6, label='Low Artists', color='coral', edgecolor='black')
    ax1.set_xlabel('Log(Play Count + 1)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Play Counts (Log Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Box plot comparison
    df_plot = df[['artist_tier', 'log_play_count']].dropna()
    box_data = [df_plot[df_plot['artist_tier'] == 'Top']['log_play_count'],
                df_plot[df_plot['artist_tier'] == 'Low']['log_play_count']]
    
    bp = ax2.boxplot(box_data, labels=['Top Artists', 'Low Artists'], patch_artist=True)
    bp['boxes'][0].set_facecolor('royalblue')
    bp['boxes'][1].set_facecolor('coral')
    ax2.set_ylabel('Log(Play Count + 1)')
    ax2.set_title('Play Count Distribution Comparison')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure1_playcount_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure1_playcount_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Figure 1")
    
    return fig

def create_figure_2_tier_distribution(df):
    """Create Figure 2: Popularity Tier Distribution by Artist Category"""
    
    print("\n" + "="*60)
    print("FIGURE 2: POPULARITY TIER DISTRIBUTION")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create crosstab for stacked bar chart
    tier_order = ['Underground', 'Emerging', 'Popular', 'Hit', 'Mega-Hit']
    
    # Ensure popularity_tier exists
    if 'popularity_tier' not in df.columns:
        def assign_tier(count):
            if count < 1_000_000:
                return 'Underground'
            elif count < 10_000_000:
                return 'Emerging'
            elif count < 100_000_000:
                return 'Popular'
            elif count < 500_000_000:
                return 'Hit'
            else:
                return 'Mega-Hit'
        df['popularity_tier'] = df['PlayCount'].apply(assign_tier)
    
    crosstab = pd.crosstab(df['popularity_tier'], df['artist_tier'])
    crosstab = crosstab.reindex(tier_order, fill_value=0)
    
    # Create stacked bar chart
    x = np.arange(len(tier_order))
    width = 0.35
    
    ax.bar(x - width/2, crosstab['Top'], width, label='Top Artists', color='royalblue', edgecolor='black')
    ax.bar(x + width/2, crosstab['Low'], width, label='Low Artists', color='coral', edgecolor='black')
    
    ax.set_xlabel('Popularity Tier')
    ax.set_ylabel('Number of Songs')
    ax.set_title('Distribution of Songs Across Popularity Tiers')
    ax.set_xticks(x)
    ax.set_xticklabels(tier_order, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (top, low) in enumerate(zip(crosstab['Top'], crosstab['Low'])):
        if top > 0:
            ax.text(i - width/2, top + 10, str(top), ha='center', va='bottom')
        if low > 0:
            ax.text(i + width/2, low + 10, str(low), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('figure2_tier_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure2_tier_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Figure 2")
    
    return fig

def create_figure_3_temporal_distribution(df):
    """Create Figure 3: Release Year Distribution"""
    
    print("\n" + "="*60)
    print("FIGURE 3: TEMPORAL DISTRIBUTION")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Subplot 1: Overlapping histograms
    top_years = df[df['artist_tier'] == 'Top']['Year'].dropna()
    low_years = df[df['artist_tier'] == 'Low']['Year'].dropna()
    
    bins = np.arange(df['Year'].min(), df['Year'].max() + 2, 2)
    
    ax1.hist(top_years, bins=bins, alpha=0.6, label='Top Artists', color='royalblue', edgecolor='black')
    ax1.hist(low_years, bins=bins, alpha=0.6, label='Low Artists', color='coral', edgecolor='black')
    ax1.set_xlabel('Release Year')
    ax1.set_ylabel('Number of Songs')
    ax1.set_title('Distribution of Songs by Release Year')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Time series of average play counts
    yearly_avg = df.groupby(['Year', 'artist_tier'])['PlayCount'].mean().reset_index()
    
    for tier in ['Top', 'Low']:
        tier_data = yearly_avg[yearly_avg['artist_tier'] == tier]
        color = 'royalblue' if tier == 'Top' else 'coral'
        ax2.plot(tier_data['Year'], tier_data['PlayCount']/1_000_000, 
                marker='o', label=f'{tier} Artists', color=color, linewidth=2)
    
    ax2.set_xlabel('Release Year')
    ax2.set_ylabel('Average Play Count (Millions)')
    ax2.set_title('Average Play Counts Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure3_temporal_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure3_temporal_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Figure 3")
    
    return fig

def create_figure_4_word_count_distribution(df):
    """Create Figure 4: Word Count Distribution"""
    
    print("\n" + "="*60)
    print("FIGURE 4: WORD COUNT DISTRIBUTION")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data
    if 'word_count' not in df.columns:
        df['word_count'] = df['Lyrics'].fillna('').str.split().str.len()
    
    top_words = df[df['artist_tier'] == 'Top']['word_count'].dropna()
    low_words = df[df['artist_tier'] == 'Low']['word_count'].dropna()
    
    # Create violin plot
    parts = ax.violinplot([top_words, low_words], positions=[1, 2], 
                          showmeans=True, showmedians=True, showextrema=True)
    
    # Customize colors
    colors = ['royalblue', 'coral']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Top Artists', 'Low Artists'])
    ax.set_ylabel('Word Count')
    ax.set_title('Distribution of Song Word Counts')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean and median labels
    for i, (data, pos) in enumerate([(top_words, 1), (low_words, 2)]):
        ax.text(pos, data.mean(), f'μ={data.mean():.0f}', 
               ha='center', va='bottom', fontsize=9)
        ax.text(pos, data.median(), f'M={data.median():.0f}', 
               ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figure4_word_count_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure4_word_count_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Figure 4")
    
    return fig

def create_figure_5_correlation_matrix(df):
    """Create Figure 5: Correlation Matrix of Key Variables"""
    
    print("\n" + "="*60)
    print("FIGURE 5: CORRELATION MATRIX")
    print("="*60)
    
    # Prepare features
    if 'log_play_count' not in df.columns:
        df['log_play_count'] = np.log1p(df['PlayCount'])
    if 'word_count' not in df.columns:
        df['word_count'] = df['Lyrics'].fillna('').str.split().str.len()
    if 'duration_seconds' not in df.columns:
        df['duration_seconds'] = df['Duration'] / 1000
    
    # Select key variables
    variables = ['log_play_count', 'word_count', 'duration_seconds', 'Year']
    
    # Filter to available columns
    available_vars = [v for v in variables if v in df.columns]
    
    # Calculate correlation matrix
    corr_matrix = df[available_vars].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, square=True, linewidths=1, 
                cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Correlation Matrix of Key Variables')
    
    plt.tight_layout()
    plt.savefig('figure5_correlation_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure5_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Figure 5")
    
    return fig, corr_matrix

def generate_summary_statistics(df):
    """Generate a comprehensive summary for the paper"""
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS FOR PAPER")
    print("="*60)
    
    summary = f"""
DATA DESCRIPTION SUMMARY
========================

1. DATASET SIZE:
   - Total songs: {len(df):,}
   - Top artists: {sum(df['artist_tier'] == 'Top'):,} ({sum(df['artist_tier'] == 'Top')/len(df)*100:.1f}%)
   - Low artists: {sum(df['artist_tier'] == 'Low'):,} ({sum(df['artist_tier'] == 'Low')/len(df)*100:.1f}%)

2. PLAY COUNT RANGES:
   - Overall: {df['PlayCount'].min():,.0f} to {df['PlayCount'].max():,.0f}
   - Top artists: {df[df['artist_tier'] == 'Top']['PlayCount'].min():,.0f} to {df[df['artist_tier'] == 'Top']['PlayCount'].max():,.0f}
   - Low artists: {df[df['artist_tier'] == 'Low']['PlayCount'].min():,.0f} to {df[df['artist_tier'] == 'Low']['PlayCount'].max():,.0f}

3. TEMPORAL COVERAGE:
   - Years spanned: {df['Year'].min():.0f} to {df['Year'].max():.0f}
   - Most common year: {df['Year'].mode().values[0]:.0f}

4. KEY DIFFERENCES (Top vs Low):
   - Median play count ratio: {df[df['artist_tier'] == 'Top']['PlayCount'].median() / df[df['artist_tier'] == 'Low']['PlayCount'].median():.1f}x
   - Average word count difference: {df[df['artist_tier'] == 'Top']['word_count'].mean() - df[df['artist_tier'] == 'Low']['word_count'].mean():.0f} words

5. DATA QUALITY:
   - Songs with lyrics: {df['Lyrics'].notna().sum():,} ({df['Lyrics'].notna().mean()*100:.1f}%)
   - Valid duration data: {df['Duration'].notna().sum():,} ({df['Duration'].notna().mean()*100:.1f}%)
   - Valid year data: {df['Year'].notna().sum():,} ({df['Year'].notna().mean()*100:.1f}%)
    """
    
    print(summary)
    
    # Save to file
    with open('data_summary_statistics.txt', 'w') as f:
        f.write(summary)
    
    print("\n✓ Summary saved to 'data_summary_statistics.txt'")
    
    return summary

def main():
    """Main execution function"""
    
    print("="*60)
    print("GENERATING DATA DESCRIPTION SECTION")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df, df_top, df_low = load_data()
    
    # Generate all tables
    print("\nGenerating tables...")
    table1 = generate_table_1_dataset_composition(df, df_top, df_low)
    table2 = generate_table_2_temporal_distribution(df)
    table3 = generate_table_3_playcount_statistics(df, df_top, df_low)
    table4, df = generate_table_4_popularity_tiers(df)
    table5, df = generate_table_5_duration_statistics(df)
    table6, df = generate_table_6_lyrics_statistics(df)
    table11, df = generate_table_11_statistical_comparison(df)
    
    # Generate all figures
    print("\nGenerating figures...")
    fig1 = create_figure_1_playcount_distribution(df)
    fig2 = create_figure_2_tier_distribution(df)
    fig3 = create_figure_3_temporal_distribution(df)
    fig4 = create_figure_4_word_count_distribution(df)
    fig5, corr_matrix = create_figure_5_correlation_matrix(df)
    
    # Generate summary
    summary = generate_summary_statistics(df)
    
    print("\n" + "="*60)
    print("DATA DESCRIPTION GENERATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  Tables (LaTeX format):")
    print("    - table1_composition.tex")
    print("    - table2_temporal.tex")
    print("    - table3_playcount.tex")
    print("    - table4_tiers.tex")
    print("    - table5_duration.tex")
    print("    - table6_lyrics.tex")
    print("    - table11_comparison.tex")
    print("\n  Figures (PDF and PNG):")
    print("    - figure1_playcount_distribution")
    print("    - figure2_tier_distribution")
    print("    - figure3_temporal_distribution")
    print("    - figure4_word_count_distribution")
    print("    - figure5_correlation_matrix")
    print("\n  Summary:")
    print("    - data_summary_statistics.txt")
    
    return df

if __name__ == "__main__":
    df = main()
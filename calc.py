from matplotlib.ticker import PercentFormatter
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

BG_COLOR = '#1a1a1a'
TEXT_COLOR = '#ffffff'
LINE_COLOR = '#40c4ff'
UP_COLOR = '#E42313'
DOWN_COLOR = '#2DB928'
NEUTRAL_COLOR = '#616161'
MISSING_COLOR = '#424242'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("fig", exist_ok=True)

df = pd.read_csv("hist.csv", encoding='gb2312')
periods = df.columns[1:].tolist()
clean_periods = []
for col in periods:
    clean = col.replace('"', '').split(')')[0].replace('(', '\n')
    clean_periods.append(clean)

all_operators = df.iloc[:, 1:].stack().dropna().unique()
full_index = pd.MultiIndex.from_product(
    [all_operators, periods],
    names=["干员", "时期"]
)

long_df = df.melt(id_vars="名次", value_vars=periods, 
                var_name="时期", value_name="干员").dropna()
merged = pd.merge(full_index.to_frame(index=False), long_df,
                on=["干员", "时期"], how="left")

total_players = long_df.groupby("时期")["名次"].max().to_dict()
merged["总人数"] = merged["时期"].map(total_players)
merged = merged.sort_values(["干员", "时期"])

merged["排名数值"] = merged["名次"]
merged["排名比例"] = (merged["名次"] / merged["总人数"]) * 100
merged["前次名次"] = merged.groupby("干员")["名次"].shift(1)
merged["排名变化"] = merged.apply(
    lambda x: 0 if pd.isna(x["前次名次"]) else x["前次名次"] - x["名次"],
    axis=1
)

last_period = periods[-1]
last_rank_map = long_df[long_df["时期"] == last_period][["干员", "名次"]].dropna()
last_rank_map["名次"] = last_rank_map["名次"].astype(int)

max_rank = last_rank_map["名次"].max()
last_rank_map["倒序名次"] = max_rank - last_rank_map["名次"] + 1
last_rank_map = last_rank_map.set_index("干员")["倒序名次"].to_dict()
def gen(operator):
    """
    Generate a trend analysis chart for a given operator.

    This function creates a detailed visualization of an operator's ranking trends over time,
    including rank changes and percentile rankings. The chart is saved as a PNG file.

    Parameters:
    operator (str): The name of the operator for which to generate the chart.

    Returns:
    None. The function saves the generated chart as a PNG file and does not return any value.

    The function performs the following main tasks:
    1. Filters and prepares data for the specified operator.
    2. Creates a matplotlib figure with two y-axes.
    3. Plots rank changes as a bar chart on the first y-axis.
    4. Plots percentile rankings as a line chart on the second y-axis.
    5. Adds various visual elements like color coding, annotations, and grid lines.
    6. Saves the resulting chart as a PNG file named after the operator's final ranking.

    Note: This function relies on several global variables and dataframes that must be
    defined before calling this function, including 'merged', 'periods', 'clean_periods',
    'last_rank_map', and various color constants.
    """
    op_data = merged[merged["干员"] == operator].sort_values("时期")
    op_data = op_data.set_index("时期").reindex(periods).reset_index()

    fig, ax1 = plt.subplots(figsize=(15, 8), facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax1.set_facecolor(BG_COLOR)

    color_map = {
        'up': UP_COLOR,
        'down': DOWN_COLOR, 
        'new': NEUTRAL_COLOR,
        'na': MISSING_COLOR
    }

    colors = []
    changes = []
    for _, row in op_data.iterrows():
        if pd.isna(row["名次"]):
            colors.append(MISSING_COLOR)
            changes.append(None)
        elif row["排名变化"] == 0:
            colors.append(NEUTRAL_COLOR)
            changes.append(0)
        elif row["排名变化"] > 0:
            colors.append(UP_COLOR)
            changes.append(row["排名变化"])
        else:
            colors.append(DOWN_COLOR)
            changes.append(row["排名变化"])

    bars = ax1.bar(range(len(periods)), 
                 op_data["排名变化"].fillna(0),
                 color=colors, 
                 alpha=0.3, 
                 width=0.8,
                 edgecolor=NEUTRAL_COLOR)

    for idx, (bar, change) in enumerate(zip(bars, changes)):
        if change is None or change == 0:
            continue

        if change > 0:
            arrow_symbol = '↑'
            y_offset = 2
            va = 'bottom'
        else:
            arrow_symbol = '↓'
            y_offset = -2
            va = 'top'

        ax1.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + y_offset,
                f"{arrow_symbol}{abs(int(change))}", 
                ha='center', 
                va=va,
                color=bar.get_facecolor(),
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor=BG_COLOR, 
                        alpha=0.8,
                        edgecolor='none',
                        boxstyle='round,pad=0.1'))

    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_ylabel("排名变化", labelpad=15, color=TEXT_COLOR)
    ax1.set_ylim(-50, 50)

    ax2 = ax1.twinx()
    ax2.set_facecolor(BG_COLOR)
    line = ax2.plot(range(len(periods)), 
                  op_data["排名比例"], 
                  marker='o', 
                  markersize=10,
                  linewidth=3, 
                  color=LINE_COLOR,
                  markerfacecolor='white',
                  markeredgewidth=2)
    sep_position = 9
    ax1.axvline(x=sep_position - 0.5,  
              color='#808080',         
              linewidth=1.5,           
              alpha=0.7,              
              linestyle='--',          
              zorder=1)               
    ax2.axhline(33.33, color='#757575', linewidth=0.5, alpha=0.3, zorder=0)
    ax2.axhline(66.66, color='#757575', linewidth=0.5, alpha=0.3, zorder=0)
    for ax in [ax1, ax2]:
        ax.spines['bottom'].set_color(TEXT_COLOR)
        ax.spines['top'].set_color(TEXT_COLOR) 
        ax.spines['right'].set_color(TEXT_COLOR)
        ax.spines['left'].set_color(TEXT_COLOR)
        ax.tick_params(axis='x', colors=TEXT_COLOR)
        ax.tick_params(axis='y', colors=TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)

    for i, (idx, row) in enumerate(op_data.iterrows()):
        if pd.notnull(row["名次"]):
            ax2.annotate(f'#{int(row["名次"])}', 
                        (i, row["排名比例"]),
                        textcoords="offset points",
                        xytext=(0,12),
                        ha='center',
                        fontsize=10,
                        fontweight='bold',
                        color=LINE_COLOR,
                        bbox=dict(facecolor=BG_COLOR, alpha=0.7, edgecolor='none'))

    ax2.set_ylim(100, 0)
    ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax2.set_ylabel("排名百分比", labelpad=15, color=TEXT_COLOR)

    plt.xticks(range(len(periods)), clean_periods, rotation=45, ha='right')
    plt.title(f"{operator} 排名趋势分析", pad=20, fontsize=18, color=TEXT_COLOR)

    ax1.grid(False)
    ax2.grid(False)

    fig.tight_layout()

    final_rank = last_rank_map.get(operator, 999)
    if final_rank == 999:
        return
    filename = f"{final_rank:03d}.png"
    # plt.savefig(f"fig/{filename}", dpi=150, bbox_inches='tight', transparent=False)
    plt.savefig(f"fig/{operator}.png", dpi=150, bbox_inches='tight', transparent=False)
    plt.close()


from tqdm import tqdm
if __name__ == "__main__":
    for operator in tqdm(all_operators, desc="生成图表"):
        gen(operator)
import matplotlib.pyplot as plt
import math
import pathlib
style_path = '../src/utilities/thesis.mplstyle'
plt.style.use(style_path)

__all__ = ['save_fig', 'lineplot', 'boxplot']

def save_fig(fig, path, title, note=None):
    path = pathlib.Path(path)
    table_notes = f"""{note}."""
    label = " ".join(title.split(" ")[:4]).replace(".", "").replace(":", "").replace("-", " ")
    name = f'{title.lower().replace(" ", "_").replace("-", "_").replace(".", "").replace(":", "")}.png'
    fig.savefig(f'{path}/{name}', bbox_inches='tight', dpi=300)
    if note is not None:
        txt = f"""\\begin{{figure}}[H]\n \
    \\centering\n \
    \\caption{{{title}}}\\label{{FIG: {label}}}\n \
    \\includegraphics[width=1\linewidth]{{{path.parent.name}/{path.name}/{name}}}\n \
    \\raggedright\n \
    \\textsubscript{{\\parbox{{0.9\\linewidth}}{{\\textit{{{table_notes}}}}}}}
\\end{{figure}}\n\
\\noindent"""
    else:
        txt = f"""\\begin{{figure}}[H]\n \
    \\centering\n \
    \\caption{{{title}}}\\label{{FIG: {label}}}\n \
    \\includegraphics[width=1\linewidth]{{{path.parent.name}/{path.name}/{name}}}\n\
\\end{{figure}}\n\
\\noindent"""
    path = f'{path}/{name.replace(".png", ".tex")}'
    with open(path, 'w') as f:
        f.write(txt)
        f.close()
        
def lineplot(df, y, x=None, ylabel=None, xlabel=None, interval=1.2, rotation=0):
    x_values = df.index if x is None else x
    fig, ax = plt.subplots()
    for i in y:
        ax.plot(x_values, df[i], label=i, alpha=0.9)
        
    ymin = df[y].min().min()
    ymax = df[y].max().max()
    ymax_rounded = math.ceil(ymax / interval) * interval
    ymin_rounded = math.floor(ymin / interval) * interval
        
    plt.ylim(ymin_rounded, ymax_rounded)
    plt.xticks(rotation=rotation, fontsize=11)
    ax.tick_params(axis='y', which='both', left=False, labelleft=True, labelsize=11)
    ax.tick_params(axis='x', which='major', bottom=False, labelbottom=True, labelsize=11)
    ax.grid(axis='y', which='minor', linestyle=':', color='#DADCE5')
    ax.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.12), fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    
    return fig

def boxplot(data_dict, ylabel=None, xlabel=None, rotation=0, outliers=False, vert=False, box_color='#5fc3cf'):
    fig, ax = plt.subplots()
    
    bp = ax.boxplot(data_dict.values(), labels=data_dict.keys(), vert=vert, showfliers=outliers, patch_artist=True)
    for box in bp['boxes']:
        box.set_facecolor(box_color)
    
    ax.tick_params(axis='y', which='both', left=False, labelleft=True, labelsize=11)
    plt.xticks(rotation=rotation, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis='y', which='major', linestyle=':', color='#DADCE5')
    
    fig.subplots_adjust(left=0.305)

    return fig

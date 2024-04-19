
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors
import seaborn as sns
import pandas as pd


def plot_components_grid(principal_components, marker_names):

    # marker_names = markers_df.columns.to_list()
    maxPCs = 12
    PC_names = [f"PC{i}" for i in range(1, maxPCs+1)]

    # make a dataframe
    components_df = pd.DataFrame.from_dict(dict(zip(PC_names, np.abs(principal_components))))
    components_df["markers"] = marker_names
    components_df = components_df.set_index("markers")

    colour_dict = {'PC1': '#B5E675',    'PC2': '#6ED8A9',   'PC3': '#51B3D4', 
                    'PC4': '#4579AA',   'PC5': '#BC96C9',   'PC6': '#917AC2', 
                    'PC7': '#5A488B',   'PC8': '#888888',   'PC9': '#888888',
                    'PC10': '#888888',  'PC11': '#888888',  'PC12': '#888888'}

    fig, ax = plt.subplots(figsize=(5, 5))

    for PC in colour_dict.keys():
        data = components_df.copy()
        # Make every column except the one we're plotting Nan
        data.loc[:, data.columns != PC] = np.nan

        colour_map = mpl.colors.LinearSegmentedColormap.from_list("", ["white",colour_dict[PC]])

        if PC == 'PC8':
                cbar_ax = fig.add_axes([0.98, 0.698, .05, .2])

                sns.heatmap(data, annot=False, fmt=".2f", linewidth=0.3,
                    cmap = colour_map, vmin = 0, vmax = 1, cbar_ax = cbar_ax, ax = ax, cbar_kws={"label": "absolute loading"})
                
        else:
            sns.heatmap(data, annot=False, fmt=".2f",
                        cmap = colour_map, vmin = 0, vmax = 1, linewidth=0.3,
                        cbar=False, ax = ax)
            

    ax.axhline(y=0, color='#333333',linewidth=1)
    ax.axhline(y=12, color='#333333',linewidth=1)
    ax.axvline(x=0, color='#333333',linewidth=1)
    ax.axvline(x=13, color='#333333',linewidth=1)
    ax.set(ylabel=None)
    ax.set(xlabel=None)

        # Vertical lines
    # ax.annotate('>95%', xy=(0.575, 0.915), xycoords='figure fraction')
    ax2 = ax.twiny()
    ax2.spines["bottom"].set_position(("axes", 1.04))
    ax2.spines["bottom"].set_linewidth(1.5)
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_tick_params(width=1.5, length=6)
    ax2.spines["bottom"].set_visible(True)
    ax2.set_xticks([-1, -0.32, 0.16, 1])
    ax2.set_xticklabels(['', '', '', ''])


    ax2.set_xlim(-1, 1)
    ax.annotate('>95%', xy=(0.33, 0.93), xycoords='figure fraction')
    ax.annotate('>98%', xy=(0.53, 0.93), xycoords='figure fraction')
    ax.annotate('>99%', xy=(0.78, 0.93), xycoords='figure fraction')

    fig.tight_layout()
    plt.show()




# def bin_horizontal_distance(horzDist,size_bin=0.05):

#     """
#     Bin the horizontal distance into bins of size size_bin.
#     horzDist is a numpy array of horizontal distances.
#     """

#     # Make a dataframe
#     df = pd.DataFrame(horzDist, columns = ['horzDist'])

#     bins = np.arange(-12.2,0.2, size_bin)
#     bins = np.around(bins, 3)
#     labels = bins.astype(str).tolist()
#     # make label one smaller
#     labels.pop(0)
    
#     df['bins'] = pd.cut(df['horzDist'], bins, right=False, labels = labels, include_lowest=True)

#     # Change bins back to numpy array of floats
#     binned_horzDist = df['bins'].to_numpy(dtype=float)

#     return binned_horzDist

# # -------------------------------------------------
# def bin_scores(scores,size_bin):
#     pass

# # -------------------------------------------------
# def plot_multicolour_heatmap():
#     pass

# # -------------------------------------------------
# def plot_components(hawk3d):


#     """
#     Plot a heat map of the components. Uses the Hawk3D class. 
#     """

#     components = hawk3d.PCA.principal_components

#     marker_names = hawk3d.keypoint_manager.names_right_keypoints

#     marker_names = [name.replace('right_', '') for name in marker_names]

#     # Add _x _y _z to each name
#     marker_names = [name + '_' + i for name in marker_names for i in ['x', 'y', 'z']]

#     PC_names = ['PC' + str(i) for i in range(1, 13)]

#     # Make a dataframe
#     components_df = pd.DataFrame.from_dict(dict(zip(PC_names, np.abs(components))))
#     components_df['markers'] = marker_names
#     components_df = components_df.set_index('markers')

#     colour_PC_dict = {'PC1': '#B5E675', 'PC2': '#6ED8A9', 'PC3': '#51B3D4', 
#                       'PC4': '#4579AA', 'PC5': '#BC96C9', 'PC6': '#917AC2', 
#                       'PC7': '#5A488B', 'PC8': '#888888', 'PC9': '#888888',
#                       'PC10': '#888888', 'PC11': '#888888', 'PC12': '#888888'}

#     fig, ax = plt.subplots(figsize=(5, 5))



#     for PC in PC_names:
#         data = components_df.copy()
#         # Every column except the second one
#         data.loc[:, data.columns != PC] = float('nan')

#         colour_map = mpl.colors.LinearSegmentedColormap.from_list("", ["white",colour_PC_dict[PC]])
#         if PC == 'PC8':
#             cbar_ax = fig.add_axes([0.98, 0.698, .05, .2])

#             sns.heatmap(data, annot=False, fmt=".2f", linewidth=0.3,
#                 cmap = colour_map, vmin = 0, vmax = 1, cbar_ax = cbar_ax, ax = ax, cbar_kws={"label": "absolute loading"})
            
#         else:
#             sns.heatmap(data, annot=False, fmt=".2f",
#                         cmap = colour_map, vmin = 0, vmax = 1, linewidth=0.3,
#                         cbar=False, ax = ax)
            

#     ax.axhline(y=0, color='#333333',linewidth=1)
#     ax.axhline(y=12, color='#333333',linewidth=1)
#     ax.axvline(x=0, color='#333333',linewidth=1)
#     ax.axvline(x=13, color='#333333',linewidth=1)
#     ax.set(ylabel=None)
#     ax.set(xlabel=None)

#      # Vertical lines
#     # ax.annotate('>95%', xy=(0.575, 0.915), xycoords='figure fraction')
#     ax2 = ax.twiny()
#     ax2.spines["bottom"].set_position(("axes", 1.04))
#     ax2.spines["bottom"].set_linewidth(1.5)
#     ax2.xaxis.set_ticks_position("bottom")
#     ax2.xaxis.set_tick_params(width=1.5, length=6)
#     ax2.spines["bottom"].set_visible(True)
#     ax2.set_xticks([-1, -0.32, 0.16, 1])
#     ax2.set_xticklabels(['', '', '', ''])

    
#     ax2.set_xlim(-1, 1)
#     ax.annotate('>95%', xy=(0.33, 0.93), xycoords='figure fraction')
#     ax.annotate('>98%', xy=(0.53, 0.93), xycoords='figure fraction')
#     ax.annotate('>99%', xy=(0.78, 0.93), xycoords='figure fraction')



#     fig.tight_layout()
#     plt.show()

#     return fig, ax

# # -------------------------------------------------
# def plot_compare_hawks_components(hawk3d):
#     pass

# # -------------------------------------------------
# def plot_score(data,PC_name,fig,ax):

#     colour_PC_dict = {'PC1': '#B5E675', 'PC2': '#6ED8A9', 'PC3': '#51B3D4', 
#                       'PC4': '#4579AA', 'PC5': '#BC96C9', 'PC6': '#917AC2', 
#                       'PC7': '#5A488B', 'PC8': '#888888', 'PC9': '#888888',
#                       'PC10': '#888888', 'PC11': '#888888', 'PC12': '#888888'}


#     score, std, horzDist, bodypitch = get_binned_scores(data, PC_name)
    

    
#     ax.axhline(y=0, color='#333333', linestyle=':', linewidth=0.5)
    
#     ax.fill_between(horzDist, score-std, score+std, color=colour_PC_dict[PC_name], alpha=0.4, edgecolor='none')
    
#     ax.plot(horzDist, score, color=colour_PC_dict[PC_name], linewidth=2)

#     return fig, ax

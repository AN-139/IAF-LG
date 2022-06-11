import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

class VisualClass():
    def __init__(self):
        pass

    def PlotStepHistory(self, title, seriesDict, metaParam):
        fig, ax = plt.subplots()

        ax.set_title(title)
        ax.set_xlabel('step')

        for seriesName in seriesDict:
            ax.plot( seriesDict[ seriesName ], label = seriesName )
        ax.legend(loc='best') # add all to the lagend

        import matplotlib.lines as mlines

        handles, labels = ax.get_legend_handles_labels()      
        handles = handles + [ mlines.Line2D( [], [], label = key + ": " + str(metaParam[key]) ) for key in metaParam]       
        ax.legend(handles = handles, loc = 'bottom', ncol = 4, frameon = True, framealpha = 0.5)
        ax.grid('on')

        plt.show()
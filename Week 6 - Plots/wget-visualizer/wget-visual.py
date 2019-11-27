import warnings
import pandas as pd
import numpy as np
from numpy import convolve
import matplotlib.pyplot as plt
from matplotlib import figure

def convert_number_to_KMG(x):
    if x < 1000:
        x = round(x,-1)
        return str(x)
    elif x < 1000000:
        x = round(x,-2) / 1000
        return str(x)+"K"
    elif x < 1000000000:
        x = round(x,-5) / 1000000
        return str(x)+"M"
    else:
        x = round(x,-8) / 1000000000
        return str(x)+"G"

def convert_KMG_to_number(x):
    total_stars = 0
    if "K" in x:
        if len(x) > 1:
            total_stars = float(x.replace('K', '')) * 1000 # convert K to a thousand
    elif "M" in x:
        if len(x) > 1:
            total_stars = float(x.replace('M', '')) * 1000000 # convert M to a million
    elif "G" in x:
        total_stars = float(x.replace('G', '')) * 1000000000 # convert G to a Billion
    else:
        total_stars = float(x) # Less than 1000    
    return int(total_stars)

def convert_speed(arr):
    newdata = []
    for entry in arr:
        newpoint = convert_KMG_to_number( entry.replace(",", ".") )
        newdata.append(newpoint)
    return np.array(newdata)
    
def load_files(filenames, cols):
    frames = {} #dictionary
    for filename in filenames:
        df = pd.read_csv(filename, skiprows=5, skipfooter=2, header=None, delim_whitespace=True, error_bad_lines=False, usecols = cols)
        df = df.dropna(how='any')        
        #print(filename)
        # print(df.tail())
        key = filename.split(".")[0]
        df.columns = [key]        
        frames[key] = df
    return frames

def clean_extreme(values, target=5):
    ratio = 10 * target
    while ratio > target:
        mean = np.mean(values)
        max = np.amax(values)
        ratio = max / mean
        if ratio > target:
            index = np.argmax(values)
            # smooth at the index
            values[index] = (values[index] + mean ) / 2
    return values

def fabric_softener(values, window=50):
    # first apply moving average smoothing
    clean = values[~np.isnan(values)] # remove nan values
    weights = np.repeat(1.0, window) / window
    ma = np.convolve(clean, weights, 'valid') # moving average
    mean = np.mean(ma)
    # clear nan values
    # for i in range(0, len(ma)):
    #    if np.isnan(ma[i]).all():
    #        ma[i] = mean
    ma = clean_extreme(ma)
    return ma

def create_plots(frames, fig):
    ymax = 0
    colors_cubic = ['#00FF00', '#00DD11', '#00AA22'] # RRGGBB, greenish colors for cubic
    colors_lead = ['#FF0000', '#DD0011', '#AA0022'] ## RRGGBB, reddish colors for lead
    next_cubic = 0
    next_lead = 0
    next_plot = 1 # should start with 1
    for key, frame in frames.items():
        df = frames[key]      
        # set dataframe size
        array = np.array(df[key].values)
        array = convert_speed(array)
        ym = np.mean(array)
        
        w = 50
        if "cubic" in key:
            w = 80
        elif "lead" in key:
            w = 80
        array = fabric_softener(array, window=w)
        size = len(array)
        x_axis = range(0,size)
        local_ymax = np.amax(array)
        #local_ymean = np.mean(array)
        if local_ymax > ymax:
            ymax = local_ymax
        # Select line style, solid for cubic, dashed for lead
        lstyle = 'solid'
        if "lead" in key:
            lstyle='dashed'
            coldata = colors_lead[0]
            colmean = colors_lead[1]
            colmedian = colors_lead[2]
        if "cubic" in key:
            lstyle='solid'
            coldata = colors_cubic[0]
            colmean = colors_cubic[1]
            colmedian = colors_cubic[2]
        # create subplot
        axi = fig.add_subplot(3, 2, next_plot) # rows, cols, plot number
        axi.plot(x_axis, array, color=coldata, linestyle=lstyle, label=key)
        # add mean line
        y_mean = [ym]*len(array)
        lbl = "Sistemlerde raporlanan ortalama hız=" + convert_number_to_KMG(ym)
        axi.plot(x_axis, y_mean, color=colmean, linestyle="-.", label=lbl)
        # add median line
        ymd = np.median(array)
        y_median = [ymd] * len(array)
        lbl = "Gerçekte hissedilen ortanca hız=" + convert_number_to_KMG(ymd)
        axi.plot(x_axis, y_median, color=colmedian, linestyle=":", label=lbl)
        # axi.set_xlabel('İndirilen parça')
        ylim = 1.1 * local_ymax 
        axi.set_ylim(0, ylim)
        axi.set_ylabel('Parçanın indirilme hızı')   
        axi.set_title(key)
        axi.legend()
        next_plot = next_plot + 1
    return ymax
        #All data series have been added to plot

warnings.filterwarnings("ignore")
names = ['astro-cubic.out', 'astro-lead.out',
             'guppy-cubic.out', 'guppy-lead.out',
             'interview-cubic.out', 'interview-lead.out']
needed = [7]
frames = load_files(filenames=names, cols=needed)
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ymax = create_plots(frames, fig)

# f, axarr = plt.subplots(6)
#for index in range(6):
#    axarr[index].set_ylim(0, ymax)
plt.show()
# plt.savefig('the_best_plot.pdf')
   
    
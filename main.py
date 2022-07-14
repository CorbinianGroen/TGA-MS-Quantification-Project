from tkinter import *
import pandas as pd
import numpy as np
import scipy.interpolate
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from tkinter import filedialog
from tkinter import ttk
import time


def Loop():
    dirref = open('dirref.txt')
    dirref_c = dirref.readlines()
    dirref_t = dirref_c[1].replace('\n', '')
    dirref.close()

    def header(filename):
        ms_header = open(filename)

        content = ms_header.readlines()

        content = content[6].split('\t')
        while ("" in content):
            content.remove("")
        content.remove('\n')

        content_t = []
        for x in range(0, len(content)):
            a = content[x]
            a1 = a + '_t'
            content_t.append(a1)

        content_x = []
        for x in range(0, len(content)):
            a = content[x]
            a1 = a + '_x'
            content_x.append(a1)
        colnames = []
        for x in range(0, len(content)):
            a = content_t[x]
            b = content_x[x]
            c = content[x]
            colnames.append(a)
            colnames.append(b)
            colnames.append(c)

        colnames[1] = 'Time/s'
        return colnames

    window = Tk()
    window.title('TGA-MS Quantification')
    window.geometry('530x250')
    window.configure(background='white')

    # create labels

    Label(window, text='Data:', bg='white', fg='black', font='none 12').grid(row=1, column=0, sticky=W)

    Label(window, text='TGA:', bg='white', fg='black', font='none 12').grid(row=2, column=0, sticky=E)
    Label(window, text='MS:', bg='white', fg='black', font='none 12').grid(row=3, column=0, sticky=E)
    Label(window, text='Normalization:', bg='white', fg='black', font='none 12').grid(row=4, column=0, sticky=E)
    various = Label(window, text='Quantification:', bg='white', fg='black', font='none 12')
    various.grid(row=5, column=0, sticky=E)
    # create entry

    TGA_file = Entry(window, width=60, bg='lightgrey')
    TGA_file.grid(row=2, column=1, sticky=W)
    MS_file = Entry(window, width=60, bg='lightgrey')
    MS_file.grid(row=3, column=1, sticky=W)
    Norm_val = Entry(window, width=5, bg='lightgrey')
    Norm_val.grid(row=4, column=1, sticky=W)
    Quan_val = Entry(window, width=15, bg='lightgrey')
    Quan_val.grid(row=5, column=1, sticky=W)

    # create open button
    def open_TGA():
        file = filedialog.askopenfilename(title='Open TGA file', initialdir=str(dirref_t),
                                          filetypes=[('Textfile', '*.txt')])
        if file is not None:
            TGA_file.insert(0, file)

    Button(window, text='Open', width=6, command=open_TGA).grid(row=2, column=2, sticky=E)

    def open_MS():
        file = filedialog.askopenfilename(title='Open MS file', initialdir=str(dirref_c[2]),
                                          filetypes=[('ASCIIfile', '*.asc')])
        if file is not None:
            MS_file.insert(0, file)

    Button(window, text='Open', width=6, command=open_MS).grid(row=3, column=2, sticky=E)

    # make check box
    def labelupdate():
        if check_var.get() == 1:
            various.config(text='n.a.')
        else:
            various.config(text='Quantification:')

    check_var = IntVar()
    check = Checkbutton(window, text='Only TGA-MS data', variable=check_var, onvalue=1, offvalue=0, bg='white',
                        activebackground='white', command=labelupdate)
    check.grid(row=6, column=1, sticky=W, pady=20)

    # create exit button
    def exit1():
        pgBar = ttk.Progressbar(window, length=300, mode='determinate', orient=HORIZONTAL)
        pgBar.grid(row=7, columnspan=2, column=1, sticky=E, pady=20)

        pgBar['value'] = 0
        window.update_idletasks()

        global mz
        mz = Quan_val.get().split(' ')
        global numberofmasses
        numberofmasses = len(mz)
        global Norm
        Norm = Norm_val.get()

        global tga_ms
        global tga_ms_quan
        global tga
        global tgafile
        global msfile

        # get TGA file
        tgafile = TGA_file.get()
        colnames_tga = ['index', 'T_sample/C', 'time/s', 'HF/mW', 'Weight/mg', 'T_ramp/C']
        tga_input = pd.read_csv(tgafile, sep='\s+', skiprows=2, header=None, skipinitialspace=True, skipfooter=1,
                                engine='python', encoding='ANSI', names=colnames_tga)

        # convert comma to decimal and float number
        tga_all = tga_input['index']
        for b in range(1, 6):
            tga_1 = tga_input[tga_input.columns[b]].str.replace(',', '.')
            tga_all = pd.merge(tga_all, tga_1, left_index=True, right_index=True)
        # normalize mass to max value
        tga_conv = tga_all.astype(float).drop(['index', 'T_sample/C', 'HF/mW'], axis=1)
        tga_norm_1 = tga_conv['Weight/mg'] / tga_conv['Weight/mg'].loc[tga_conv['Weight/mg'].nlargest(1).index[0]] * 100
        tga = pd.merge(tga_conv, tga_norm_1, left_index=True, right_index=True)
        tga.rename(columns={'Weight/mg_y': 'Weight/%', 'Weight/mg_x': 'Weight/mg'}, inplace=True)

        # Derivative of the TGA Curve

        spline_tga = scipy.interpolate.splrep(tga['time/s'], tga['Weight/mg'])
        tga_der = scipy.interpolate.splev(tga['time/s'], spline_tga, der=1)
        data = pd.DataFrame([tga_der]).transpose() * (-1)
        data.rename(columns={data.columns[0]: 'Smoothed_Diff/mgs-1'}, inplace=True)
        tga = pd.merge(tga, data, left_index=True, right_index=True)

        pgBar['value'] = 15
        window.update_idletasks()

        # get ms file and header
        msfile = MS_file.get()
        msheader = header(msfile)
        ms_input = pd.read_csv(msfile, sep='\t', skiprows=8, names=msheader, header=None)

        # sort MS file
        ms_notime = ms_input.filter(regex=r'.*(?<!_t)$')
        ms = ms_notime.filter(regex=r'.*(?<!_x)$')
        X = 'Time/s'

        # Normalize MS
        ms_norm_1 = ms.drop(X, axis=1)
        ms_norm_noX = ms_norm_1.div(ms[Norm], axis=0)
        ms_norm = pd.concat([pd.Series(ms[X], index=ms_norm_noX.index, name=X), ms_norm_noX], axis=1)

        pgBar['value'] = 30
        window.update_idletasks()

        # interpolate all data
        #   upper limit for all interpolations
        global upper
        upper = int(ms_norm[X].loc[ms_norm[X].nlargest(2).index[1]]) - 1

        #   interpolate TGA diff
        tga_inter_X_np = np.linspace(0, upper, upper + 1)
        tga_ms = pd.DataFrame(tga_inter_X_np, columns=['time/s'])
        tga_header = list(tga)

        for x in range(1, tga.shape[1]):
            a = tga_header[x]
            tga_inter = interpolate.interp1d(x=tga['time/s'], y=tga[a], kind='linear')
            tga_inter_np = tga_inter(tga_ms['time/s'])
            tga_inter_pd = pd.DataFrame(tga_inter_np, columns=[a])
            tga_ms = pd.merge(tga_ms, tga_inter_pd, left_index=True, right_index=True)

        #   interpolate MS
        ms_header = list(ms_norm)
        for x in range(0, ms_norm.shape[1]):
            a = ms_header[x]
            ms_inter = interpolate.interp1d(x=ms_norm[X], y=ms_norm[a], kind='linear')
            ms_inter_np = ms_inter(tga_ms['time/s'])
            ms_inter_pd = pd.DataFrame(ms_inter_np, columns=[a])
            tga_ms = pd.merge(tga_ms, ms_inter_pd, left_index=True, right_index=True)
        tga_ms = tga_ms.drop('Time/s', axis=1)

        pgBar['value'] = 60
        window.update_idletasks()

        check_value = check_var.get()

        if check_value == 0:

            # smooth ms data
            global tga_ms_smooth
            tga_ms_smooth = tga_ms.loc[:, 'time/s':'Smoothed_Diff/mgs-1']

            for x in range(0, numberofmasses):
                a = mz[x]
                ms_smooth_1 = tga_ms[a]
                tga_ms_smooth = pd.merge(tga_ms_smooth, ms_smooth_1, left_index=True, right_index=True)

            pgBar['value'] = 80
            window.update_idletasks()

            # Quantification
            x0_2 = np.ones(numberofmasses * 2)

            def func(x):
                tga_ms_quan_1 = tga_ms_smooth.loc[:, 'time/s':'Smoothed_Diff/mgs-1']
                for a in range(0, numberofmasses):
                    b = mz[a]
                    quan_mz_1 = tga_ms_smooth[b] * x[a * 2] - x[a * 2 + 1]
                    tga_ms_quan_1 = pd.merge(tga_ms_quan_1, quan_mz_1, left_index=True, right_index=True)

                tga_ms_quan_1['SumMasses'] = tga_ms_quan_1[mz[0]]
                for a in range(1, numberofmasses):
                    c = mz[a]
                    tga_ms_quan_1['SumMasses'] = tga_ms_quan_1['SumMasses'] + tga_ms_quan_1[c]

                tga_ms_quan_1['Difference**2'] = (tga_ms_quan_1['Smoothed_Diff/mgs-1'] - tga_ms_quan_1[
                    'SumMasses']) ** 2

                negative_sum = 0
                for a in range(0, numberofmasses):
                    b = mz[a]
                    negative = tga_ms_quan_1[b].loc[tga_ms_quan_1[b].index[tga_ms_quan_1[b] < 0]]
                    if negative.shape[0] != 0:
                        negative = negative ** 2
                        negative_sum_1 = negative.sum()
                        negative_sum += negative_sum_1

                penalty = 10
                Sum = tga_ms_quan_1['Difference**2'].sum() + (penalty * negative_sum)

                return Sum

            global f
            f = minimize(func, x0=x0_2, method='SLSQP')
            tga_ms_quan = tga_ms_smooth.loc[:, 'time/s':'Smoothed_Diff/mgs-1']

            for x in range(0, numberofmasses):
                b = mz[x]
                quan_mz = tga_ms_smooth[b] * f.x[x * 2] - f.x[x * 2 + 1]
                tga_ms_quan = pd.merge(tga_ms_quan, quan_mz, left_index=True, right_index=True)

            tga_ms_quan['SumMasses'] = tga_ms_quan[mz[0]]
            for x in range(1, numberofmasses):
                c = mz[x]
                tga_ms_quan['SumMasses'] = tga_ms_quan['SumMasses'] + tga_ms_quan[c]

            tga_ms_quan['Difference**2'] = (tga_ms_quan['Smoothed_Diff/mgs-1'] - tga_ms_quan['SumMasses']) ** 2

            pgBar['value'] = 100
            window.update_idletasks()
            time.sleep(0.3)
            window.destroy()

        else:
            pgBar['value'] = 100
            window.update_idletasks()
            time.sleep(0.3)
            window.destroy()

    Button(window, text='Finished', width=6, command=exit1).grid(row=6, column=1, sticky=E)

    def on_closing():
        global x7
        x7 = 1
        window.destroy()
        exit()

    window.protocol("WM_DELETE_WINDOW", on_closing)
    window.mainloop()

    check_value = check_var.get()
    if check_value == 1:
        window = Tk()
        window.title('TGA-MS Save')
        window.geometry('520x110')
        window.configure(background='white')

        Label(window, text='dir:  ', bg='white', fg='black', font='none 12').grid(row=0, column=0, sticky=E)
        Label(window, text='Name:  ', bg='white', fg='black', font='none 12').grid(row=1, column=0, sticky=E)

        tgadir = tgafile.split('/')
        del tgadir[-1]
        tgadir = '/'.join(tgadir) + '/'

        msdir = msfile.split('/')
        del msdir[-1]
        msdir = '/'.join(msdir) + '/'

        tganame = tgafile.split('/')
        tganame = tganame[-1].removesuffix('.txt')

        dir_entry = Entry(window, width=60, bg='lightgrey')
        dir_entry.grid(row=0, column=1, sticky=E)
        dir_entry.insert(0, tgadir)
        name_entry = Entry(window, width=60, bg='lightgrey')
        name_entry.grid(row=1, column=1, sticky=E)
        name_entry.insert(0, tganame)

        # create exit button
        def dirsave():

            path = filedialog.askdirectory(title='Save Results', initialdir=tgadir)
            if path is not None:
                dir_entry.delete(0, 'end')
                dir_entry.insert(0, path)

        Button(window, text='Dir', width=6, command=dirsave).grid(row=0, column=2, sticky=W)

        def exit2():

            path_name = dir_entry.get()
            filename = name_entry.get()
            savefile1 = path_name + filename + '_tga-ms' + '.txt'
            # savefile2 = path_name + filename + '_with_smoothed_masses' + '.txt'
            tga_ms.to_csv(savefile1, sep='\t', index=False, header=True)
            # tga_ms_smooth.to_csv(savefile2, sep='\t', index=False, header=True)
            # plt.plot(tga_ms_smooth['time/s'], tga_ms_smooth['32'])
            # plt.show()
            dirref = open('dirref.txt', 'w')
            dirref.write('\n' + tgadir + '\n' + msdir)
            dirref.close()
            global x7
            x7 = 1
            window.destroy()
            exit()

        Button(window, text='Save + Exit', width=12, command=exit2).grid(row=2, column=2, sticky=E, pady=20)

        def exit3():

            path_name = dir_entry.get()
            filename = name_entry.get()
            savefile1 = path_name + filename + '_tga-ms' + '.txt'
            # savefile2 = path_name + filename + '_with_smoothed_masses' + '.txt'
            tga_ms.to_csv(savefile1, sep='\t', index=False, header=True)
            # tga_ms_smooth.to_csv(savefile2, sep='\t', index=False, header=True)
            # plt.plot(tga_ms_smooth['time/s'], tga_ms_smooth['32'])
            # plt.show()
            dirref = open('dirref.txt', 'w')
            dirref.write('\n' + tgadir + '\n' + msdir)
            dirref.close()
            global x7
            x7 = 0
            window.destroy()

        Button(window, text='Save + Do Another', width=16, command=exit3).grid(row=2, column=1, sticky=E, pady=20)

        def on_closing():
            global x7
            x7 = 1
            window.destroy()
            exit()

        window.protocol("WM_DELETE_WINDOW", on_closing)

        window.mainloop()

        if x7 == 0:
            Loop()

    else:
        pass

    # create the Window
    window = Tk()
    window.title('TGA-MS Quantification')
    window.geometry('1200x700')
    window.configure(background='white')

    global tga_ms_quan

    # the figure that will contain the plot
    fig = plt.figure()
    plt.subplots_adjust(bottom=0.25)
    #   Slider
    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    cons_factor = Slider(ax_slide, 'upper', valmin=0, valmax=upper + 1, valinit=upper, valstep=1)

    ax_slide_2 = plt.axes([0.25, 0.15, 0.65, 0.03])
    cons_factor_2 = Slider(ax_slide_2, 'lower', valmin=0, valmax=upper + 1, valinit=0, valstep=1)

    ax: plt.Axes = fig.subplots()
    p0 = ax.plot(tga_ms_quan['time/s'], tga_ms_quan['Smoothed_Diff/mgs-1'], label='Diff')
    p1 = ax.plot(tga_ms_quan['time/s'], tga_ms_quan['SumMasses'], label='Sum')
    for x in range(0, numberofmasses):
        a = mz[x]
        p = ax.plot(tga_ms_quan['time/s'], tga_ms_quan[a], label=a)
    ax.legend(loc='upper left', frameon=False)
    zeroline = ax.plot([0, upper], [0, 0], color='black', linestyle='dashed', linewidth=0.5)
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False).grid(row=0, column=0, sticky=W)
    # toolbar.update()
    canvas.get_tk_widget().grid(row=0, column=0, sticky=W, rowspan=numberofmasses * 2 + numberofmasses + 4)
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().grid(row=1, column=0, sticky=E, rowspan=numberofmasses * 2 + numberofmasses + 4)

    Filter = Entry(window, width=10, bg='lightgrey')
    Filter.grid(row=0, column=2, sticky=W)
    Filter.insert(0, '3')

    def smoothing():
        windowfilter = int(Filter.get())

        if windowfilter % 2 == 0:
            windowfilter += 1
        else:
            pass
        global tga_ms_smooth
        tga_ms_smooth = tga_ms.loc[:, 'time/s':'Weight/%']

        tga_smooth_0 = savgol_filter(tga_ms['Smoothed_Diff/mgs-1'], windowfilter, 2)
        tga_smooth_1 = pd.DataFrame([tga_smooth_0]).transpose()
        tga_smooth_1.rename(columns={tga_smooth_1.columns[0]: 'Smoothed_Diff/mgs-1'}, inplace=True)
        tga_ms_smooth = pd.merge(tga_ms_smooth, tga_smooth_1, left_index=True, right_index=True)

        for x in range(0, numberofmasses):
            a = mz[x]
            ms_smooth_0 = savgol_filter(tga_ms[a], windowfilter, 2)
            ms_smooth_1 = pd.DataFrame([ms_smooth_0]).transpose()
            ms_smooth_1.rename(columns={ms_smooth_1.columns[0]: a}, inplace=True)
            tga_ms_smooth = pd.merge(tga_ms_smooth, ms_smooth_1, left_index=True, right_index=True)

        tga_ms_quan = tga_ms_smooth.loc[:, 'time/s':'Smoothed_Diff/mgs-1']

        for x in range(0, numberofmasses):
            a = mz[x]
            quan_mz = tga_ms_smooth[a] * f.x[x * 2] - f.x[x * 2 + 1]
            tga_ms_quan = pd.merge(tga_ms_quan, quan_mz, left_index=True, right_index=True)

        tga_ms_quan['SumMasses'] = tga_ms_quan[mz[0]]
        for x in range(1, numberofmasses):
            c = mz[x]
            tga_ms_quan['SumMasses'] = tga_ms_quan['SumMasses'] + tga_ms_quan[c]

        tga_ms_quan['Difference**2'] = (tga_ms_quan['Smoothed_Diff/mgs-1'] - tga_ms_quan['SumMasses']) ** 2
        cons2 = cons_factor_2.val
        cons = cons_factor.val
        extend = ax.axis()
        ax.clear()
        ax.axis(extend)
        p0 = ax.plot(tga_ms_quan['time/s'].iloc[cons2:cons], tga_ms_quan['Smoothed_Diff/mgs-1'].iloc[cons2:cons],
                     label='Diff')
        p1 = ax.plot(tga_ms_quan['time/s'].iloc[cons2:cons], tga_ms_quan['SumMasses'].iloc[cons2:cons], label='Sum')
        for a in range(0, numberofmasses):
            b = mz[a]
            p = ax.plot(tga_ms_quan['time/s'].iloc[cons2:cons], tga_ms_quan[b].iloc[cons2:cons], label=b)
        zeroline = ax.plot([0, upper], [0, 0], color='black', linestyle='dashed', linewidth=0.5)

        ax.legend(loc='upper left', frameon=False)

        fig.canvas.draw()

    Button(window, text='Smooth', width=6, command=smoothing).grid(row=0, column=3, sticky=E)

    def func1(x):
        cons = cons_factor.val
        cons2 = cons_factor_2.val
        tga_ms_quan_1 = tga_ms_smooth.loc[cons2:cons, 'time/s':'Smoothed_Diff/mgs-1']
        for a in range(0, numberofmasses):
            b = mz[a]
            quan_mz_1 = tga_ms_smooth[b].iloc[cons2:cons] * x[a * 2] - x[a * 2 + 1]
            tga_ms_quan_1 = pd.merge(tga_ms_quan_1, quan_mz_1, left_index=True, right_index=True)

        tga_ms_quan_1['SumMasses'] = tga_ms_quan_1[mz[0]]
        for a in range(1, numberofmasses):
            c = mz[a]
            tga_ms_quan_1['SumMasses'] = tga_ms_quan_1['SumMasses'] + tga_ms_quan_1[c]

        tga_ms_quan_1['Difference**2'] = (tga_ms_quan_1['Smoothed_Diff/mgs-1'] - tga_ms_quan_1['SumMasses']) ** 2

        negative_sum = 0
        for a in range(0, numberofmasses):
            b = mz[a]
            negative = tga_ms_quan_1[b].loc[tga_ms_quan_1[b].index[tga_ms_quan_1[b] < 0]]
            if negative.shape[0] != 0:
                negative = negative ** 2
                negative_sum_1 = negative.sum()
                negative_sum += negative_sum_1

        penalty = 10
        Sum = tga_ms_quan_1['Difference**2'].sum() + (penalty * negative_sum)

        return Sum


    x0_2 = f.x

    # create Label
    Label(window, text='Smoothing Window:', bg='white', fg='black', font='none 12').grid(row=0, column=1, sticky=E)
    Label(window, text='Quantification:', bg='white', fg='black', font='none 12').grid(row=1, column=1, sticky=E)
    Label(window, text='Parameters:', bg='white', fg='black', font='none 12').grid(row=2, column=1, sticky=E)
    global h
    h = {}
    for x in range(0, numberofmasses):
        g = mz[x]
        h['Label_{0}'.format(x * 2)] = Label(window, text=g + ' ' + 'factor', bg='white', fg='black', font='none 12')
        h['Label_{0}'.format(x * 2)].grid(row=4 + x * 2, column=1,sticky=E)
        h['Label_{0}'.format(x * 2 + 1)] = Label(window, text=g + ' ' + 'baseline', bg='white', fg='black', font='none 12')
        h['Label_{0}'.format(x * 2 + 1)].grid(row=5 + x * 2, column=1,sticky=E)

    # create text entry box
    global d
    d = {}
    for x in range(0, numberofmasses):
        d['textentry_{0}'.format(x * 2)] = Entry(window, width=30, bg='lightgrey')
        d['textentry_{0}'.format(x * 2)].grid(row=4 + x * 2, column=2, sticky=W)
        d['textentry_{0}'.format(x * 2)].insert(0, x0_2[x * 2])
        d['textentry_{0}'.format(x * 2 + 1)] = Entry(window, width=30, bg='lightgrey')
        d['textentry_{0}'.format(x * 2 + 1)].grid(row=5 + x * 2, column=2, sticky=W)
        d['textentry_{0}'.format(x * 2 + 1)].insert(0, x0_2[x * 2 + 1])

    def update_labels():
        global h
        global d
        #destroy old labels

        for x in range(0, old_number):
            h['Label_{0}'.format(x * 2)].destroy()
            h['Label_{0}'.format(x * 2 + 1)].destroy()

        for x in range(0, old_number):
            d['textentry_{0}'.format(x * 2)].destroy()
            d['textentry_{0}'.format(x * 2 + 1)].destroy()

        #create new labels

        h = {}
        for x in range(0, numberofmasses):
            g = mz[x]
            h['Label_{0}'.format(x * 2)] = Label(window, text=g + ' ' + 'factor', bg='white', fg='black', font='none 12')
            h['Label_{0}'.format(x * 2)].grid(row=4 + x * 2, column=1, sticky=E)
            h['Label_{0}'.format(x * 2 + 1)] = Label(window, text=g + ' ' + 'baseline', bg='white', fg='black', font='none 12')
            h['Label_{0}'.format(x * 2 + 1)].grid(row=5 + x * 2, column=1, sticky=E)

        # create text entry box

        d = {}
        for x in range(0, numberofmasses):
            d['textentry_{0}'.format(x * 2)] = Entry(window, width=30, bg='lightgrey')
            d['textentry_{0}'.format(x * 2)].grid(row=4 + x * 2, column=2, sticky=W)
            d['textentry_{0}'.format(x * 2)].insert(0, '1')
            d['textentry_{0}'.format(x * 2 + 1)] = Entry(window, width=30, bg='lightgrey')
            d['textentry_{0}'.format(x * 2 + 1)].grid(row=5 + x * 2, column=2, sticky=W)
            d['textentry_{0}'.format(x * 2 + 1)].insert(0, '1')


    # create submit button
    def click():
        c = {}
        for x in range(0, numberofmasses):
            c['entered_text_{0}'.format(x * 2)] = d['textentry_{0}'.format(x * 2)].get()
            c['entered_text_{0}'.format(x * 2 + 1)] = d['textentry_{0}'.format(x * 2 + 1)].get()
        e = {}
        for x in range(0, numberofmasses):
            e['x0_2_{0}'.format(x)] = [c['entered_text_{0}'.format(x * 2)], c['entered_text_{0}'.format(x * 2 + 1)]]

        x0_fin = e['x0_2_{0}'.format(0)]
        for x in range(1, numberofmasses):
            x0_fin = x0_fin + e['x0_2_{0}'.format(x)]
        x0_fin = [float(x) for x in x0_fin]
        global f
        f = minimize(func1, x0=x0_fin, method='SLSQP')

        tga_ms_quan = tga_ms_smooth.loc[:, 'time/s':'Smoothed_Diff/mgs-1']

        for x in range(0, numberofmasses):
            b = mz[x]
            quan_mz = tga_ms_smooth[b] * f.x[x * 2] - f.x[x * 2 + 1]
            tga_ms_quan = pd.merge(tga_ms_quan, quan_mz, left_index=True, right_index=True)

        tga_ms_quan['SumMasses'] = tga_ms_quan[mz[0]]
        for x in range(1, numberofmasses):
            c = mz[x]
            tga_ms_quan['SumMasses'] = tga_ms_quan['SumMasses'] + tga_ms_quan[c]

        tga_ms_quan['Difference**2'] = (tga_ms_quan['Smoothed_Diff/mgs-1'] - tga_ms_quan['SumMasses']) ** 2

        for x in range(0, numberofmasses):
            d['textentry_{0}'.format(x * 2)].delete(0, 'end')
            d['textentry_{0}'.format(x * 2 + 1)].delete(0, 'end')

            d['textentry_{0}'.format(x * 2)].insert(0, f.x[x * 2])
            d['textentry_{0}'.format(x * 2 + 1)].insert(0, f.x[x * 2 + 1])
        extend = ax.axis()
        ax.clear()
        ax.axis(extend)
        cons = cons_factor.val
        cons2 = cons_factor_2.val
        p0 = ax.plot(tga_ms_quan['time/s'].iloc[cons2:cons], tga_ms_quan['Smoothed_Diff/mgs-1'].iloc[cons2:cons],
                     label='Diff')
        p1 = ax.plot(tga_ms_quan['time/s'].iloc[cons2:cons], tga_ms_quan['SumMasses'].iloc[cons2:cons], label='Sum')
        for a in range(0, numberofmasses):
            b = mz[a]
            p = ax.plot(tga_ms_quan['time/s'].iloc[cons2:cons], tga_ms_quan[b].iloc[cons2:cons], label=b)
        zeroline = ax.plot([0, upper], [0, 0], color='black', linestyle='dashed', linewidth=0.5)

        ax.legend(loc='upper left', frameon=False)

        fig.canvas.draw()
        # print(x0_2)
        pressed = 1
        global x1
        x1 = f.x
        global x2
        x2 = 1
        global x3
        x3 = 0

    Retry = Button(window, text='Retry', width=6, command=click)
    Retry.grid(row=numberofmasses * 2 + 5, column=2, sticky=E)

    # create integrate button
    def integrate():
        global x3
        x3 = 1
        global tga_ms_quan
        massloss = tga_ms_smooth['Weight/mg'].loc[tga_ms_smooth['Weight/mg'].nlargest(1).index[0]] - \
                   tga_ms_smooth['Weight/mg'].loc[tga_ms_smooth['Weight/mg'].nsmallest(1).index[0]]

        Label(window, text='{0:.4f} mg'.format(massloss), bg='white', fg='black', font='none 12').grid(
            row=numberofmasses * 2 + 8, column=2, sticky=E)
        Label(window, text='TGA:  ', bg='white', fg='black', font='none 12').grid(row=numberofmasses * 2 + 9, column=1,
                                                                                  sticky=E)
        global tga_int
        tga_int = Entry(window, width=30, bg='white')
        tga_int.grid(row=numberofmasses * 2 + 9, column=2, sticky=W)
        global k
        k = {}
        for x in range(0, numberofmasses):
            k['intentry_{0}'.format(x)] = Entry(window, width=30, bg='white')
            k['intentry_{0}'.format(x)].grid(row=numberofmasses * 2 + 10 + x, column=2, sticky=W)
            g = mz[x]
            Label(window, text=g + ':  ', bg='white', fg='black', font='none 12').grid(row=numberofmasses * 2 + 10 + x,
                                                                                       column=1, sticky=E)

        tga_ms_quan = tga_ms_smooth.loc[:, 'time/s':'Smoothed_Diff/mgs-1']

        for x in range(0, numberofmasses):
            b = mz[x]
            quan_mz = tga_ms_smooth[b] * x1[x * 2] - x1[x * 2 + 1]
            tga_ms_quan = pd.merge(tga_ms_quan, quan_mz, left_index=True, right_index=True)

        tga_ms_quan['SumMasses'] = tga_ms_quan[mz[0]]
        for x in range(1, numberofmasses):
            c = mz[x]
            tga_ms_quan['SumMasses'] = tga_ms_quan['SumMasses'] + tga_ms_quan[c]
        tga_ms_quan['Difference'] = (tga_ms_quan['Smoothed_Diff/mgs-1'] - tga_ms_quan['SumMasses'])
        tga_ms_quan['Difference**2'] = tga_ms_quan['Difference'] ** 2

        ax.clear()
        cons = cons_factor.val
        cons2 = cons_factor_2.val
        p0 = ax.plot(tga_ms_quan['time/s'], tga_ms_quan['Smoothed_Diff/mgs-1'], label='Diff')
        p1 = ax.plot(tga_ms_quan['time/s'], tga_ms_quan['SumMasses'], label='Sum')
        for a in range(0, numberofmasses):
            d = mz[a]
            p = ax.plot(tga_ms_quan['time/s'], tga_ms_quan[d], label=d)
        zeroline = ax.plot([0, upper], [0, 0], color='black', linestyle='dashed', linewidth=0.5)
        max_height = tga_ms_quan['Smoothed_Diff/mgs-1'].loc[tga_ms_quan['Smoothed_Diff/mgs-1'].nlargest(1).index[0]]

        line_lower = ax.plot([cons2, cons2], [0, max_height], color='black', linestyle='dashed', linewidth=0.5)
        line_upper = ax.plot([cons, cons], [0, max_height], color='black', linestyle='dashed', linewidth=0.5)
        ax.legend(loc='upper left', frameon=False)

        fig.canvas.draw()

        x_int = tga_ms_quan['time/s'].iloc[cons2:cons]
        y_int = tga_ms_quan['Smoothed_Diff/mgs-1'].iloc[cons2:cons]
        diff_int = np.trapz(x=x_int, y=y_int)
        tga_int.insert(0, diff_int)
        for x in range(0, numberofmasses):
            a = mz[x]
            y_int_ms = tga_ms_quan[a].iloc[cons2:cons]
            diff_int_ms = np.trapz(x=x_int, y=y_int_ms)
            k['intentry_{0}'.format(x)].insert(0, diff_int_ms)

    Integrate = Button(window, text='Integrate', width=6, command=integrate)
    Integrate.grid(row=numberofmasses * 2 + 5, column=3, sticky=W)

    def pressed(value):
        if 'x2' in globals():
            if x3 == 1:
                global tga_ms_quan
                tga_ms_quan = tga_ms_smooth.loc[:, 'time/s':'Smoothed_Diff/mgs-1']

                for x in range(0, numberofmasses):
                    b = mz[x]
                    quan_mz = tga_ms_smooth[b] * x1[x * 2] - x1[x * 2 + 1]
                    tga_ms_quan = pd.merge(tga_ms_quan, quan_mz, left_index=True, right_index=True)

                tga_ms_quan['SumMasses'] = tga_ms_quan[mz[0]]
                for x in range(1, numberofmasses):
                    c = mz[x]
                    tga_ms_quan['SumMasses'] = tga_ms_quan['SumMasses'] + tga_ms_quan[c]
                tga_ms_quan['Difference'] = (tga_ms_quan['Smoothed_Diff/mgs-1'] - tga_ms_quan['SumMasses'])
                tga_ms_quan['Difference**2'] = tga_ms_quan['Difference'] ** 2

                extend = ax.axis()
                ax.clear()
                ax.axis(extend)
                cons = cons_factor.val
                cons2 = cons_factor_2.val
                p0 = ax.plot(tga_ms_quan['time/s'], tga_ms_quan['Smoothed_Diff/mgs-1'], label='Diff')
                p1 = ax.plot(tga_ms_quan['time/s'], tga_ms_quan['SumMasses'], label='Sum')
                for a in range(0, numberofmasses):
                    d = mz[a]
                    p = ax.plot(tga_ms_quan['time/s'], tga_ms_quan[d], label=d)
                zeroline = ax.plot([0, upper], [0, 0], color='black', linestyle='dashed', linewidth=0.5)
                max_height = tga_ms_quan['Smoothed_Diff/mgs-1'].loc[
                    tga_ms_quan['Smoothed_Diff/mgs-1'].nlargest(1).index[0]]

                line_lower = ax.plot([cons2, cons2], [0, max_height], color='black', linestyle='dashed', linewidth=0.5)
                line_upper = ax.plot([cons, cons], [0, max_height], color='black', linestyle='dashed', linewidth=0.5)
                ax.legend(loc='upper left', frameon=False)

                fig.canvas.draw()
                tga_int.delete(0, 'end')
                x_int = tga_ms_quan['time/s'].iloc[cons2:cons]
                y_int = tga_ms_quan['Smoothed_Diff/mgs-1'].iloc[cons2:cons]
                diff_int = np.trapz(x=x_int, y=y_int)
                tga_int.insert(0, diff_int)
                for x in range(0, numberofmasses):
                    a = mz[x]
                    k['intentry_{0}'.format(x)].delete(0, 'end')
                    y_int_ms = tga_ms_quan[a].iloc[cons2:cons]
                    diff_int = np.trapz(x=x_int, y=y_int_ms)
                    k['intentry_{0}'.format(x)].insert(0, diff_int)

            else:
                tga_ms_quan = tga_ms_smooth.loc[:, 'time/s':'Smoothed_Diff/mgs-1']

                for x in range(0, numberofmasses):
                    b = mz[x]
                    quan_mz = tga_ms_smooth[b] * x1[x * 2] - x1[x * 2 + 1]
                    tga_ms_quan = pd.merge(tga_ms_quan, quan_mz, left_index=True, right_index=True)

                tga_ms_quan['SumMasses'] = tga_ms_quan[mz[0]]
                for x in range(1, numberofmasses):
                    c = mz[x]
                    tga_ms_quan['SumMasses'] = tga_ms_quan['SumMasses'] + tga_ms_quan[c]

                tga_ms_quan['Difference**2'] = (tga_ms_quan['Smoothed_Diff/mgs-1'] - tga_ms_quan['SumMasses']) ** 2
                extend = ax.axis()
                ax.clear()
                ax.axis(extend)
                cons = cons_factor.val
                cons2 = cons_factor_2.val
                tga_ms_quan_border = tga_ms_quan.iloc[cons2:cons]
                p0 = ax.plot(tga_ms_quan_border['time/s'], tga_ms_quan_border['Smoothed_Diff/mgs-1'], label='Diff')
                p1 = ax.plot(tga_ms_quan_border['time/s'], tga_ms_quan_border['SumMasses'], label='Sum')
                for a in range(0, numberofmasses):
                    d = mz[a]
                    p = ax.plot(tga_ms_quan_border['time/s'], tga_ms_quan_border[d], label=d)
                zeroline = ax.plot([0, upper], [0, 0], color='black', linestyle='dashed', linewidth=0.5)
                ax.legend(loc='upper left', frameon=False)
                fig.canvas.draw()

        else:
            extend = ax.axis()
            ax.clear()
            ax.axis(extend)
            cons = cons_factor.val
            cons2 = cons_factor_2.val
            tga_ms_quan = tga_ms_smooth.loc[:, 'time/s':'Smoothed_Diff/mgs-1']

            for x in range(0, numberofmasses):
                b = mz[x]
                quan_mz = tga_ms_smooth[b] * f.x[x * 2] - f.x[x * 2 + 1]
                tga_ms_quan = pd.merge(tga_ms_quan, quan_mz, left_index=True, right_index=True)

            tga_ms_quan['SumMasses'] = tga_ms_quan[mz[0]]
            for x in range(1, numberofmasses):
                c = mz[x]
                tga_ms_quan['SumMasses'] = tga_ms_quan['SumMasses'] + tga_ms_quan[c]

            tga_ms_quan['Difference**2'] = (tga_ms_quan['Smoothed_Diff/mgs-1'] - tga_ms_quan['SumMasses']) ** 2
            tga_ms_quan_border = tga_ms_quan.iloc[cons2:cons]
            p0 = ax.plot(tga_ms_quan_border['time/s'], tga_ms_quan_border['Smoothed_Diff/mgs-1'], label='Diff')
            p1 = ax.plot(tga_ms_quan_border['time/s'], tga_ms_quan_border['SumMasses'], label='Sum')
            for a in range(0, numberofmasses):
                d = mz[a]
                p = ax.plot(tga_ms_quan_border['time/s'], tga_ms_quan_border[d], label=d)
            zeroline = ax.plot([0, upper], [0, 0], color='black', linestyle='dashed', linewidth=0.5)
            ax.legend(loc='upper left', frameon=False)
            fig.canvas.draw()

    cons_factor.on_changed(pressed)
    cons_factor_2.on_changed(pressed)

    # create exit button
    def exit8():
        window.destroy()

    Save = Button(window, text='Save', width=6, command=exit8)
    Save.grid(row=numberofmasses * 2 + 5, column=4, sticky=W)

    Quantification = Entry(window, width=20, bg='lightgrey')
    Quantification.grid(row=1, column=2, sticky=W)
    mz1 = str(mz).replace(',', ' ')
    mz1 = mz1.replace('[', '')
    mz1 = mz1.replace(']', '')
    mz1 = mz1.replace('\'', '')
    mz1 = mz1.replace('  ', ' ')
    Quantification.insert(0, mz1)

    def quantification():
        global numberofmasses
        global old_number
        old_number = numberofmasses
        global mz
        mz = Quantification.get().split(' ')
        numberofmasses = len(mz)

        update_labels()
        Retry.grid(row=numberofmasses * 2 + 5, column=2, sticky=E)
        Integrate.grid(row=numberofmasses * 2 + 5, column=3, sticky=W)
        Save.grid(row=numberofmasses * 2 + 5, column=4, sticky=W)

        canvas.get_tk_widget().grid(row=0, column=0, sticky=W, rowspan=numberofmasses * 2 + numberofmasses + 4)
        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().grid(row=1, column=0, sticky=E, rowspan=numberofmasses * 2 + numberofmasses + 4)

        windowfilter = int(Filter.get())

        if windowfilter % 2 == 0:
            windowfilter += 1
        else:
            pass
        global tga_ms_smooth
        tga_ms_smooth = tga_ms.loc[:, 'time/s':'Weight/%']

        tga_smooth_0 = savgol_filter(tga_ms['Smoothed_Diff/mgs-1'], windowfilter, 2)
        tga_smooth_1 = pd.DataFrame([tga_smooth_0]).transpose()
        tga_smooth_1.rename(columns={tga_smooth_1.columns[0]: 'Smoothed_Diff/mgs-1'}, inplace=True)
        tga_ms_smooth = pd.merge(tga_ms_smooth, tga_smooth_1, left_index=True, right_index=True)

        for x in range(0, numberofmasses):
            a = mz[x]
            ms_smooth_0 = savgol_filter(tga_ms[a], windowfilter, 2)
            ms_smooth_1 = pd.DataFrame([ms_smooth_0]).transpose()
            ms_smooth_1.rename(columns={ms_smooth_1.columns[0]: a}, inplace=True)
            tga_ms_smooth = pd.merge(tga_ms_smooth, ms_smooth_1, left_index=True, right_index=True)
        click()

    Button(window, text='Quantification', width=10, command=quantification).grid(row=1, column=3, sticky=E)

    def on_closing():
        window.destroy()
        exit()

    window.protocol("WM_DELETE_WINDOW", on_closing)

    window.mainloop()

    # save window

    window = Tk()
    window.title('TGA-MS Quantification')
    window.geometry('600x150')
    window.configure(background='white')

    Label(window, text='dir:  ', bg='white', fg='black', font='none 12').grid(row=0, column=0, sticky=E)
    Label(window, text='Name:  ', bg='white', fg='black', font='none 12').grid(row=1, column=0, sticky=E)

    tgadir = tgafile.split('/')
    del tgadir[-1]
    tgadir = '/'.join(tgadir) + '/'

    msdir = msfile.split('/')
    del msdir[-1]
    msdir = '/'.join(msdir) + '/'

    tganame = tgafile.split('/')
    tganame = tganame[-1].removesuffix('.txt')

    dir_entry = Entry(window, width=60, bg='lightgrey')
    dir_entry.grid(row=0, column=1, sticky=E)
    dir_entry.insert(0, tgadir)
    name_entry = Entry(window, width=60, bg='lightgrey')
    name_entry.grid(row=1, column=1, sticky=E)
    name_entry.insert(0, tganame)

    # create exit button
    def dirsave():
        path = filedialog.askdirectory(title='Save Results', initialdir=tgadir)
        if path is not None:
            dir_entry.delete(0, 'end')
            dir_entry.insert(0, path)

    Button(window, text='Dir', width=6, command=dirsave).grid(row=0, column=2, sticky=W)

    def exit2():
        # create factor and int file
        max_weight = tga_ms_smooth['Weight/mg'].loc[tga_ms_smooth['Weight/mg'].nlargest(1).index[0]]
        massloss = max_weight - tga_ms_smooth['Weight/mg'].loc[tga_ms_smooth['Weight/mg'].nsmallest(1).index[0]]
        massloss_perc = (massloss / max_weight) * 100
        cons = cons_factor.val
        cons2 = cons_factor_2.val
        x_int = tga_ms_quan['time/s'].iloc[cons2:cons]
        y_int = tga_ms_quan['Smoothed_Diff/mgs-1'].iloc[cons2:cons]
        diff_int = np.trapz(x=x_int, y=y_int)
        results = pd.DataFrame(
            {'name': ['weightloss/mg : ', 'weightloss/% : ', 'area_tga_diff/mg : ', 'area_tga_diff/% : '],
             'result': [massloss, massloss_perc, diff_int, (diff_int / max_weight) * 100]})
        for x in range(0, numberofmasses):
            a = mz[x]
            y_int_ms = tga_ms_quan[a].iloc[cons2:cons]
            diff_int_ms = np.trapz(x=x_int, y=y_int_ms)
            results.loc[len(results.index)] = ['{0}/mg : '.format(a), diff_int_ms]
            results.loc[len(results.index)] = ['{0}/% : '.format(a), (diff_int_ms / max_weight) * 100]
        for x in range(0, numberofmasses):
            a = mz[x]
            results.loc[len(results.index)] = ['{0}_factor : '.format(a), '%.5E' % x1[x * 2]]
            results.loc[len(results.index)] = ['{0}_baseline : '.format(a), '%.5E' % x1[x * 2 + 1]]

        path_name = dir_entry.get()
        filename = name_entry.get()
        savefile1 = path_name + filename + '_tga-ms' + '.txt'
        savefile2 = path_name + filename + '_quantification' + '.txt'
        savefile3 = path_name + filename + '_results' + '.txt'
        tga_ms.to_csv(savefile1, sep='\t', index=False, header=True)
        tga_ms_quan.to_csv(savefile2, sep='\t', index=False, header=True)
        results.to_csv(savefile3, sep='\t', index=False, header=True)
        dirref = open('dirref.txt', 'w')
        dirref.write('\n' + tgadir + '\n' + msdir)
        dirref.close()
        global x7
        x7 = 1
        window.destroy()
        exit()

    Button(window, text='Save + Exit', width=12, command=exit2).grid(row=2, column=2, sticky=E, pady=20)

    def exit3():
        # create factor and int file
        max_weight = tga_ms_smooth['Weight/mg'].loc[tga_ms_smooth['Weight/mg'].nlargest(1).index[0]]
        massloss = max_weight - tga_ms_smooth['Weight/mg'].loc[tga_ms_smooth['Weight/mg'].nsmallest(1).index[0]]
        massloss_perc = (massloss / max_weight) * 100
        cons = cons_factor.val
        cons2 = cons_factor_2.val
        x_int = tga_ms_quan['time/s'].iloc[cons2:cons]
        y_int = tga_ms_quan['Smoothed_Diff/mgs-1'].iloc[cons2:cons]
        diff_int = np.trapz(x=x_int, y=y_int)
        results = pd.DataFrame(
            {'name': ['weightloss/mg : ', 'weightloss/% : ', 'area_tga_diff/mg : ', 'area_tga_diff/% : '],
             'result': [massloss, massloss_perc, diff_int, (diff_int / max_weight) * 100]})
        for x in range(0, numberofmasses):
            a = mz[x]
            y_int_ms = tga_ms_quan[a].iloc[cons2:cons]
            diff_int_ms = np.trapz(x=x_int, y=y_int_ms)
            results.loc[len(results.index)] = ['{0}/mg : '.format(a), diff_int_ms]
            results.loc[len(results.index)] = ['{0}/% : '.format(a), (diff_int_ms / max_weight) * 100]
        for x in range(0, numberofmasses):
            a = mz[x]
            results.loc[len(results.index)] = ['{0}_factor : '.format(a), '%.5E' % x1[x * 2]]
            results.loc[len(results.index)] = ['{0}_baseline : '.format(a), '%.5E' % x1[x * 2 + 1]]

        path_name = dir_entry.get()
        filename = name_entry.get()
        savefile1 = path_name + filename + '_tga-ms' + '.txt'
        savefile2 = path_name + filename + '_quantification' + '.txt'
        savefile3 = path_name + filename + '_results' + '.txt'
        tga_ms.to_csv(savefile1, sep='\t', index=False, header=True)
        tga_ms_quan.to_csv(savefile2, sep='\t', index=False, header=True)
        results.to_csv(savefile3, sep='\t', index=False, header=True)
        dirref = open('dirref.txt', 'w')
        dirref.write('\n' + tgadir + '\n' + msdir)
        dirref.close()
        global x7
        x7 = 0
        window.destroy()

    Button(window, text='Save + Do Another', width=17, command=exit3).grid(row=2, column=1, sticky=E, pady=20)

    def on_closing():
        global x7
        x7 = 1
        window.destroy()
        exit()

    window.protocol("WM_DELETE_WINDOW", on_closing)

    window.mainloop()

    if x7 == 0:
        Loop()


Loop()

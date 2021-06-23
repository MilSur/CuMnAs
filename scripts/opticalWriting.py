import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import kmeans1d

class DataProcessing:

    def __init__(self, file):
        dat_file = np.loadtxt(file + '.dat')
        pulses_file = np.loadtxt(file + '.pulses')
        t = dat_file[:,0]
        res4 = dat_file[:,3]/ dat_file[:,2]
        temp = dat_file[:,4]
        t_p = pulses_file[:,0]
        I_p = pulses_file[:,1]
        self.R4 = np.stack((t, res4), axis = -1)
        self.T = np.stack((t, temp), axis = -1)
        self.pulses = np.stack((t_p, I_p), axis = -1)
        self.initGuess = [8, 0.005*np.average(res4), np.average(res4)]

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    @staticmethod
    def get_statistics(array):
        return np.mean(array), np.std(array)

    def plotFull(self):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01)
        X = self.R4[:, 0]
        R4 = self.R4[:, 1]
        self.pulses[0,0]
        fig.append_trace(go.Scatter(
            x= (self.R4[:, 0]-self.pulses[0,0]),
            y=self.R4[:, 1],
        ), row=1, col=1)

        fig.append_trace(go.Bar(
            x= (self.pulses[:,0]-self.pulses[0,0]),
            y=self.pulses[:,1],
            width = 0.3,
        ), row=2, col=1)

        fig.append_trace(go.Scatter(
            x= (self.T[:, 0]-self.pulses[0,0]),
            y= self.T[:, 1]
        ), row=3, col=1)

        fig.update_yaxes(title_text="Resistivity (ohms)", row=1, col=1)
        fig.update_yaxes(title_text="Intensity (mW)", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (Celsius)", row=3, col=1)
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)


        fig.update_layout(height=1000, width=900, showlegend=False)
        fig.show()

    def groupData(self):
        inds = [self.find_nearest(self.R4[:,0], i) for i in self.pulses[:,0]]
        R4_splitted = np.split(self.R4, inds)
        R4_splitted = np.array(R4_splitted[1:]) # Cut signal before first pulse
        R4_splitted = np.array([np.stack((p[:,0] - self.pulses[i,0], p[:,1]), axis = -1) for i, p in enumerate(R4_splitted)]) # substract timepoint of pulse from each pulse
        unique_pulses = np.unique(self.pulses[:,1])
        return np.array([[I, R4_splitted[np.array([i == I for i in self.pulses[:,1]])]] for I in unique_pulses])

    def processWOFitting(self, A_from, A_to, error_from, error_to):
        inds = [self.find_nearest(self.R4[:,0], i) for i in self.pulses[:,0]]
        R4_splitted = np.split(self.R4, inds)
        R4_baseline = np.mean(R4_splitted[0][:,1])
        R4_splitted = np.array(R4_splitted[1:]) # Cut signal before first pulse
        R4_splitted = np.array([np.stack((p[:,0] - self.pulses[i,0], p[:,1]), axis = -1) for i, p in enumerate(R4_splitted)]) # substract timepoint of pulse from each pulse
        return np.array([[np.mean(pulse[:,1][(A_from < pulse[:,0])*(pulse[:,0] < A_to)]) - R4_baseline, np.max(pulse[:,1])-R4_baseline, np.std(pulse[:,1][(error_from < pulse[:,0])*(pulse[:,0] < error_to)])] for pulse in R4_splitted])

    def groupDataByTemp(self):
        inds = [self.find_nearest(self.R4[:,0], i) for i in self.pulses[:,0]]
        R4_splitted = np.split(self.R4, inds)
        R4_splitted = np.array(R4_splitted[1:]) # Cut signal before first pulse
        T_splitted = np.split(self.T, inds)
        T_splitted = np.array(T_splitted[1:]) # Cut signal before first pulse
        np.average(np.array(T_splitted[i])[:,1])
        np.ptp(np.array(R_splitted[i])[:,1])
        return np.array([[np.average(np.array(T_splitted[i])[:,1]), np.ptp(np.array(R_splitted[i])[:,1])] for i, val in enumerate(T_splitted)])

    def fitTemperature(self, toMins, plot=True):
        def temp(x, a, b):
            return a * x + b
        t = (self.R4[:, 0]-self.pulses[0,0])/60
        timeFilter = t < toMins
        popt, pcov = curve_fit(temp, self.T[:, 1][timeFilter], self.R4[:, 1][timeFilter])
        self.tempFit = popt

        if plot:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=self.T[:, 1][timeFilter], y=self.R4[:, 1][timeFilter]))
            fig1.add_trace(go.Scatter(x=self.T[:, 1][timeFilter], y=temp(self.T[:, 1][timeFilter], *self.tempFit)))
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=t, y=self.R4[:, 1]))
            fig2.add_trace(go.Scatter(x=t, y=temp(self.T[:, 1], *self.tempFit)))

            fig1.update_xaxes(title_text="Temperature (°C)")
            fig1.update_yaxes(title_text="Resistivity (ohms)")
            fig2.update_xaxes(title_text="Time (min)")
            fig2.update_yaxes(title_text="Resistivity (ohms)")
            fig1.update_layout(showlegend=False, width=900, margin=dict(b=0))
            fig2.update_layout(showlegend=False, width=900, margin=dict(t=0))

            fig1.show()
            fig2.show()

    def substTemperature(self):
        def temp(x, a, b):
            return a * x + b
        self.R4 = np.stack((self.R4[:, 0], self.R4[:, 1]-temp(self.T[:, 1], *self.tempFit)+temp(np.average(self.T[:, 1]), *self.tempFit)), axis = -1)
        self.initGuess[1] = 0.005*temp(np.average(self.T[:, 1]), *self.tempFit)
        self.initGuess[2] = temp(np.average(self.T[:, 1]), *self.tempFit)

    def powerDependence(self, fitFrom = 1, errLim = 5, skipFirst = 0, plot=True):
        # errLim in percent
        # fitFrom in seconds

        def decay(x, tau, A, off):
            return A * np.exp(-((x)/tau)**0.6) + off

        taus = []
        for g in self.groupData()[skipFirst:]:
            group = g[1]
            Int = g[0]
            for i, p in enumerate(group):
                xp = p[:,0]
                yp = p[:,1]
                popt, pcov = curve_fit(decay, xp[xp>fitFrom], yp[xp>fitFrom], p0=self.initGuess)
                perr = np.sqrt(np.diag(pcov))
                if 100*perr[0]/popt[0]<errLim:
                    taus = np.append(taus, popt[0])

        clusters, centroids = kmeans1d.cluster(taus, 1)

        def decayFixed(x, A, off):
            tau = centroids
            return A * np.exp(-((x)/tau)**0.6) + off

        vysledek = np.empty((1,2))
        for g in self.groupData():
            group = g[1]
            Int = g[0]
            for i, p in enumerate(group):
                xp = p[:,0]
                yp = p[:,1]
                popt, pcov = curve_fit(decayFixed, xp[xp>fitFrom], yp[xp>fitFrom], p0=self.initGuess[1:])
                perr = np.sqrt(np.diag(pcov))
                vysledek = np.append(vysledek, [[Int, popt[0]]], axis=0)
        if plot:
            self.pd_data = vysledek
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vysledek[1:,0], y=vysledek[1:,1], mode="markers"))
            fig.update_xaxes(title_text="Intensity (mW)")
            fig.update_yaxes(title_text="Amplitude (ohms)")
            fig.update_layout(width=900, title = 'Intensity dependence:') #annotations = [dict(xref='paper', yref='paper', x=0.5, y=-0.20, showarrow=False, text ='This is my caption for the Plotly figure')])
            fig.show()
        else:
            return vysledek[1:]

def plotSpot(w, orientation = 'vertical'):
    X = np.linspace(-15,15,101)
    Y = np.linspace(-15,15,101)
    def Gauss(x,y,w):
        return np.exp(-(2*(x**2+y**2)/w**2))
    Data = ([[Gauss(x,y,w) for x in X] for y in Y])
    fig = go.Figure()
    if orientation == 'horizontal':
        vert = 5
    if orientation == 'vertical':
        vert = 0
    fig.add_trace(go.Contour(
            z=Data,
            x=X, # horizontal axis
            y=Y, # vertical axis
            contours_coloring='heatmap',
            line_width=0,
            colorscale=[[0.0, "rgba(255,255,255,0)"],
                    [1.0, "rgba(255,0,0,1)"]],
            showscale=False
        ))
    fig.add_shape(
                type="rect",
                x0=-15,
                y0=-15,
                x1=-5-vert,
                y1=15,
                fillcolor="rgba(255,255,255,1)",
                #opacity=0.5,
                layer="below",
                line_width=0,
            )
    fig.add_shape(
                type="rect",
                x0=5+vert,
                y0=-15,
                x1=15,
                y1=15,
                fillcolor="rgba(255,255,255,1)",
                #opacity=0.5,
                layer="below",
                line_width=0,
            )
    fig.add_shape(
                type="rect",
                x0=-15,
                y0=15,
                x1=15,
                y1=10-vert,
                fillcolor="rgba(255,255,255,1)",
                #opacity=0.5,
                layer="below",
                line_width=0,
            )
    fig.add_shape(
                type="rect",
                x0=-15,
                y0=-15,
                x1=15,
                y1=-10+vert,
                fillcolor="rgba(255,255,255,1)",
                #opacity=0.5,
                layer="below",
                line_width=0,
            )
    fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=-0.5*w,
                y0=-0.5*w,
                x1=0.5*w,
                y1=0.5*w,
                line_color="rgba(255,255,255,0.25)",
                line_width=5,
            )
    fig.update_layout(
        title={
            'text': str(w)+" um",
            'y':0.48,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'middle',
            'font': {
                'size':15,
                'color': '#ffffff'
                },
            },
        width=500,
        height=500,
        #plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(title_text="um")
    fig.update_yaxes(title_text="um")
    fig.show()

def centerOutputs():
    from IPython.core.display import HTML
    display(HTML("""
<style>
.output {
    display: flex;
    align-items: center;
    text-align: center;
}
</style>
"""))

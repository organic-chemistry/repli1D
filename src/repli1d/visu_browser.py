
import plotly
import plotly.offline as off
import plotly.graph_objs as go

try:
    plotly.offline.init_notebook_mode(connected=True)
except:
    pass
from plotly import tools
from pylab import *

def plot_without_border(img,w=10,h=10,dpi=100,tmpname="mat.png"):
    from PIL import Image


    fig = figure(frameon=False)
    fig.set_size_inches(w,h)


    ax = Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)


    ax.imshow(img, aspect='auto')
    fig.savefig(tmpname, dpi=dpi)
    Im = Image.open(tmpname)
    return Im

def plotly_blocks(x,group_data,name="report.html", save=False,default="markers+lines",Im=None):


    PData = []
    Nblock = len(group_data)
    if Im != None:
        Nblock += 1

    for ig,gdata in enumerate(group_data):
        for data in gdata:
            if len(data)==2:
                y,named=data
                x0=x
            elif len(data)==3:
                x0,y,named=data

            else:
                named="unkown"
                y=data[0]
                #print(data)
                x0=x

            PData.append(go.Scatter(
                x=x0,
                y=y,
                xaxis='x',
                yaxis='y%i'%(Nblock-ig),
                name=named,
                mode=default,

                )
            )

    delta = 1 / Nblock
    marge = 0.05
    def axis_name(i):
        if i==1:
            return "yaxis"
        return "yaxis%i"%i
    layout ={ "xaxis":dict(
            domain=[0, 1]) }
    layout.update({axis_name(i):{"domain":[1-(Nblock-i+1)*delta,1-(Nblock-i)*delta]} for i in range(1,Nblock+1)})
    #print(layout)
    if Im is not None:
        simage = 4
        layout.update({"images":[dict(
                      source= Im,
                      xref= "x",
                      yref= "y",
                      x= x[0],
                      y= simage-1,
                      sizex= x[-1]-x[0],
                      sizey= simage,
                      sizing= "stretch",
                      opacity= 1,
                      layer= "below")]})
    layout = go.Layout(**layout)


    fig = go.Figure(data=PData, layout=layout)

    kwargs = {}
    if save:
        kwargs = {"image_height": 1000, "image_width": 1500, "image": "svg"}
    print(name)
    off.plot(fig, filename=name, auto_open=False, **kwargs)

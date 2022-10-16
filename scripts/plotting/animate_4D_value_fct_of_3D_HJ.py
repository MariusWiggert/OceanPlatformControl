import numpy as np
import plotly.graph_objects as go

vol4= np.load("volume_frames.npy")

a, b, c = vol4.shape[1:]
X, Y, Z = np.mgrid[0:a:a*1j, 0:b:b*1j, 0:c:c*1j]


fig = go.Figure(data=go.Isosurface(x=X.flatten(),
                                 y=Y.flatten(),
                                 z=Z.flatten(),
                                 value=vol4[0].flatten(),
                                 colorscale='jet',
                                 isomin=0,
                                 surface_count=1,
                                 isomax=0
    ))



frames=[go.Frame(data=go.Volume(
                               value=vol4[k].flatten()), 
                 name=str(k)) for k in range(len(vol4))]
updatemenus = [dict(
        buttons = [
            dict(
                args = [None, {"frame": {"duration": 201, "redraw": True},
                                "fromcurrent": True, "transition": {"duration": 0}}],
                label = "Play",
                method = "animate"
                ),
            dict(
                 args = [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                label = "Pause",
                method = "animate"
                )
        ],
        direction = "left",
        pad = {"r": 10, "t": 87},
        showactive = False,
        type = "buttons",
        x = 0.21,
        xanchor = "right",
        y = -0.075,
        yanchor = "top"
    )] 

sliders = [dict(steps = [dict(method= 'animate',
                              args= [[f'{k}'],                           
                              dict(mode= 'immediate',
                                   frame= dict(duration=201, redraw=True),
                                   transition=dict(duration= 0))
                                 ],
                              label=f'{k+1}'
                             ) for k in range(len(vol4))], 
                active=0,
                transition= dict(duration= 0 ),
                x=0, # slider starting position  
                y=0, 
                currentvalue=dict(font=dict(size=12), 
                                  prefix='frame: ', 
                                  visible=True, 
                                  xanchor= 'center'
                                 ),  
                len=1.0) #slider length
           ]

fig.update_layout(width=700, height=700, updatemenus=updatemenus, sliders=sliders)
fig.update(frames=frames)
fig.update_traces(showscale=False)
ax = fig.show()
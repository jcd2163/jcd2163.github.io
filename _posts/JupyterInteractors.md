
Basic Interactor Demo
---------------------

This demo shows off an interactive visualization using [Bokeh](https://bokeh.pydata.org) for plotting, and Ipython interactors for widgets. The demo runs entirely inside the Ipython notebook, with no Bokeh server required.

The dropdown offers a choice of trig functions to plot, and the sliders control the frequency, amplitude, and phase. 

To run, click on, `Cell->Run All` in the top menu, then scroll to the bottom and move the sliders. 


```python
from ipywidgets import interact
import numpy as np

from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
output_notebook()
```



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="965eca3b-1fe6-467a-b065-0d24972df842">Loading BokehJS ...</span>
    </div>





```python
x = np.linspace(0, 2*np.pi, 2000)
y = np.sin(x)
```


```python
p = figure(title="simple line example", plot_height=300, plot_width=600, y_range=(-5,5))
r = p.line(x, y, color="#2222aa", line_width=3)
```


```python
def update(f, w=1, A=1, phi=0):
    if   f == "sin": func = np.sin
    elif f == "cos": func = np.cos
    elif f == "tan": func = np.tan
    r.data_source.data['y'] = A * func(w * x + phi)
    push_notebook()
```


```python
show(p, notebook_handle=True)
```



<div class="bk-root">
    <div class="bk-plotdiv" id="834d5569-56c1-4ed2-9a5d-c00734a7c1ab"></div>
</div>







<p><code>&lt;Bokeh Notebook handle for <strong>In[5]</strong>&gt;</code></p>




```python
interact(update, f=["sin", "cos", "tan"], w=(0,100), A=(1,5), phi=(0, 20, 0.1))
```


<p>Failed to display Jupyter Widget of type <code>interactive</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>






    <function __main__.update>



angled_friction_model
---------------------

Vibrothermography is vibration induced crack heating. This Python package
attempts to calculate vibrothermography heating power from frictional 
heating of the rubbing crack surfaces. The crack is presumed to be under
closure stress and excited by an external cyclic load. It is (for the 
moment) limited to semicircular cracks under combined normal (mode I) 
and shear (mode II) loading. 


Requirements
------------
(older versions may work but have not been tested)
  * Python 2.7.5 or 3.4.9 or newer. 
  * scipy 1.0.0 or newer
  * numpy 1.14.3 or newer
  * matplotlib 1.5.3 or newer
  * IPython/Jupyter (recommended)
  * git 2.17.1 or newer 
    (you may delete the .git directory if you prefer not to use version
    control.)
  * crackclosuresim2 v0.4.1 or newer
  * crackclosuresim2 requires Cython and a C compiler. See the README.txt
    with crackclosuresim2 for more information. 

On Linux these components, with the exception of crackclosuresim2,  are 
usually available as packages from your operating system vendor. 

On Windows/Macintosh it is usually easiest to use a Python distribution 
such as Anaconda https://www.anaconda.com or Canopy 
https://www.enthought.com/product/canopy/ 

These distributions typically provide the full 
Python/Numpy/Matplotlib/IPython stack by default, so you only need
a few more pieces such as Cython, git, and the C compiler. 
64-bit versions of the distributions are recommended

Installing angled_friction_model
--------------------------------
From a terminal, command prompt, Anaconda or Canopy terminal, etc. 
change to the angled_friction_model source directory and type:
  python setup.py build
  python setup.py install

Depending on your configuration the 'install' step might require
root or administrator permissions. You can also explicitly specify 
a different Python version or binary. 

Running angled_friction_model
-----------------------------

Try the examples in the 'demos/' subdirectory. 
   e.g. python simple_afm_demo.py

We recommend using an IPython/Jupyter 
Qt console or similar. Usually you will want to 
start your session by initializing matplotlib mode: 
  %matplotlib qt

Then run one of the demos:
  %run simple_afm_demo.py

When writing your own Python code, you can import the angled_friction_model package
with: 
  import angled_friction_model

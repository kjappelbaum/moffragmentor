Getting started with moffragmentor
====================================

Installation
---------------

We recommend installing moffragmentor in a clean virtual environment environment (e.g., a `conda environment <https://docs.conda.io/projects/conda/en/latest/index.html>`_)


You can install the latest stable release from PyPi using

.. code-block:: bash

    pip install moffragmentor


or the latest development version using

.. code-block:: bash

    pip install git+https://github.com/kjappelbaum/moffragmentor.git


Extensions
...........

In case you want to use the :py:meth:`~moffragmentor.SBU.Node.show_molecule`function in Jupyter lab you have to

.. code-block:: bash
    jupyter-labextension install @jupyter-widgets/jupyterlab-manager
    jupyter-labextension install nglview-js-widgets
    jupyter-nbextension enable nglview --py --sys-prefix

You also might find the `debugging help in the nglview documentation <https://github.com/nglviewer/nglview/blob/master/docs/FAQ.md#widget-not-shown>`_ useful.

Fragmenting a MOF
-------------------


Calculating descriptors
--------------------------


Modifying a building block
-------------------------------

Getting started with moffragmentor
====================================

Installation
---------------

We recommend installing moffragmentor in a clean virtual environment
(e.g., a `conda environment <https://docs.conda.io/projects/conda/en/latest/index.html>`_)


You can install the latest stable release from PyPi using

.. code-block:: bash

    pip install moffragmentor


or the latest development version using

.. code-block:: bash

    pip install git+https://github.com/kjappelbaum/moffragmentor.git

.. note::

    If you install via pip you will need to manually install openbabel (e.g. `conda install openbabel -c conda-forge`).

If you want to determine RCSR codes,
you will also need to install a `Java runtime environment (JRE) of version 1.5.0 or later <https://www.java.com/en/>`_
on your machine as we use the `Systre <http://gavrog.org>`_  (Symmetry, Structure (Recognition) and Refinement)
code to perform analysis of the labeled quotient graphs we construct in moffragmentor.

Extensions
...........

In case you want to use the :py:meth:`~moffragmentor.SBU.Node.show_molecule`,
or :py:meth:`~moffragmentor.mof.MOF.show_structure` function in Jupyter lab you have to

.. code-block:: bash

    jupyter-labextension install @jupyter-widgets/jupyterlab-manager
    jupyter-labextension install nglview-js-widgets
    jupyter-nbextension enable nglview --py --sys-prefix

You also might find the
`debugging help in the nglview documentation <https://github.com/nglviewer/nglview/blob/master/docs/FAQ.md#widget-not-shown>`_ useful.

Fragmenting a MOF
-------------------

To fragment a MOF you need to create an instance of :py:class:`~moffragmentor.mof.MOF`
and then call :py:meth:`~moffragmentor.mof.MOF.fragment`.

.. code-block:: python

    from moffragmentor import MOF

    mof = MOF.from_cif(<my_cif.cif>)
    fragmentation_result = mof.fragment()

The result is a :code:`FragmentationResult` :code:`namedtuple` with the fields :code:`nodes`, :code:`linkers`,
both subclasses of a :py:class:`moffragmentor.sbu.SBUCollection` and  :code:`bound_solvent`, :code:`unbound_solvent`, both :py:class:`moffragmentor.molecule.NonSbuMoleculeCollection`, and a :py:class:`moffragmentor.net.Net`.

.. warning::

    If you use the :code:`MOF.from_cif` method, we will run :py:obj:`pymatgen.analysis.spacegroup.SpacegroupAnalyzer` on the input structure.
    This might take some time, and we also have encountered cases where it can be really slow.
    If you do not want this, you can either "manually" call the constructor or tune the tolerance parameters.

.. warning::

    Note that moffragmentor currently does not automatically delete bound solvent. This is due to two observations:

    1. We have very little understanding of what solvent we can remove without affecting the structural integrity.
    2. We (currently) do not have a way to estimate if a solvent is charged.
       We explore different implementation strategies, but we do not have a robust one at this moment.


You might want a quick overview of the composition of the different components.
You can access this via the :code:`composition` properties

.. code-block:: python

    solvent_collection.composition

which will return a dictionary of the counts of the compositions, for example :code:`{'C3 H7 N1 O1': 3, 'H2 O1': 4}`.

Clearly, we do not consider floating solvent for the computation of the net.


.. admonition:: Known issues
    :class: warning

    For some structures in the CSD MOF subset, there will be problems with the fragmentation.
    One example is :code:`CAYSIE`, which is a metalloporphyrinate.
    Here, the code struggles to distinguish nodes and linkers
    as a core routine of the moffragmentor is to check if a metal atom is inside another, potential linker, molecule.

    .. figure:: _static/RSM2943.png
        :alt: RSM2943
        :width: 400px
        :align: center

        Example of a metalloporphyrinate for which the fragmentor fails.

    Also note that there are problems with analyzing the combinatorial topology of 1D rod MOFs.
    `There only recently has been an algorithm proposed that is implemented in ToposPro <https://link.springer.com/article/10.1007/s11224-016-0774-1>`_.


API documentation
===================

Core API
----------

Most users will only need to deal with the :code:`MOF` class.

.. automodule:: moffragmentor.mof
    :members:

Command line interface
-----------------------

.. automodule:: moffragmentor.cli
    :members:

SBU subpackage
----------------

Defines datastructures for the building blocks as well as collections of building blocks

.. automodule:: moffragmentor.sbu.sbu
    :members:

.. automodule:: moffragmentor.sbu.sbucollection
    :members:

.. automodule:: moffragmentor.sbu.node
    :members:

.. automodule:: moffragmentor.sbu.nodecollection
    :members:

.. automodule:: moffragmentor.sbu.linker
    :members:

.. automodule:: moffragmentor.sbu.linkercollection
    :members:

molecule subpackage
-----------------------

Defines datastructures for the non-building-block molecules (e.g. solvent) as well as collections of such molecules

.. automodule:: moffragmentor.molecule.nonsbumolecule
    :members:

.. automodule:: moffragmentor.molecule.nonsbumoleculecollection
    :members:


Fragmentor subpackage
-------------------------

This subpackage is not optimized for end-users.
It is intended for developers who wish to customize the behavior of the fragmentor.

.. automodule:: moffragmentor.fragmentor.branching_points
    :members:

.. automodule:: moffragmentor.fragmentor.nodelocator
    :members:

.. automodule:: moffragmentor.fragmentor.linkerlocator
    :members:

.. automodule:: moffragmentor.fragmentor.solventlocator
    :members:

.. automodule:: moffragmentor.fragmentor.splitter
    :members:

.. automodule:: moffragmentor.fragmentor.filter
    :members:

.. automodule:: moffragmentor.fragmentor.molfromgraph
    :members:

Utils subpackage
-----------------

Also the :code:`utils` subpackage is not optimized for end-users.

.. automodule:: moffragmentor.utils
    :members:

.. automodule:: moffragmentor.utils.errors
    :members:

.. automodule:: moffragmentor.utils.mol_compare
    :members:

.. automodule:: moffragmentor.utils.periodic_graph
    :members:

.. automodule:: moffragmentor.utils.systre
    :members:

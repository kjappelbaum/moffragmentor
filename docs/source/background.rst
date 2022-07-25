Background information
========================


Fragmentation
---------------
For the fragmentation of a MOF structure we rely on a structure graph.
In moffragmentor we use heuristics in pymatgen to construct the structure graph
(which is a networkx multigraph with annotation about the periodic neighbors).
If the structure graph does not contain a pair of bonds,
moffragmentor cannot consider it in the fragmentation.

.. warning::
    The current implementation of the fragementation is in parts
    relatively inefficient as we perform multiple traversals of the structure graph.

For the fragmentation, there are a few definitions we have to make:

.. topic:: **Bridge**

    In a graph, a `bridge <https://en.wikipedia.org/wiki/Bridge_(graph_theory)>`_ is an edge, if removed, would increase the number of connected components.

.. topic:: **Bound solvent**

    A bound solvent molecule is bound via a bridge edge to one metal.
    According to this definition, M-OCH\ :sub:`3`, M-OH\ :sub:`2`, etc. are all bound solvents, whereas a bridging formate is not.

.. topic:: **Floating solvent**

    Floating solvent is an isolated connected component
    in the structure graph that does not lie across periodic boundaries in a supercell.

.. topic:: **Branching site**

    * has at minimum coordination number 3
    * at least one path with maximum 2 edges that leads to metal and does not contain a bride
    * has at minimum 2 non-metal connections that are not bridges
       (note that in case the connection to a metal goes via 2 edges,
       then the first node of this path to the metal contributes to this count)

    If there are multiple neighboring sites selecting according this definition, we pick the one closest to the metal (the fewest number of edges).

.. topic:: **Capping site**

    A capping site is part of a cycle with one of more metals that does not contain a branching site.



For the fragmentation branching sites define the places at which
we make the split between node and linker.
The fragmentation algorithm goes through the following steps:

1. Extracting floating solvent.
2. From metal sites perform depth-first search on the structure graph up to a branching site.
3. "Complete graph" by traversing the graph from all non-branching sites traversed in step 1
    up to a leaf node.
4. Extracting nodes as separate connected components.
5. Deleting nodes from the structure graph and extracting linkers as connected components.


SBU dimensionality
--------------------

For many applications, the dimensionality of the SBUs can be of interest [Rosi2005]_.
For example, one can hypothesize that 1D nodes can have favorable charge conductance properties.
Also, such rod SBUs may prevent interpenetration [Rosi2005]_.

To compute the dimensionality of the building blocks we use the algorithm proposed by Larsen et al. [Larsen2019]_.


Net Embedding
----------------

A key concept in reticular chemistry is the one of the net.
Computing the topology of the net embedding is not entirely trivial
as there is no specific rule of clusters of atoms should be condensed to a vertex [Bureekaew2015]_
(for example, `one might place vertices on subfragments of large linkers <https://www.mofplus.org/content/show/generalnetinfo>`_.
In moffragmentor, we use the centers of node and linker clusters as vertices.
Using the `Systre code <http://gavrog.org/Systre-Help.html>`_ [DelagoFriedrichs2003]_,
we can then determine the `RCSR <http://rcsr.anu.edu.au/rcsr_nets>`_ code of this net.


References
-------------

.. [Rosi2005] Rosi, N. L. et al.
    Rod Packings and Metal−Organic Frameworks
    Constructed from Rod-Shaped Secondary Building Units.
    J. Am. Chem. Soc. 127, 1504–1518 (2005).

.. [Larsen2019] Larsen, P. M., Pandey, M., Strange, M. & Jacobsen, K. W.
    Definition of a scoring parameter to identify low-dimensional materials components.
    Phys. Rev. Materials 3, (2019).

.. [Bureekaew2015] Bureekaew, S., Balwani, V., Amirjalayer, S. & Schmid, R.
    Isoreticular isomerism in 4,4-connected paddle-wheel metal–organic frameworks:
    structural prediction by the reverse topological approach.
    CrystEngComm 17, 344–352 (2015).

.. [DelagoFriedrichs2003] Delgado-Friedrichs, O. & O’Keeffe, M.
    Identification of and symmetry computation for crystal nets.
    Acta Cryst Sect A 59, 351–360 (2003).

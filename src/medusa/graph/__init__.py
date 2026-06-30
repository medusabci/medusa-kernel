"""Graph-theoretic metrics for connectivity matrices.

Each function takes a square weighted adjacency matrix ``W``
(channels x channels) and returns a graph metric: node degree and
strength, density, clustering, transitivity, complexity, path length,
efficiency, modularity and the betweenness / eigenvector / participation
centralities. :func:`surrogate_graph` builds a weight-shuffled surrogate
network for statistical testing.
"""

# ``assortativity``, ``density`` and ``participation_coefficient`` import
# their sibling *modules* (``degree``, ``modularity``) at import time, so load
# them first: the ``degree`` and ``modularity`` re-exports below bind functions
# of the same name into this package namespace, shadowing those submodules.
from medusa.graph.assortativity import assortativity
from medusa.graph.betweeenness_centrality import betweenness
from medusa.graph.clustering_coefficient import clustering_coefficient
from medusa.graph.complexity import complexity
from medusa.graph.degree import degree
from medusa.graph.density import density
from medusa.graph.efficiency import efficiency
from medusa.graph.eigen_centrality import eigen_centrality
from medusa.graph.modularity import modularity
from medusa.graph.participation_coefficient import participation_coefficient
from medusa.graph.path_length import path_length
from medusa.graph.surrogate_graph import surrogate_graph
from medusa.graph.transitivity import transitivity

__all__ = [
    "assortativity",
    "betweenness",
    "clustering_coefficient",
    "complexity",
    "degree",
    "density",
    "efficiency",
    "eigen_centrality",
    "modularity",
    "participation_coefficient",
    "path_length",
    "surrogate_graph",
    "transitivity",
]

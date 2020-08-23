library(igraph)
library(readr)
library(zeallot)
library(tidyverse)

data <- read_csv('edgelist.csv')

n_time_steps <- 8
n_layers <- 2
n_nodes <- 242
k_offset <- 0
t_offset <- 8

unique_nodes <- union(data$node_a, data$node_b)

Y = array(0, dim = c(n_layers, n_time_steps, n_nodes, n_nodes))
for (k in 1:n_layers) {
    for (t in 1:n_time_steps) {
        edgelist <- data %>%
            filter(day == (k + k_offset)) %>%
            filter(hour == (t + t_offset)) %>%
            select(node_a, node_b)
        G <- igraph::graph_from_edgelist(as.matrix(edgelist), directed = FALSE)
        G <- igraph::simplify(G)
        G <- as.undirected(G)

        missing_vertices <- setdiff(unique_nodes, V(G))
        G <- G + vertices(setdiff(unique_nodes, V(G)))
        Y[k, t, ,] <- as.matrix(igraph::as_adjacency_matrix(G))
    }
}

# write to disk
for (k in 1:n_layers) {
    for (t in 1:n_time_steps) {
        Ykt <- as.matrix(Y[k, t,,])
        write.table(Ykt, file=paste0('primaryschool_', k, '_', t, '.npy'),
                    col.names = FALSE, row.names = FALSE)
    }
}

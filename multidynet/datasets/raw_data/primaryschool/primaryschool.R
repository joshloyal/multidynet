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
n_blocks <- 2

unique_nodes <- union(data$node_a, data$node_b)

Y = array(0, dim = c(n_layers, n_blocks * n_time_steps, n_nodes, n_nodes))
for (k in 1:n_layers) {
    ts <- 1
    for (t in 1:n_time_steps) {
        for (b in 1:n_blocks) {
            edgelist <- data %>%
                filter(day == (k + k_offset)) %>%
                filter(hour == (t + t_offset)) %>%
                filter(time_block == b) %>%
                select(node_a, node_b)
            G <- igraph::graph_from_edgelist(as.matrix(edgelist), directed = FALSE)
            G <- igraph::simplify(G)
            G <- as.undirected(G)

            missing_vertices <- setdiff(unique_nodes, V(G))
            G <- G + vertices(setdiff(unique_nodes, V(G)))
            Y[k, ts, ,] <- as.matrix(igraph::as_adjacency_matrix(G))
            ts <- ts + 1
        }
    }
}

# write to disk
for (k in 1:n_layers) {
    ts <- 1
    for (t in 1:n_time_steps) {
        for (b in 1:n_blocks) {
            Ykt <- as.matrix(Y[k, ts,,])
            write.table(Ykt, file=paste0('primaryschool_', k, '_', ts, '.npy'),
                        col.names = FALSE, row.names = FALSE)
            ts <- ts + 1
        }
    }
}

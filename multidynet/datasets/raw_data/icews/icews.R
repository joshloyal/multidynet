library(igraph)
library(readr)
library(zeallot)
library(tidyverse)

data <- read_csv('~/myworkspace/gdelt/events_2011.csv')


filter_relation <- function(data, type = "quadclass", n_countries = 50) {

    # filter most active countries over the year
    edgelist <- data %>%
        group_by(source, target) %>%
        count() %>%
        ungroup() %>%
        rename(weight = n)

    g <- igraph::graph_from_edgelist(as.matrix(edgelist %>%
                                                   select(source, target)),
                                     directed = TRUE)
    g <- igraph::simplify(g)
    g <- igraph::as.undirected(g)

    countries <- names(
        sort(igraph::strength(g, loops = FALSE, mode = 'all'),
             decreasing = TRUE)[1:n_countries])

    n_time_steps <- 12
    if (type == "quadclass") {
        Y = array(0, dim = c(4, n_time_steps, n_countries, n_countries))
        for (k in 1:4) {
            for (t in 1:n_time_steps) {
                edgelist <- data %>%
                    filter(quadclass == k) %>%
                    filter(month == t) %>%
                    filter(source %in% countries) %>%
                    filter(target %in% countries) %>%
                    select(source, target)

                G <- igraph::graph_from_edgelist(as.matrix(edgelist))
                G <- igraph::simplify(G)
                G <- G + vertex(countries[!countries %in% V(G)$name])
                Y[k, t,,] <- as.matrix(
                    igraph::as_adjacency_matrix(G))[countries, countries]
                G <- as.undirected(G)
            }
        }
    } else {
        Y = array(0, dim = c(20, n_countries, n_countries))
        for (k in 1:20) {
            edgelist <- data %>%
                filter(cameo == k) %>%
                filter(source %in% countries) %>%
                filter(target %in% countries)
            if (!is.null(month)) {
                edgelist <- edgelist %>%
                    filter(month == month)
            }
            edgelist <- edgelist %>%
                select(source, target)

            G <- igraph::graph_from_edgelist(as.matrix(edgelist))
            G <- igraph::simplify(G)
            G <- G + vertex(countries[!countries %in% V(G)$name])
            G <- as.undirected(G)

            Y[k,,] <- as.matrix(
                igraph::as_adjacency_matrix(G))[countries, countries]

        }
    }

    list(Y=Y, countries=countries)
}


# 4 quad code realtions
c(Y_cameo, countries) %<-% filter_relation(data, type = 'quadclass', n_countries = 50)

# write to disk
for (k in 1:4) {
    for (t in 1:12) {
        Y <- as.matrix(Y_cameo[k, t,,])
        write.table(Y, file=paste0('icews_2011_', k, '_', t, '.npy'),
                    col.names = FALSE, row.names = FALSE)
    }
}

write.table(countries, file='icews_countries.txt',
            col.names = FALSE, row.names = FALSE,
            quote = FALSE)

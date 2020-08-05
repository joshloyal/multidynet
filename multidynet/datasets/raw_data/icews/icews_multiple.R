library(igraph)
library(readr)
library(zeallot)
library(tidyverse)

#data <- read_csv('processed_data/events_2011.csv')


load_files <- function() {
    data <- NULL
    for (file_name in list.files('processed_data')) {
        data <- bind_rows(data, read_csv(paste0('processed_data/',file_name)))
    }

    data
}


filter_relation <- function(type = "quadclass", n_countries = 50) {
    # load data
    data <- load_files()

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

    n_months <- 12
    years <- c(2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016)
    n_time_steps <- n_months * length(years)

    Y = array(0, dim = c(4, n_time_steps, n_countries, n_countries))
    for (k in 1:4) {
        t <- 1
        for (y in years) {
            for (m in 1:n_months) {
                edgelist <- data %>%
                    filter(quadclass == k) %>%
                        filter(year == y) %>%
                        filter(month == m) %>%
                        filter(source %in% countries) %>%
                        filter(target %in% countries) %>%
                        select(source, target)

                G <- igraph::graph_from_edgelist(as.matrix(edgelist))
                G <- igraph::simplify(G)
                G <- G + vertex(countries[!countries %in% V(G)$name])
                Y[k, t, ,] <- as.matrix(
                    igraph::as_adjacency_matrix(G))[countries, countries]
                G <- as.undirected(G)

                t <- t + 1
            }
        }
    }

    list(Y=Y, countries=countries)
}


# 4 quad code realtions
c(Y_cameo, countries) %<-% filter_relation(type = 'quadclass', n_countries = 65)

n_time_steps <- dim(Y_cameo)[2]

# write to disk
for (k in 1:4) {
    for (t in 1:n_time_steps) {
        Y <- as.matrix(Y_cameo[k, t,,])
        write.table(Y, file=paste0('icews_', k, '_', t, '.npy'),
                    col.names = FALSE, row.names = FALSE)
    }
}

write.table(countries, file='icews_countries.txt',
            col.names = FALSE, row.names = FALSE,
            quote = FALSE)

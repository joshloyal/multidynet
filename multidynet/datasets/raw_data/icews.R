library(igraph)
library(readr)

data <- read_csv('~/myworkspace/gdelt/events_2012.csv')


filter_relation <- function(data, type = "quadclass", n_countries = 50) {
    
    # filter most active countries over the year
    edgelist <- data %>% 
        group_by(source, target) %>% 
        count() %>% 
        ungroup() %>% 
        rename(weight = n) 
    
    countries <- names(
        sort(igraph::strength(g, loops = FALSE, mode = 'all'), 
             decreasing = TRUE)[1:n_countries])
    
    n_time_steps <- 12
    if (type == "quadclass") {
        Y = array(0, dim = c(4, n_time_steps, n_countries, n_countries))
        for (k in 1:4) {
            for (t in 1:12) {
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


# 20 cameo code realtions
c(Y_cameo, countries) %<-% filter_relation(data, type = 'quadclass', n_countries = 50)


G_cameo <- igraph::graph_from_adjacency_matrix(Y_cameo[4,2,,], mode='undirected')
V(G_cameo)$name <- countries
plot(G_cameo, vertex.size = 5, vertex.color = 'steelblue',
     #edge.arrow.size = 0.4, edge.arrow.width = 0.5,
     edge.width = 0.5,
     vertex.frame.color='white',
     vertex.label.color = 'black', vertex.label.cex = 0.7)

# Statistical-Analysis-High-D-DSTI
## Question 1 :
### 1.1 Describe the general setup of resampling techniques and explain how it can be used for parameter tuning.

In high-dimensional statistical analysis, resampling techniques like cross-validation and bootstrapping are employed to assess model performance and improve parameter tuning.

Cross-Validation: The dataset is divided into multiple folds. The model is trained on a subset of these folds and tested on the remaining fold. This process is repeated for each fold, and performance metrics are averaged to estimate model effectiveness.
Bootstrapping: Multiple samples are drawn with replacement from the original dataset. Each sample is used to train and test the model. This technique helps estimate the variability of model performance and adjust parameters accordingly.
Both methods help in selecting optimal parameters by providing a robust estimate of model performance and its variability in high-dimensional settings.

Moreover, other significant resampling techniques include :
Leave-One-Out Cross-Validation (LOOCV): A special case of cross-validation where each training set is used with a single observation left out for testing. It’s computationally intensive but provides a nearly unbiased estimate of model performance.
Leave-P-Out Cross-Validation: Generalizes LOOCV by leaving out p observations for testing. Useful for larger datasets where LOOCV may be impractical.
K-Fold Cross-Validation: The dataset is divided into K subsets or folds. The model is trained on K−1 folds and tested on the remaining fold, repeated K times. This balances between computational efficiency and performance estimation.
Repeated Cross-Validation: Repeats K-Fold Cross-Validation multiple times with different random splits to increase reliability.
Subsampling: Similar to bootstrapping, but without replacement. It involves taking random subsets of the data for training and testing to estimate model performance and variability.

These techniques help in parameter tuning by providing insights into model stability and generalization across different data partitions or samples.

### 1.2 Describe some techniques allowing to select the number of clusters with Gaussian mixture
models, when using the EM algorithm for inference.

When using Gaussian Mixture Models (GMMs) with the Expectation-Maximization (EM) algorithm, several techniques can help determine the optimal number of clusters:

Bayesian Information Criterion (BIC): Evaluates models based on the likelihood of the data and penalizes for model complexity. Lower BIC values indicate a better balance between model fit and complexity.
Akaike Information Criterion (AIC): Similar to BIC but with a different penalty term for model complexity. It also helps in selecting the number of clusters by comparing different models.
Cross-Validation: Splits the data into training and validation sets, trains the GMM on the training set, and evaluates its performance on the validation set. Metrics like log-likelihood or clustering accuracy are used to determine the optimal number of clusters.
Likelihood Ratio Test: Compares the fit of two models with different numbers of clusters. It tests whether adding more clusters significantly improves the model.
Silhouette Score: Measures how similar an object is to its own cluster compared to other clusters. Higher silhouette scores indicate better-defined clusters.
Cluster Stability: Evaluates the consistency of the clustering results across different samples or perturbations of the data. Stable clusters are considered more reliable.

These techniques help in choosing the number of clusters by balancing model fit, complexity, and stability.


## Question 2 :
### 1.1 :

The quality of clustering can be related to variance in several ways:
Within-Cluster Variance: In a good clustering solution, the variance within each cluster should be low, indicating that the data points within each cluster are close to each other. This means that each cluster is compact and well-defined.
Between-Cluster Variance: The variance between clusters should be high, suggesting that clusters are well-separated from each other. High between-cluster variance means that the clusters are distinct and not overlapping.
In summary, a high-quality clustering solution has low within-cluster variance and high between-cluster variance, indicating tight, well-separated clusters.

### 1.2 :
# Load necessary package
library(ggplot2)

- Prepare the Data:
# Define the data
var1 <- c(0, 0, 1, 3, 3.5, 1, 3, 4)
var2 <- c(1, 2, 1, 1, 1, 5, 4, 5)
data <- data.frame(var1, var2)

- Compute Distance Matrix:
# Create a distance matrix
dist_matrix <- dist(data)

- Hierarchical Clustering:
# Perform hierarchical clustering using single linkage (minimum distance)
hc_single <- hclust(dist_matrix, method = "single")

- Plot the Dendrogram:
# Plot the dendrogram
plot(hc_single, main = "Dendrogram of Single Linkage Hierarchical Clustering")

- Cut the Dendrogram:
# Cut the tree into 3 clusters
clusters <- cutree(hc_single, k = 4)

- Add Cluster Information:
# Add cluster information to the data frame
data$cluster <- as.factor(clusters)

- Plot the Clusters:
# Plot the clustered data
ggplot(data, aes(x = var1, y = var2, color = cluster)) +
  geom_point(size = 3) +
  labs(title = "Hierarchical Clustering with Single Linkage",
       x = "Variable 1", y = "Variable 2")
![image](https://github.com/user-attachments/assets/7d8d9e95-d950-4bb3-b2b4-94ad9bf1a1f7)


## Question 3 :
### 3.1
Loading the data
```
# Load the Vélib data
load("velib.Rdata")
```

### 3.2 Data Preparation and Descriptive Analysis
Data Preparation and Descriptive Analysis
Set Up Data
```
# Extract data and setup row and column names
bike_data <- velib$data
colnames(bike_data) <- velib$dates
rownames(bike_data) <- paste(1:NROW(velib$names), velib$names, sep = "_")
```

Check for Missing Values and Data Distribution
```
# Check for missing values

sum(is.na(bike_data))

# Plot distribution of bike availability
boxplot(bike_data, main = "Distribution of Bike Availability")
```

Remove Incomplete Data and Compute Row Means
```
# Remove incomplete data from the first Sunday
bike_data <- bike_data[, -seq_len(13)]

# Add a column for the average number of bikes available
bike_data$mean_availability <- rowMeans(bike_data, na.rm = TRUE)
```

Separate Weekdays and Weekends
```
# Split data into weekdays and weekends
weekdays_data <- bike_data[, 1:120]
weekends_data <- bike_data[, 121:168]
station_positions <- velib$position
station_positions$has_bonus <- velib$bonus
```

Time Series Analysis
```
# Calculate the average bike availability by hour
mean_hourly_availability <- data.frame(mean_availability = colMeans(bike_data))

# Convert to time series object
time_series_data <- ts(mean_hourly_availability, start = c(1, 1), end = c(7, 24), frequency = 24)
plot(time_series_data, ylab = 'Mean Bike Availability (%)', xlab = 'Hour')

# Plot seasonal trends
library(ggplot2)
library(forecast)

ggseasonplot(time_series_data, polar = FALSE) +
  ggtitle("Seasonal Trends of Bike Availability") + 
  xlab("Hour") + 
  scale_color_discrete(name = "Day", labels = c("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")) +
  theme_bw()
```
![image](https://github.com/user-attachments/assets/11459699-4c11-40c2-8156-0a0ff94cf402)


### 3.3
Principal Component Analysis (PCA)
```
# Apply PCA to the bike data
pca_result <- prcomp(bike_data[, -ncol(bike_data)], scale. = TRUE)

# Scree plot to decide the number of components
library(factoextra)
fviz_eig(pca_result)
```
![image](https://github.com/user-attachments/assets/e55fb178-62a1-42ff-9879-fbd06f3cdb2b)


Determine Number of Components
```
# Cattell’s scree test
eigenvalue_diff <- abs(diff(pca_result$sdev))
plot(eigenvalue_diff, type = 'b', main = "Cattell's Scree Test")
abline(h = 0.1 * max(eigenvalue_diff), lty = 2, col = 'blue')
```
![image](https://github.com/user-attachments/assets/35f5da9c-5dc5-4e84-bcc4-13516a9bc018)


Visualize PCA Results
# Plot individuals in PCA space
```
fviz_pca_ind(pca_result,
             col.ind = "cos2",
             gradient.cols = c("#0000FF", "#FFA500", "#FF0000"),
             repel = TRUE)
```

# Plot variable contributions to PCs
```
fviz_pca_var(pca_result,
             col.var = "coord",
             gradient.cols = c("#0000FF", "#FFA500", "#FF0000"),
             repel = TRUE)
```

### 3.4
Clustering Analysis
#### 3.4.1
Hierarchical Clustering
```
# Compute distance matrix and apply hierarchical clustering
dist_matrix <- dist(bike_data[, -ncol(bike_data)])
hclust_result <- hclust(dist_matrix, method = 'complete')
```

# Plot dendrogram
```
plot(hclust_result, main = "Hierarchical Clustering Dendrogram")
```

# Cut tree to get clusters
```
hierarchical_clusters <- cutree(hclust_result, k = 4)
```
# Plot clusters on the map
```
leaflet(station_positions) %>% 
  addProviderTiles(providers$CartoDB.Positron) %>%
  addCircleMarkers(radius = bike_data$mean_availability * 6,
                   color = palette(hierarchical_clusters),
                   stroke = ~ifelse(station_positions$has_bonus == "1", TRUE, FALSE), 
                   label = ~paste(rownames(bike_data), " - Cluster:", hierarchical_clusters),
                   fillOpacity = 0.9)
```

#### 3.4.2
K-Means Clustering
# Determine optimal number of clusters
```
wss <- numeric(15)
for (k in 1:15) {
  kmeans_result <- kmeans(bike_data[, -ncol(bike_data)], centers = k, nstart = 10)
  wss[k] <- kmeans_result$betweenss / kmeans_result$totss
}

# Plot the WSS to decide the number of clusters
plot(wss, type = 'b', main = "Optimal Number of Clusters (K-Means)")
```

Apply K-Means Clustering
# Choose 4 clusters based on the plot
```
k <- 4
kmeans_result <- kmeans(bike_data[, -ncol(bike_data)], centers = k, nstart = 10)

# Define color palette for clusters
library(RColorBrewer)
library(leaflet)

colors <- brewer.pal(4, "Dark2")
palette <- colorFactor(colors, domain = NULL)
```

# Plot clusters on the map
```
leaflet(station_positions) %>% 
  addProviderTiles(providers$CartoDB.Positron) %>%
  addCircleMarkers(radius = bike_data$mean_availability * 5,
                   color = palette(kmeans_result$cluster),
                   stroke = ~ifelse(station_positions$has_bonus == "1", TRUE, FALSE), 
                   label = ~paste(rownames(bike_data), " - Cluster:", kmeans_result$cluster),
                   fillOpacity = 0.9)
```
![image](https://github.com/user-attachments/assets/83447f74-8c6e-4d92-bac9-f4d37654be6e)


### 4
Summary
# Compare mean availability across clusters
```
boxplot(mean_availability ~ hierarchical_clusters, data = bike_data,
        ylab = 'Mean Availability', xlab = 'Cluster', main = 'Bike Availability by Cluster')

# Visualize cluster dynamics by hour
library(reshape2)
library(plotly)
```
![image](https://github.com/user-attachments/assets/3ce034d6-b162-4b2c-85e3-d1f3d4cc1a0a)


# Prepare data for plotting
```
hourly_data <- data.frame(rowMeans(bike_data[, -ncol(bike_data)]))
hourly_data$station <- rownames(bike_data)
hourly_data$cluster <- hierarchical_clusters

melted_data <- melt(hourly_data, id.vars = c('station', 'cluster'))

plot_ly(data = melted_data, x = ~longitude, y = ~latitude, size = ~value, color = ~cluster, 
        frame = ~variable, text = ~station, type = 'scatter', mode = 'markers', 
        title = "Cluster Dynamics by Hour")
```

Analysis Summary
Cluster Analysis: The K-means and hierarchical clustering results show similar patterns, grouping stations based on average bike availability and location.
PCA Interpretation: The first principal component captures work commute dynamics, while the second component reflects general bike demand.

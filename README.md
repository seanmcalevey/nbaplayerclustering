# nbaplayerclustering
Clustering NBA Players Based on Offensive Advanced Stats

Summary of the code:
1. Import relevant libraries: numpy, urllib.request, bs4, pandas

2. Scrape web data from Basketball-Reference.com to get advanced stats for 2018-19 season

3. Get headers and rows for DataFrame

4. Initialize DataFrame with the above data, and then pre-process the data

5. Keep only relevant columns for analysis: ['Player', 'TS%', '3PAr', 'FTr', 'ORB%', 'AST%', 'TOV%', 'USG%']
  a. Convert DataFrame into matrix

6. Import scale from sklearn.preprocessing, and scale the data

7. Import PCA from sklearn.decomposition, and use PCA (Principal Component Analysis) to reduce dimensionality
  a. Reduce the data to 3 dimensions, which explain 62.422% of the variance.

8. Import KMeans from sklearn.cluster

9. Collect the statistics of all players and average them within each cluster

10. Create a DataFrame for all the clusters and their averages with their PCA1, PCA2, PCA3 data

11. Graph clusters in 3D with matplotlib.pyplot and mplot3d for better visualization

12. Take player name (input from user) and find their three closest L2-norm players by PCA1, PCA2, PCA3

13. Create DataFrame for comp players and rank them according to their similarity (L2-norm difference) to the subject player

# nbaplayerclustering
Clustering NBA Players Based on Advanced Stats from Basketball-Reference.com and FiveThirtyEight's DRAYMOND data.

The goal here is to group players effectively so that conclusions made about one player/playstyle can be applied to a group of others. The aim is to cluster players according their skillesets, play style, and success on the court. In other words, the goal is to identify players who play similarly to other players.

Take for example a player like LeBron James. Which NBAer approaches the game most similarly to him? The short answer is not many, because he's Lebron. Nevertheless there are players who are skilled scorers, passers, and defenders, and they will compare *best* to LeBron. Note: this isn't simply about comparing similar play styles, although it is about that too.

Consider player like Andre Iguodala, who approaches the game of basketball almost identically to LeBron: he's a servicable scorer, decent passer, decent rebounder, and a great defender. He's also similar in body size and plays the same position. In terms of how well-rounded Iguodala's game is, he's a dead ringer for LeBron. The problem is that he's only relatively decent at all the things LeBron transcends at. So although the two would compare favorably, Iguodala still might not be the best comp for LeBron.

On the other hand, someone like Kevin Durant, whose game has aspects that are a bit different than LeBron's (Durant's a better shooter, for one), could be a better comp. Durant adds similarly high value as an all-around player, has a high usage rate, is a good defender, and can hold his own passing the ball. In other words, he could be a better comp for LeBron.

This player clustering algorithm generates an arbitrary number of groupings of similar NBA players for a given target based on a 13-dimensional advanced-stat dataset pulled from Basketball-Ref.com and FiveThirtyEight's DRAYMOND data. The dataset was scaled, reduced a first time with PCA (Principal Component Analysis), scaled again, reduced a second time with LDA (Linear Discriminant Analysis) which preserved the clusters from the first clustering, and finally that result was scaled to produce the 2d output. There is a graph of the results at the end of section 5.

Advanced metrics considered in this algorithm: true shooting %, three point attempt rate, free throw rate, o-rebound %, assist %, turnover %, usage %, steal %, block %, defensive box plus/minus, and DRAYMOND, which is FiveThirtyEight's proprietary metric for measuring the ability to disrupt opponents' shots without blocking them, a crucial skill that's absent from standard box score info and even many advanced stats. Position and age are also considered, primarily to give the algorithm a base idea of the physical build and energy level of players.

The algorithm scrapes its data from basketball-reference.com and also directly imports its DRAYMOND data from FiveThirtyEight.

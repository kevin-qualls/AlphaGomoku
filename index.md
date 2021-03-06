#### Stefan Stanojevic, Kevin Qualls
#### DATA 2040: Deep Learning and Advanced Topics in Data Science

Hello! You have reached our website for our DATA 2040 Final project!

We are graduate students at Brown University building an AI version of Gomoku.

To view the machine-learning techniuqes implemented for this project, check out the following [Github Repository](https://github.com/stefs92/AlphaGomoku.git). Also below are blog posts that describe the implementations. 

# Understanding the Playing Field  
## Initial Blog Post - Stefan Stanojevic, Kevin Qualls

For this project, we set out a goal of implementing algorithms something akin to Google's DeepMind to devise an AI capable of playing the game of Gomoku at a competitive level. 

Similar to Connect Four, Gomoku is like Connect Five (Go means 5 in Japanese, and moku means pieces [[1]](http://www.opengames.com.ar/en/rules/Gomoku) ), but played on a horizontal Go board. An illustration is shown below (Image adapted from [[2]](https://www.researchgate.net/publication/312325842_Move_prediction_in_Gomoku_using_deep_learning)).

<p align="center">
<img width="249" alt="Screen Shot 2020-04-13 at 8 59 59 PM" src="https://user-images.githubusercontent.com/54907300/79174814-d85b9300-7dc9-11ea-9377-9cc909485ad2.png">
</p>
<p align="center">
  <b>Fig. 1: Gomoku is Played on a 15 x 15 Board</b><br>
</p>


There are ample documentations online for how to build a Connect Four AI, as the game is relatively simple and requires less computational power [[3]](http://www.opengames.com.ar/en/rules/Gomoku). On the other hand, documentation for how to build an AI Gomoku is sparse, likely due to its abstract strategy on a 15 x 15 gameboard setting [[2]](https://www.researchgate.net/publication/312325842_Move_prediction_in_Gomoku_using_deep_learning). Rather than seeing this as a setback, we welcomed the challenge to deepen our understanding of neural networks and machine learning AI.  

Our starting goal was to simply train a neural network to predict the next move a skilled human player would make in Gomoku. In order to do so, we've started with a dataset of competitive games pieced together from different sources on [[4]](http://mostovlyansky.narod.ru/iedown.html), mostly from Russian tournament archives. We've managed to paste together around 11000 games overall. The game entries came in the following form:

1997,382=[marik,om-by,-,88FFFE98798A6A975B4C59999A7BA86C5D5C3C7A4B896BA7B6A99687,?,?]

This entry first specifies the year of the competion, the players and the winner (- corresponds to the second player winning). Then, the long string of penta-decimal numbers specifies the board coordinates of Gomoku moves. The sequence of moves in this game of Gomoku is shown in the following figure. Note that while no player has managed to connect 5 tokens yet, the white has already won by constructing two unrestricteed sequences of 3 tokens (26,4,28 and 28,8,24), intersecting at token 28.


<p align="center">
<img width="249" alt="Screen Shot 2020-04-13 at 8 59 59 PM" src="https://user-images.githubusercontent.com/31740043/79678687-fa8b5180-81cb-11ea-9943-343c38e5bf97.PNG">
</p>
<p align="center">
  <b>Fig. 2: Sequence of Moves in an Example Game</b><br>
</p>

Next, we turned this game string into a sequence of 28 images representing the states of the board at different times during the game. Those would correspond to inputs to our neural network. The output was a single number specifying one of 15^2 = 225 possible next moves. Some additional preprocessing included removing the duplicate board game states from the dataset. This was done by first sorting the list of board states, and then iterating through this dataset and collecting neighboring identical boards. Tokens were one-hot encoded, with (1,0,0) corresponding to first player's token, (0,1,0) corresponding to second player's token and (0,0,1) corresponding to empty space.    

Since this is essentially an image classification task, it makes sense to try to use a convolutional neural network. The neural network architecture we used came from [[2]](https://www.researchgate.net/publication/312325842_Move_prediction_in_Gomoku_using_deep_learning), and took the following form:

![model](https://user-images.githubusercontent.com/31740043/79679151-57d5d180-81d1-11ea-95d3-1f3b453120d3.PNG)
<p align="center">
  <b>Fig. 3: Neural Network of Example Game from Fig 2</b><br>
</p>

This neural network achieved a decent validation accuracy of around 55% pretty quickly, as shown in the following graphs:

<p align="center">
<img width="261" alt="Screen Shot 2020-05-12 at 3 44 43 PM" src="https://user-images.githubusercontent.com/54907300/81739172-dc321200-9468-11ea-896d-1c30eeb4b2dd.png">
</p>
<p align="center">
  <b>Fig. 4: Accuracies and Losses of Example Game</b><br>
</p>


The model is shown to be overfitted, as the accuracies don't sync with one another. Compared to the training data set, the validation data set shows more stability, as the accuracy and loss number keep a relatively horizontal shape. 

## Reinforcement Learning Approach

We have now obtained a neural network capable of imitating human players by predicting their next move, and plan to use this to jump-start the training of our AI Gomoku player using DeepMind's reinforcement learning algorithm. Let us briefly describe how this would work. 

In the language of DeepMind, our model is a "policy head", advising AI which next moves to take under consideration. Another quantity that a player needs to consider is the "value" of board states, roughly the measure of how desirable they are. Our AI player can then perform a number of simulations of possible games guided by its policy and value estimates (something called "Monte Carlo Tree Search"), and decide what move to make based on the result of those simulations. We can thus play out a number of AI vs AI games, and train our neural network on the replays. This results in a very large training set of games between progressively more advanced players, which given sufficient computational power can be used to train our neural network to reach an expert level in Gomoku.

## References

[1] Rules for Gomoku - http://www.opengames.com.ar/en/rules/Gomoku

-- Gives an overview of origins of Gomoku as well as the rules of the game.

[2] Shao, Kun & Zhao, Dongbin & Tang, Zhentao & Zhu, Yuanheng. (2016). Move prediction in Gomoku using deep learning. 292-297. 10.1109/YAC.2016.7804906.  - https://www.researchgate.net/publication/312325842_Move_prediction_in_Gomoku_using_deep_learning

-- Describes methodology of how to predict moves in Gomoku, using a convolutional neural network model. 

[3] From-scratch implementation of AlphaZero for Connect4 - https://towardsdatascience.com/from-scratch-implementation-of-alphazero-for-connect4-f73d4554002a

-- Describes how to implement Google DeepMind's AlphaZero approach for Connect4. Methodology can be applied to Gomoku.

[4] Gomoku datasets http://mostovlyansky.narod.ru/iedown.html

-- Archives datasets of Gomoku games. Data is stored in a .bdt file.

[5] AlphaGomoku: An AlphaGo-based Gomoku Artificial Intelligence using Curriculum Learning - Zheng Xie, Xing Yu Fu, Jin Yuan Yu, Likelihood Lab, https://arxiv.org/pdf/1809.10595.pdf 

-- Shows how to implement curriculum learning - a technique that builds the AI Gomoku's strategy and knowledge of the game through progressively difficult tasks. 

...

# Some Attempts at Self-Play Reinforcement Learning
## Midway Blog Post - Stefan Stanojevic, Kevin Qualls

In our initial blog post, we wrote about training a neural network on the dataset of Gomoku games, in order to predict the next move a human player would make. Since then, we have taken our project a step further and coded an AI Gomoku player that can gradually improve its skill through self-play and reinforcement learning.

Since we were curious about whether our neural network has actually learned important elements of the game or not, we decided to quickly code a self-playing module and visualize its performance, before fully implementing the AlphaZero algorithm. We used our trained "policy head", giving us the probability distribution over possible moves, to iteratively generate the next move until one of the AI players was in the position to win by connecting 5 tokens. Python's ipywidget library proved to be very useful for visualizing the games, specifically its objects interact and Play. You can see one of the sample games below:

<p align="center">
<img src="https://user-images.githubusercontent.com/31740043/80993579-4c0d2080-8e09-11ea-9149-3533c65e79e7.gif" >
</p>
<p align="center">
  <b>Fig. 5: Sample Game of AI Gomoku Playing Against Itself</b><br>
</p>



Predictably, our AI is pretty bad at playing the game, as it is missing key components - the "value head" evaluating the chance of winning for different board states, as well as the ability to simulate the future. In order to rectify the first problem, we went back to our dataset of human games. Keeping track of the winner of each game, we assigned a score of +1, -1 or 0 (in case of a draw) to each board state of a given game and averaged those out over the dataset. Then, a neural network of the same architecture as in the initial blog post (except the final layer, adopted to the new regression task) is trained to predict the board state value. We obtain a not great, not terrible performance as seen in the plot below:

<p align="center">
<img width="469" alt="Fig 6 final project" src="https://user-images.githubusercontent.com/54907300/81741226-38e2fc00-946c-11ea-9120-3e620d4a3dd2.png">
</p>
<p align="center">
  <b>Fig. 6: Model Performance of Predicting the Winner for Each Game</b><br>
</p>

As we can see, the best validation accuracy is achieved very early in the training process, and there is significant overfitting later on.

At this point, we were in a position to try out a slightly more sophisticated algorithm. Our agent can decide on which move to make considering the value functions of different moves, as well as its probability distribution over the space of moves. The first one is related to "exploitation" of its knowledge, more useful in the later stages of the game, and the second one to "exploration" of possibilities, in theory useful for innovation early in the game. 

We wanted to emulate the AlphaZero algorithm, which worked in the following way. Prior to making each move, simulations of the remainder of the game are made. Since the space of possible games is generally way too large to efficiently cover with a search algorithm, the search is done by looking at a set of the likely games randomly sampled using the probabilities from "policy head" and values from "value head". While AlphaGo does around 1000 simulations at each step, even a much smaller number of simulations looks too computationally intensive for us, with each simulation taking on the order of a minute to finish.


## Adjustments to Consider for Reinforcement Learning Approach

We feel that with more computational power, we can use reinforcement learning to better train an AI Gomoku. We also would like to see if reinforcement learning can lead to the AI learning any unconventional strategies that can help it be more successful. For example, in our course lecture, we learned how reinforcement learning helped an AI develop a strategy to get ~20% more points in the game CoastRunners 7, without the AI having to complete the game [8]. Additionally, reinforcement learning also helped an AI rack up more points in the game Atari to the point where the bottom paddle didn't have to move itself [8]. Ultimately, it would require more time and GPU power for our AI Gomoku study a pattern like this.

## References

[1] Rules for Gomoku - http://www.opengames.com.ar/en/rules/Gomoku

-- Gives an overview of origins of Gomoku as well as the rules of the game.

[2] Shao, Kun & Zhao, Dongbin & Tang, Zhentao & Zhu, Yuanheng. (2016). Move prediction in Gomoku using deep learning. 292-297. 10.1109/YAC.2016.7804906.  - https://www.researchgate.net/publication/312325842_Move_prediction_in_Gomoku_using_deep_learning

-- Describes methodology of how to predict moves in Gomoku, using a convolutional neural network model. 

[3] From-scratch implementation of AlphaZero for Connect4 - https://towardsdatascience.com/from-scratch-implementation-of-alphazero-for-connect4-f73d4554002a

-- Describes how to implement Google DeepMind's AlphaZero approach for Connect4. Methodology can be applied to Gomoku.

[4] Gomoku datasets http://mostovlyansky.narod.ru/iedown.html

-- Archives datasets of Gomoku games. Data is stored in a .bdt file.

[5] AlphaGomoku: An AlphaGo-based Gomoku Artificial Intelligence using Curriculum Learning - Zheng Xie, Xing Yu Fu, Jin Yuan Yu, Likelihood Lab, https://arxiv.org/pdf/1809.10595.pdf 

-- Shows how to implement curriculum learning - a technique that builds the AI Gomoku's strategy and knowledge of the game through progressively difficult tasks. 

[6] Wang, Y. (n.d.). Mastering the Game of Gomoku without Human Knowledge. doi: 10.15368/theses.2018.47

-- Describes implementation of an AI Gomoku without prior knowlege from humans. Provides explanation of the Monte Carlo Tree Search algorithm, which we plan to use for our model. 

[7] Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., … Hassabis, D. (2017). Mastering the game of Go without human knowledge. Nature, 550(7676), 354–359. doi: 10.1038/nature24270

-- Describes reinforcement learning algorithm used for AlphaGo Zero to defeat AlphaGo 100 times to 0. 

[8] DATA 2040 Lecture 21 - Lecture 22: Reinforcement Learning 

-- Describes how reinforcement learning can pick up helpful, unconventional strategies in AI games. 

...


# Some Final Exploration 
## Final Blog Post - Stefan Stanojevic, Kevin Qualls

After training our neural networks on a dataset of Gomoku games and setting up the code that would run self-play, we seemed to be in a good place. However, the computational complexity of the task ahead of us proved to be a significant difficulty. AlphaZero algorithm runs a number of game simulations (in AphaGo's case, around 1000) at each game step in order to determine which move to make. Furthermore, their game agent has spent a human equivalent of several thousand years playing Go. On the other hand, due to the fact that we are running an unparallelized Python code, a single simulation takes of the order of a minute to complete. So, we have decided to take a step back and think of some other ways we can improve our understanding of Gomoku and related games through deep learning.

Our DATA 2040 Professor, Dr. Potter, let us know that since Connect-4 is a solved game, there exists a great dataset that might be worthwhile for us to explore. This dataset, due to Dr. John Tromp, contains winner information for a full set of 8-token board states in which neither player has won yet. Since we were curious how good machine learning can be in capturing this information, we decided to go ahead and train a neural net to predict a winner given a board state. This is similar to the analysis we've done for Gomoku, but this time we had an advantage of starting from a clean dataset for an exactly solved game. 

After applying the same CNN architecture from the previous two blog posts (inspired by [[2]](https://www.researchgate.net/publication/312325842_Move_prediction_in_Gomoku_using_deep_learning)), we managed to squeeze some extra accuracy out of our model by going deeper and applying skip connections, in the spirit of AlphaGo architecture from [[9]](https://doi.org/10.1038/nature16961). Our best-performing model consisted of a stack of 10 residual blocks, each of the following architecture,

<p align="center">
<img width="200" alt="accuracy" src="https://user-images.githubusercontent.com/31740043/81613680-e68ed600-93ac-11ea-840a-c94590e37cbc.png">
</p>
<p align="center">
  <b>Fig. 7: Schematic of Best Performing Model</b><br>
</p>

This stack of residual blocks was preceeded by a single convolutional layer, and followed by two dense layers, with 100 hidden neurons. Every convolutional layer in this network had 16 filters. We found that using both dropouts and L1 regularizations yields the best results, and tuned the dropout parameter to 0.3 and L1 regularization parameter to 0.5 in all convolutional layers.

Batch normalizations played a pretty important role in our Gomoku model. During our initial experimentation with this model, we also used batch normalizations as a part of our residual blocks. However, this produced slightly suboptimal results compared to our final model, which got rid of batch normalizations altogether. After doing a bit of research on this phenomonon, we realized that this is actually a pretty common theme, among several different neural network architectures, as documented in [[10]](https://arxiv.org/pdf/1801.05134.pdf). Dropouts (when implemented before batch normalization) mess with batch statistics, causing a "variance shift", ultimately dropping the performance [[10]](https://arxiv.org/pdf/1801.05134.pdf).

We got the following accuracy and loss curves:

<p align="center">
<img width="316" alt="fig 8 final project" src="https://user-images.githubusercontent.com/54907300/81743367-80b75280-946f-11ea-9c9a-98521a0d261b.png">
</p>
<p align="center">
  <b>Fig. 8: Accuracy and Loss Curves of Final Model</b><br>
</p>


Our neural network is able to predict the winner of a given 8-token board state with a validation accuracy of around 87%. Furthermore, the validation and accuracy curves came closer together and look more in sync compared to our first model in Fig. 4. 

Since tinkering with different types of CNN architectures seems unable to produce a better accuracy on this dataset, it can be interesting to contemplate what this might mean for the applications of deep learning to game theory. Algorithms such as AplhaZero do not directly use neural nets to decide on the next move; rather, they perform a version of minimax algorithm in which the neural network is used to select moves to base simulations on. In order for this to be successful, it may not be necessary to have neural network calculate the value of the state with extreme precision.

## Future Work


Although we didn't implement reinforcement learning as planned - due to limitations with  computational power - we were able to still achieve a validation accuracy of around 87% for prediciting the next winner in Connect-4, given an 8-token board state. This result can be used to efficiently jump start a great Connect-4 player.

More importantly, we plan to continue to work on implementing a Gomoku player. The most important task going forward is to speed up the self-play process by removing inefficiencies in our code and using parallel processing. Learning more about Cuda library seems like a good place to start.

Something else to explore is experimenting with discounted state values for states that are far removed from the end of the game. Authors of AlphaGo assigned +1 to every state in the winning game and -1 to every losing state; however, due to their immense computational power, they were able to generate huge training datasets. With our more limited resources, however, this may not be the best way to go. Something else to consider if we are unable to achieve significant speedups is doing away with the "policy head" entirely and focusing on just the values. In this case, every "simulation" of the game could serve as a training entry to further improve the neural network estimating the value of game states. 

Another idea to play with is trying to incorporate the symmetries of the problem into our neural network, in order to effectively decrease the size of the game configuration space, along the lines of [[11]](https://arxiv.org/pdf/1602.02660.pdf).  Our 15x15 board has reflection symmetries around a horizontal, vertical, and two diagonal axes containing this point. An option to consider if we are unable to construct a neural network that encodes those symmetries is simply enlarging our training sets by applying those symmetry transformations.

Then, we have encountered several interesting ideas  in the literature that we would like to implement. One potentially very useful idea is called curriculum learning [[5]](https://arxiv.org/pdf/1809.10595.pdf), and has to do with introducing simpler ideas to our player first and then building up its skill. For example, like the authors of this paper, we may try to generate a dataset of important moves ourselves, and feed it to our neural network. Additionally, we can use non - deep learning based mentor AI to generate more training data, and play against our model.

In any case, we believe that with more computational time and coffee we can eventually make AlphaGomoku work.



## References

[1] Rules for Gomoku - http://www.opengames.com.ar/en/rules/Gomoku

-- Gives an overview of origins of Gomoku as well as the rules of the game.

[2] Shao, Kun & Zhao, Dongbin & Tang, Zhentao & Zhu, Yuanheng. (2016). Move prediction in Gomoku using deep learning. 292-297. 10.1109/YAC.2016.7804906.  - https://www.researchgate.net/publication/312325842_Move_prediction_in_Gomoku_using_deep_learning

-- Describes methodology of how to predict moves in Gomoku, using a convolutional neural network model. 

[3] From-scratch implementation of AlphaZero for Connect4 - https://towardsdatascience.com/from-scratch-implementation-of-alphazero-for-connect4-f73d4554002a

-- Describes how to implement Google DeepMind's AlphaZero approach for Connect4. Methodology can be applied to Gomoku.

[4] Gomoku datasets http://mostovlyansky.narod.ru/iedown.html

-- Archives datasets of Gomoku games. Data is stored in a .bdt file.

[5] AlphaGomoku: An AlphaGo-based Gomoku Artificial Intelligence using Curriculum Learning - Zheng Xie, Xing Yu Fu, Jin Yuan Yu, Likelihood Lab, https://arxiv.org/pdf/1809.10595.pdf 

-- Shows how to implement curriculum learning - a technique that builds the AI Gomoku's strategy and knowledge of the game through progressively difficult tasks. 

[6] Wang, Y. (n.d.). Mastering the Game of Gomoku without Human Knowledge. doi: 10.15368/theses.2018.47

-- Describes implementation of an AI Gomoku without prior knowlege from humans. Provides explanation of the Monte Carlo Tree Search algorithm, which we plan to use for our model. 

[7] Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., … Hassabis, D. (2017). Mastering the game of Go without human knowledge. Nature, 550(7676), 354–359. doi: 10.1038/nature24270

-- Describes reinforcement learning algorithm used for AlphaGo Zero to defeat AlphaGo 100 times to 0. 

[8] DATA 2040 Lecture 21 - Lecture 22: Reinforcement Learning 

-- Describes how reinforcement learning can pick up helpful, unconventional strategies in AI games. 

[9] Silver, D., Huang, A., Maddison, C. et al. Mastering the game of Go with deep neural networks and tree search. Nature 529, 484–489 (2016). https://doi.org/10.1038/nature16961

-- Describes how their AlphaGo program considers which moves are advantageous by evaluating different moves.    

[10] Li, Xiang, et al. "Understanding the disharmony between dropout and batch normalization by variance shift." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019. https://arxiv.org/pdf/1801.05134.pdf

-- Justifies why the performance of our model could be poor, due to dropouts being used before applying batch normalization. 

[11] Dieleman, S., De Fauw, J., Kavukcuoglu, K. (2016). Exploiting Cyclic Symmetry in Convolutional Neural Networks, https://arxiv.org/pdf/1602.02660.pdf

-- Describes how to train a neural network model for images that demonstrate symmetry. This is helpful, since the Gomoku Board is 15x15 and has symmetric properties.

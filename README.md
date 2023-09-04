# QRL_project
Dimensionality reduction in quantum reinforcement learning agents. 

Quantum reinforcement learning is typically done with a reuploading PQC. This scheme has been proven to work quite well, however, 
for a n dimensional state space one would need an n qubit circuit. This can become problematic, especially with the currnt size
of quantum computers. 

![Image](https://i.imgur.com/LLFTvMP.jpg)

# how to run?

Install everything etcetera and run the main.ipynb for a simple run


# why is this so slow

I think the main problem that causes speed issues is the custom tensorflow model/layers. 
Simon used the regular tensorflow model which u compile and then run. 
Here we implement the circuit as a custom tensorflow layer which sits inside a custom tensorflow model.(this is a must i think if you want to assign optimizers to each trainable variable)  
This makes it, unfortunatly, very slow. 




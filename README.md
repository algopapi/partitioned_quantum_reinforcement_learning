# QRL_project
Dimensionality reduction in quantum reinforcement learning agents. 

Quantum reinforcement learning is typically done with a reuploading PQC. This scheme has been proven to work quite well, however, 
for a n dimensional state space one would need an n qubit circuit. This can become problematic, especially with the currnt size
of quantum computers. 

In this project I explore the idea of trading classical and quantum recourses. We take a simple REINFORCE agent with 
a reuploading PQC and "chop it in halve" (in a clever way). Then, we attempt to train it in the cartpole environment. 


![Image](https://i.imgur.com/LLFTvMP.jpg)

I think the main problem that causes speed issues is the custom tensorflow model/layers. 
Simon used the regular tensorflow model which u compile and then run. 
Here we implement the circuit as a custom tensorflow layer which sits inside a custom tensorflow model.(this is a must i think if you want to assign optimizers to each trainable variable)  
This makes it, unfortunatly, very slow. 


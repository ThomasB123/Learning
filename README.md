# Learning
Year 3 Coursework

## The Mythical Pegasus and Gravitar

Deep Learning and Reinforcement Learning Summative Assignment Michaelmas Term, 2020

## Introduction

The assignment is split into two parts: (1) Deep Learning (50 marks) and (2) Reinforcement Learning (50 marks) accordingly, for a combined total of 100 marks available.

In short, you are to build a generative model that creates unique and diverse images of winged horses that all look like a Pegasus, the mythical divine horse in Greek mythology. This is usually depicted as pure white, with clear outspread wings. The second part of the assignment is to train a deep reinforcement learning agent to play the Atari 2600 game Gravitar, published in 1983. This is one of the most difficult Atari games where you have to fly to different planets as you get pulled towards deadly stars by gravity, shoot enemies, conserve and pick up fuel with a tractor beam that also deflects bullets.

For the Pegasus part, you are to write a short scientific paper for the method, results, and limitations in a provided LATEX template that closely follows parts of the ICLR 2021 conference style guidelines. Whereas for the Gravitar part, you are to provide a video of the agent playing the game, code, and log file (no report is required). These files must all be zipped together like this:

  - submission.zip
    - pegasus-paper.pdf
    - pegasus-code.ipynb (or .py)
    - gravitar-code.ipynb (or .py)
    - gravitar-video-episode-1210-score-85.mp4
    - gravitar-log.txt

To assist in this, the following templates are provided to build on:

- Pegasus Paper LATEX Template - login with durham email, on ‘overleaf.com’ click ‘make a copy’ to edit
- Google Colab Pegasus Starter Code
- Google Colab Gravitar Starter Code

## Part 1: The Mythical Pegasus

Using the CIFAR-10 and/or STL-10 datasets, you are to train a deep generative model to synthesise a unique batch of 64 images that all looked like winged horses (of any colour). There are no such images of winged horses in these two datasets. You should then select, from the batch, the single best image that looks most like a white Pegasus. The paper should use the provided LATEX template, where you are to write up the methodology, results, and limitations of your approach alongside a short abstract.

The report should be written like an academic paper, where mathematical notation should try to follow the ICLR 2021 guidelines (see the template for more information). This means your discussions should be short, clear, and concise —less is more. Where appropriate, it is recommended to include a high-level architectural diagram in the paper to help explain your approach.

You can use any architecture that you like, and you can use any sampling strategy. For example you could train an autoencoder, and condition it on a birds and horses which have been manually identified from the dataset. These datasets mostly have annotated class labels available for the images, which you can use to help identify the horses and birds. This will give two latent codes via the encoder network, that you could linearly interpolate between to give the final outputs via the decoder network. Alternatively, you could train a GAN and randomly sample a batch of 64 images from the model. These two ideas are not necessarily the best solution. Also, there are penalties and bonuses that will influence your design:

- Use any adversarial training (e.g. GAN) method  -4 marks
- Your best Pegasus is non-white (e.g. brown or black)  -2 marks
- Nearly all winged horses in the batch are white +1 mark
- Train only on CIFAR-10  -2 marks
- Train with STL-10 resized to 48x48 pixels +1 mark
- Train with STL-10 resized to 64x64 pixels +1 mark
- Train with STL-10 at the full 96x96 pixels +3 marks
- Manually edit (paint) any images or outputs -50 marks
- Train on any other data outside of the datasets -50 marks
- Use or modify someones code without referencing it -50 marks
- Use pre-trained weights from another model -50 marks
- Every page over 4 pages in the paper (excluding references) -5 marks
 
Table 1: Penalties and bonuses accumulate, and are added onto the final mark.

Please state at the end of the paper the total bonus or penalty you are expecting, based on this table. For example if you successfully train a GAN to produce a white Pegasus from a batch of mixed-colour winged horses, using both data from CIFAR-10 and from STL-10 at the full resolution, you can expect to receive: -4 marks (as its a GAN), then 1+1+3 = +5 marks for STL-10 at 96x96 resolution, for a total bonus of +1 mark. If you submit a paper that is 7 pages long, you will receive an additional -15 mark penalty.

Figure 1: Generating a realistic and recognisable Pegasus is very difficult within the dataset constraints, therefore outstanding attempts with unimpressive results can still get good marks. These are some ex- cellent results when training only on CIFAR-10.


## Pegasus paper marking scheme

The paper will be marked as follows:

- [20 marks] Scientific quality and mathematical rigor for the paper and solution
  – Communication, application, and presentation of the underpinning mathematical theory
  – Architectural design, sophistication, appropriateness, and novelty
  – Clarity, simplicity and frugality of both the scientific writing and the implementation
  
- [10 marks] Recognisability of the single best output
  – Can I tell this is an image of a Pegasus?
  – How much do I need to stretch my imagination to see a winged horse?
  
- [10 marks] Realism of the sampled batch of model outputs
  – Are the generated images blurry?
  – Do the objects in them have realistic shapes or textures? Do they look real?
  
- [10 marks] Uniqueness of the sampled batch of model outputs
  – How different are the images from their nearest neighbours in the dataset? – How diverse are the samples within the batch of 64 provided?
  – Do all the winged horses look the same? Is there any mode collapse?


 Part 2: Gravitar
Gravitar is a notoriously difficult Atari 2600 game, with complex controls and changing environments where you always have to navigate away from gravity. This is what the beautiful Atari 2600 console looked like, which was released in 1977. It had an 8-bit 1.19 MHz CPU with just 128 bytes of RAM!
Figure 2: Atari 2600 console Figure 3: Gravitar cartridge
I recommend that before you begin, you have a go at playing Gravitar online to understand the game mechanics. Try to visit a couple of planets and use the tractor beam (down key) to pick up some more fuel. My score is 7,250 - can you do any better?
W [Play Gravitar with an online Atari 2600 emulator] spacebar to start/fire and move with arrow keys W [More information, including the full manual and cartridge images]
     Task
Your task is to build and train a deep reinforcement learning agent to obtain the highest score possible, using the provided starter code linked on the first page. OpenAI gym provides two Gravitar environments ‘Gravitar-ram-v0’ and ‘Gravitar-v0’ respectively. You can use either of them, but pay attention to the size of the input array shapes as documented W here and W here.
You can use any reinforcement learning algorithm that you like, including ones not covered in the lectures.
Submission
The following figure explains how and what to submit. You can print whatever data you like into the log, but you must keep the string which starts with ‘marking’. This string prints the mean and standard devi- ation of the score over every 100 episodes. You must not change this. The submitted video should show the highest score that you are able to successfully record. Recording a video is quite slow, so you may wish
3

 to do it every 25 or more episodes. Increasing this interval may slightly improve training performance, but you may miss recording a good episode.
This is the best episode with a recorded video
Therefore download episode 250 and rename it Make sure these lines are printed gravitar-video-episode-250-score-700.mp4 Copy the full log to gravitar-log.txt
The code submission must be written with clarity and minimalism. You can submit an .ipynb or .py file. You do not need to follow PEP-8 or any departmental guidelines for code quality with this assignment. Try to keep comments to a minimum, as good code should speak for itself. If you have tried many impressive designs and ideas, you may briefly explain these in a large comment (or text in .ipynb if you prefer) at the bottom of the submission, but do not expect this to significantly influence marking of your final solution.
Anti-tampering
There will be statistical analysis of the submitted log files. Any log files with outliers in the convergence behaviour will have their associated code retrained multiple times using SLURM scripts on NCC, learning a distribution over its expected convergence behaviour. If the likelihood of the submitted log data shows evidence of tampering, a departmental investigation will be conducted. If interested, this method for de- tecting tampering is similar to parts of the solution used to solve W Sploosh Kaboom in Zelda Windwaker.
Marking
The code, video, and log submission will be marked as follows: • [30 marks] Convergence and score
– How efficiently does your agent learn?
– Does it just get lucky after lots and lots of training? – How often does it get a good score?
• [10 marks] Sophistication and appropriateness of the solution
– How well have you applied the relevant theory to the problem?
– How hackish is your implementation, or is it robust and well-designed?
– Have you just cited and pasted code, or is their evidence of comprehension with further study and novel design extending beyond the lecture materials?
• [10 marks] Intuition of video
– Does the video look like the agent is just randomly getting lucky after lots of episodes?
       4

– Is it chasing and hunting down the enemies efficiently?
– Is it navigating gravity gracefully, or is it like it’s powered by an infinite improbability drive? – Has it learnt any surprising behaviours?
PyTorch Training
This assignment can be completed entirely using Google Colab, or you may wish to register for an ac- count and train on NCC: http://ncc.clients.dur.ac.uk/ (only available on the internal university net- work). Users without remote server experience or job queuing system experience (such as SLURM) are recommended to continue to use Google Colab, which is just as fast for PyTorch training. If using NCC, please carefully read the documentation and respect other users on the job queuing system.
Closing Comment
I hope that you enjoy this coursework. If you are struggling, please join one of the weekly zoom meetings to ask questions and discuss any issues, such as with programming or relevant theory.

# SB-GAN
<!-- <img src='imgs/teaser_SBGAN.jpg' align="right" width=384> -->
<center><h2>Semantic Bottleneck Scene Generation</h2></center>
Samaneh Azadi, Michael Tschannen, Eric Tzeng, Sylvain Gelly, Trevor Darrell, Mario Lucic
<img src='imgs/SB-GAN-samples.jpg' align="center">
Coupling the high-fidelity generation capabilities of label-conditional image synthesis methods with the flexibility of unconditional generative models, we propose a semantic bottleneck GAN model for unconditional synthesis of complex scenes. We assume pixel-wise segmentation labels are available during training and use them to learn the scene structure. During inference, our model first synthesizes a realistic segmentation layout from scratch, then synthesizes a realistic scene conditioned on that layout. For the former, we use an unconditional progressive segmentation generation network that captures the distribution of realistic semantic scene layouts. For the latter, we use a conditional segmentation-to-image synthesis network that captures the distribution of photo-realistic images conditioned on the semantic layout. When trained end-to-end, the resulting model outperforms state-of-the-art generative models in unsupervised image synthesis on two challenging domains in terms of the Frechet Inception Distance and user-study evaluations. Moreover, we demonstrate the generated segmentation maps can be used as additional training data to strongly improve recent segmentation-to-image synthesis networks.

<table align=center width=850px>
  <center><h1>Paper</h1></center>
  <tr>
  <td width=400px align=center>
  <!-- <p style="margin-top:4px;"></p> -->
  <a href="https://people.eecs.berkeley.edu/~sazadi/SemanticGAN/thumbnail.jpg"><img style="height:200px" src="https://people.eecs.berkeley.edu/~sazadi/SemanticGAN/thumbnail.jpg"/></a>
  <center>
  <span style="font-size:20pt"><a href="https://people.eecs.berkeley.edu/~sazadi/SemanticGAN/main.pdf">[Paper 2MB]</a>&nbsp;
  <span style="font-size:20pt"><a href="https://arxiv.org/abs/">[arXiv]</a>
  </center>
  </td>
  </tr>
  </table>
<center><h1>Code</h1></center>
Code will be released soon.


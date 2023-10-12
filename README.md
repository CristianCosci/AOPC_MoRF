# Area Over Perturbation Curve using Most Relevan feature

## Introduction
Since we cannot trust the XAI methods blindly, it is necessary to define and have some **metrics capable of providing a quality index of each explanation method**. However, there are still no metrics used as standard in the state of the art. The objective assessment of the explanations quality is still an active field of research. The various metrics defined are not general but depend on the specific model task on which the XAI methods are used. <br>
Many efforts have been made to define **quality measures for heatmaps and saliency maps** which explain individual predictions of an Image Classification model. One of the main metrics used to evaluate explanations in the form of saliency maps is called **FaithFulness**. <br> 
The faithfulness of an explanation refers to whether the relevance scores reflect the true importance. The assumption of this evaluation metric is that perturbation of relevant (according to the heatmap) input variables should lead to a steeper decline of the prediction score than perturbation on the less important ones. Thus, the average decline of the prediction score after several rounds of perturbation (starting from the most relevant input variables) defines an objective measure of heatmap quality. If the explanation identifies the truly relevant input variables, then the decline should be large. <br>
There are several methods to compute the Faithfulness, one of them is a metric called **Area Over the Perturbation Curve - (AOPC)**, described below.

## Metric Description
The **AOPC** approach, measures the change in classifier output as pixels are sequentially perturbed (flipped in binary images, or set to a different value for RGB images) in order of their relevance as estimated by the explanation method. It can be see as a greedy iterative procedure that consists of measuring how the class encoded in the image disappears when we progressively remove information from the image $x$. <br>
The classification output should decrease more rapidly for methods that provide more accurate estimates of pixel relevance. This approach can be done in two ways:
- **Most Relevant First (MoRF)**: The pixels are perturbed starting from the most important according to the heatmap (rapid decrease in classification output).
- **Least Relevant First (LeRF)**: The least relevant pixels are perturbed first. In this case, the classification output should change more slowly the more accurate the saliency method.

In this work, we decided to use the MoRF version. It is computed as follow:
$$
    \text{AOPC}_M = \frac{1}{L + 1} \Biggl \langle \sum_{k=1}^{L} f(x^{(0)}_M) - f(x^{(k)}_M) \biggr \rangle_{p(X)} 
$$

where $M$ is the pixel deletion procedure (MoRF or LeRF), $L$ is the number of pixel deletion steps, $f(x)$ is the output value of the classifier for input image $x$ (i.e. the probability assigned to the highest-probability class),  $x^{(0)}_M$ is the input image after $0$ perturbation steps (i.e. $x^{(0)}_M = x$), $x^{(k)}_M$ is the input image after $k$ perturbation steps, and $\bigl \langle . \bigr \rangle_{p(X)}$ denotes the mean over all images in the dataset.

## Usage

### Example of Usage 

## Requiriments Installation

### Questions and Issues

### References
\cite{tomsett2020sanity, samek2016evaluating}

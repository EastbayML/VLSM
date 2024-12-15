# West Coast Machine Learning
## Very Small Language Model

### Experiments Brainstorming
This is just a list of ideas of things we might want to experiment with.

#### Visualize token level loss over a sequence
- Explore how this changes during training
- Loss should start constant over the sequence
- We are looking for ‘context learning’ where the loss decreases as the sequence grows

#### Architecture
- **Root of the residual stream**
  - Could the residual stream start with a learned value instead of the last token? Would this allow changing the residual stream size throughout the model?
- **Experiment with different size layers**
  - Does every layer need to be the same? If the first layers are parsing the sentence and the last layers are creating the sentence, could they be smaller?
- **Model dimension**
  - Smaller dimensions in the beginning as the sentence is parsed, larger dimensions in the middle for larger context, smaller dimensions in the end for sentence generation
- **Context window**
  - Smaller context in the beginning for sentence parsing, larger context in the middle, and smaller context in the end.
- **Weight sharing**
  - Is it possible to share weights between layers? Is it possible that some recursion could occur? Could the same weights be applied to the residual stream after state has been added to perform the next step of the prediction
- **Quality of dataset**
  - Observe the effect of training on different datasets
    - Tinystories
    - Webtext
    - Books
    - Wikitext
    - Dolma
    - SQuAD

#### Separation of Reason and Knowledge
- If we define knowledge as the facts, and reason as the processing of the facts to drive inference, can we make a smaller model that is able to reason using external facts, either from a RAG, a LoRA or a context window?
  - Could the facts be automatically and easily predicted and loaded based on the context.
  - Keep the knowledge out of the model.
  - Train the model using a preloaded context. The first part of the input is the preloaded context and the second part can be derived using only general reasoning and the preloaded context. This is both a methodology and a new dataset
  - Distill the knowledge out of the model
    - Using a pretrained model, create a loss function that will ‘forget’ knowledge if it is available externally. I.e. RAG, LoRA, context window. This migrates knowledge out of the model and presumably can be distilled into a smaller window.

#### Training
- Test out ‘Grant Descent’ aka ‘Tangent Descent’ on a language model

#### Power scaling of sum of the attention score values

#### Watch the mean token probability early during training. 
I think it should go from flat, to matching the natural language distribution?

####  Shaping the probability distribution
How does the probability distribution of a sequence change given different prompts?
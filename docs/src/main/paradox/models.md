# Models

## Encoders

- **Uni-directional RNN:** This represents one of the 
  simplest encoders; a uni-directional (i.e., 
  left-to-right) RNN encoder. This encoder takes as input 
  a source sequence in some language and produces a tensor 
  containing the RNN outputs for each time step and a final 
  state which is a sequence of last computed RNN states 
  in layer order containing the states for each layer 
  (e.g., `Seq(state0, state1, ...)`).

@@@ note

Currently the encoders and the decoders are bridged using a 
copy mechanism that simply copies the final encoder state 
as the initial decoder state. This means that the total 
number of encoder layers must match the total number of 
decoder layers. This limitation will be lifted once we add
support for more bridging mechanisms.

@@@

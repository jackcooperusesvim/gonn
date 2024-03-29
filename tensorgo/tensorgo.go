package tensorgo

import (
	"errors"
	"math/rand"
)

type MultiLayeredPerceptron struct {
	layers  []Layer
	act_fun ActivationFunction
}

type ActivationFunction struct {
	eval       func(float64) float64
	derivative func(float64) float64
}

type gradient struct {
	layer_index int
}
type Layer interface {
	evaluate(input []float64) (output []float64, err error)
	inputShape() int
}

type PerceptronLayer struct {
	next_layer  Layer
	input_shape int
	out_weights [][]float64
	biases      []float64
	act_fun     ActivationFunction
	initialized bool
}

type InputLayer struct {
	out_weights []float64
}

func (pl PerceptronLayer) inputShape() int {
	return pl.input_shape
}
func build_PerceptronLayer(neuron_count int, act_fun ActivationFunction) {
	pl := PerceptronLayer{}
	pl.input_shape = neuron_count
	pl.act_fun = act_fun
	pl.biases = make([]float64, neuron_count)
	for index := 0; index < neuron_count; index++ {
		pl.biases[index] = rand.Float64()

	}

}

func (pl PerceptronLayer) validate() error {
	if !pl.initialized {
		return errors.New("Layer is not initialized")
	}
	if pl.biases == nil {
		return errors.New("PerceptronLayer is labeled as initialized but has no biases")
	}
	if pl.biases == nil {
		return errors.New("PerceptronLayer is labeled as initialized but has no output_weights")
	}
	if pl.next_layer == nil {
		return errors.New("PerceptronLayer is labeled as initialized but has no output layer")
	}

}
func (pl PerceptronLayer) evaluate(input []float64) (output []float64, err error) {
	output = make([]float64, pl.next_layer.inputShape())

	neuron_output := make([]float64, len(input))
	for index, in := range input {
		neuron_output[index] = pl.act_fun.eval(in + pl.biases[index])
	}
	// the idea with the out_weights, is that the output weights are
	//incorporated into the layer. We do the summing in this layer as well.
	//For that reason, we have access to the next layer so that we can
	//find the input shape the next layer needs (equal to the len(biases))

	//for speed it will probably be useful to not have to count len(biases) every time, so we will store that in the input_shape variable

	for out_index := 0; out_index < pl.next_layer.inputShape(); out_index++ {
		for index, out := range neuron_output {
			output[out_index] += out * pl.out_weights[index][out_index]
		}
	}
	return output, nil
}

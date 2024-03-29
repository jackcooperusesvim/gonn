package tensorgo

import (
	"errors"
	"fmt"
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
	is_ready() (bool, error)
}

type OutputLayer struct {
	input_shape int
	biases      []float64
	act_fun     ActivationFunction
}

func build_OutputLayer(neuron_count int, act_fun ActivationFunction) OutputLayer {
	ol := OutputLayer{}
	ol.input_shape = neuron_count
	ol.act_fun = act_fun
	ol.biases = make([]float64, neuron_count)
	for index := 0; index < neuron_count; index++ {
		ol.biases[index] = rand.Float64()
	}
	return ol
}

func (ol OutputLayer) is_ready() (bool, error) {
	if ol.input_shape == len(ol.biases) {
		return false, errors.New("number of biases does not match the input shape")
	}
	return true, nil
}

func (ol OutputLayer) inputShape() int {
	return ol.input_shape
}

func (ol OutputLayer) evaluate(input []float64) (output []float64, err error) {
	if len(input) != ol.input_shape {
		return nil, errors.New(fmt.Sprintf("input does not match input shape \n\texpected input shape:%v\n\tactual input shape:%v", ol.input_shape, len(input)))
	}

	output = make([]float64, ol.input_shape)

	for i := 0; i < ol.input_shape; i++ {
		output = append(output, rand.Float64())
	}

	return output, nil
}

type PerceptronLayer struct {
	next_layer  Layer
	input_shape int
	out_weights [][]float64
	biases      []float64
	act_fun     ActivationFunction
	initialized bool
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

func (pl PerceptronLayer) inputShape() int {
	return pl.input_shape
}

func (pl PerceptronLayer) initialize(next_layer Layer) PerceptronLayer {
	pl.out_weights = make([][]float64, pl.input_shape)
	pl.next_layer = next_layer
	for i := 0; i < pl.input_shape; i++ {
		nodeslice := make([]float64, pl.next_layer.inputShape())
		for j := 0; j < pl.next_layer.inputShape(); j++ {
			nodeslice = append(nodeslice, rand.Float64())
		}
		pl.out_weights = append(pl.out_weights, nodeslice)
	}
	pl.initialized = true
	return pl
}

func (pl PerceptronLayer) is_ready() (bool, error) {
	if pl.biases == nil {
		return false, errors.New("PerceptronLayer has no biases / neurons")
	}
	if pl.input_shape != len(pl.biases) {
		return false, errors.New("PerceptronLayer input_shape does not match # of biases / neurons")
	}

	var err_base string
	var err_conn string
	if pl.initialized {
		err_base = "PerceptronLayer is labeled as initialized"
		err_conn = "but"
	} else {
		err_base = "PerceptronLayer is un-initialized"
		err_conn = "and"
	}
	if pl.next_layer == nil {
		return false, errors.New(fmt.Sprintf("%s %s has no output layer", err_base, err_conn))
	}
	if pl.out_weights == nil {
		return false, errors.New(fmt.Sprintf("%s %s has no output_weights", err_base, err_conn))
	}
	weight_current_layer_dim := len(pl.out_weights)
	if weight_current_layer_dim == pl.inputShape() {
		weight_next_layer_dim := len(pl.out_weights[0])
		if weight_next_layer_dim != pl.next_layer.inputShape() {
			return false, errors.New(fmt.Sprintf("Incorrect output shape: \n\tNext Layer Input: %v \n\tWeight Matrix output shape :%v", weight_next_layer_dim, pl.next_layer.inputShape()))
		}
	} else {
		return false, errors.New(fmt.Sprintf("Incorrect weight input shape: \n\tNodes: %v \n\tWeight Matrix input shape :%v", pl.inputShape(), weight_current_layer_dim))
	}

	return true, nil
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

package tensorgo

import (
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
)

type MultiLayeredPerceptron struct {
	layers   []PerceptronLayer
	outlayer OutputLayer
}

func (mlp MultiLayeredPerceptron) evaluate(input []float64) (output []float64, err error) {
	inputs := make([]float64, mlp.layers[0].inputShape())
	var outputs []float64
	for layerIndex := 0; layerIndex < len(mlp.layers)-1; layerIndex++ {
		outputs, err = mlp.layers[layerIndex].evaluate(inputs)
		if err != nil {
			return nil, err
		}

		inputs = outputs
	}
	return outputs, nil
}

func (mlp MultiLayeredPerceptron) link() MultiLayeredPerceptron {
	//FIX: There is something wrong here where the last layer touched by this loop fails to link properly
	// and throws an "uninitialized and no output layer"
	finalindex := len(mlp.layers) - 1
	for i := 0; i <= finalindex-1; i++ {
		mlp.layers[i] = mlp.layers[i].link(mlp.layers[i+1])
	}
	mlp.layers[finalindex] = mlp.layers[finalindex].link(mlp.outlayer)
	return mlp
}

func (mlp MultiLayeredPerceptron) inputShape() int {
	return mlp.layers[0].inputShape()
}

func (mlp MultiLayeredPerceptron) is_ready() (outb bool, erindex int, oute error) {
	for i, layer := range mlp.layers {
		outb, oute = layer.is_ready()
		if !outb {
			return outb, i, oute
		}
	}
	return outb, 0, oute
}

type ActivationFunction interface {
	eval(float64) float64
	derivative(float64) float64
}

type SigmoidActFun struct{}

func (saf SigmoidActFun) eval(in float64) float64 {
	return 1 / (1 + math.Exp(0-in))
}

func (saf SigmoidActFun) derivative(in float64) float64 {
	return saf.eval(in) * (1 - saf.eval(in))
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

func build_OutputLayer(neuron_count int, act_fun ActivationFunction) (ol OutputLayer) {
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
		output[i] = ol.act_fun.eval(input[i])
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

func build_PerceptronLayer(neuron_count int, act_fun ActivationFunction) (pl PerceptronLayer) {
	pl.input_shape = neuron_count
	pl.act_fun = act_fun
	pl.biases = make([]float64, neuron_count)
	for index := 0; index < neuron_count; index++ {
		pl.biases[index] = rand.Float64()
	}
	return pl

}

func (pl PerceptronLayer) inputShape() int {
	return pl.input_shape
}

func (pl PerceptronLayer) summaryStr() (out string) {
	out = fmt.Sprintf("Summary: \n\tinput_shape: %v\n\tcount of out_weights neuron connections: %v\n\tcount of out_weights output connections: %v\n\tcount of biases: %v\n\tnext layer input shape: %v\n\tinitialized? : %v\n\nweight summary:\n",
		pl.input_shape, len(pl.out_weights), len(pl.out_weights[0]), len(pl.biases), pl.next_layer.inputShape(), pl.initialized)

	for i, node := range pl.out_weights {
		out = fmt.Sprintf("%s\tnode #%v outputs\n", out, i)
		for j, outweight := range node {
			out = fmt.Sprintf("%s\t\tweight #%v : %v\n", out, j, outweight)
		}
	}
	return out
}

func (pl PerceptronLayer) link(next_layer Layer) PerceptronLayer {
	if pl.initialized {
		return pl
	}

	pl.out_weights = make([][]float64, pl.input_shape)
	pl.next_layer = next_layer
	for i := 0; i < pl.input_shape; i++ {
		pl.out_weights[i] = make([]float64, pl.next_layer.inputShape())
		for j := 0; j < pl.next_layer.inputShape(); j++ {
			pl.out_weights[i][j] = rand.Float64()
		}
	}
	pl.initialized = true
	// fmt.Printf(pl.summaryStr())
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

func InitMLP(node_count []int, output_shape int) (mlp MultiLayeredPerceptron, _ error) {
	saf := SigmoidActFun{}
	for _, nodes := range node_count {
		mlp.layers = append(mlp.layers, build_PerceptronLayer(nodes, saf))
	}

	mlp.outlayer = build_OutputLayer(output_shape, saf)
	mlp.link()
	_, _, err := mlp.is_ready()
	if err != nil {
		return mlp, err
	}
	return mlp, nil
}

func CompleteMLP() {
	saf := SigmoidActFun{}
	mlp := MultiLayeredPerceptron{}
	mlp.layers = []PerceptronLayer{
		build_PerceptronLayer(3, saf),
		build_PerceptronLayer(5, saf),
		build_PerceptronLayer(7, saf),
		build_PerceptronLayer(5, saf),
		build_PerceptronLayer(3, saf),
		build_PerceptronLayer(3, saf),
	}
	mlp.outlayer = build_OutputLayer(3, saf)
	mlp.link()
	ready, layerer, err := mlp.is_ready()
	if !ready {
		fmt.Printf("bad layer:%v\n", layerer)
		log.Fatal(err)
	}
	input := []float64{1, 1, 1}
	output, err := mlp.evaluate(input)
	if err != nil {
		log.Fatal(err)
	}
	for index, out := range output {
		fmt.Printf("index %v: %v\n", index, out)
	}
}

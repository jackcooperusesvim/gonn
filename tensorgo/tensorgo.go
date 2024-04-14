package tensorgo

import (
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
)

const sep = "/"
const supsep = "|"

type MultiLayeredPerceptron struct {
	layers   []PerceptronLayer
	outlayer OutputLayer
}
type layer_gradient struct {
	weight_gradient   []float64
	bias_gradient     []float64
	backprop_gradient []float64
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

func retnone() int {
	return -1
}
func mlp_from_binary([]byte) (mlp MultiLayeredPerceptron) {
	return
}

func (mlp MultiLayeredPerceptron) to_binary() ([]byte, error) {
	//TODO: ADD THE EXTRA INFORMATION: Output Layer, initialized, etc...

	// This is the function which turns model weights into a file, which can be later read to re-create the model.
	// The file is split into two main sections the header and parameter sections.
	//	Header: Contains important data for parsing the rest of the function
	//		Integers representing the size of each layer are held in byte form and are
	//		seperated by the number -1 in byte form, which functions as a seperator.
	//
	//		To seperate the header from the parameter, the header section ends with a seperator byte equivilant
	//			to the rune '|'.

	//	Parameter: Contains the data about the weights and biases of the model. These are seperated by the byte equivilant
	//		to the float64 420.0. Nice.
	//
	//		Weights: This section contains the layer weights. It is contiguous. Seperation has to be
	//			inferenced from the header
	//
	//		Biases: This section contains the layer biases. It is contiguous. Seperation has to be
	//			inferenced from the header

	supersep := byte('|')
	paramsep := Float64ToBytes(420)

	_, _, err := mlp.is_ready()
	if err != nil {
		return nil, err
	}
	var tot_weights int
	var tot_biases int
	for _, layer := range mlp.layers {
		tot_weights += len(layer.out_weights) * len(layer.out_weights[0])
	}

	filebytes := []byte{}
	headerbytes := []byte{}

	weightbytes := [][8]byte{}
	biasbytes := [][8]byte{}
	weightbytes = make([][8]byte, tot_weights)
	biasbytes = make([][8]byte, tot_biases)

	//arrange all the information by section
	for _, layer := range mlp.layers {

		headerbytes = append(headerbytes, byte(layer.inputShape()))

		size := len(layer.out_weights) * len(layer.out_weights[0])

		weightbytes = make([][8]byte, size)

		for _, bias := range layer.biases {
			biasbytes = append(biasbytes, Float64ToBytes(bias))
		}

		for _, subarr := range layer.out_weights {
			for _, weight := range subarr {
				weightbytes = append(weightbytes, Float64ToBytes(weight))
			}
		}

	}

	//compile sections
	copy(filebytes, headerbytes)
	filebytes = append(filebytes, supersep)

	for _, float := range weightbytes {
		filebytes = append(filebytes, float[:]...)
	}

	filebytes = append(filebytes, paramsep[:]...)
	for _, float := range biasbytes {
		filebytes = append(filebytes, float[:]...)
	}

	return filebytes, nil
}

func Float64ToBytes(f float64) [8]byte {
	out := [8]byte{}
	binary.PutUvarint(out[:], math.Float64bits(f))
	return out
}

func BytesToFloat64(b [8]byte) (float64, error) {
	uint, errint := binary.Uvarint(b[:])
	if errint < 0 {
		return 0, errors.New("Value is too large for binary.Uvarint")
	}
	if errint == 0 {
		return 0, errors.New("input buffer to small for binary.Uvarint. Size matters...")
	}
	return math.Float64frombits(uint), nil
}

func (mlp MultiLayeredPerceptron) calculate_gradients(next_layer_grad []layer_gradient) []layer_gradient {
}

func (l PerceptronLayer) calculate_gradients(next_layer_grad []layer_gradient) []layer_gradient {
}

func (mlp MultiLayeredPerceptron) link() MultiLayeredPerceptron {
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

//I figured it would be easier for the backprop if we just measured indivitual costs with a
//function and then added them together when needed

func IndividualCostMSE(actual []float64, expected []float64) []float64 {
	costs := make([]float64, len(actual))
	for index := range actual {
		costs[index] = (expected[index] - actual[index]) * (expected[index] - actual[index])
	}
	return costs
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

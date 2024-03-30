package tensorgo_test

import "testing"

func TestMLPCreation(t *testing.T){
	mlp = MultiLayeredPerceptron{}
	t.Logf("MultiLayeredPerceptron created.\nAdding input layer...\n")
	mlp.layers = []Layer{
	build_PerceptronLayer(3,SigmoidActFun),
	build_PerceptronLayer(3,SigmoidActFun),
	build_PerceptronLayer(5,SigmoidActFun),
	build_PerceptronLayer(5,SigmoidActFun),
	build_PerceptronLayer(3,SigmoidActFun),
	build_OutputLayer(3,SigmoidActFun)}
	for layerIndex:=0; layerIndex<len(mlp.layers)-1,layerIndex++{
		mlp[layerIndex].initialize(mlp[layerIndex])
	}
}

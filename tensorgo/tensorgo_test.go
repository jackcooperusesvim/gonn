package tensorgo_test

import (
	. "gonn/tensorgo"
	"testing"
)

func TestCompleteMLP(t *testing.T) {
	CompleteMLP()
}

func TestInitMLP(t *testing.T) {

	_, err := InitMLP([]int{9, 4, 4}, 3)
	if err != nil {
		t.Log("small model failed build")
		t.Fail()
	}
	_, err = InitMLP([]int{9, 6, 13, 25, 6, 1}, 6)
	if err != nil {
		t.Log("medium model failed build")
		t.Fail()
	}

	_, err = InitMLP([]int{9, 6, 13, 95, 6, 19, 6, 13, 95, 6, 1}, 20)
	if err != nil {
		t.Log("large model failed build")
		t.Fail()
	}
}

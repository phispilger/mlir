module {
func @test_invert(%arg0: tensor<*xf64>) {
	%0 = "bss2.invert"(%arg0) : (tensor<*xf64>) -> tensor<*xf64>
	%1 = "bss2.invert"(%0) : (tensor<*xf64>) -> tensor<*xf64>
	%2 = "bss2.invert"(%1) : (tensor<*xf64>) -> tensor<*xf64>
	"std.return"(%2) : (tensor<*xf64>) -> tensor<*xf64>
}
}
